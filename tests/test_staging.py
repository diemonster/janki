"""Staging files, and the import path that diverts reading-less kanji rows.

A reading is part of the record ID, so a kanji row without one can never be
imported and then repaired — it has to wait in staging for a human.
"""

from __future__ import annotations

import difflib
import errno
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import japanese_anki.io as janki_io
from japanese_anki import cli
from japanese_anki import staging as staging_module
from japanese_anki.io import DataError, load_records
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import (
    StagingError,
    annotate,
    annotations,
    prune_staging,
    read_staging,
    read_staging_text,
    rewrite_staging,
    write_staging,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(**overrides: object) -> VocabularyRecord:
    values: dict[str, object] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak", "to talk"],
        "part_of_speech": "verb",
        "examples": [ExampleSentence(japanese="毎日話す。", english="I speak every day.")],
        "tags": ["shirabe"],
        "source": SourceReference(
            type="shirabe", imported_from="export.csv", row=2, raw_fields={"Word": "話す"}
        ),
    }
    values.update(overrides)
    return VocabularyRecord(**values)  # type: ignore[arg-type]


# --- annotations ------------------------------------------------------------


def test_annotations_live_in_raw_fields_as_strings() -> None:
    staged = annotate(_record(), hold_reason="missing reading", already_known=True)

    assert staged.source.raw_fields == {
        "Word": "話す",
        "hold_reason": "missing reading",
        "already_known": "true",
    }
    assert annotations(staged) == {"hold_reason": "missing reading", "already_known": "true"}
    # The original record is untouched — annotate copies.
    assert _record().source.raw_fields == {"Word": "話す"}


def test_annotate_with_none_clears_an_annotation() -> None:
    staged = annotate(_record(), hold_reason="missing reading")

    resolved = annotate(staged, hold_reason=None)

    assert annotations(resolved) == {}
    assert resolved.source.raw_fields == {"Word": "話す"}


def test_annotate_rejects_names_outside_the_pinned_set() -> None:
    with pytest.raises(StagingError) as excinfo:
        annotate(_record(), reviewed_by="brandon")

    assert "reviewed_by" in str(excinfo.value)


# --- write_staging / read_staging -------------------------------------------


def test_round_trip_preserves_records_and_metadata(tmp_path: Path) -> None:
    staged = annotate(_record(), hold_reason="missing reading", suggested_reading="はなす")
    meta = {"source_file": "export.csv", "extracted_at": "2026-08-06", "model": "claude-opus-5"}

    write_staging(tmp_path / "candidates.yaml", [staged], meta)
    records, read_meta = read_staging(tmp_path / "candidates.yaml")

    assert read_meta == meta
    assert len(records) == 1
    # Round-trips through VocabularyRecord.from_dict with nothing lost.
    assert records[0].to_dict() == staged.to_dict()
    assert annotations(records[0]) == {
        "hold_reason": "missing reading",
        "suggested_reading": "はなす",
    }


def test_field_replacement_block_round_trips_the_old_values_it_authorizes(
    tmp_path: Path,
) -> None:
    """A staged rewrite is authority to replace one exact old field value.

    The proposal itself is deliberately not fingerprinted: a reviewer may edit
    the proposed wording in staging.  What must stay fixed while that review is
    open is the value it is about to replace in ``vocabulary.json``.
    """
    original = _record(usage_notes="old note")
    proposed = _record(
        meanings=["to converse"],
        examples=[ExampleSentence(japanese="友達と話す。", english="I talk with a friend.")],
        usage_notes="new note",
    )
    changes = {
        original.id: {
            "meanings": (original.meanings, proposed.meanings),
            "examples": (original.examples, proposed.examples),
            "usage_notes": (original.usage_notes, proposed.usage_notes),
        }
    }

    block = staging_module.field_replacement_block([original], changes)
    path = tmp_path / "ai.yaml"
    write_staging(
        path,
        [proposed],
        {staging_module.FIELD_REPLACEMENTS_KEY: block},
    )

    _records, meta = read_staging(path)
    assert meta[staging_module.FIELD_REPLACEMENTS_KEY] == block
    fields = block["records"][original.id]
    assert fields == {
        "meanings": staging_module.replacement_fingerprint(original, "meanings"),
        "examples": staging_module.replacement_fingerprint(original, "examples"),
        "usage_notes": staging_module.replacement_fingerprint(
            original, "usage_notes"
        ),
    }


def test_replacement_fingerprints_are_bound_to_the_record_and_field() -> None:
    """A digest cannot be moved to another record or another same-valued field."""
    first = _record(furigana="same", romaji="same")
    second = _record(
        id="word:聞く:きく", expression="聞く", reading="きく",
        furigana="same", romaji="same",
    )

    fingerprints = {
        staging_module.replacement_fingerprint(first, "furigana"),
        staging_module.replacement_fingerprint(first, "romaji"),
        staging_module.replacement_fingerprint(second, "furigana"),
    }

    assert len(fingerprints) == 3


@pytest.mark.parametrize("field", ["id", "expression", "reading", "source", "tags"])
def test_replacement_metadata_refuses_fields_that_merge_cannot_replace(
    field: str,
) -> None:
    item = _record()

    with pytest.raises(StagingError, match=field):
        staging_module.field_replacement_block(
            [item], {item.id: {field: ("old", "new")}}
        )


def test_the_file_is_a_records_mapping_load_records_already_understands(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"

    write_staging(path, [_record()], {"source_file": "export.csv", "review_notes": "check these"})

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(payload) == ["source_file", "review_notes", "records"]
    assert payload["records"][0]["id"] == "word:話す:はなす"
    # Japanese stays readable rather than \u-escaped, and the loader in io.py
    # reads the file as-is — no staging-aware loader needed.
    assert "話す" in path.read_text(encoding="utf-8")
    assert [record.id for record in load_records(path)] == ["word:話す:はなす"]


def test_write_staging_refuses_to_overwrite_review_edits(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    write_staging(path, [_record()], {"source_file": "export.csv"})
    edited = path.read_text(encoding="utf-8")

    with pytest.raises(StagingError) as excinfo:
        write_staging(path, [_record(id="word:食べる:たべる", expression="食べる")], {})

    assert str(path) in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == edited


def test_a_metadata_key_outside_the_contract_is_written_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # META_KEYS is the contract M3.3 and M2.6 write against. A warning is the
    # right strength: read_staging hands every non-`records` key back, so a note
    # a reviewer added by hand must survive a round trip — what this catches is
    # a *writer* inventing a key nothing downstream reads.
    path = tmp_path / "candidates.yaml"

    write_staging(path, [_record()], {"source_file": "export.csv", "reviewer": "me"})

    err = capsys.readouterr().err
    assert "reviewer" in err
    assert "source_file" not in err.split("is not one of")[0]
    _, meta = read_staging(path)
    assert meta["reviewer"] == "me"


def test_provider_is_recognized_staging_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "ai.yaml"

    write_staging(path, [_record()], {"provider": "anthropic"})

    assert capsys.readouterr().err == ""
    _, meta = read_staging(path)
    assert meta["provider"] == "anthropic"


def test_candidate_accounting_is_recognized_staging_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extract.yaml"
    accounting = {
        "version": 1,
        "parsed_candidate_count": 0,
        "canonical_record_count": 0,
        "unusable_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "collision_group_count": 0,
        "collision_groups": [],
        "candidate_accounting_fingerprint": "a" * 64,
    }

    write_staging(path, [_record()], {"candidate_accounting": accounting})

    assert capsys.readouterr().err == ""
    _, meta = read_staging(path)
    assert meta["candidate_accounting"] == accounting


def test_a_generated_review_run_id_is_recognized_and_round_trips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "ai.yaml"
    run_id = staging_module.new_review_run_id()

    write_staging(path, [_record()], {"review_run_id": run_id})

    assert capsys.readouterr().err == ""
    _, meta = read_staging(path)
    assert staging_module.review_run_id(meta) == run_id


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-uuid",
        "11111111-1111-1111-8111-111111111111",
        "11111111-1111-4111-1111-111111111111",
        "11111111-1111-4111-8111-11111111111A",
        42,
    ],
)
def test_write_staging_refuses_a_malformed_review_run_id(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "ai.yaml"

    with pytest.raises(StagingError, match="review-run-id-invalid"):
        write_staging(path, [_record()], {"review_run_id": value})


def test_read_staging_refuses_a_hand_edited_malformed_review_run_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ai.yaml"
    path.write_text("review_run_id: not-a-uuid\nrecords: []\n", encoding="utf-8")

    with pytest.raises(StagingError, match="review-run-id-invalid"):
        read_staging(path)


def test_force_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    write_staging(path, [_record()], {"source_file": "export.csv"})

    write_staging(
        path,
        [_record(id="word:食べる:たべる", expression="食べる", reading="たべる")],
        {"source_file": "other.csv"},
        force=True,
    )

    records, meta = read_staging(path)
    assert [record.id for record in records] == ["word:食べる:たべる"]
    assert meta == {"source_file": "other.csv"}


def test_metadata_cannot_shadow_the_records_key(tmp_path: Path) -> None:
    with pytest.raises(StagingError):
        write_staging(tmp_path / "candidates.yaml", [_record()], {"records": "nope"})


def test_read_staging_reports_a_malformed_file(tmp_path: Path) -> None:
    listed = tmp_path / "listed.yaml"
    listed.write_text("- id: word:話す:はなす\n", encoding="utf-8")
    empty = tmp_path / "no-records.yaml"
    empty.write_text("source_file: export.csv\n", encoding="utf-8")
    scalars = tmp_path / "scalars.yaml"
    scalars.write_text("records:\n  - 話す\n", encoding="utf-8")

    with pytest.raises(StagingError):
        read_staging(listed)
    with pytest.raises(StagingError):
        read_staging(empty)
    with pytest.raises(StagingError):
        read_staging(scalars)
    with pytest.raises(DataError):
        read_staging(tmp_path / "missing.yaml")


def test_a_new_staging_file_is_atomically_bound_to_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A staging file holds hand-edited readings that exist nowhere else — not
    # in git, not in the source CSV once it is gone. A direct write_text here
    # leaves no .tmp file either, so only the call itself proves the contract.
    path = tmp_path / "candidates.yaml"
    calls: list[tuple[Path, str, dict[str, object]]] = []

    def observed_write(target: Path, text: str, **kwargs: object) -> None:
        calls.append((target, text, kwargs))

    monkeypatch.setattr(
        staging_module,
        "atomic_write_text_bound",
        observed_write,
    )

    write_staging(path, [_record()], {"source_file": "export.csv"})

    assert [target for target, _text, _kwargs in calls] == [path]
    assert yaml.safe_load(calls[0][1])["records"][0]["id"] == "word:話す:はなす"
    assert calls[0][2] == {"expected_revision": None, "expected_absent": True}
    assert not path.exists()


def test_forced_staging_write_refuses_a_final_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidates.yaml"
    write_staging(path, [_record()], {"source_file": "export.csv"})
    outside = tmp_path / "owner-data.yaml"
    outside.write_bytes(b"must not change\n")
    real_write = staging_module.atomic_write_text_bound

    def swap_then_write(target: Path, text: str, **kwargs: object) -> None:
        Path(target).unlink()
        Path(target).symlink_to(outside)
        real_write(target, text, **kwargs)

    monkeypatch.setattr(staging_module, "atomic_write_text_bound", swap_then_write)

    with pytest.raises(DataError, match="non-regular|Bound target changed"):
        write_staging(
            path,
            [_record(expression="聞く", reading="きく")],
            {"source_file": "export.csv"},
            force=True,
        )

    assert path.is_symlink()
    assert outside.read_bytes() == b"must not change\n"


def test_expected_revision_retains_an_edit_made_at_the_atomic_commit_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidates.yaml"
    confirmed = b"confirmed review\n"
    edited = b"new human edit\n"
    path.write_bytes(confirmed)
    real_exchange = getattr(janki_io, "_exchange_entries", None)
    injected = False

    def edit_then_exchange(*args: object, **kwargs: object) -> None:
        nonlocal injected
        path.write_bytes(edited)
        injected = True
        assert real_exchange is not None
        real_exchange(*args, **kwargs)

    monkeypatch.setattr(
        janki_io, "_exchange_entries", edit_then_exchange, raising=False
    )

    with pytest.raises(DataError, match="both names were retained"):
        janki_io.atomic_write_text_bound(
            path,
            "paid replacement\n",
            expected_revision=hashlib.sha256(confirmed).hexdigest(),
        )

    assert injected
    assert path.read_bytes() == b"paid replacement\n"
    retained = list(tmp_path.glob(f".{path.name}.*.tmp"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == edited
    with pytest.raises(DataError, match="bound evidence"):
        janki_io.read_bytes_bound(path)


def test_expected_revision_refuses_a_parent_detached_at_the_atomic_commit_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    path = staging_dir / "candidates.yaml"
    confirmed = b"confirmed review\n"
    path.write_bytes(confirmed)
    held = tmp_path / "held-staging"
    real_exchange = getattr(janki_io, "_exchange_entries", None)
    swapped = False

    def detach_then_exchange(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            staging_dir.rename(held)
            staging_dir.mkdir()
            swapped = True
        assert real_exchange is not None
        real_exchange(*args, **kwargs)

    monkeypatch.setattr(
        janki_io, "_exchange_entries", detach_then_exchange, raising=False
    )

    with pytest.raises(DataError, match="directory changed"):
        janki_io.atomic_write_text_bound(
            path,
            "paid replacement\n",
            expected_revision=hashlib.sha256(confirmed).hexdigest(),
        )

    assert swapped
    assert not path.exists()
    assert (held / path.name).read_bytes() == confirmed


def test_expected_revision_retains_both_files_if_rollback_cannot_be_proven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidates.yaml"
    confirmed = b"confirmed review\n"
    replacement = b"paid replacement\n"
    path.write_bytes(confirmed)
    real_exchange = janki_io._exchange_entries
    calls = 0

    def edit_then_block_rollback(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_exchange(*args, **kwargs)
            return
        raise OSError(errno.EIO, "rollback unavailable")

    monkeypatch.setattr(janki_io, "_exchange_entries", edit_then_block_rollback)

    def fail_commit_marker(*_args: object, **_kwargs: object) -> str:
        raise OSError(errno.EIO, "commit marker unavailable")

    monkeypatch.setattr(janki_io, "_move_cas_marker", fail_commit_marker)

    with pytest.raises(DataError, match="both names were retained"):
        janki_io.atomic_write_text_bound(
            path,
            replacement.decode("utf-8"),
            expected_revision=hashlib.sha256(confirmed).hexdigest(),
        )

    retained = list(tmp_path.glob(f".{path.name}.*.tmp"))
    assert calls == 2
    assert path.read_bytes() == replacement
    assert len(retained) == 1
    assert retained[0].read_bytes() == confirmed


def test_expected_revision_fails_closed_without_atomic_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidates.yaml"
    confirmed = b"confirmed review\n"
    path.write_bytes(confirmed)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOTSUP, "atomic exchange unavailable")

    monkeypatch.setattr(janki_io, "_exchange_entries", unavailable)

    with pytest.raises(DataError, match="atomic exchange unavailable"):
        janki_io.atomic_write_text_bound(
            path,
            "paid replacement\n",
            expected_revision=hashlib.sha256(confirmed).hexdigest(),
        )

    assert path.read_bytes() == confirmed
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("operation", ["rewrite", "coverage", "prune"])
def test_in_place_staging_mutations_preserve_a_final_seam_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / "candidates.yaml"
    first = _record()
    records = [first]
    meta: dict[str, object] = {"source_file": "export.csv"}
    if operation == "coverage":
        meta["coverage"] = {}
    if operation == "prune":
        records.append(
            _record(
                id="word:聞く:きく",
                expression="聞く",
                reading="きく",
            )
        )
    write_staging(path, records, meta)
    captured = path.read_bytes()
    human = captured + b"# final human edit\n"
    real_write = staging_module.atomic_write_text_bound
    observed_revision = ""

    def edit_then_write(target: Path, text: str, **kwargs: object) -> None:
        nonlocal observed_revision
        observed_revision = str(kwargs.get("expected_revision", ""))
        path.write_bytes(human)
        real_write(target, text, **kwargs)

    monkeypatch.setattr(staging_module, "atomic_write_text_bound", edit_then_write)

    with pytest.raises(DataError, match="changed content"):
        if operation == "rewrite":
            staging_module.rewrite_staging(
                path,
                [replace(first, meanings=["changed meaning"])],
            )
        elif operation == "coverage":
            staging_module.record_coverage_approval(
                path,
                {"authority": "human", "reason": "reviewed"},
            )
        else:
            staging_module.prune_staging(path, [True, False])

    assert observed_revision == hashlib.sha256(captured).hexdigest()
    assert path.read_bytes() == human


def test_coverage_approval_uses_one_exact_expected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidates.yaml"
    write_staging(path, [_record()], {"source_file": "export.csv", "coverage": {}})
    rendered = path.read_bytes()
    transient = rendered + b"# transient workbench edit\n"
    expected_revision = hashlib.sha256(rendered).hexdigest()

    path.write_bytes(transient)
    with pytest.raises(StagingError, match="coverage-review-stale"):
        staging_module.record_coverage_approval(
            path,
            {"authority": "human", "reason": "reviewed"},
            expected_revision=expected_revision,
        )
    assert path.read_bytes() == transient
    path.write_bytes(rendered)

    real_load = staging_module._load_document_snapshot

    def load_transient(target: Path) -> tuple[object, str, str]:
        target.write_bytes(transient)
        try:
            return real_load(target)
        finally:
            target.write_bytes(rendered)

    monkeypatch.setattr(
        staging_module,
        "_load_document_snapshot",
        load_transient,
    )

    with pytest.raises(DataError, match="changed content"):
        staging_module.record_coverage_approval(
            path,
            {"authority": "human", "reason": "reviewed"},
        )

    assert path.read_bytes() == rendered
    _records, meta = read_staging(path)
    assert "approval" not in meta["coverage"]


@pytest.mark.parametrize("operation", ["rewrite", "coverage", "prune"])
def test_in_place_staging_mutations_report_a_file_removed_after_review(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / "candidates.yaml"
    first = _record()
    records = [first]
    meta: dict[str, object] = {"source_file": "export.csv"}
    if operation == "coverage":
        meta["coverage"] = {}
    if operation == "prune":
        records.append(
            _record(id="word:聞く:きく", expression="聞く", reading="きく")
        )
    write_staging(path, records, meta)
    reviewed, _reviewed_meta = read_staging(path)
    path.unlink()

    with pytest.raises(StagingError, match="no longer exists"):
        if operation == "rewrite":
            staging_module.rewrite_staging(path, reviewed)
        elif operation == "coverage":
            staging_module.record_coverage_approval(
                path,
                {"authority": "human", "reason": "reviewed"},
            )
        else:
            staging_module.prune_staging(path, [True, False])


@pytest.mark.parametrize("name", ["candidates.json", "candidates", "candidates.txt"])
def test_write_staging_refuses_a_name_outside_the_yaml_review_contract(
    tmp_path: Path, name: str
) -> None:
    # The repository contract keeps every review artifact visibly YAML even
    # though the bound reader parses the captured wire rather than its suffix.
    with pytest.raises(StagingError) as error:
        write_staging(tmp_path / name, [_record()], {})

    assert "YAML" in str(error.value)
    assert not (tmp_path / name).exists()


def test_an_empty_records_list_reads_back_as_no_records(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    write_staging(path, [], {"source_file": "export.csv"})

    records, meta = read_staging(path)

    assert records == []
    assert meta == {"source_file": "export.csv"}


# --- janki import-shirabe: the needs-reading diversion -----------------------

NEEDS_READING_CSV = "Word,Reading,Definition\n話す,,to speak\n電話,でんわ,telephone\n"


def _project(tmp_path: Path, csv_text: str = NEEDS_READING_CSV) -> tuple[Path, Path]:
    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\nstaging_dir = "staging"\n',
        encoding="utf-8",
    )
    source = tmp_path / "export.csv"
    source.write_text(csv_text, encoding="utf-8")
    return tmp_path, source


def _stored_ids(root: Path) -> list[str]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return [record["id"] for record in payload]


def test_import_stages_reading_less_kanji_rows_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    out = capsys.readouterr().out
    assert str(staged_path) in out
    assert "reading janki cannot use" in out
    assert "話す contains kanji but has no reading" in out
    # The review signal is named, so a reviewer can reach it without guessing.
    assert f"janki validate {staged_path}" in out

    # The malformed row never reaches vocabulary.json.
    assert _stored_ids(root) == ["word:電話:でんわ"]

    records, meta = read_staging(staged_path)
    assert [record.id for record in records] == ["word:話す:"]
    assert annotations(records[0]) == {"hold_reason": "missing reading"}
    assert meta["source_file"] == "export.csv"
    assert "extracted_at" in meta and "review_notes" in meta


def test_import_never_overwrites_an_existing_needs_reading_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.parent.mkdir(parents=True)
    reviewed = "records:\n  - id: word:話す:はなす\n    expression: 話す\n    reading: はなす\n"
    staged_path.write_text(reviewed, encoding="utf-8")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    # Hand edits survive, the count and path are reported, and the rest of the
    # import still lands.
    assert staged_path.read_text(encoding="utf-8") == reviewed
    assert "Held 1 row(s)" in out
    assert str(staged_path) in out
    assert _stored_ids(root) == ["word:電話:でんわ"]


def test_a_resolved_staging_file_is_told_it_is_finished_not_to_resolve_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The nag with no exit. Re-importing the same CSV can never consume the
    # staging file, so "Resolve that file, then re-run this import" repeats
    # forever over a file that is already resolved — while the file's own
    # review_notes say to keep it. Once `validate` is happy with it, the import
    # says the one thing that ends the loop.
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_text(
        "records:\n"
        "  - id: word:話す:はなす\n"
        "    expression: 話す\n"
        "    reading: はなす\n"
        "    meanings: [to speak]\n",
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "Its review is finished" in out
    # The one exit that exists, and no competing manual recipe beside it.
    assert "janki promote" in out
    assert "status --rebuild" not in out
    assert "Resolve that file" not in out


def test_a_staging_file_with_work_left_is_still_asked_to_be_resolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)
    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0
    capsys.readouterr()

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "Resolve that file, then re-run this import." in out
    assert "Its review is finished" not in out


@pytest.mark.parametrize(
    "content",
    [
        "records: not-a-list\n",  # read_staging refuses it
        "records:\n  - [not, a, mapping]\n",  # likewise, one level down
        "records:\n  - id: word:話す:\n    expression: 話す\n",  # validate refuses it
    ],
)
def test_a_staging_file_that_cannot_be_read_keeps_the_cautious_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    # The branch that decides between "keep this file" and "move its records
    # into your records file and delete it", over a file of hand-typed readings
    # nothing can regenerate. Any failure to read has to land on the advice
    # that keeps it: flipping `except JankiError: return False` to `return
    # True` used to leave the whole suite green.
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_text(content, encoding="utf-8")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "Resolve that file, then re-run this import." in out
    assert "review is finished" not in out
    assert "delete it" not in out
    assert staged_path.read_text(encoding="utf-8") == content


def test_a_staging_file_with_no_rows_is_not_called_a_finished_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # README step 3 is "delete the rows not worth keeping"; a reviewer who kept
    # none leaves `records: []`. `validate` calls that a warning, not an error,
    # so it used to read as a finished review — and the import told its owner to
    # move records that do not exist.
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.parent.mkdir(parents=True)
    original = (
        "source_file: export.csv\n"
        "extracted_at: '2026-08-20'\n"
        "review_notes: Fill the missing reading, then promote.\n"
        "records: []\n"
    )
    staged_path.write_text(original, encoding="utf-8")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "it holds no rows" in out
    assert "tracked staging data" in out
    assert "no safe automatic completion route" in out
    assert "Delete it" not in out
    assert "review is finished" not in out
    assert "move its records" not in out
    assert staged_path.read_text(encoding="utf-8") == original


def test_a_finished_review_is_described_the_way_validate_reports_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # "'janki validate' finds nothing wrong with it" is not what the user sees
    # when they run it: validate reports warnings as well as errors, and this
    # branch only ever checked for errors.
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_text(
        "records:\n"
        "  - id: word:話す:はなす\n"
        "    expression: 話す\n"
        "    reading: はなす\n"
        "    meanings: [to speak]\n",
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "'janki validate' reports no errors for it" in out
    assert "finds nothing wrong" not in out


def test_import_without_reading_less_rows_writes_no_staging_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path, "Word,Reading,Definition\n電話,でんわ,telephone\n")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    assert not (root / "staging").exists()


def test_the_staging_notes_lead_to_a_state_validate_accepts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The instructions embedded in every staging file have to terminate. Filling
    # in the reading alone leaves the malformed `id:` behind, and validate keeps
    # erroring on the very review the reviewer just performed — so the notes
    # must say to delete the id line, and doing so must actually clear it.
    root, source = _project(tmp_path)
    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    capsys.readouterr()

    payload = yaml.safe_load(staged_path.read_text(encoding="utf-8"))
    notes = payload["review_notes"]
    assert "'id:' line" in notes
    assert "janki validate" in notes

    # Follow them literally: fill in the reading, drop the id, keep the row.
    for record in payload["records"]:
        record["reading"] = "はなす"
        del record["id"]
    staged_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

    assert cli.main(["--root", str(root), "validate", str(staged_path)]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    assert [record.id for record in load_records(staged_path)] == ["word:話す:はなす"]


def test_a_staging_failure_leaves_the_records_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Staging runs before the records are written. Otherwise the command
    # replaces vocabulary.json, then fails to stage, then exits non-zero — so
    # the caller is told the import did not happen while the held rows exist
    # nowhere on disk at all.
    root, source = _project(tmp_path)
    (root / "staging").write_text("not a directory\n", encoding="utf-8")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 1

    assert "error:" in capsys.readouterr().err
    assert not (root / "vocabulary.json").exists()


def test_a_directory_at_the_staging_path_is_reported_as_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A directory `exists()`, so the review-edits branch claimed to be
    # protecting hand edits that cannot be there while the held rows were
    # silently never staged and the import still exited 0.
    root, source = _project(tmp_path)
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"
    staged_path.mkdir(parents=True)

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 1

    err = capsys.readouterr().err
    assert "is a directory" in err
    assert "review edits" not in err
    assert not (root / "vocabulary.json").exists()
    assert "needs-reading" not in capsys.readouterr().out


def test_the_staging_notes_give_an_exit_that_exists_today(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every instruction in the notes has to be one the reviewer can actually
    # run today. Before M3.4 that meant naming the three manual steps; now that
    # 'janki promote' exists it means naming that, and *not* leaving the manual
    # recipe behind as a second, diverging set of instructions.
    root, source = _project(tmp_path)
    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0
    capsys.readouterr()
    staged_path = root / "staging" / "shirabe-export-needs-reading.yaml"

    notes = yaml.safe_load(staged_path.read_text(encoding="utf-8"))["review_notes"]

    assert "janki validate" in notes
    assert "janki promote" in notes
    assert "keep this file" not in notes
    # No forward-reference to a command that now exists, and no leftover manual
    # recipe competing with it.
    assert "Milestone" not in notes
    assert "status --rebuild" not in notes


def test_rewrite_touches_only_the_row_whose_field_changed(tmp_path: Path) -> None:
    """`rewrite_staging` promises to leave the rest of the file "byte-for-byte
    as it was". It did not: every staging file is created by `write_staging`
    through PyYAML, which spells an absent value `null`, while ruamel's
    round-trip representer spells it as an empty scalar — so re-dumping
    rewrote that line on every record, and annotating one row produced a diff
    touching rows nobody edited.
    """
    records = [
        _record(id="word:走る:はしる", expression="走る", reading="はしる",
                meanings=["to run"]),
        _record(id="word:食べる:たべる", expression="食べる", reading="たべる",
                meanings=["to eat"]),
        _record(id="word:飲む:のむ", expression="飲む", reading="のむ",
                meanings=["to drink"]),
    ]
    path = tmp_path / "source.pdf.yaml"
    write_staging(path, records, {"source_file": "source.pdf"})
    before = path.read_text(encoding="utf-8").split("\n")

    loaded, _meta = read_staging(path)
    edited = list(loaded)
    edited[1] = replace(edited[1], meanings=["to eat (corrected)"])
    rewrite_staging(path, edited)
    after = path.read_text(encoding="utf-8").split("\n")

    changed = [
        line
        for line in difflib.unified_diff(before, after, lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert changed == ["-  - to eat", "+  - to eat (corrected)"], changed


def test_the_two_yaml_writers_agree_byte_for_byte(tmp_path: Path) -> None:
    """Staging files are *created* by PyYAML and *edited* by ruamel, and the
    two must produce identical bytes for identical data.

    They did not. Both wrapped at width 100 but chose different break points,
    so re-emitting a long plain scalar re-folded it and left a trailing space
    at the break. A bare load-and-dump with nothing edited changed 54 lines of
    a real staging file — which made `rewrite_staging`'s promise to leave the
    rest "byte-for-byte as it was" untrue, and churned the diff of every
    promotion, since `promote` calls both `rewrite_staging` and
    `prune_staging`. Neither writer folds now; both read `STAGING_YAML_WIDTH`.
    """
    long_note = (
        "Used when the speaker or an in-group member gives something to "
        "someone else. The receiver is not the speaker, and the giver is "
        "marked with the particle が in a neutral description of the event."
    )
    records = [_record(usage_notes=long_note)]
    path = tmp_path / "source.pdf.yaml"
    write_staging(path, records, {"source_file": "source.pdf"})
    written = path.read_text(encoding="utf-8")

    # PyYAML wrote it; a ruamel round trip must reproduce it exactly.
    document = staging_module._load_document(path)
    buffer = io.StringIO()
    staging_module._parser().dump(document, buffer)
    assert buffer.getvalue() == written

    # ...and so must every writer built on that round trip.
    loaded, _meta = read_staging(path)
    rewrite_staging(path, loaded)
    assert path.read_text(encoding="utf-8") == written
    prune_staging(path, [True])
    assert path.read_text(encoding="utf-8") == written

    assert long_note in written, "a long scalar stays on one line"
    assert not any(
        line.rstrip() != line for line in written.split("\n")
    ), "no writer may leave trailing whitespace"


def test_both_writers_read_the_same_width_constant() -> None:
    """The constant exists so the two cannot drift apart again; a literal in
    either writer would reintroduce the divergence silently."""
    source = (
        Path(staging_module.__file__).read_text(encoding="utf-8")
    )
    assert "width=STAGING_YAML_WIDTH" in source
    assert "parser.width = STAGING_YAML_WIDTH" in source
    assert "width=100" not in source


def test_read_staging_text_parses_with_libyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The largest YAML janki reads is a staging file, and it reads them all per turn."""
    seen: list[object] = []
    real = yaml.load

    def spy(stream: object, Loader: object) -> object:  # noqa: N803 - PyYAML's name
        seen.append(Loader)
        return real(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "load", spy)

    records, meta = read_staging_text(
        "source_file: lesson.pdf\n"
        "records:\n"
        "  - id: word:話す:はなす\n"
        "    expression: 話す\n"
        "    reading: はなす\n"
        "    meaning: to speak\n"
    )

    assert [record.expression for record in records] == ["話す"]
    assert meta["source_file"] == "lesson.pdf"
    assert seen == [yaml.CSafeLoader]


def test_repository_staging_and_deck_files_parse_identically_under_both_loaders() -> None:
    """libyaml is a faster scanner, not a different dialect — proven on real files.

    The fixture is this checkout's own staging and deck YAML rather than an
    invented document, because the risk being ruled out is a construction the
    project actually writes and nobody thought to imagine.
    """
    paths = [
        *sorted((REPO_ROOT / "data" / "staging").rglob("*.yaml")),
        *sorted((REPO_ROOT / "data" / "decks").rglob("*.yaml")),
    ]
    assert paths, "the repository ships the staging and deck files this reads"

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert yaml.load(text, Loader=janki_io.YAML_LOADER) == yaml.load(
            text, Loader=yaml.SafeLoader
        ), path
