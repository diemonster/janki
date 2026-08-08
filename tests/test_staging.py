"""Staging files, and the import path that diverts reading-less kanji rows.

A reading is part of the record ID, so a kanji row without one can never be
imported and then repaired — it has to wait in staging for a human.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from japanese_anki import cli
from japanese_anki import staging as staging_module
from japanese_anki.io import DataError, load_records
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import (
    StagingError,
    annotate,
    annotations,
    read_staging,
    write_staging,
)


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


def test_a_staging_file_is_written_through_the_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A staging file holds hand-edited readings that exist nowhere else — not
    # in git, not in the source CSV once it is gone. A direct write_text here
    # leaves no .tmp file either, so only the call itself proves the contract.
    path = tmp_path / "candidates.yaml"
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        staging_module,
        "atomic_write_text",
        lambda target, text: calls.append((target, text)),
    )

    write_staging(path, [_record()], {"source_file": "export.csv"})

    assert [target for target, _ in calls] == [path]
    assert yaml.safe_load(calls[0][1])["records"][0]["id"] == "word:話す:はなす"
    assert not path.exists()


@pytest.mark.parametrize("name", ["candidates.json", "candidates", "candidates.txt"])
def test_write_staging_refuses_a_suffix_nothing_could_read_back(
    tmp_path: Path, name: str
) -> None:
    # The content is YAML whatever the name says, and read_staging dispatches on
    # the suffix — so a .json staging file writes fine and every read of it
    # fails with a parse error far from the call that created it.
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
    assert "janki status --rebuild" in out
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
    staged_path.write_text("records: []\n", encoding="utf-8")

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert "it holds no rows" in out
    assert "review is finished" not in out
    assert "move its records" not in out


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
