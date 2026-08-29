"""The exact-approval write transaction.

These are the tests that survived W2b's fold. The localhost page they were
written against is deleted; what they still cover is the part that mattered —
that an approval binds the exact sentences shown, that a stale snapshot or a
swapped symlink refuses before either write, that the two writes are ordered
so a failure between them reports precisely which one landed, and that the
locks are taken in a deterministic order. The workbench surface that calls
this is tested in ``tests/test_workbench.py``.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from japanese_anki import extract, patterns, staging
from japanese_anki import io as data_io
from japanese_anki.io import atomic_write_text_bound
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    EXAMPLE_AUTHORITY_STAGING,
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
    example_accepted,
    set_example_flags,
)
from japanese_anki.workbench import review as panel_module
from japanese_anki.workbench.review import (
    PanelRequestError,
    ReviewPanel,
    ReviewPanelError,
    StaleReviewError,
)

RUN_ID = "11111111-1111-4111-8111-111111111111"
OTHER_RUN_ID = "22222222-2222-4222-8222-222222222222"


@dataclass(frozen=True)
class PanelFiles:
    staging_dir: Path
    staging_path: Path
    patterns_path: Path
    records: tuple[VocabularyRecord, ...]
    pattern_set: patterns.PatternSet


def _provenance() -> dict[str, object]:
    return {
        "source_sha256": "1" * 64,
        "mode": "prose",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "response_schema_version": 3,
        "system_prompt_fingerprint": "2" * 64,
        "style_guide_fingerprint": "3" * 64,
        "user_prompt_fingerprint": "4" * 64,
        "response_schema_fingerprint": "5" * 64,
        "request_fingerprint": "6" * 64,
    }


def _record(name: str, reading: str, *, authority: str = "") -> VocabularyRecord:
    raw_fields = {
        "page": "7",
        "context": "<img src=x onerror=alert(1)>",
        "inclusion_reason": "The source explicitly glosses this word.",
        "provisional_fields": "meanings:machine-owned",
    }
    if authority:
        raw_fields[EXAMPLE_AUTHORITY_KEY] = authority
    return VocabularyRecord(
        id=f"word:{name}:{reading}",
        expression=name,
        reading=reading,
        furigana=f"{name}[{reading}]",
        romaji="neko",
        meanings=["cat <pet>"],
        part_of_speech="noun",
        verb_group="none",
        transitivity="intransitive",
        examples=[
            ExampleSentence(
                japanese=f"{name}<script>alert(1)</script>",
                furigana=f"{name}[{reading}]です。",
                romaji="neko desu",
                english="It is a cat.",
                audio="example.mp3",
                spoken_japanese=f"{reading}<exact>.",
                register="polite",
            ),
            ExampleSentence(
                japanese=f"{name}だよ。",
                furigana=f"{name}[{reading}]だよ。",
                romaji="neko da yo",
                english="It's a cat.",
                register="casual",
            ),
        ],
        conjugations={"test": "<form>"},
        tags=["lesson<tag>"],
        usage_notes="Use <carefully>.",
        audio="word.mp3",
        image="image.png",
        pitch_accent=["LH"],
        audio_accent="LH",
        frequency_rank=1234,
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            row=8,
            raw_fields=raw_fields,
        ),
    )


def _panel_files(
    tmp_path: Path,
    *,
    records: tuple[VocabularyRecord, ...] | None = None,
    reviewed: bool = False,
) -> PanelFiles:
    source = "lesson.pdf"
    provenance = _provenance()
    proposed = patterns.PatternSet(
        source=source,
        kind="lesson",
        title="Staged title",
        patterns=(patterns.Pattern("staged template", "staged gloss"),),
    )
    staged = replace(
        patterns.with_prompt_provenance(proposed, provenance),
        review_run_id=RUN_ID,
    )
    current = replace(
        staged,
        title="Current <corrected> title",
        patterns=(
            patterns.Pattern(
                "<script>current template</script>",
                "current <gloss>",
                ("例文<script>alert(2)</script>",),
                "page <9>",
            ),
        ),
        reviewed=reviewed,
    )
    result = extract.ExtractionResult(candidates=(), source_units=(), model_reported_unit_count=0)
    meta = {
        "source_file": source,
        "extracted_at": "2026-08-20",
        "model": "claude-opus-5",
        "review_run_id": RUN_ID,
        "prompt_provenance": provenance,
        "pattern_set": staged.to_dict(),
        "coverage": extract.coverage_block(
            result,
            source_sha256=str(provenance["source_sha256"]),
            mode=None,
        ),
    }
    rows = (
        records
        if records is not None
        else (
            _record("猫", "ねこ"),
            _record("犬", "いぬ"),
        )
    )
    staging_dir = tmp_path / "data" / "staging"
    staging_path = staging_dir / "lesson.pdf.yaml"
    patterns_path = tmp_path / "data" / "patterns.json"
    staging.write_staging(staging_path, rows, meta)
    patterns.save_store(patterns_path, {source: current})
    return PanelFiles(staging_dir, staging_path, patterns_path, rows, current)


def _open(files: PanelFiles) -> ReviewPanel:
    return ReviewPanel.open(
        files.staging_path,
        staging_dir=files.staging_dir,
        patterns_path=files.patterns_path,
    )


def test_selected_record_binds_exact_examples_and_unselected_stays_untouched(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    before_patterns = files.patterns_path.read_bytes()
    panel = _open(files)

    outcome = panel.submit(record_ids=[files.records[0].id], review_patterns=False)

    after, _meta = staging.read_staging(files.staging_path)
    expected = set_example_flags(
        files.records[0],
        EXAMPLE_AUTHORITY_KEY,
        (example.japanese for example in files.records[0].examples if example.japanese),
    )
    assert after == [expected, files.records[1]]
    assert after[0].source.raw_fields[EXAMPLE_AUTHORITY_KEY] != EXAMPLE_AUTHORITY_STAGING
    assert all(example_accepted(after[0], example) for example in after[0].examples)
    changed = replace(
        after[0].examples[0],
        japanese=after[0].examples[0].japanese + " changed",
    )
    assert not example_accepted(after[0], changed)
    assert files.patterns_path.read_bytes() == before_patterns
    assert outcome.accepted_record_ids == (files.records[0].id,)
    assert not outcome.pattern_reviewed


def test_captured_wire_authority_update_is_one_surgical_line_on_rich_staging(
    tmp_path: Path,
) -> None:
    stay = _record("泊まる", "とまる")
    files = _panel_files(tmp_path, records=(stay, _record("犬", "いぬ")))
    wire = files.staging_path.read_text(encoding="utf-8")
    wire = wire.replace(
        "- id: word:泊まる:とまる\n  expression: 泊まる\n",
        "- expression: 泊まる\n",
        1,
    )
    wire = wire.replace(
        "records:\n",
        "# preserve this reviewer comment\n"
        "review_notes: >-\n"
        "  This deliberately long metadata value must retain its exact wrapping "
        "while one authority line is inserted far below it.\n"
        "records:\n",
        1,
    )
    wire = wire.replace(
        "      provisional_fields: meanings:machine-owned\n",
        "      provisional_fields: meanings:machine-owned\n"
        "      # keep this raw-fields comment byte for byte\n",
        1,
    )
    captured, captured_meta = staging.read_staging_text(
        wire,
        source=str(files.staging_path),
    )
    updated = list(captured)
    updated[0] = set_example_flags(
        captured[0],
        EXAMPLE_AUTHORITY_KEY,
        (example.japanese for example in captured[0].examples if example.japanese),
    )

    rendered = staging.render_example_authority_updates(
        wire,
        updated,
        source=str(files.staging_path),
    )

    authority = updated[0].source.raw_fields[EXAMPLE_AUTHORITY_KEY]
    inserted = f"      example_authority: {authority}\n"
    assert rendered.count(inserted) == 1
    assert not inserted.rstrip("\n").endswith(" ")
    assert rendered.replace(inserted, "", 1) == wire
    reparsed, reparsed_meta = staging.read_staging_text(
        rendered,
        source=str(files.staging_path),
    )
    assert reparsed == updated
    assert reparsed_meta == captured_meta


@pytest.mark.parametrize("shape", ["flow", "duplicate"])
def test_captured_wire_authority_update_refuses_ambiguous_yaml_shapes(
    tmp_path: Path, shape: str
) -> None:
    files = _panel_files(tmp_path, records=(_record("猫", "ねこ"),))
    wire = files.staging_path.read_text(encoding="utf-8")
    if shape == "flow":
        start = wire.index("    raw_fields:\n")
        wire = wire[:start] + "    raw_fields: {page: '7'}\n"
    else:
        wire = wire.replace(
            "      page: '7'\n",
            "      page: '7'\n      page: duplicate\n",
            1,
        )
    captured, _meta = staging.read_staging_text(wire, source=str(files.staging_path))
    updated = [
        set_example_flags(
            captured[0],
            EXAMPLE_AUTHORITY_KEY,
            (example.japanese for example in captured[0].examples if example.japanese),
        )
    ]

    with pytest.raises(staging.StagingError, match="block-style|duplicate"):
        staging.render_example_authority_updates(
            wire,
            updated,
            source=str(files.staging_path),
        )


def test_pattern_review_changes_only_the_current_store_entry(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    before_staging = files.staging_path.read_bytes()

    outcome = _open(files).submit(record_ids=[], review_patterns=True)

    assert files.staging_path.read_bytes() == before_staging
    stored = patterns.load_store(files.patterns_path)
    assert stored == {files.pattern_set.source: replace(files.pattern_set, reviewed=True)}
    assert outcome.accepted_record_ids == ()
    assert outcome.pattern_reviewed


def test_pattern_review_surgically_changes_only_its_captured_boolean(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    raw = json.loads(files.patterns_path.read_text(encoding="utf-8"))
    raw["lesson.pdf"]["owner_extension"] = {"spacing": [1, 2, 3]}
    raw["unrelated.pdf"] = {
        "kind": "unknown",
        "title": "leave this exact wire alone",
        "reviewed": False,
        "patterns": [],
        "owner_extension": "preserve me",
    }
    captured = json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n"
    files.patterns_path.write_text(captured, encoding="utf-8")
    entry_start = captured.index('"lesson.pdf":')
    token_start = captured.index('"reviewed":false', entry_start) + len('"reviewed":')
    expected = captured[:token_start] + "true" + captured[token_start + len("false") :]

    outcome = _open(files).submit(record_ids=[], review_patterns=True)

    assert outcome.pattern_reviewed
    assert files.patterns_path.read_text(encoding="utf-8") == expected
    assert expected.replace('"reviewed":true', '"reviewed":false', 1) == captured


@pytest.mark.parametrize("changed", ["staging", "patterns"])
def test_any_stale_snapshot_refuses_before_either_write(tmp_path: Path, changed: str) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    if changed == "staging":
        files.staging_path.write_bytes(files.staging_path.read_bytes() + b"# changed\n")
    else:
        files.patterns_path.write_bytes(files.patterns_path.read_bytes() + b" \n")
    stale_staging = files.staging_path.read_bytes()
    stale_patterns = files.patterns_path.read_bytes()

    with pytest.raises(StaleReviewError, match="changed"):
        panel.submit(record_ids=[files.records[0].id], review_patterns=True)

    assert files.staging_path.read_bytes() == stale_staging
    assert files.patterns_path.read_bytes() == stale_patterns


def test_malformed_nested_staged_pattern_answer_refuses_the_panel(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    records, meta = staging.read_staging(files.staging_path)
    malformed = dict(meta["pattern_set"])
    malformed["patterns"] = "not a pattern list"
    meta["pattern_set"] = malformed
    staging.write_staging(files.staging_path, records, meta, force=True)

    with pytest.raises(ReviewPanelError, match="nested staged pattern"):
        _open(files)


def test_source_file_must_be_the_exact_store_key_not_a_normalized_path(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    staged = files.staging_path.read_text(encoding="utf-8")
    files.staging_path.write_text(
        staged.replace("source_file: lesson.pdf", "source_file: nested/lesson.pdf"),
        encoding="utf-8",
    )

    with pytest.raises(ReviewPanelError, match="basename-shaped"):
        _open(files)


def test_only_a_direct_regular_active_staging_file_is_accepted(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    link = files.staging_dir / "link.yaml"
    link.symlink_to(files.staging_path)
    done = files.staging_dir / "done"
    done.mkdir()
    archived = done / files.staging_path.name
    archived.write_bytes(files.staging_path.read_bytes())

    for path in (link, archived, files.staging_dir):
        with pytest.raises(ReviewPanelError, match="active staging"):
            ReviewPanel.open(
                path,
                staging_dir=files.staging_dir,
                patterns_path=files.patterns_path,
            )


@pytest.mark.parametrize("target_name", ["staging", "patterns"])
def test_open_capture_refuses_a_final_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    files = _panel_files(tmp_path)
    target = files.staging_path if target_name == "staging" else files.patterns_path
    moved = tmp_path / f"captured-{target.name}"
    swapped = False

    real_read_bound = data_io._read_bound_bytes

    def swap_before_read(directory_fd: int, name: str):
        nonlocal swapped
        if name == target.name and not swapped:
            target.replace(moved)
            target.symlink_to(moved)
            swapped = True
        return real_read_bound(directory_fd, name)

    monkeypatch.setattr(data_io, "_read_bound_bytes", swap_before_read)

    with pytest.raises(ReviewPanelError, match="regular non-symlink|capture"):
        _open(files)

    assert swapped
    assert target.is_symlink()


@pytest.mark.parametrize("target_name", ["staging", "patterns"])
def test_open_capture_refuses_a_same_bytes_path_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    files = _panel_files(tmp_path)
    target = files.staging_path if target_name == "staging" else files.patterns_path
    moved = tmp_path / f"opened-{target.name}"
    original = target.read_bytes()
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    swapped = False
    real_read = os.read

    def replace_during_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        opened = os.fstat(descriptor)
        if not swapped and (opened.st_dev, opened.st_ino) == target_identity:
            target.replace(moved)
            target.write_bytes(original)
            swapped = True
        return real_read(descriptor, size)

    monkeypatch.setattr(data_io.os, "read", replace_during_read)

    with pytest.raises(ReviewPanelError, match="capture|path changed"):
        _open(files)

    assert swapped
    assert target.read_bytes() == original


def test_combined_pattern_failure_keeps_card_approval_and_reports_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    before_patterns = files.patterns_path.read_bytes()

    def fail_second(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.patterns_path:
            raise OSError("simulated pattern write failure")
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(
        panel_module,
        "atomic_write_text_bound",
        fail_second,
        raising=False,
    )

    with pytest.raises(panel_module.PartialReviewError, match="card approval was saved") as caught:
        panel.submit(record_ids=[files.records[0].id], review_patterns=True)

    accepted, _meta = staging.read_staging(files.staging_path)
    assert all(example_accepted(accepted[0], example) for example in accepted[0].examples)
    assert caught.value.outcome.accepted_record_ids == (files.records[0].id,)
    assert not caught.value.outcome.pattern_reviewed
    assert files.patterns_path.read_bytes() == before_patterns


def test_direct_staging_edit_at_final_write_seam_is_never_approved_or_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    before_patterns = files.patterns_path.read_bytes()
    human_bytes = panel.staging_bytes + b"# direct human edit\n"

    def edit_then_write(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.staging_path:
            files.staging_path.write_bytes(human_bytes)
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(
        panel_module,
        "atomic_write_text_bound",
        edit_then_write,
        raising=False,
    )

    with pytest.raises(StaleReviewError, match="staging"):
        panel.submit(record_ids=[files.records[0].id], review_patterns=True)

    assert files.staging_path.read_bytes() == human_bytes
    changed, _meta = staging.read_staging(files.staging_path)
    assert EXAMPLE_AUTHORITY_KEY not in changed[0].source.raw_fields
    assert files.patterns_path.read_bytes() == before_patterns


def test_direct_pattern_edit_at_final_write_seam_survives_partial_card_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    human_patterns = panel.patterns_bytes + b" \n"

    def edit_then_write(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.patterns_path:
            files.patterns_path.write_bytes(human_patterns)
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(
        panel_module,
        "atomic_write_text_bound",
        edit_then_write,
        raising=False,
    )

    with pytest.raises(panel_module.PartialReviewError):
        panel.submit(record_ids=[files.records[0].id], review_patterns=True)

    assert files.patterns_path.read_bytes() == human_patterns
    accepted, _meta = staging.read_staging(files.staging_path)
    assert all(example_accepted(accepted[0], example) for example in accepted[0].examples)


def test_concurrent_staging_edit_during_later_pattern_failure_is_not_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)

    def edit_staging_then_fail(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.patterns_path:
            files.staging_path.write_bytes(
                files.staging_path.read_bytes() + b"# later human edit\n"
            )
            raise OSError("simulated later pattern failure")
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(
        panel_module,
        "atomic_write_text_bound",
        edit_staging_then_fail,
        raising=False,
    )

    with pytest.raises(panel_module.PartialReviewError):
        panel.submit(record_ids=[files.records[0].id], review_patterns=True)

    assert files.staging_path.read_bytes().endswith(b"# later human edit\n")
    accepted, _meta = staging.read_staging(files.staging_path)
    assert all(example_accepted(accepted[0], example) for example in accepted[0].examples)


def test_pattern_post_replace_failure_reports_the_landed_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)

    def land_then_fail(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )
        if Path(path) == files.patterns_path:
            raise OSError("simulated pattern post-replace failure")

    monkeypatch.setattr(panel_module, "atomic_write_text_bound", land_then_fail)

    with pytest.raises(panel_module.PartialReviewError) as caught:
        panel.submit(record_ids=[], review_patterns=True)

    assert caught.value.outcome.accepted_record_ids == ()
    assert caught.value.outcome.pattern_reviewed
    assert patterns.load_store(files.patterns_path)[files.pattern_set.source].reviewed


def test_symlink_swap_at_final_staging_write_cannot_touch_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    target = tmp_path / "must-not-change.yaml"
    target.write_bytes(b"owner data\n")

    def swap_then_write(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.staging_path:
            files.staging_path.unlink()
            files.staging_path.symlink_to(target)
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(
        panel_module,
        "atomic_write_text_bound",
        swap_then_write,
        raising=False,
    )

    with pytest.raises(StaleReviewError, match="staging"):
        panel.submit(record_ids=[files.records[0].id], review_patterns=False)

    assert files.staging_path.is_symlink()
    assert target.read_bytes() == b"owner data\n"


def test_staging_and_pattern_paths_are_locked_in_sorted_realpath_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    entered: list[Path] = []

    @contextmanager
    def observe(path: Path):
        entered.append(Path(path))
        yield

    monkeypatch.setattr(panel_module, "exclusive_path_lock", observe)

    panel = _open(files)
    expected = sorted(
        (files.staging_path, files.patterns_path),
        key=lambda path: os.fspath(Path(os.path.realpath(path))),
    )
    assert entered == [Path(os.path.realpath(path)) for path in expected]

    entered.clear()
    panel.submit(record_ids=[files.records[0].id], review_patterns=False)
    assert entered == [Path(os.path.realpath(path)) for path in expected]


def test_submit_rejects_actions_the_rendered_page_did_not_offer(tmp_path: Path) -> None:
    existing = _record("猫", "ねこ", authority=EXAMPLE_AUTHORITY_STAGING)
    files = _panel_files(tmp_path, records=(existing,))
    panel = _open(files)

    with pytest.raises(PanelRequestError, match="not reviewable"):
        panel.submit(record_ids=[existing.id], review_patterns=False)


def test_an_enrichment_review_written_before_the_marker_is_still_a_model_pass(
    tmp_path: Path,
) -> None:
    """`enrich --ai` files written before the `ai_enrichment` block existed
    name the collection as their source and carry nothing else. Reading them
    as an ordinary import would offer re-identification — telling a row it is
    a different word, when the model's answer was produced for the word it was
    actually asked about."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / "ai-enrichment.yaml"
    staging.write_staging(
        path,
        [_record("走る", "はしる")],
        {"source_file": "vocabulary.json", "model": "claude-opus-5"},
    )

    panel = ReviewPanel.open(
        path,
        staging_dir=staging_dir,
        patterns_path=tmp_path / "patterns.json",
        collection_name="vocabulary.json",
    )

    assert panel.provenance_kind == "model-pass"
    assert panel.reidentifiable is False
