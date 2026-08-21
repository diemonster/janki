"""The single-file localhost extraction review panel."""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlencode

import pytest

from japanese_anki import cli, extract, patterns, staging
from japanese_anki import review_panel as panel_module
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
from japanese_anki.review_panel import (
    MAX_BODY_BYTES,
    PanelRequestError,
    ReviewPanel,
    ReviewPanelError,
    StaleReviewError,
    make_server,
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
                instructions="Keep <exact>.",
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


def test_render_is_complete_escaped_and_existing_authority_is_display_only(
    tmp_path: Path,
) -> None:
    accepted = _record("猫", "ねこ", authority=EXAMPLE_AUTHORITY_STAGING)
    files = _panel_files(tmp_path, records=(accepted, _record("犬", "いぬ")))

    html = _open(files).render()

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Current &lt;corrected&gt; title" in html
    assert "&lt;script&gt;current template&lt;/script&gt;" in html
    for value in (
        "word:猫:ねこ",
        "猫[ねこ]",
        "cat &lt;pet&gt;",
        "noun",
        "none",
        "intransitive",
        "neko desu",
        "It is a cat.",
        "lesson&lt;tag&gt;",
        "Use &lt;carefully&gt;.",
        "lesson.pdf",
        "The source explicitly glosses this word.",
    ):
        assert value in html
    for noise in (
        "example.mp3",
        "Keep &lt;exact&gt;.",
        "word.mp3",
        "image.png",
        "Frequency rank",
        "Prompt provenance",
        "machine-owned",
    ):
        assert noise not in html
    assert "autocomplete=off" in html
    assert "Already reviewed" in html
    assert 'value="word:猫:ねこ"' not in html
    assert html.count('value="word:犬:いぬ"') == 1
    assert "犬だよ。" in html
    assert "I reviewed every Japanese example above" in html
    assert "I reviewed the current pattern-store entry" in html
    assert "restart the panel" in html
    assert "reload" not in html.lower()
    assert '<h3 lang="ja">猫 <small>ねこ</small></h3>' in html
    assert 'aria-label="Review examples for 犬 (いぬ)"' not in html
    assert (
        "I reviewed every Japanese example above for 犬 (いぬ) and accept it "
        "as teaching content."
    ) in html
    assert "<details open><summary>Current pattern-store entry</summary>" in html
    assert "<details open><summary>Staged pattern answer</summary>" not in html
    assert "<script" not in html.lower()
    assert "https://" not in html and "http://" not in html


def test_empty_optional_card_fields_are_not_rendered(tmp_path: Path) -> None:
    base = _record("猫", "ねこ")
    examples = tuple(replace(example, audio="", instructions="") for example in base.examples)
    sparse = replace(
        base,
        examples=examples,
        conjugations={},
        audio="",
        image="",
        pitch_accent=[],
        audio_accent="",
        frequency_rank=None,
    )
    files = _panel_files(tmp_path, records=(sparse,))

    html = _open(files).render()

    for label in (
        "Conjugations",
        "Word audio",
        "Image",
        "Pitch accent",
        "Audio accent",
        "Frequency rank",
        "Example audio",
        "Instructions",
        "Hold",
    ):
        assert f"<dt>{label}</dt>" not in html


def test_only_exact_or_fully_bound_existing_authority_is_display_only(
    tmp_path: Path,
) -> None:
    near_miss = _record("猫", "ねこ", authority=" staging-review")
    unbound = _record("犬", "いぬ")
    bound = set_example_flags(
        unbound,
        EXAMPLE_AUTHORITY_KEY,
        (example.japanese for example in unbound.examples if example.japanese),
    )
    files = _panel_files(tmp_path, records=(near_miss, bound))

    html = _open(files).render()

    assert "not a recognized review value" in html
    assert html.count("not a recognized review value") == 1
    assert html.count("Already reviewed") == 1
    assert 'value="word:猫:ねこ"' not in html
    assert 'value="word:犬:いぬ"' not in html


def test_partial_or_stale_bound_authority_remains_reviewable_with_a_warning(
    tmp_path: Path,
) -> None:
    record = _record("猫", "ねこ")
    partial = set_example_flags(
        record,
        EXAMPLE_AUTHORITY_KEY,
        [record.examples[0].japanese],
    )
    files = _panel_files(tmp_path, records=(partial,))

    html = _open(files).render()

    assert 'value="word:猫:ねこ"' in html
    assert "do not cover every current Japanese example" in html
    assert "replace them with fingerprints for exactly the examples shown" in html


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


def test_combined_submit_and_fresh_noop_are_idempotent(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    _open(files).submit(record_ids=[row.id for row in files.records], review_patterns=True)
    settled_staging = files.staging_path.read_bytes()
    settled_patterns = files.patterns_path.read_bytes()

    fresh = _open(files)
    html = fresh.render()
    assert html.count("Already reviewed") >= 3
    outcome = fresh.submit(record_ids=[], review_patterns=False)

    assert outcome.accepted_record_ids == ()
    assert not outcome.pattern_reviewed
    assert files.staging_path.read_bytes() == settled_staging
    assert files.patterns_path.read_bytes() == settled_patterns


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


@pytest.mark.parametrize("mismatch", ["run", "provenance", "missing", "absent"])
def test_stale_pattern_lineage_disables_only_pattern_review(tmp_path: Path, mismatch: str) -> None:
    files = _panel_files(tmp_path)
    if mismatch == "run":
        replacement = replace(files.pattern_set, review_run_id=OTHER_RUN_ID)
    elif mismatch == "provenance":
        replacement = replace(
            files.pattern_set,
            prompt_provenance={**_provenance(), "model": "other"},
        )
    elif mismatch == "missing":
        replacement = replace(
            files.pattern_set,
            review_run_id=None,
            prompt_provenance={},
        )
    else:
        replacement = None
    patterns.save_store(
        files.patterns_path,
        {} if replacement is None else {replacement.source: replacement},
    )
    panel = _open(files)
    before_patterns = files.patterns_path.read_bytes()

    html = panel.render()

    assert "Staged pattern answer" in html
    assert "staged template" in html
    if replacement is None:
        assert "No current pattern-store entry" in html
        assert "no entry for the exact key" in html
    else:
        assert "Current pattern-store entry" in html
        assert "current template" in html
        assert "does not exactly match this staging review run" in html
    assert 'name="patterns"' not in html
    assert 'value="word:猫:ねこ"' in html
    with pytest.raises(PanelRequestError, match="lineage"):
        panel.submit(record_ids=[], review_patterns=True)
    assert files.patterns_path.read_bytes() == before_patterns

    panel.submit(record_ids=[files.records[0].id], review_patterns=False)
    accepted, _meta = staging.read_staging(files.staging_path)
    assert all(example_accepted(accepted[0], example) for example in accepted[0].examples)


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

    def swap_before_open(path: Path) -> int:
        nonlocal swapped
        candidate = Path(path)
        if candidate == target and not swapped:
            target.replace(moved)
            target.symlink_to(moved)
            swapped = True
        return os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))

    monkeypatch.setattr(panel_module, "_open_no_follow", swap_before_open, raising=False)

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
    swapped = False

    def replace_during_read(descriptor: int) -> bytes:
        nonlocal swapped
        target.replace(moved)
        target.write_bytes(original)
        swapped = True
        chunks = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)

    monkeypatch.setattr(panel_module, "_read_open_fd", replace_during_read, raising=False)

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


def test_zero_row_pattern_only_file_can_be_reviewed(tmp_path: Path) -> None:
    files = _panel_files(tmp_path, records=())
    panel = _open(files)

    assert 'name="record"' not in panel.render()
    outcome = panel.submit(record_ids=[], review_patterns=True)

    assert outcome.pattern_reviewed
    assert patterns.load_store(files.patterns_path)["lesson.pdf"].reviewed


def _running_server(panel: ReviewPanel):
    server = make_server(panel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(
    server: object,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, payload


def _valid_form(panel: ReviewPanel, *record_ids: str, patterns_too: bool = False) -> bytes:
    fields = [
        ("csrf", panel.csrf_token),
        ("staging_snapshot", panel.staging_fingerprint),
        ("patterns_snapshot", panel.patterns_fingerprint),
        ("action", "save"),
        *(("record", record_id) for record_id in record_ids),
    ]
    if patterns_too:
        fields.append(("patterns", "review"))
    return urlencode(fields).encode()


def test_partial_post_body_times_out_without_blocking_server_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    monkeypatch.setattr(panel_module, "REQUEST_TIMEOUT_SECONDS", 0.05, raising=False)
    server, thread = _running_server(panel)
    client = socket.create_connection(server.server_address, timeout=1)
    client.settimeout(1)
    try:
        request = (
            f"POST /review HTTP/1.1\r\nHost: {server.expected_host}\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            "Content-Length: 32\r\nConnection: close\r\n\r\nx"
        )
        client.sendall(request.encode("ascii"))
        response = client.recv(4096)
        assert b" 408 " in response.split(b"\r\n", 1)[0]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_boundary_has_security_headers_and_success_exits(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    server, thread = _running_server(panel)
    try:
        assert server.daemon_threads is False
        assert server.block_on_close is True
        status, headers, body = _request(server, "GET", "/")
        assert status == 200
        assert headers["content-security-policy"].startswith("default-src 'none'")
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["cache-control"] == "no-store"
        assert b"<script" not in body.lower()

        status, _headers, _body = _request(
            server,
            "GET",
            "/",
            headers={"Host": "localhost:1"},
        )
        assert status == 403

        status, error_headers, _body = _request(
            server,
            "GET",
            "/",
            headers={
                "Host": server.expected_host,
                "Origin": "http://evil.test",
            },
        )
        assert status == 403
        assert error_headers["x-frame-options"] == "DENY"

        form = _valid_form(panel, files.records[0].id, patterns_too=True)
        request_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": f"http://{server.expected_host}",
        }
        status, _headers, body = _request(
            server, "POST", "/review", body=form, headers=request_headers
        )
        assert status == 200
        assert b"Review saved" in body
        thread.join(timeout=3)
        assert not thread.is_alive(), "a successful one-submit panel exits"
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_stale_refusal_is_terminal_and_says_to_restart(tmp_path: Path) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    files.staging_path.write_bytes(files.staging_path.read_bytes() + b"# changed\n")
    server, thread = _running_server(panel)
    try:
        status, _headers, body = _request(
            server,
            "POST",
            "/review",
            body=_valid_form(panel, files.records[0].id),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 409
        assert b"Restart the panel" in body
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_indeterminate_write_reports_partial_and_is_terminal(
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
        if Path(path) == files.staging_path:
            raise OSError("simulated post-replace failure")

    monkeypatch.setattr(panel_module, "atomic_write_text_bound", land_then_fail)
    server, thread = _running_server(panel)
    try:
        status, _headers, body = _request(
            server,
            "POST",
            "/review",
            body=_valid_form(panel, files.records[0].id),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 409
        assert b"Partial review result" in body
        assert b"Card example approval saved" in body
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    accepted, _meta = staging.read_staging(files.staging_path)
    assert all(example_accepted(accepted[0], example) for example in accepted[0].examples)


@pytest.mark.parametrize("outcome", ["success", "stale", "partial"])
def test_terminal_submit_shuts_down_even_when_response_write_breaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    form = _valid_form(
        panel,
        files.records[0].id,
        patterns_too=outcome == "partial",
    )
    if outcome == "stale":
        files.staging_path.write_bytes(files.staging_path.read_bytes() + b"# changed\n")
    elif outcome == "partial":

        def fail_pattern(
            path: Path,
            text: str,
            *,
            expected_revision: str | None = None,
            **kwargs: object,
        ) -> None:
            if Path(path) == files.patterns_path:
                raise OSError("simulated pattern failure")
            atomic_write_text_bound(
                path,
                text,
                expected_revision=expected_revision,
                **kwargs,
            )

        monkeypatch.setattr(panel_module, "atomic_write_text_bound", fail_pattern)

    def broken_response(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError("client disconnected")

    monkeypatch.setattr(panel_module._ReviewHandler, "_send", broken_response)
    server, thread = _running_server(panel)
    try:
        with suppress(OSError, http.client.HTTPException):
            _request(
                server,
                "POST",
                "/review",
                body=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        thread.join(timeout=3)
        assert not thread.is_alive(), f"{outcome} must stop after a broken response"
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_retry_uses_same_token_after_proven_no_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    failed = False

    def fail_once(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        nonlocal failed
        if Path(path) == files.staging_path and not failed:
            failed = True
            raise OSError("simulated pre-replace failure")
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    monkeypatch.setattr(panel_module, "atomic_write_text_bound", fail_once)
    server, thread = _running_server(panel)
    form = _valid_form(panel, files.records[0].id)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        status, _headers, body = _request(
            server,
            "POST",
            "/review",
            body=form,
            headers=headers,
        )
        assert status == 500
        assert b"Nothing was proven written" in body
        assert thread.is_alive()

        status, _headers, body = _request(
            server,
            "POST",
            "/review",
            body=form,
            headers=headers,
        )
        assert status == 200
        assert b"Review saved" in body
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("no_open", [False, True])
def test_cli_opens_or_prints_one_ephemeral_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_open: bool,
) -> None:
    files = _panel_files(tmp_path)
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    opened: list[str] = []

    class FakeServer:
        expected_host = "127.0.0.1:43123"
        served = False
        closed = False

        def serve_forever(self) -> None:
            self.served = True

        def server_close(self) -> None:
            self.closed = True

    server = FakeServer()
    monkeypatch.setattr(panel_module, "make_server", lambda _panel: server)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
    argv = ["--root", str(tmp_path), "review-panel", str(files.staging_path)]
    if no_open:
        argv.append("--no-open")

    assert cli.main(argv) == 0

    assert server.served and server.closed
    assert opened == ([] if no_open else ["http://127.0.0.1:43123/"])
    assert "Review panel: http://127.0.0.1:43123/" in capsys.readouterr().out


def test_http_rejects_oversize_and_duplicate_actions_without_writes(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    before_staging = files.staging_path.read_bytes()
    before_patterns = files.patterns_path.read_bytes()

    oversize_panel = _open(files)
    server, thread = _running_server(oversize_panel)
    try:
        status, _headers, _body = _request(
            server,
            "POST",
            "/review",
            body=b"x" * (MAX_BODY_BYTES + 1),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    duplicate_panel = _open(files)
    server, thread = _running_server(duplicate_panel)
    try:
        valid = _valid_form(duplicate_panel, files.records[0].id)
        duplicate = valid + b"&action=save"
        status, _headers, _body = _request(
            server,
            "POST",
            "/review",
            body=duplicate,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert files.staging_path.read_bytes() == before_staging
    assert files.patterns_path.read_bytes() == before_patterns


def test_csrf_is_one_shot_even_when_the_first_submit_changes_nothing(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    form = _valid_form(panel)

    assert panel.consume_form(form).accepted_record_ids == ()
    with pytest.raises(PanelRequestError, match="already used"):
        panel.consume_form(form)


@pytest.mark.parametrize(
    ("tamper", "value"),
    [
        ("missing_csrf", None),
        ("csrf", "wrong-token"),
        ("staging_snapshot", "0" * 64),
        ("patterns_snapshot", "f" * 64),
    ],
)
def test_invalid_token_or_hidden_snapshot_writes_nothing_and_token_remains_valid(
    tmp_path: Path,
    tamper: str,
    value: str | None,
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    fields = [
        ("csrf", panel.csrf_token),
        ("staging_snapshot", panel.staging_fingerprint),
        ("patterns_snapshot", panel.patterns_fingerprint),
        ("action", "save"),
        ("record", files.records[0].id),
    ]
    key = "csrf" if tamper == "missing_csrf" else tamper
    fields = [(name, item) for name, item in fields if name != key]
    if value is not None:
        fields.append((key, value))
    before_staging = files.staging_path.read_bytes()
    before_patterns = files.patterns_path.read_bytes()

    with pytest.raises(PanelRequestError):
        panel.consume_form(urlencode(fields).encode())

    assert files.staging_path.read_bytes() == before_staging
    assert files.patterns_path.read_bytes() == before_patterns
    outcome = panel.consume_form(_valid_form(panel, files.records[0].id))
    assert outcome.accepted_record_ids == (files.records[0].id,)


def test_stale_submit_does_not_consume_token_if_exact_snapshot_is_restored(
    tmp_path: Path,
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    form = _valid_form(panel, files.records[0].id)
    snapshot = files.staging_path.read_bytes()
    files.staging_path.write_bytes(snapshot + b"# temporary edit\n")

    with pytest.raises(StaleReviewError):
        panel.consume_form(form)

    files.staging_path.write_bytes(snapshot)
    outcome = panel.consume_form(form)
    assert outcome.accepted_record_ids == (files.records[0].id,)


def test_concurrent_double_submit_can_enter_the_write_path_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _panel_files(tmp_path)
    panel = _open(files)
    form = _valid_form(panel, files.records[0].id)
    first_write_entered = threading.Event()
    allow_first_write = threading.Event()
    second_started = threading.Event()
    outcomes: list[object] = []
    failures: list[BaseException] = []

    def slow_write(
        path: Path,
        text: str,
        *,
        expected_revision: str | None = None,
        **kwargs: object,
    ) -> None:
        if Path(path) == files.staging_path:
            first_write_entered.set()
            assert allow_first_write.wait(timeout=3)
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_revision,
            **kwargs,
        )

    def submit(*, second: bool = False) -> None:
        if second:
            second_started.set()
        try:
            outcomes.append(panel.consume_form(form))
        except BaseException as exc:  # captured for assertions from the worker
            failures.append(exc)

    monkeypatch.setattr(panel_module, "atomic_write_text_bound", slow_write)
    first = threading.Thread(target=submit)
    second = threading.Thread(target=lambda: submit(second=True))
    first.start()
    assert first_write_entered.wait(timeout=3)
    second.start()
    assert second_started.wait(timeout=3)
    allow_first_write.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], PanelRequestError)


def test_submit_rejects_actions_the_rendered_page_did_not_offer(tmp_path: Path) -> None:
    existing = _record("猫", "ねこ", authority=EXAMPLE_AUTHORITY_STAGING)
    files = _panel_files(tmp_path, records=(existing,))
    panel = _open(files)

    with pytest.raises(PanelRequestError, match="not reviewable"):
        panel.submit(record_ids=[existing.id], review_patterns=False)
