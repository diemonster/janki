"""Exact Assistant confirmations over the existing staging review writer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import extract, patterns, staging
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    ProposalContext,
)
from japanese_anki.application.assistant_staging_review import (
    AssistantStagingReviewError,
    AssistantStagingReviewPartialError,
    execute_staging_review,
    plan_staging_review,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
    example_accepted,
)
from japanese_anki.workbench.review import PartialReviewError, ReviewOutcome, ReviewPanel

RUN_ID = "11111111-1111-4111-8111-111111111111"


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


def _record(expression: str, reading: str, sentence: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["fixture"],
        examples=[
            ExampleSentence(
                japanese=sentence,
                furigana=f"{sentence}[{reading}]",
                romaji="fixture desu",
                english="A fixture sentence.",
                spoken_japanese=f"{sentence}。",
                register="polite",
            )
        ],
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            raw_fields={"page": "7"},
        ),
    )


def _project(
    tmp_path: Path,
) -> tuple[ProjectConfig, Path, tuple[VocabularyRecord, ...], patterns.PatternSet]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(
        "[project]\nname = \"Review fixture\"\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.deck_dir.mkdir(parents=True)
    config.normalized_file.parent.mkdir(parents=True)
    config.normalized_file.write_text("[]\n", encoding="utf-8")

    provenance = _provenance()
    staged = replace(
        patterns.with_prompt_provenance(
            patterns.PatternSet(
                source="lesson.pdf",
                kind="lesson",
                title="Staged lesson title",
                patterns=(
                    patterns.Pattern(
                        "〜てもいいですか",
                        "ask permission",
                        ("ここで食べてもいいですか。",),
                        "page 7",
                    ),
                ),
            ),
            provenance,
        ),
        review_run_id=RUN_ID,
    )
    # The existing transaction marks the current store entry, so the Assistant
    # must render this exact corrected value rather than the stale nested copy.
    current = replace(
        staged,
        title="Current corrected lesson title",
        patterns=(
            patterns.Pattern(
                "〜てもいいですか",
                "ask permission politely",
                ("ここで食べてもいいですか。", "写真を撮ってもいいですか。"),
                "page 7, note 1",
            ),
        ),
    )
    records = (
        _record("食べる", "たべる", "ここで食べてもいいですか。"),
        _record("撮る", "とる", "写真を撮ってもいいですか。"),
    )
    result = extract.ExtractionResult(
        candidates=(),
        source_units=(),
        model_reported_unit_count=0,
    )
    metadata = {
        "source_file": "lesson.pdf",
        "extracted_at": "2026-09-02",
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
    staging_path = config.staging_dir / "lesson.pdf.yaml"
    staging.write_staging(staging_path, records, metadata)
    patterns.save_store(config.patterns_file, {"lesson.pdf": current})
    return config, staging_path, records, current


def _proposal_resource(config: ProjectConfig) -> str:
    broker = AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)
    proposals = [
        item
        for item in catalog["data"]["resources"]
        if item["kind"] == "proposal"
    ]
    assert len(proposals) == 1
    return str(proposals[0]["resource_id"])


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_plan_renders_exact_selected_japanese_current_patterns_and_both_snapshots(
    tmp_path: Path,
) -> None:
    config, staging_path, records, current = _project(tmp_path)
    resource_id = _proposal_resource(config)
    before = _files(tmp_path)

    plan = plan_staging_review(
        config,
        proposal_resource_id=resource_id,
        record_ids=(records[1].id, records[0].id),
        review_patterns=True,
    )

    assert _files(tmp_path) == before
    assert plan.proposal_path == staging_path.resolve()
    assert plan.record_ids == (records[1].id, records[0].id)
    assert plan.projection["selection"]["records"] == [
        {
            "record_id": records[1].id,
            "expression": "撮る",
            "reading": "とる",
            "examples": [
                {
                    "japanese": "写真を撮ってもいいですか。",
                    "furigana": "写真を撮ってもいいですか。[とる]",
                    "romaji": "fixture desu",
                    "english": "A fixture sentence.",
                    "spoken_japanese": "写真を撮ってもいいですか。。",
                    "register": "polite",
                }
            ],
        },
        {
            "record_id": records[0].id,
            "expression": "食べる",
            "reading": "たべる",
            "examples": [
                {
                    "japanese": "ここで食べてもいいですか。",
                    "furigana": "ここで食べてもいいですか。[たべる]",
                    "romaji": "fixture desu",
                    "english": "A fixture sentence.",
                    "spoken_japanese": "ここで食べてもいいですか。。",
                    "register": "polite",
                }
            ],
        },
    ]
    assert plan.projection["selection"]["patterns"] == {
        "kind": current.kind,
        "title": "Current corrected lesson title",
        "source_name": "lesson.pdf",
        "patterns": [
            {
                "template": "〜てもいいですか",
                "gloss": "ask permission politely",
                "examples": [
                    "ここで食べてもいいですか。",
                    "写真を撮ってもいいですか。",
                ],
                "where": "page 7, note 1",
            }
        ],
    }
    assert plan.projection["snapshots"] == {
        "staging_sha256": hashlib.sha256(staging_path.read_bytes()).hexdigest(),
        "patterns_sha256": hashlib.sha256(config.patterns_file.read_bytes()).hexdigest(),
    }
    assert plan.fingerprint == hashlib.sha256(
        plan.projection_wire.encode("utf-8")
    ).hexdigest()


def test_execute_replans_then_uses_review_panel_for_only_explicit_decisions(
    tmp_path: Path,
) -> None:
    config, _staging_path, records, _current = _project(tmp_path)
    plan = plan_staging_review(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[1].id,),
        review_patterns=True,
    )

    result = execute_staging_review(config, plan)

    staged, _metadata = staging.read_staging(plan.proposal_path)
    by_id = {record.id: record for record in staged}
    assert EXAMPLE_AUTHORITY_KEY not in by_id[records[0].id].source.raw_fields
    assert example_accepted(
        by_id[records[1].id], by_id[records[1].id].examples[0]
    )
    assert result.outcome.accepted_record_ids == (records[1].id,)
    assert result.outcome.pattern_reviewed is True
    assert patterns.load_store(config.patterns_file)["lesson.pdf"].reviewed is True


def test_execute_refuses_when_exact_rendered_japanese_changed(
    tmp_path: Path,
) -> None:
    config, staging_path, records, _current = _project(tmp_path)
    plan = plan_staging_review(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[0].id,),
        review_patterns=False,
    )
    current_records, metadata = staging.read_staging(staging_path)
    current_records[0] = replace(
        current_records[0],
        examples=[
            replace(
                current_records[0].examples[0],
                japanese="ここで食べてもよろしいですか。",
            )
        ],
    )
    staging.write_staging(staging_path, current_records, metadata, force=True)
    before = staging_path.read_bytes()

    with pytest.raises(AssistantStagingReviewError, match="changed after confirmation"):
        execute_staging_review(config, plan)

    assert staging_path.read_bytes() == before
    assert EXAMPLE_AUTHORITY_KEY not in staging.read_staging(staging_path)[0][0].source.raw_fields


def test_execute_refuses_when_exact_pattern_store_snapshot_changed(
    tmp_path: Path,
) -> None:
    config, _staging_path, records, current = _project(tmp_path)
    plan = plan_staging_review(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[0].id,),
        review_patterns=False,
    )
    patterns.save_store(
        config.patterns_file,
        {"lesson.pdf": replace(current, title="Changed after rendering")},
    )

    with pytest.raises(AssistantStagingReviewError, match="changed after confirmation"):
        execute_staging_review(config, plan)

    staged = staging.read_staging(plan.proposal_path)[0]
    assert EXAMPLE_AUTHORITY_KEY not in staged[0].source.raw_fields


def test_execute_refuses_an_identical_plan_from_another_repository(
    tmp_path: Path,
) -> None:
    first, _first_path, first_records, _first_patterns = _project(tmp_path / "first")
    second, second_path, _second_records, _second_patterns = _project(
        tmp_path / "second"
    )
    plan = plan_staging_review(
        first,
        proposal_resource_id=_proposal_resource(first),
        record_ids=(first_records[0].id,),
        review_patterns=False,
    )
    before = second_path.read_bytes()

    with pytest.raises(AssistantStagingReviewError, match="another repository"):
        execute_staging_review(second, plan)

    assert second_path.read_bytes() == before
    assert EXAMPLE_AUTHORITY_KEY not in staging.read_staging(second_path)[0][
        0
    ].source.raw_fields


@pytest.mark.parametrize(
    ("record_ids", "review_patterns", "match"),
    [
        ((), False, "selected no cards or patterns"),
        (("missing",), False, "not reviewable"),
        (("duplicate", "duplicate"), False, "supplied twice"),
        ("one-id", False, "explicit list"),
        ((), 1, "explicit boolean"),
    ],
)
def test_plan_never_infers_a_review_selection(
    tmp_path: Path,
    record_ids: object,
    review_patterns: object,
    match: str,
) -> None:
    config, _path, _records, _current = _project(tmp_path)

    with pytest.raises(AssistantStagingReviewError, match=match):
        plan_staging_review(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_ids=record_ids,  # type: ignore[arg-type]
            review_patterns=review_patterns,  # type: ignore[arg-type]
        )


def test_plan_accepts_only_opaque_current_proposal_resource(tmp_path: Path) -> None:
    config, staging_path, records, _current = _project(tmp_path)

    with pytest.raises(AssistantStagingReviewError, match="Unknown Assistant proposal"):
        plan_staging_review(
            config,
            proposal_resource_id=str(staging_path),
            record_ids=(records[0].id,),
            review_patterns=False,
        )


def test_plan_refuses_a_different_proposal_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staging_path, records, _current = _project(tmp_path)
    resource_id = _proposal_resource(config)
    monkeypatch.setattr(
        AssistantContextBroker,
        "proposal_context",
        lambda _broker, received: ProposalContext(
            resource_id=received,
            proposal_kind="card_revision",
            path=staging_path,
        ),
    )

    with pytest.raises(AssistantStagingReviewError, match="different review workflow"):
        plan_staging_review(
            config,
            proposal_resource_id=resource_id,
            record_ids=(records[0].id,),
            review_patterns=False,
        )


def test_plan_refuses_bytes_changed_after_opaque_proposal_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, staging_path, records, _current = _project(tmp_path)
    resource_id = _proposal_resource(config)
    monkeypatch.setattr(
        AssistantContextBroker,
        "proposal_context",
        lambda _broker, received: ProposalContext(
            resource_id=received,
            proposal_kind="source_extraction",
            path=staging_path,
            proposal_sha256="0" * 64,
        ),
    )

    with pytest.raises(AssistantStagingReviewError, match="changed while Janki"):
        plan_staging_review(
            config,
            proposal_resource_id=resource_id,
            record_ids=(records[0].id,),
            review_patterns=False,
        )


def test_execute_preserves_exact_partial_review_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _staging_path, records, _current = _project(tmp_path)
    plan = plan_staging_review(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[0].id,),
        review_patterns=True,
    )
    landed = ReviewOutcome((records[0].id,), False)

    def fail_partly(
        _panel: ReviewPanel,
        *,
        record_ids: object,
        review_patterns: object,
    ) -> ReviewOutcome:
        assert record_ids == (records[0].id,)
        assert review_patterns is True
        raise PartialReviewError(
            "The card approval was saved, but the pattern review was not.",
            landed,
        )

    monkeypatch.setattr(ReviewPanel, "submit", fail_partly)

    with pytest.raises(AssistantStagingReviewPartialError) as raised:
        execute_staging_review(config, plan)

    assert raised.value.outcome == landed
