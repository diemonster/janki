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
    PreparedStagingReview,
    PreparedStagingReviewBatch,
    ReviewBatchRequest,
    apply_prepared_staging_review,
    apply_prepared_staging_review_batch,
    execute_staging_review,
    plan_staging_review,
    prepare_staging_review,
    prepare_staging_review_batch,
    recover_prepared_staging_review,
    recover_prepared_staging_review_batch,
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


# --- the prepared review entrypoints and their configuration binding ----------
#
# A prepared intent is a plan, never authority. It names paths, and a path in a
# durable record is the one part of it an editor or a later configuration can
# move. Every entry that writes therefore re-resolves each component's role
# through the *live* configuration and refuses anything else, before effects.


def _prepared(config: ProjectConfig, staging_path: Path, records, **kwargs):
    return prepare_staging_review(
        config,
        staging_path,
        record_ids=kwargs.pop("record_ids", (records[0].id,)),
        review_patterns=kwargs.pop("review_patterns", True),
        **kwargs,
    )


def test_prepared_review_refuses_a_configuration_that_moved_its_pattern_store(
    tmp_path: Path,
) -> None:
    """Same repository root, different configured store: still the wrong file.

    The prepared component names the store the preparation read. A second
    configuration in the same root points `patterns_file` somewhere else, and
    applying the intent under it would compare-and-swap the *first* store —
    writing a review mark into a file this configuration does not use, and
    leaving the one it does use unmarked.
    """
    config, staging_path, records, _current = _project(tmp_path)
    prepared = _prepared(config, staging_path, records)
    moved = replace(config, patterns_file=tmp_path / "data" / "other-patterns.json")
    moved.patterns_file.write_bytes(config.patterns_file.read_bytes())
    before_patterns = config.patterns_file.read_bytes()
    before_moved = moved.patterns_file.read_bytes()
    before_staging = staging_path.read_bytes()

    with pytest.raises(AssistantStagingReviewError, match="pattern store"):
        apply_prepared_staging_review(moved, prepared)

    assert config.patterns_file.read_bytes() == before_patterns
    assert moved.patterns_file.read_bytes() == before_moved
    assert staging_path.read_bytes() == before_staging

    with pytest.raises(AssistantStagingReviewError, match="pattern store"):
        recover_prepared_staging_review(moved, prepared)

    assert config.patterns_file.read_bytes() == before_patterns
    assert staging_path.read_bytes() == before_staging

    # The configuration it was prepared under still applies exactly.
    outcome = apply_prepared_staging_review(config, prepared)
    assert outcome.accepted_record_ids == (records[0].id,)
    assert patterns.load_store(config.patterns_file)["lesson.pdf"].reviewed


def test_prepared_review_refuses_a_staging_target_outside_the_configured_directory(
    tmp_path: Path,
) -> None:
    """An altered intent cannot re-point the review at another file.

    The substituted target holds exactly the bytes the component binds, so
    every digest check passes; only the configured-role check separates it
    from the file the owner reviewed.
    """
    config, staging_path, records, _current = _project(tmp_path)
    prepared = _prepared(config, staging_path, records, review_patterns=False)
    elsewhere = tmp_path / "elsewhere" / staging_path.name
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(staging_path.read_bytes())
    wire = prepared.to_dict()
    wire["staging_path"] = str(elsewhere)
    wire["review"]["components"][0]["path"] = str(elsewhere)
    altered = PreparedStagingReview.from_dict(wire)
    before = elsewhere.read_bytes()

    with pytest.raises(AssistantStagingReviewError, match="staging"):
        apply_prepared_staging_review(config, altered)

    assert elsewhere.read_bytes() == before
    assert staging_path.read_bytes() == before

    with pytest.raises(AssistantStagingReviewError, match="staging"):
        recover_prepared_staging_review(config, altered)

    assert elsewhere.read_bytes() == before


def test_prepared_review_refuses_a_symlinked_substitution_of_its_own_target(
    tmp_path: Path,
) -> None:
    """A path alias is not the path. The name is what the intent bound."""
    config, staging_path, records, _current = _project(tmp_path)
    prepared = _prepared(config, staging_path, records, review_patterns=False)
    target = tmp_path / "outside.yaml"
    target.write_bytes(staging_path.read_bytes())
    staging_path.unlink()
    staging_path.symlink_to(target)
    before = target.read_bytes()

    with pytest.raises(AssistantStagingReviewError, match="staging"):
        apply_prepared_staging_review(config, prepared)

    assert target.read_bytes() == before
    assert staging_path.is_symlink()


def test_prepared_review_binds_a_part_whose_owner_decided_nothing(
    tmp_path: Path,
) -> None:
    """A fully resolved part still gets a bound, writeless intent."""
    config, staging_path, records, _current = _project(tmp_path)
    before_staging = staging_path.read_bytes()
    before_patterns = config.patterns_file.read_bytes()

    prepared = prepare_staging_review(
        config,
        staging_path,
        record_ids=(),
        review_patterns=False,
    )

    assert prepared.review.record_ids == ()
    assert not any(component.writes for component in prepared.review.components)
    assert staging_path.read_bytes() == before_staging

    outcome = apply_prepared_staging_review(config, prepared)

    assert outcome == ReviewOutcome()
    assert staging_path.read_bytes() == before_staging
    assert config.patterns_file.read_bytes() == before_patterns


def test_prepared_review_round_trips_and_binds_its_configured_paths(
    tmp_path: Path,
) -> None:
    config, staging_path, records, _current = _project(tmp_path)
    prepared = _prepared(config, staging_path, records)

    restored = PreparedStagingReview.from_dict(json.loads(json.dumps(prepared.to_dict())))

    assert restored == prepared
    assert Path(restored.review.staging.path) == staging_path.resolve()
    assert Path(restored.review.patterns.path) == config.patterns_file.resolve()
    outcome = apply_prepared_staging_review(config, restored)
    assert outcome.accepted_record_ids == (records[0].id,)


# --- the aggregate reviewed phase ---------------------------------------------
#
# `docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §7.2 and §7.6. Several parts, each
# with its own staging file and its own owner choices, and **one** pattern store
# they all write.


def _part_source(index: int) -> str:
    return f"lesson-{index:02d}.pdf"


def _run_id(index: int) -> str:
    return f"{index:08d}-1111-4111-8111-111111111111"


def _staged_part(
    config: ProjectConfig,
    index: int,
    *,
    expression: str,
    reading: str,
    sentence: str,
) -> tuple[Path, VocabularyRecord, patterns.PatternSet]:
    """One part's staging file and the store entry its run proposed."""
    source = _part_source(index)
    run_id = _run_id(index)
    provenance = dict(_provenance())
    provenance["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    staged = replace(
        patterns.with_prompt_provenance(
            patterns.PatternSet(
                source=source,
                kind="lesson",
                title=f"{source} title",
                patterns=(
                    patterns.Pattern(
                        "〜てもいいですか", "ask permission", (sentence,), "page 7"
                    ),
                ),
            ),
            provenance,
        ),
        review_run_id=run_id,
    )
    proposed = replace(
        _record(expression, reading, sentence),
        source=SourceReference(
            type="extract", imported_from=source, raw_fields={"page": "7"}
        ),
    )
    result = extract.ExtractionResult(
        candidates=(), source_units=(), model_reported_unit_count=0
    )
    metadata = {
        "source_file": source,
        "extracted_at": "2026-09-02",
        "model": "claude-opus-5",
        "review_run_id": run_id,
        "prompt_provenance": provenance,
        "pattern_set": staged.to_dict(),
        "coverage": extract.coverage_block(
            result, source_sha256=str(provenance["source_sha256"]), mode=None
        ),
    }
    path = config.staging_dir / f"{source}.yaml"
    staging.write_staging(path, [proposed], metadata)
    return path, proposed, staged


def _batch_project(
    tmp_path: Path, *, parts: int = 2, extra_store: dict[str, patterns.PatternSet] | None = None
) -> tuple[ProjectConfig, list[Path], list[VocabularyRecord]]:
    """A repository with several reviewable parts over one pattern store."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(
        '[project]\nname = "Review fixture"\n', encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)
    config.deck_dir.mkdir(parents=True)
    config.normalized_file.parent.mkdir(parents=True)
    config.normalized_file.write_text("[]\n", encoding="utf-8")
    words = [
        ("食べる", "たべる", "ここで食べてもいいですか。"),
        ("撮る", "とる", "写真を撮ってもいいですか。"),
    ]
    paths: list[Path] = []
    records: list[VocabularyRecord] = []
    store: dict[str, patterns.PatternSet] = dict(extra_store or {})
    for index in range(1, parts + 1):
        expression, reading, sentence = words[(index - 1) % len(words)]
        path, proposed, staged = _staged_part(
            config, index, expression=expression, reading=reading, sentence=sentence
        )
        paths.append(path)
        records.append(proposed)
        store[_part_source(index)] = staged
    patterns.save_store(config.patterns_file, store)
    return config, paths, records


def _requests(
    paths: list[Path], records: list[VocabularyRecord], *, patterns_for: set[int]
) -> list[ReviewBatchRequest]:
    return [
        ReviewBatchRequest(
            part_name=f"p{index:02d}",
            staging_path=path,
            record_ids=(records[index - 1].id,),
            review_patterns=index in patterns_for,
            expected_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for index, path in enumerate(paths, start=1)
    ]


def test_an_aggregate_review_composes_every_part_mark_over_one_pattern_store(
    tmp_path: Path,
) -> None:
    """§7.6: one final pattern payload, not one per part.

    Two parts each mark their own store entry. Prepared independently, each
    would bind the same path at a different after-digest and whichever landed
    second would find the store at neither of its own — so its review, and its
    staging half with it, could never be applied or recovered.
    """
    config, paths, records = _batch_project(tmp_path)
    before = _files(tmp_path)

    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1, 2})
    )

    assert _files(tmp_path) == before
    assert [part.part_name for part in prepared.batch.parts] == ["p01", "p02"]
    assert [part.source for part in prepared.batch.parts] == [
        "lesson-01.pdf",
        "lesson-02.pdf",
    ]
    assert len(prepared.batch.components) == 3
    assert prepared.batch.patterns.writes
    composed = patterns.load_store_text(
        prepared.batch.patterns.after_text or "",
        source=str(config.patterns_file),
    )
    assert {key: entry.reviewed for key, entry in composed.items()} == {
        "lesson-01.pdf": True,
        "lesson-02.pdf": True,
    }

    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), True),
    }
    saved = patterns.load_store(config.patterns_file)
    assert {key: entry.reviewed for key, entry in saved.items()} == {
        "lesson-01.pdf": True,
        "lesson-02.pdf": True,
    }
    for path, proposed in zip(paths, records, strict=True):
        rows, _meta = staging.read_staging(path)
        assert example_accepted(rows[0], rows[0].examples[0])
        assert rows[0].id == proposed.id


def test_an_aggregate_review_recovers_a_crash_after_its_staging_writes(
    tmp_path: Path,
) -> None:
    """§7.13's between-the-review-writes mutant, with two parts before the store."""
    config, paths, records = _batch_project(tmp_path)
    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1, 2})
    )
    store_before = config.patterns_file.read_bytes()
    for part in prepared.batch.parts:
        Path(part.staging.path).write_text(
            part.staging.after_text or "", encoding="utf-8"
        )
    assert config.patterns_file.read_bytes() == store_before

    outcome = recover_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), True),
    }
    saved = patterns.load_store(config.patterns_file)
    assert [entry.reviewed for entry in saved.values()] == [True, True]
    # Idempotent: the same intent applied again finds every path finished.
    assert recover_prepared_staging_review_batch(config, prepared).parts == outcome.parts


def test_an_aggregate_review_refuses_a_stale_last_target_before_any_write(
    tmp_path: Path,
) -> None:
    """The whole vector is measured first, so no earlier part lands."""
    config, paths, records = _batch_project(tmp_path, parts=2)
    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1})
    )
    paths[1].write_text(
        paths[1].read_text(encoding="utf-8") + "\n# a reviewer edited this\n",
        encoding="utf-8",
    )
    before = _files(tmp_path)

    with pytest.raises(AssistantStagingReviewError) as caught:
        apply_prepared_staging_review_batch(config, prepared)

    assert "p02" in str(caught.value)
    assert _files(tmp_path) == before


def test_an_aggregate_review_never_unreviews_and_leaves_other_entries_alone(
    tmp_path: Path,
) -> None:
    """A false choice is not an unreview, and an unrelated mark is untouched.

    `patterns.render_reviewed_update` is applied once per *selected* entry whose
    mark is false. The part that did not select its pattern set keeps its
    entry's exact `false`, and a source nobody in this batch reviewed keeps its
    existing `true`.
    """
    reviewed_elsewhere = replace(
        patterns.PatternSet(source="other.pdf", kind="lesson", title="Other"),
        reviewed=True,
    )
    config, paths, records = _batch_project(
        tmp_path, extra_store={"other.pdf": reviewed_elsewhere}
    )
    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1})
    )

    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), False),
    }
    saved = patterns.load_store(config.patterns_file)
    assert {key: entry.reviewed for key, entry in saved.items()} == {
        "other.pdf": True,
        "lesson-01.pdf": True,
        "lesson-02.pdf": False,
    }


def test_an_aggregate_review_binds_a_part_that_decides_nothing(tmp_path: Path) -> None:
    """A part an earlier pass already resolved is bound and unwritten."""
    config, paths, records = _batch_project(tmp_path)
    requests = _requests(paths, records, patterns_for={1})
    requests[1] = replace(requests[1], record_ids=(), review_patterns=False)
    prepared = prepare_staging_review_batch(config, requests)

    quiet = prepared.batch.part("p02")
    assert quiet is not None
    assert not quiet.staging.writes
    assert quiet.staging.after_text is None

    live_before = paths[1].read_bytes()
    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts["p02"] == ReviewOutcome()
    assert paths[1].read_bytes() == live_before


def test_an_aggregate_review_keeps_an_absent_pattern_store_apart_from_an_empty_one(
    tmp_path: Path,
) -> None:
    """Absence is its own before-state; a present `{}` is a file someone wrote."""
    config, paths, records = _batch_project(tmp_path)
    requests = _requests(paths, records, patterns_for=set())
    config.patterns_file.unlink()

    absent = prepare_staging_review_batch(config, requests)

    assert absent.batch.patterns.expected_before is None
    assert absent.batch.patterns.expected_after is None
    assert not absent.batch.patterns.writes

    config.patterns_file.write_text("{}", encoding="utf-8")
    present = prepare_staging_review_batch(config, requests)

    assert present.batch.patterns.expected_before == hashlib.sha256(b"{}").hexdigest()
    assert not present.batch.patterns.writes


def test_an_aggregate_review_refuses_a_part_whose_bytes_moved(tmp_path: Path) -> None:
    config, paths, records = _batch_project(tmp_path)
    requests = _requests(paths, records, patterns_for={1})
    requests[0] = replace(requests[0], expected_revision="0" * 64)
    before = _files(tmp_path)

    with pytest.raises(AssistantStagingReviewError, match="staging-review-stale"):
        prepare_staging_review_batch(config, requests)

    assert _files(tmp_path) == before


def test_an_aggregate_review_refuses_a_target_outside_the_configuration(
    tmp_path: Path,
) -> None:
    """Every part is re-bound through the live configuration before effects."""
    config, paths, records = _batch_project(tmp_path)
    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1})
    )
    moved = replace(config, staging_dir=config.staging_dir.parent / "elsewhere")
    before = _files(tmp_path)

    with pytest.raises(AssistantStagingReviewError):
        apply_prepared_staging_review_batch(moved, prepared)

    assert _files(tmp_path) == before


def test_a_prepared_aggregate_review_round_trips_through_its_durable_wire(
    tmp_path: Path,
) -> None:
    config, paths, records = _batch_project(tmp_path)
    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1, 2})
    )

    restored = PreparedStagingReviewBatch.from_dict(
        json.loads(json.dumps(prepared.to_dict(), ensure_ascii=False))
    )

    assert restored == prepared
    assert restored.fingerprint == prepared.fingerprint


# --- §7.2: a selected mark applies while false, and leaves a true one alone ---


def _reviewed_elsewhere() -> patterns.PatternSet:
    """A store entry no part of these batches names."""
    return replace(
        patterns.PatternSet(source="other.pdf", kind="lesson", title="Other"),
        reviewed=True,
    )


def _mark_reviewed(config: ProjectConfig, *sources: str) -> None:
    store = patterns.load_store(config.patterns_file)
    for source in sources:
        store[source] = replace(store[source], reviewed=True)
    patterns.save_store(config.patterns_file, store)


def test_an_aggregate_review_retains_a_selected_already_reviewed_mark(
    tmp_path: Path,
) -> None:
    """§7.2: apply each selected mark **while false**, leave a true one unchanged.

    An ordinary batch: one part whose grammar an earlier pass already reviewed,
    one still waiting, and the owner selects both. Refusing the whole batch for
    the settled part loses the part that still needs its mark, and it is the
    single-panel page's display-only rule applied where it does not belong.
    """
    config, paths, records = _batch_project(
        tmp_path, extra_store={"other.pdf": _reviewed_elsewhere()}
    )
    _mark_reviewed(config, "lesson-01.pdf")
    before = _files(tmp_path)

    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1, 2})
    )

    assert _files(tmp_path) == before, "preparation publishes nothing"
    settled = prepared.batch.part("p01")
    assert settled is not None
    # The owner's choice is recorded as the owner made it. Normalizing it to
    # false in the durable intent would record a decision nobody took.
    assert settled.review_patterns is True
    assert prepared.batch.patterns.writes
    composed = patterns.load_store_text(
        prepared.batch.patterns.after_text or "", source=str(config.patterns_file)
    )
    assert {key: entry.reviewed for key, entry in composed.items()} == {
        "other.pdf": True,
        "lesson-01.pdf": True,
        "lesson-02.pdf": True,
    }
    assert PreparedStagingReviewBatch.from_dict(
        json.loads(json.dumps(prepared.to_dict(), ensure_ascii=False))
    ) == prepared

    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), True),
    }
    saved = patterns.load_store(config.patterns_file)
    assert {key: entry.reviewed for key, entry in saved.items()} == {
        "other.pdf": True,
        "lesson-01.pdf": True,
        "lesson-02.pdf": True,
    }
    for path, proposed in zip(paths, records, strict=True):
        rows, _meta = staging.read_staging(path)
        assert example_accepted(rows[0], rows[0].examples[0])
        assert rows[0].id == proposed.id


def test_an_aggregate_review_binds_an_all_reviewed_selection_as_no_change(
    tmp_path: Path,
) -> None:
    """Every selected mark already true: one bound, unwritten store component.

    Not a refusal and not a rewrite — there is nothing to change, so the store
    is bound at the bytes the parts were prepared over and an external edit to
    it still refuses the apply. The staging halves still land.
    """
    config, paths, records = _batch_project(
        tmp_path, extra_store={"other.pdf": _reviewed_elsewhere()}
    )
    _mark_reviewed(config, "lesson-01.pdf", "lesson-02.pdf")
    store_before = config.patterns_file.read_bytes()

    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1, 2})
    )

    assert not prepared.batch.patterns.writes
    assert prepared.batch.patterns.after_text is None
    assert (
        prepared.batch.patterns.expected_before
        == prepared.batch.patterns.expected_after
        == hashlib.sha256(store_before).hexdigest()
    )

    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), True),
    }
    assert config.patterns_file.read_bytes() == store_before
    for path in paths:
        rows, _meta = staging.read_staging(path)
        assert example_accepted(rows[0], rows[0].examples[0])


def test_an_aggregate_review_leaves_an_unselected_false_mark_alone(
    tmp_path: Path,
) -> None:
    """A false choice beside a selected already-true one is still not a write.

    The retained mark contributes nothing, so nothing rewrites the store — and
    the part that chose `false` keeps its own exact `false`.
    """
    config, paths, records = _batch_project(
        tmp_path, extra_store={"other.pdf": _reviewed_elsewhere()}
    )
    _mark_reviewed(config, "lesson-01.pdf")
    store_before = config.patterns_file.read_bytes()

    prepared = prepare_staging_review_batch(
        config, _requests(paths, records, patterns_for={1})
    )

    assert not prepared.batch.patterns.writes

    outcome = apply_prepared_staging_review_batch(config, prepared)

    assert outcome.parts == {
        "p01": ReviewOutcome((records[0].id,), True),
        "p02": ReviewOutcome((records[1].id,), False),
    }
    assert config.patterns_file.read_bytes() == store_before
    saved = patterns.load_store(config.patterns_file)
    assert {key: entry.reviewed for key, entry in saved.items()} == {
        "other.pdf": True,
        "lesson-01.pdf": True,
        "lesson-02.pdf": False,
    }
