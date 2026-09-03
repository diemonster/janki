"""Paid card revisions stop at exact-bound staging."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import promote, staging
from japanese_anki.application import assistant_card_revision_review, card_change_staging
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.application.promotion import plan_promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging, replacement_fingerprint

OPERATION_ID = "b65df166-dcd7-4c1b-b264-c79041410c55"
REQUEST_FINGERPRINT = "a" * 64


def _record(**overrides: object) -> VocabularyRecord:
    values: dict[str, object] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "verb_group": "godan",
        "tags": ["lesson-8"],
        "usage_notes": "Used for speaking a language.",
        "examples": [
            ExampleSentence(
                japanese="日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
                register="polite",
            )
        ],
        "source": SourceReference(type="manual", imported_from="lesson-8.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)  # type: ignore[arg-type]


def _project(tmp_path: Path, record: VocabularyRecord) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    normalized = tmp_path / "data" / "normalized"
    normalized.mkdir(parents=True)
    (normalized / "vocabulary.json").write_text(
        json.dumps([record.to_dict()], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    decks = tmp_path / "data" / "decks"
    decks.mkdir(parents=True)
    (decks / "all.yaml").write_text(
        "deck:\n"
        "  name: All words\n"
        "  deck_id: 1234\n"
        "  model_id: 5678\n"
        '  source: "../normalized/vocabulary.json"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _provenance() -> card_change_staging.CardRevisionProvenance:
    return card_change_staging.CardRevisionProvenance(
        operation_id=OPERATION_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
        provider="claude-code",
        model="claude-opus-5",
    )


def test_complete_record_plan_is_pure_and_stages_existing_promotion_metadata(
    tmp_path: Path,
) -> None:
    current = _record()
    config = _project(tmp_path, current)
    proposed = replace(
        current,
        part_of_speech="godan verb",
        usage_notes="Use this for the act of speaking; 話せる expresses ability.",
    )

    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        proposed_records=[proposed],
        provenance=_provenance(),
        focus_resource_id="deck-opaque-17",
    )
    view = card_change_staging.project_card_change_staging(plan)

    assert not config.staging_dir.exists()
    assert view.record_ids == (current.id,)
    assert view.changed_fields == {
        current.id: ("part_of_speech", "usage_notes"),
    }
    assert view.focus_resource_id == "deck-opaque-17"
    assert view.operation_id == OPERATION_ID
    assert view.request_fingerprint == REQUEST_FINGERPRINT

    result = card_change_staging.stage_card_change_staging(
        config,
        plan,
        expected_fingerprint=view.plan_fingerprint,
    )

    assert result.staging_path == plan.staging_path
    assert result.record_ids == (current.id,)
    staged, meta = read_staging(result.staging_path)
    assert staged == [proposed]
    assert meta["review_run_id"] == OPERATION_ID
    assert result.state == "staged"
    assert meta["card_revision"] == {
        "version": 1,
        "operation_id": OPERATION_ID,
        "request_fingerprint": REQUEST_FINGERPRINT,
        "provider": "claude-code",
        "attribution_provider": "anthropic",
        "model": "claude-opus-5",
        "input_fingerprints": dict(plan.input_fingerprints),
        "fields": {current.id: ["part_of_speech", "usage_notes"]},
        "focus_resource_id": "deck-opaque-17",
    }
    assert meta["field_replacements"]["records"] == {
        current.id: {
            "part_of_speech": replacement_fingerprint(current, "part_of_speech"),
            "usage_notes": replacement_fingerprint(current, "usage_notes"),
        }
    }
    assert "acceptance" not in meta
    assert not config.ledger_file.exists()
    assert not config.operations_file.exists()
    assert not config.scan_inbox.parent.exists()

    # The shared writer refuses until exact owner review is durable.
    promotion = plan_promotion(config, result.staging_path)
    assert promotion.is_blocked
    assert "no exact durable owner review" in promotion.blocked

    resource = next(
        item
        for item in json.loads(AssistantContextBroker(config).catalog().wire)["data"]["resources"]
        if item.get("proposal_kind") == "card_revision"
    )
    with pytest.raises(
        assistant_card_revision_review.AssistantCardRevisionReviewError,
        match="belong to this proposal",
    ):
        assistant_card_revision_review.plan_card_revision_review(
            config,
            resource_id=resource["resource_id"],
            record_ids=["word:unknown:unknown"],
        )
    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=resource["resource_id"],
        record_ids=[current.id],
    )
    assert review.projection["selection"]["changes"] == [
        {
            "record_id": current.id,
            "expression": current.expression,
            "field": "part_of_speech",
            "old_value": "verb",
            "proposed_value": "godan verb",
        },
        {
            "record_id": current.id,
            "expression": current.expression,
            "field": "usage_notes",
            "old_value": "Used for speaking a language.",
            "proposed_value": ("Use this for the act of speaking; 話せる expresses ability."),
        },
    ]
    assistant_card_revision_review.execute_card_revision_review(config, review)
    _reviewed_records, reviewed_meta = read_staging(result.staging_path)
    assert reviewed_meta["card_revision_review"]["accepted_record_ids"] == [current.id]

    promotion = plan_promotion(config, result.staging_path)
    assert not promotion.is_blocked
    assert len(promotion.merging) == 1
    assert promotion.merging[0].landing.part_of_speech == "godan verb"
    assert promotion.merging[0].landing.usage_notes.endswith("expresses ability.")

    # The durable mark cannot be reused for a value the owner did not see.
    tampered = replace(proposed, usage_notes="A different, unreviewed value.")
    staging.write_staging(
        result.staging_path,
        [tampered],
        reviewed_meta,
        force=True,
    )
    refused = plan_promotion(config, result.staging_path)
    assert refused.is_blocked
    assert "changed after owner review" in refused.blocked


def test_partial_archive_cannot_reuse_owner_review_for_tampered_live_revision(
    tmp_path: Path,
) -> None:
    first = _record()
    second = _record(
        id="word:食べる:たべる",
        expression="食べる",
        reading="たべる",
        meanings=["to eat"],
        usage_notes="Used for eating.",
    )
    config = _project(tmp_path, first)
    config.normalized_file.write_text(
        json.dumps([first.to_dict(), second.to_dict()], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    proposed_first = replace(first, usage_notes="Reviewed first proposal.")
    proposed_second = replace(second, usage_notes="Reviewed second proposal.")
    staged_plan = card_change_staging.plan_card_change_staging(
        config,
        [first, second],
        proposed_records=[proposed_first, proposed_second],
        provenance=_provenance(),
    )
    staged_result = card_change_staging.stage_card_change_staging(
        config,
        staged_plan,
        expected_fingerprint=staged_plan.fingerprint,
    )
    resource = next(
        item
        for item in json.loads(AssistantContextBroker(config).catalog().wire)["data"]["resources"]
        if item.get("proposal_kind") == "card_revision"
    )
    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=resource["resource_id"],
        record_ids=[first.id, second.id],
    )
    assistant_card_revision_review.execute_card_revision_review(config, review)
    _reviewed, reviewed_meta = read_staging(staged_result.staging_path)

    done = config.staging_dir / "done"
    done.mkdir(parents=True)
    staging.write_staging(
        done / staged_result.staging_path.name,
        [proposed_first],
        promote.archive_meta(reviewed_meta, 1),
    )
    staging.write_staging(
        staged_result.staging_path,
        [proposed_second],
        reviewed_meta,
        force=True,
    )

    unchanged = plan_promotion(config, staged_result.staging_path)

    assert not unchanged.is_blocked

    staging.write_staging(
        staged_result.staging_path,
        [replace(proposed_second, usage_notes="Unreviewed value after partial archive.")],
        reviewed_meta,
        force=True,
    )

    refused = plan_promotion(config, staged_result.staging_path)

    assert refused.is_blocked
    assert "changed after owner review" in refused.blocked


def test_field_replacement_input_builds_complete_valid_records(tmp_path: Path) -> None:
    current = _record()
    config = _project(tmp_path, current)

    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        field_replacements={
            current.id: {
                "meanings": ["to speak", "to talk"],
                "examples": [
                    {
                        "japanese": "日本語を話します。",
                        "furigana": "日本語[にほんご]を 話[はな]します。",
                        "romaji": "nihongo o hanashimasu.",
                        "english": "I speak Japanese.",
                        "audio": "",
                        "spoken_japanese": "",
                        "register": "polite",
                    },
                    {
                        "japanese": "あとで話す？",
                        "furigana": "あとで 話[はな]す？",
                        "romaji": "ato de hanasu?",
                        "english": "Want to talk later?",
                        "audio": "",
                        "spoken_japanese": "",
                        "register": "casual",
                    },
                ],
            }
        },
        provenance=_provenance(),
    )

    assert plan.proposed_records[0].meanings == ["to speak", "to talk"]
    assert plan.proposed_records[0].examples[1].register == "casual"
    assert tuple(change.field for change in plan.changes) == ("meanings", "examples")


@pytest.mark.parametrize("field", ["id", "expression", "reading", "source", "tags"])
def test_field_replacement_input_refuses_every_protected_field(
    tmp_path: Path,
    field: str,
) -> None:
    current = _record()
    config = _project(tmp_path, current)

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="cannot change protected field",
    ):
        card_change_staging.plan_card_change_staging(
            config,
            [current],
            field_replacements={current.id: {field: current.to_dict()[field]}},
            provenance=_provenance(),
        )

    assert not config.staging_dir.exists()


def test_invalid_or_identity_changed_complete_record_is_refused(tmp_path: Path) -> None:
    current = _record()
    config = _project(tmp_path, current)

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="cannot change protected field 'reading'",
    ):
        card_change_staging.plan_card_change_staging(
            config,
            [current],
            proposed_records=[replace(current, reading="しゃべる")],
            provenance=_provenance(),
        )

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="fails structural validation.*at least one English meaning",
    ):
        card_change_staging.plan_card_change_staging(
            config,
            [current],
            proposed_records=[replace(current, meanings=[])],
            provenance=_provenance(),
        )

    assert not config.staging_dir.exists()


def test_stage_rejects_stale_canonical_or_tampered_plan_without_writing(
    tmp_path: Path,
) -> None:
    current = _record()
    config = _project(tmp_path, current)
    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        field_replacements={current.id: {"usage_notes": "A clearer note."}},
        provenance=_provenance(),
    )

    canonical = load_records(config.normalized_file)
    canonical[0] = replace(canonical[0], usage_notes="A concurrent owner edit.")
    config.normalized_file.write_text(
        json.dumps([record.to_dict() for record in canonical], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="Canonical record.*changed",
    ):
        card_change_staging.stage_card_change_staging(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
        )
    assert not plan.staging_path.exists()

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="plan fingerprint",
    ):
        card_change_staging.stage_card_change_staging(
            config,
            replace(plan, focus_resource_id="tampered"),
            expected_fingerprint=plan.fingerprint,
        )
    assert not plan.staging_path.exists()


def test_stage_never_overwrites_an_existing_review(tmp_path: Path) -> None:
    current = _record()
    config = _project(tmp_path, current)
    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        field_replacements={current.id: {"usage_notes": "A clearer note."}},
        provenance=_provenance(),
    )
    plan.staging_path.parent.mkdir(parents=True)
    plan.staging_path.write_text("owner work\n", encoding="utf-8")

    with pytest.raises(
        card_change_staging.CardChangeStagingError,
        match="already exists",
    ):
        card_change_staging.stage_card_change_staging(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
        )

    assert plan.staging_path.read_text(encoding="utf-8") == "owner work\n"


def test_stage_exact_retry_adopts_its_existing_review_without_rewriting(
    tmp_path: Path,
) -> None:
    current = _record()
    config = _project(tmp_path, current)
    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        field_replacements={current.id: {"usage_notes": "A clearer note."}},
        provenance=_provenance(),
    )
    first = card_change_staging.stage_card_change_staging(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
    )
    original = plan.staging_path.read_bytes()

    second = card_change_staging.stage_card_change_staging(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
    )

    assert first.state == "staged"
    assert second.state == "already_staged"
    assert plan.staging_path.read_bytes() == original
