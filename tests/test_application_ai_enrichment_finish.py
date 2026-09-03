"""One exact owner review and finish for an AI-enrichment proposal."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_application_ai_enrichment import _answer as enrichment_answer
from test_application_audio import RecordingProvider
from test_application_card_revision import SELECTED, _project
from test_application_revision_finish import _RealtimeTransport

from japanese_anki import staging
from japanese_anki.application import ai_enrichment, card_revision_finish
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.io import load_records
from japanese_anki.tts import openai_realtime

OPERATION_ID = "f1f9dc5d-b9b5-4ea0-93ac-9967d73ff81a"


def _staged_enrichment(
    tmp_path: Path,
    *,
    persist_focus: bool = True,
    extract_source: bool = False,
    wrong_focus: bool = False,
) -> tuple[object, Path, str]:
    config, deck = _project(tmp_path)
    if extract_source:
        payload = json.loads(config.normalized_file.read_text(encoding="utf-8"))
        payload[0]["source"]["type"] = "extract"
        payload[0]["source"]["raw_fields"] = {}
        config.normalized_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    focus = AssistantContextBroker(config).resource_id_for_deck(deck)
    if wrong_focus:
        other = deck.with_name("other.yaml")
        other.write_text(
            "deck:\n"
            "  name: Other\n"
            "  deck_id: 2234\n"
            "  model_id: 6678\n"
            '  source: "../normalized/vocabulary.json"\n'
            "  include_ids:\n"
            "    - word:遊ぶ:あそぶ\n",
            encoding="utf-8",
        )
        focus = AssistantContextBroker(config).resource_id_for_deck(other)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [SELECTED],
        force_fields=["meanings", "examples", "usage_notes"],
        operation_ids=[OPERATION_ID],
        focus_resource_id=focus if persist_focus else None,
    )
    answer = enrichment_answer(plan.calls[0], gloss="to speak or talk")
    ai_enrichment.run_ai_enrichment(
        config,
        plan,
        client=object(),
        api_call=lambda *_args, capture, **_kwargs: capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": answer.model_dump_json()}],
            }
        ),
    )
    resource = next(
        item
        for item in json.loads(AssistantContextBroker(config).catalog().wire)["data"][
            "resources"
        ]
        if item.get("proposal_kind") == "ai_enrichment"
    )
    return config, deck, str(resource["resource_id"])


def test_ai_enrichment_one_confirmation_records_exact_review_then_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, resource_id = _staged_enrichment(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_ai_enrichment_finish(
        config,
        resource_id,
        "Review this exact enrichment, apply it, voice it, and build its deck.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )

    assert plan.deck_path == deck.absolute()
    assert plan.record_ids == (SELECTED,)
    assert plan.review is not None
    assert plan.promotion is None
    assert plan.authority["proposal_kind"] == "ai_enrichment"
    selection = plan.authority["review"]["projection"]["selection"]
    assert selection["record_ids"] == [SELECTED]
    assert {change["field"] for change in selection["changes"]} == {
        "examples",
        "meanings",
        "usage_notes",
    }
    assert selection["examples"][0]["current"] == []
    assert len(selection["examples"][0]["proposed"]) == 2
    assert not plan.record_path.exists()

    real_review = card_revision_finish.assistant_ai_enrichment_review.execute_ai_enrichment_review
    real_replan = card_revision_finish._replan_live_promotion
    calls: list[str] = []

    def observe_review(*args: object, **kwargs: object):
        calls.append("review")
        assert card_revision_finish.inspect_card_revision_finish(
            config, plan.fingerprint
        ).state == "authorized"
        return real_review(*args, **kwargs)

    def stop_after_review(*_args: object, **_kwargs: object):
        raise card_revision_finish.CardRevisionFinishError("stop after AI review")

    monkeypatch.setattr(
        card_revision_finish.assistant_ai_enrichment_review,
        "execute_ai_enrichment_review",
        observe_review,
    )
    monkeypatch.setattr(card_revision_finish, "_replan_live_promotion", stop_after_review)
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="stop after AI review",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )

    reviewed, meta = staging.read_staging(plan.review.proposal_path)
    assert meta[staging.AI_ENRICHMENT_REVIEW_KEY]["accepted_record_ids"] == [
        SELECTED
    ]
    assert "example_authority" not in reviewed[0].source.raw_fields

    monkeypatch.setattr(card_revision_finish, "_replan_live_promotion", real_replan)
    result = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=words,
    )

    assert result.state == "complete"
    assert calls == ["review"]
    landed = {record.id: record for record in load_records(config.normalized_file)}[
        SELECTED
    ]
    assert landed.meanings == ["to speak or talk"]
    assert len(landed.examples) == 2
    assert result.output_path.exists()


def test_ai_enrichment_review_binds_exact_examples_on_extract_sourced_card(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_enrichment(
        tmp_path,
        extract_source=True,
    )
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_ai_enrichment_finish(
        config,
        resource_id,
        "Review and finish this exact enrichment.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )

    result = card_revision_finish.execute_card_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=words,
    )

    assert result.state == "complete"
    landed = {record.id: record for record in load_records(config.normalized_file)}[
        SELECTED
    ]
    authority = landed.source.raw_fields["example_authority"]
    assert len(authority.split(",")) == 2


def test_ai_enrichment_finish_refuses_without_persisted_focus(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_enrichment(
        tmp_path,
        persist_focus=False,
    )
    words = RecordingProvider("voicevox", 7)

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="persisted.*focus|focused deck",
    ):
        card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            "Apply and finish this enrichment.",
            record_ids=[SELECTED],
            word_provider=words,
            sentence_provider=words,
        )


def test_ai_enrichment_finish_refuses_focus_that_does_not_own_selected_card(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_enrichment(
        tmp_path,
        wrong_focus=True,
    )
    words = RecordingProvider("voicevox", 7)

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="outside its persisted focused deck",
    ):
        card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            "Apply and finish this enrichment.",
            record_ids=[SELECTED],
            word_provider=words,
            sentence_provider=words,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("furigana", "話[はな]す"), ("tags", ["lesson-8", "hidden-tag"])],
)
def test_ai_enrichment_finish_refuses_an_undeclared_landing_change(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, _deck, resource_id = _staged_enrichment(tmp_path)
    proposal = AssistantContextBroker(config).proposal_context(resource_id).path
    records, meta = staging.read_staging(proposal)
    staging.write_staging(
        proposal,
        [replace(records[0], **{field: value})],
        meta,
        force=True,
    )

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match=f"exact owner review did not display.*{field}",
    ):
        card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            "Review and finish this exact enrichment.",
            record_ids=[SELECTED],
        )


def test_ai_enrichment_finish_refuses_example_authority_tampered_after_review(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_enrichment(
        tmp_path,
        extract_source=True,
    )
    review = card_revision_finish.assistant_ai_enrichment_review.plan_ai_enrichment_review(
        config,
        resource_id=resource_id,
        record_ids=[SELECTED],
    )
    card_revision_finish.assistant_ai_enrichment_review.execute_ai_enrichment_review(
        config,
        review,
    )
    records, meta = staging.read_staging(review.proposal_path)
    source = replace(
        records[0].source,
        raw_fields={
            **records[0].source.raw_fields,
            "example_authority": "owner-never-reviewed-this-fingerprint",
        },
    )
    staging.write_staging(
        review.proposal_path,
        [replace(records[0], source=source)],
        meta,
        force=True,
    )

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="source.example_authority|example authority changed after owner review",
    ):
        card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            "Finish the already reviewed enrichment.",
        )


def test_ai_enrichment_finish_refuses_when_visible_proposal_changes(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_enrichment(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_ai_enrichment_finish(
        config,
        resource_id,
        "Apply and finish this enrichment.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )
    assert plan.review is not None
    plan.review.proposal_path.write_text(
        plan.review.proposal_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="changed after it was displayed|resource",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )

    assert not plan.record_path.exists()
    assert not words.said


def test_ai_enrichment_promotion_receipt_recovery_uses_exact_reviewed_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, resource_id = _staged_enrichment(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_ai_enrichment_finish(
        config,
        resource_id,
        "Apply and finish this enrichment.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )
    real_advance = card_revision_finish._advance

    def interrupt_after_promotion(*args: object, **kwargs: object):
        if kwargs.get("to_state") == "promoted":
            raise card_revision_finish.CardRevisionFinishError(
                "stop before AI promotion receipt"
            )
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(card_revision_finish, "_advance", interrupt_after_promotion)
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="stop before AI promotion receipt",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )
    assert not plan.review.proposal_path.exists()
    assert card_revision_finish.inspect_card_revision_finish(
        config, plan.fingerprint
    ).state == "authorized"

    monkeypatch.setattr(card_revision_finish, "_advance", real_advance)
    archive = config.staging_dir / "done" / plan.review.proposal_path.name
    archive_bytes = archive.read_bytes()
    archived, archived_meta = staging.read_staging(archive)
    tampered_source = replace(
        archived[0].source,
        raw_fields={
            **archived[0].source.raw_fields,
            "example_authority": "owner-never-reviewed-this-archive-value",
        },
    )
    staging.write_staging(
        archive,
        [replace(archived[0], source=tampered_source)],
        archived_meta,
        force=True,
    )
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="example authority changed after owner review",
    ):
        card_revision_finish.resume_card_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=words,
        )
    assert not words.said
    archive.write_bytes(archive_bytes)

    result = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=words,
    )

    assert result.state == "complete"
    assert len(words.said) == 3


def test_ai_enrichment_package_retry_never_repeats_paid_example_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, resource_id = _staged_enrichment(tmp_path)
    words = RecordingProvider("voicevox", 7)
    transport = _RealtimeTransport()
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = card_revision_finish.plan_ai_enrichment_finish(
        config,
        resource_id,
        "Apply and finish this enrichment.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=sentences,
    )
    assert plan.authority["audio"]["max_provider_calls"] == 2
    real_build = card_revision_finish.deck_package.execute_deck_package_locked

    def fail_build(*_args: object, **_kwargs: object):
        raise card_revision_finish.deck_package.DeckPackageError(
            "stop after paid AI-enrichment audio"
        )

    monkeypatch.setattr(
        card_revision_finish.deck_package,
        "execute_deck_package_locked",
        fail_build,
    )
    with pytest.raises(
        card_revision_finish.deck_package.DeckPackageError,
        match="stop after paid AI-enrichment audio",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    assert len(transport.calls) == 2
    assert card_revision_finish.inspect_card_revision_finish(
        config, plan.fingerprint
    ).state == "audio_complete"

    monkeypatch.setattr(
        card_revision_finish.deck_package,
        "execute_deck_package_locked",
        real_build,
    )
    result = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
    )

    assert result.state == "complete"
    assert len(transport.calls) == 2
