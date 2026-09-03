"""One receipt-backed promotion/audio/package finish for generic revisions."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_application_audio import RecordingProvider
from test_application_card_revision import (
    OPERATION_ID,
    SELECTED,
    _answer,
    _api_call,
    _project,
)
from test_application_revision_finish import _RealtimeTransport

from japanese_anki import staging
from japanese_anki.application import (
    assistant_card_revision_review,
    card_revision,
    card_revision_finish,
)
from japanese_anki.application import promotion as promotion_application
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records
from japanese_anki.models import EXAMPLE_AUTHORITY_KEY
from japanese_anki.tts import openai_realtime


def _reviewed_revision(
    tmp_path: Path,
    *,
    answer: object | None = None,
) -> tuple[ProjectConfig, Path, str]:
    config, deck, resource_id = _staged_revision(tmp_path, answer=answer)
    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=resource_id,
        record_ids=[SELECTED],
    )
    assistant_card_revision_review.execute_card_revision_review(config, review)
    return config, deck, resource_id


def _staged_revision(
    tmp_path: Path,
    *,
    answer: object | None = None,
) -> tuple[ProjectConfig, Path, str]:
    config, deck = _project(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    revision = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card's usage note.",
        operation_id=OPERATION_ID,
        focus_resource_id="deck:lesson-8",
    )
    card_revision.run_card_revision(
        config,
        revision,
        client=object(),
        api_call=_api_call(config, revision, [], answer=answer),
    )
    resource = next(
        item
        for item in json.loads(AssistantContextBroker(config).catalog().wire)["data"][
            "resources"
        ]
        if item.get("proposal_kind") == "card_revision"
    )
    resource_id = str(resource["resource_id"])
    return config, deck, resource_id


def test_one_confirmation_persists_receipt_before_review_and_resumes_that_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, resource_id = _staged_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Review this visible change, apply it, voice it, and build its deck.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )
    assert plan.review is not None
    assert plan.promotion is None
    assert plan.authority["review"]["projection"] == plan.review.projection
    assert not plan.record_path.exists()
    calls: list[str] = []
    real_review = (
        card_revision_finish.assistant_card_revision_review.execute_card_revision_review
    )
    real_replan = card_revision_finish._replan_live_promotion

    def observe_review(*args: object, **kwargs: object):
        calls.append("review")
        receipt = card_revision_finish.inspect_card_revision_finish(
            config, plan.fingerprint
        )
        assert receipt.state == "authorized"
        return real_review(*args, **kwargs)

    def interrupt_after_review(*_args: object, **_kwargs: object):
        raise card_revision_finish.CardRevisionFinishError(
            "simulated interruption after owner review"
        )

    monkeypatch.setattr(
        card_revision_finish.assistant_card_revision_review,
        "execute_card_revision_review",
        observe_review,
    )
    monkeypatch.setattr(
        card_revision_finish,
        "_replan_live_promotion",
        interrupt_after_review,
    )
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="simulated interruption after owner review",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )
    assert calls == ["review"]

    monkeypatch.setattr(card_revision_finish, "_replan_live_promotion", real_replan)
    resumed = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=words,
    )

    assert resumed.state == "complete"
    assert calls == ["review"]


def test_plan_is_pure_and_execute_reuses_promotion_audio_and_package_services(
    tmp_path: Path,
) -> None:
    config, deck, resource_id = _reviewed_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    first = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply the reviewed card and finish this deck.",
        word_provider=words,
        sentence_provider=words,
    )
    second = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply the reviewed card and finish this deck.",
        word_provider=words,
        sentence_provider=words,
    )

    assert first == second
    assert first.deck_path == deck.absolute()
    assert first.record_ids == (SELECTED,)
    assert first.audio.words is True
    assert first.audio.examples is True
    assert first.audio.force is False
    assert first.provider_required_count == 1
    assert first.build.kind == "vocabulary"
    assert first.record_path.name == f"card-finish-{first.fingerprint}.json"
    assert not first.record_path.exists()
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before

    result = card_revision_finish.execute_card_revision_finish(
        config,
        first,
        word_provider=words,
        sentence_provider=words,
    )

    assert result.state == "complete"
    assert result.receipt_id == first.fingerprint
    assert result.record_path == first.record_path
    assert result.output_path.exists()
    selected = {record.id: record for record in load_records(config.normalized_file)}[
        SELECTED
    ]
    assert selected.usage_notes == "Use this when someone speaks or talks."
    assert selected.audio.startswith("audio/janki-")
    assert len(words.said) == 1

    repeated = card_revision_finish.resume_card_revision_finish(
        config,
        result.receipt_id,
    )
    assert repeated == result
    assert len(words.said) == 1


@pytest.mark.parametrize("source_type", ["extract", "manual"])
def test_generic_example_revision_finish_scopes_example_authority_to_extract(
    tmp_path: Path, source_type: str
) -> None:
    config, deck = _project(tmp_path)
    payload = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    payload[0]["source"]["type"] = source_type
    payload[0]["source"]["raw_fields"] = {}
    config.normalized_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    revision = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Replace this card's polite and casual examples.",
        operation_id=OPERATION_ID,
        focus_resource_id="deck:lesson-8",
    )
    examples = [
        {
            "japanese": "日本語を話します。",
            "furigana": "日本語[にほんご]を 話[はな]します。",
            "romaji": "nihongo o hanashimasu",
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
    ]
    card_revision.run_card_revision(
        config,
        revision,
        client=object(),
        api_call=_api_call(
            config,
            revision,
            [],
            answer=_answer(
                revision,
                field="examples",
                value_json=json.dumps(examples, ensure_ascii=False),
            ),
        ),
    )
    resource_id = str(
        next(
            item
            for item in json.loads(AssistantContextBroker(config).catalog().wire)[
                "data"
            ]["resources"]
            if item.get("proposal_kind") == "card_revision"
        )["resource_id"]
    )
    provider = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Review, apply, voice, and build these exact examples.",
        record_ids=[SELECTED],
        word_provider=provider,
        sentence_provider=provider,
    )

    result = card_revision_finish.execute_card_revision_finish(
        config,
        plan,
        word_provider=provider,
        sentence_provider=provider,
    )

    assert result.succeeded
    landed = {record.id: record for record in load_records(config.normalized_file)}[
        SELECTED
    ]
    assert len(landed.source.raw_fields[EXAMPLE_AUTHORITY_KEY].split(",")) == 2
    assert result.package_sha256 == hashlib.sha256(
        result.output_path.read_bytes()
    ).hexdigest()
    assert plan.review is not None
    if source_type != "extract":
        return
    archive = config.staging_dir / "done" / plan.review.proposal_path.name
    archived, meta = staging.read_staging(archive)
    tampered_source = replace(
        archived[0].source,
        raw_fields={
            **archived[0].source.raw_fields,
            EXAMPLE_AUTHORITY_KEY: "owner-never-reviewed-this-archive-value",
        },
    )
    with pytest.raises(
        promotion_application.PromoteError,
        match="example authority changed after owner review",
    ):
        promotion_application.staged_card_revision(
            meta,
            (),
            archived_ids=(SELECTED,),
            archived_records=(replace(archived[0], source=tampered_source),),
            require_owner_review=True,
        )


def test_execute_refuses_a_stale_review_before_creating_finish_authority(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _reviewed_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply and finish.",
        word_provider=words,
        sentence_provider=words,
    )
    proposal = plan.promotion.proposal_path
    proposal.write_text(proposal.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="changed after it was displayed|owner review",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )

    assert not plan.record_path.exists()
    assert not words.said


def test_execute_refuses_a_stale_visible_pending_review_before_receipt(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Review, apply, voice, and build this exact visible change.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )
    assert plan.review is not None
    proposal = plan.review.proposal_path
    proposal.write_text(proposal.read_text(encoding="utf-8") + "\n", encoding="utf-8")

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


def test_execute_replans_and_refuses_a_tampered_display_fingerprint(
    tmp_path: Path,
) -> None:
    config, _deck, resource_id = _staged_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Review, apply, voice, and build this exact visible change.",
        record_ids=[SELECTED],
        word_provider=words,
        sentence_provider=words,
    )
    tampered = replace(plan, fingerprint="0" * 64)

    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="changed after it was displayed",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            tampered,
            word_provider=words,
            sentence_provider=words,
        )

    assert not plan.record_path.exists()
    assert not words.said


def test_resume_recovers_exact_promotion_if_receipt_update_was_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, resource_id = _reviewed_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply and finish.",
        word_provider=words,
        sentence_provider=words,
    )
    real_advance = card_revision_finish._advance

    def interrupt_after_promotion(*args: object, **kwargs: object):
        if kwargs.get("to_state") == "promoted":
            raise card_revision_finish.CardRevisionFinishError(
                "simulated promotion receipt interruption"
            )
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(card_revision_finish, "_advance", interrupt_after_promotion)
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="simulated promotion receipt interruption",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )
    assert not plan.promotion.proposal_path.exists()
    assert card_revision_finish.inspect_card_revision_finish(
        config, plan.fingerprint
    ).state == "authorized"

    monkeypatch.setattr(card_revision_finish, "_advance", real_advance)
    resumed = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=words,
    )

    assert resumed.state == "complete"
    assert len(words.said) == 1


def test_package_failure_resumes_without_repeating_finished_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, resource_id = _reviewed_revision(tmp_path)
    words = RecordingProvider("voicevox", 7)
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply and finish.",
        word_provider=words,
        sentence_provider=words,
    )
    real_build = card_revision_finish.deck_package.execute_deck_package_locked

    def fail_build(*_args: object, **_kwargs: object):
        raise card_revision_finish.deck_package.DeckPackageError(
            "simulated package failure"
        )

    monkeypatch.setattr(
        card_revision_finish.deck_package,
        "execute_deck_package_locked",
        fail_build,
    )
    with pytest.raises(
        card_revision_finish.deck_package.DeckPackageError,
        match="simulated package failure",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=words,
        )
    assert card_revision_finish.inspect_card_revision_finish(
        config, plan.fingerprint
    ).state == "audio_complete"
    assert len(words.said) == 1

    monkeypatch.setattr(
        card_revision_finish.deck_package,
        "execute_deck_package_locked",
        real_build,
    )
    resumed = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
    )

    assert resumed.state == "complete"
    assert len(words.said) == 1


def test_resume_after_paid_audio_does_not_dispatch_the_same_clips_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One revision deliberately introduces two examples, which makes this
    # fixture exercise the paid Realtime half as well as the local word clip.
    config, deck = _project(tmp_path)
    revision = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Add exact polite and casual examples.",
        operation_id=OPERATION_ID,
    )
    examples = [
        {
            "japanese": "日本語を話します。",
            "furigana": "日本語[にほんご]を 話[はな]します。",
            "romaji": "nihongo o hanashimasu",
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
    ]
    answer = _answer(
        revision,
        field="examples",
        value_json=json.dumps(examples, ensure_ascii=False),
    )
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    card_revision.run_card_revision(
        config,
        revision,
        client=object(),
        api_call=_api_call(config, revision, [], answer=answer),
    )
    resource = next(
        item
        for item in json.loads(AssistantContextBroker(config).catalog().wire)["data"][
            "resources"
        ]
        if item.get("proposal_kind") == "card_revision"
    )
    resource_id = str(resource["resource_id"])
    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=resource_id,
        record_ids=[SELECTED],
    )
    assistant_card_revision_review.execute_card_revision_review(config, review)
    transport = _RealtimeTransport()
    words = RecordingProvider("voicevox", 7)
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = card_revision_finish.plan_card_revision_finish(
        config,
        resource_id,
        "Apply the reviewed examples, voice them, and build the deck.",
        word_provider=words,
        sentence_provider=sentences,
    )
    assert plan.provider_required_count == 3
    assert plan.authority["audio"]["max_provider_calls"] == 2
    real_advance = card_revision_finish._advance

    def interrupt_after_audio(*args: object, **kwargs: object):
        if kwargs.get("to_state") == "audio_complete":
            raise card_revision_finish.CardRevisionFinishError(
                "simulated receipt interruption"
            )
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(card_revision_finish, "_advance", interrupt_after_audio)
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="simulated receipt interruption",
    ):
        card_revision_finish.execute_card_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    assert len(transport.calls) == 2
    assert len(words.said) == 1

    monkeypatch.setattr(card_revision_finish, "_advance", real_advance)
    paid_target = next(
        clip.target
        for clip in plan.audio.clips
        if clip.kind == "example" and clip.provider.access == "paid-network"
    )
    paid_path = config.media_dir / "audio" / paid_target
    paid_bytes = paid_path.read_bytes()
    paid_path.unlink()
    with pytest.raises(
        card_revision_finish.CardRevisionFinishError,
        match="operation evidence was removed; refusing to bill",
    ):
        card_revision_finish.resume_card_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )
    assert len(transport.calls) == 2
    paid_path.write_bytes(paid_bytes)

    resumed = card_revision_finish.resume_card_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert resumed.state == "complete"
    assert len(transport.calls) == 2
    assert len(words.said) == 1
