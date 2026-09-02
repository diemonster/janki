"""Paid audio dispatch is reserved durably before transport can spend."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_application_audio import Provider, _drill_project
from test_application_revision_finish import _fixture

from japanese_anki import audio_cmd, operations
from japanese_anki.application import audio as audio_application
from japanese_anki.application import revision_finish
from japanese_anki.errors import JankiError
from japanese_anki.tts import openai_realtime, sentence_profile_for

_PCM = b"\x01\x00" * 240
_SOURCE = "word:止む:やむ"
_SOURCE_SHA = "a" * 64
_TEXT = "雨、まだ止まないの？"


def _events(request: dict[str, Any]) -> list[dict[str, Any]]:
    voice = request["response"]["audio"]["output"]["voice"]
    return [
        {
            "type": "session.created",
            "session": {"model": openai_realtime.MODEL},
        },
        {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(_PCM).decode("ascii"),
        },
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": openai_realtime.SAMPLE_RATE,
                        },
                        "voice": voice,
                    }
                },
            },
        },
    ]


def _direct_provider(
    operations_path: Path,
    transport: Any,
) -> openai_realtime.OpenAiRealtimeProvider:
    return openai_realtime.OpenAiRealtimeProvider(
        record_id=_SOURCE,
        api_key="test-key",
        transport=transport,
        operations_path=operations_path,
    )


def _synthesize(
    provider: openai_realtime.OpenAiRealtimeProvider,
    *,
    before_dispatch: Any = None,
) -> str:
    return provider.synthesize_journaled(
        _TEXT,
        forced_accent=False,
        source_file=_SOURCE,
        source_sha256=_SOURCE_SHA,
        persist=lambda _wav: "pending-key",
        before_dispatch=before_dispatch,
    )


def _audio_operations(path: Path) -> list[operations.Operation]:
    return [
        operation
        for operation in operations.OperationJournal.load(path).operations.values()
        if operation.kind == "audio-realtime"
    ]


def test_realtime_callback_sees_authorized_operation_and_bound_spool_before_send(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "operations.json"
    callback_ids: list[str] = []
    transport_calls = 0

    def before_dispatch(operation_id: str) -> None:
        callback_ids.append(operation_id)
        held = operations.OperationJournal.load(journal_path).operations[operation_id]
        assert held.state == "authorized"
        assert held.response_spool is not None
        assert held.response_spool.frame_count == 0
        assert held.response_spool.committed_size == 0
        spool = journal_path.parent / held.response_spool.relative_name
        assert spool.is_file()
        assert spool.read_bytes() == b""

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal transport_calls
        transport_calls += 1
        assert len(callback_ids) == 1
        held = operations.OperationJournal.load(journal_path).operations[callback_ids[0]]
        assert held.state == "running"
        return _events(request)

    provider = _direct_provider(journal_path, transport)

    assert _synthesize(provider, before_dispatch=before_dispatch) == "pending-key"
    assert transport_calls == 1
    assert len(callback_ids) == 1
    assert _audio_operations(journal_path) == []


def test_realtime_callback_refusal_never_opens_transport(tmp_path: Path) -> None:
    journal_path = tmp_path / "operations.json"
    transport_calls = 0

    def transport(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("refused dispatch reached transport")

    def refuse(_operation_id: str) -> None:
        raise RuntimeError("finish reservation could not be saved")

    provider = _direct_provider(journal_path, transport)

    with pytest.raises(RuntimeError, match="reservation could not be saved"):
        _synthesize(provider, before_dispatch=refuse)

    assert transport_calls == 0
    held = _audio_operations(journal_path)
    assert len(held) == 1
    assert held[0].state == "failed_before_send"
    assert held[0].response_spool is not None


def test_captured_realtime_recovery_never_repeats_dispatch_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "operations.json"
    first_calls = 0

    def first_transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal first_calls
        first_calls += 1
        return _events(request)

    first = _direct_provider(journal_path, first_transport)
    real_decode = openai_realtime.OpenAiRealtimeProvider.decode_response
    monkeypatch.setattr(
        openai_realtime.OpenAiRealtimeProvider,
        "decode_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            openai_realtime.TtsError("local decode interrupted")
        ),
    )
    with pytest.raises(openai_realtime.TtsError, match="decode interrupted"):
        _synthesize(first)

    assert first_calls == 1
    assert len(_audio_operations(journal_path)) == 1
    assert _audio_operations(journal_path)[0].state == "result_captured"
    monkeypatch.setattr(
        openai_realtime.OpenAiRealtimeProvider,
        "decode_response",
        real_decode,
    )
    resumed_calls = 0

    def resumed_transport(*_args: object, **_kwargs: object) -> object:
        nonlocal resumed_calls
        resumed_calls += 1
        raise AssertionError("captured recovery opened transport")

    resumed = _direct_provider(journal_path, resumed_transport)

    assert (
        _synthesize(
            resumed,
            before_dispatch=lambda _operation_id: pytest.fail(
                "captured recovery repeated before_dispatch"
            ),
        )
        == "pending-key"
    )
    assert resumed_calls == 0
    assert _audio_operations(journal_path) == []


def _raw_dispatch(
    operation_id: str,
    clip: audio_application.AudioClipPlan,
    provider: Any,
) -> audio_cmd.AudioPaidDispatch:
    return audio_cmd.AudioPaidDispatch(
        operation_id=operation_id,
        record_id=clip.record_id,
        kind=clip.kind,
        target=clip.target,
        request_input=clip.request_input,
        forced_accent=clip.forced_accent,
        content_fingerprint=clip.content_fingerprint,
        provider=provider,
    )


def test_application_maps_only_the_exact_provider_required_paid_clip(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        operations_path=config.operations_file,
    )
    plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=sentences,
    )
    clip = plan.clips[0]
    provider = sentence_profile_for(sentences, clip.record_id)
    received: list[audio_application.PaidAudioDispatch] = []
    reserve = audio_application._paid_dispatch_adapter(plan, received.append)
    assert reserve is not None

    reserve(_raw_dispatch("exact-operation", clip, provider))

    assert received == [
        audio_application.PaidAudioDispatch(
            operation_id="exact-operation",
            clip=clip,
        )
    ]

    with pytest.raises(audio_application.AudioPlanError, match="no longer matches"):
        reserve(
            replace(
                _raw_dispatch("wrong-target", clip, provider),
                target="janki-unconfirmed.wav",
            )
        )


def test_application_rejects_a_matching_local_network_dispatch_as_nonpaid(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    sentences = Provider("voicevox", 8)
    plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=sentences,
    )
    clip = plan.clips[0]
    reserve = audio_application._paid_dispatch_adapter(
        plan,
        lambda _dispatch: pytest.fail("local audio consumed paid authority"),
    )
    assert reserve is not None

    with pytest.raises(audio_application.AudioPlanError, match="no longer matches"):
        reserve(_raw_dispatch("local-operation", clip, sentences))


def _finish_providers(
    config: Any,
    transport: Any,
) -> tuple[Provider, openai_realtime.OpenAiRealtimePool]:
    return (
        Provider("voicevox", 7),
        openai_realtime.OpenAiRealtimePool(
            api_key="test-key",
            transport=transport,
            operations_path=config.operations_file,
        ),
    )


def _authorize_clip_operation(
    config: Any,
    clip: audio_application.AudioClipPlan,
    sentences: Any,
    operation_id: str,
) -> None:
    provider = sentence_profile_for(sentences, clip.record_id)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        operation_id,
        kind="audio-realtime",
        source_file=audio_cmd.audio_journal_source(
            clip.record_id,
            of=clip.kind,
            target=clip.target,
        ),
        source_sha256=clip.content_fingerprint,
        request_fp=provider.request_fingerprint(
            clip.request_input,
            forced_accent=clip.forced_accent,
        ),
        model=openai_realtime.MODEL,
    )
    operations.OperationJournal.load(config.operations_file).begin_response_capture(operation_id)


def test_finish_reservation_is_durable_before_each_transport_dispatch(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    holder: dict[str, revision_finish.RevisionFinishPlan] = {}
    calls: list[str] = []

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        plan = holder["plan"]
        clip = plan.audio.clips[len(calls)]
        record = json.loads(plan.record_path.read_text(encoding="utf-8"))
        matching = [
            item for item in record["paid_clip_reservations"] if item["target"] == clip.target
        ]
        assert len(matching) == 1
        operation_id = matching[0]["operation_id"]
        held = operations.OperationJournal.load(config.operations_file).operations[operation_id]
        assert held.state == "running"
        assert held.response_spool is not None
        assert (config.operations_file.parent / held.response_spool.relative_name).is_file()
        assert record["state"] == "revision_applied"
        calls.append(clip.target)
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    holder["plan"] = plan

    result = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert result.succeeded
    assert calls == [clip.target for clip in plan.audio.clips]
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert [item["target"] for item in record["paid_clip_reservations"]] == calls


def test_finish_reservation_refuses_redispatch_when_operation_evidence_is_lost(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def crashing_transport(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt("simulated process death at transport")

    words, sentences = _finish_providers(config, crashing_transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )

    with pytest.raises(KeyboardInterrupt, match="process death"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["state"] == "revision_applied"
    assert len(record["paid_clip_reservations"]) == 1
    operation_id = record["paid_clip_reservations"][0]["operation_id"]
    assert (
        operations.OperationJournal.load(config.operations_file).operations[operation_id].state
        == "outcome_unknown"
    )

    # Simulate irrecoverable external evidence loss. The finish receipt must
    # still prevent its already-consumed authority from becoming a new call.
    journal = json.loads(config.operations_file.read_text(encoding="utf-8"))
    del journal["operations"][operation_id]
    config.operations_file.write_text(
        json.dumps(journal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_finish.RevisionFinishError, match="operation evidence"):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert calls == 1


def test_finish_resumes_an_exact_captured_operation_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    spoken: list[str] = []

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        spoken.append(request["response"]["input"][0]["content"][0]["text"])
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    first_text = plan.audio.clips[0].request_input
    real_decode = openai_realtime.OpenAiRealtimeProvider.decode_response
    monkeypatch.setattr(
        openai_realtime.OpenAiRealtimeProvider,
        "decode_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            openai_realtime.TtsError("local decode interrupted")
        ),
    )

    partial = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert partial.state == "revision_applied"
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert len(record["paid_clip_reservations"]) == 1
    operation_id = record["paid_clip_reservations"][0]["operation_id"]
    assert (
        operations.OperationJournal.load(config.operations_file).operations[operation_id].state
        == "result_captured"
    )
    assert spoken == [first_text]
    monkeypatch.setattr(
        openai_realtime.OpenAiRealtimeProvider,
        "decode_response",
        real_decode,
    )

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert spoken.count(first_text) == 1
    assert len(spoken) == plan.provider_required_count


def test_finish_resumes_exact_recoverable_wal_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_promote = audio_application.audio_cmd.promote_pending_audio
    monkeypatch.setattr(
        audio_application.audio_cmd,
        "promote_pending_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            audio_application.JankiError("simulated publication interruption")
        ),
    )

    partial = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert partial.state == "revision_applied"
    assert partial.audio is not None and partial.audio.pending_recovery
    calls_before_resume = calls
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert len(record["paid_clip_reservations"]) == plan.provider_required_count
    assert _audio_operations(config.operations_file) == []
    monkeypatch.setattr(
        audio_application.audio_cmd,
        "promote_pending_audio",
        real_promote,
    )

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == calls_before_resume


def test_finish_resumes_current_media_after_receipt_transition_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_advance = revision_finish._advance_record

    def interrupt_audio_receipt(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("to_state") == "audio_complete":
            raise KeyboardInterrupt("simulated death before audio receipt")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(
        revision_finish,
        "_advance_record",
        interrupt_audio_receipt,
    )
    with pytest.raises(KeyboardInterrupt, match="before audio receipt"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["state"] == "revision_applied"
    assert len(record["paid_clip_reservations"]) == plan.provider_required_count
    assert all(
        clip.state == "current"
        for clip in audio_application.plan_deck_audio(
            config,
            plan.revision.deck_path,
            record_ids=plan.audio.record_ids,
            word_provider=words,
            sentence_provider=sentences,
        ).clips
    )
    calls_before_resume = calls
    monkeypatch.setattr(revision_finish, "_advance_record", real_advance)

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == calls_before_resume


def test_finish_releases_a_callback_bound_before_send_failure_then_retries_exactly(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "type": "session.created",
                    "session": {"model": "wrong-realtime-model"},
                }
            ]
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    authority = dict(plan.authority)

    partial = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert partial.state == "revision_applied"
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["authority"] == authority
    assert len(record["paid_clip_reservations"]) == 1
    first_reservation = dict(record["paid_clip_reservations"][0])
    assert first_reservation["status"] == "reserved"
    first_operation = operations.OperationJournal.load(config.operations_file).operations[
        first_reservation["operation_id"]
    ]
    assert first_operation.state == "failed_before_send"
    assert first_operation.response_spool is not None

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count + 1
    assert (
        first_reservation["operation_id"]
        not in operations.OperationJournal.load(config.operations_file).operations
    )
    final = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert final["authority"] == authority
    assert len(final["paid_clip_reservations"]) == plan.provider_required_count
    retried = [
        item
        for item in final["paid_clip_reservations"]
        if item["target"] == first_reservation["target"]
    ]
    assert len(retried) == 1
    assert retried[0]["operation_id"] != first_reservation["operation_id"]
    assert retried[0]["status"] == "reserved"


def test_finish_releases_an_authorized_callback_crash_then_retries_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    authority = dict(plan.authority)
    real_advance = operations.OperationJournal.advance

    def crash_before_dispatch(
        journal: operations.OperationJournal,
        operation_id: str,
        state: str,
        *,
        detail: str = "",
    ) -> operations.Operation:
        if state == "dispatching":
            raise KeyboardInterrupt("simulated death after reservation")
        return real_advance(journal, operation_id, state, detail=detail)

    monkeypatch.setattr(
        operations.OperationJournal,
        "advance",
        crash_before_dispatch,
    )
    with pytest.raises(KeyboardInterrupt, match="after reservation"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert calls == 0
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["authority"] == authority
    assert len(record["paid_clip_reservations"]) == 1
    first_reservation = dict(record["paid_clip_reservations"][0])
    first_operation = operations.OperationJournal.load(config.operations_file).operations[
        first_reservation["operation_id"]
    ]
    assert first_operation.state == "authorized"
    assert first_operation.response_spool is not None
    monkeypatch.setattr(operations.OperationJournal, "advance", real_advance)

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count
    assert (
        first_reservation["operation_id"]
        not in operations.OperationJournal.load(config.operations_file).operations
    )
    final = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert final["authority"] == authority
    retried = [
        item
        for item in final["paid_clip_reservations"]
        if item["target"] == first_reservation["target"]
    ]
    assert len(retried) == 1
    assert retried[0]["operation_id"] != first_reservation["operation_id"]


def test_missing_key_after_authorized_reservation_refuses_before_revision_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)

    def transport(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("missing-key resume opened transport")

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_execute = revision_finish._execute_record_locked

    def stop_after_authority(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("simulated stop after finish authority")

    monkeypatch.setattr(
        revision_finish,
        "_execute_record_locked",
        stop_after_authority,
    )
    with pytest.raises(RuntimeError, match="after finish authority"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    monkeypatch.setattr(revision_finish, "_execute_record_locked", real_execute)

    clip = plan.audio.clips[0]
    provider = sentence_profile_for(sentences, clip.record_id)
    operation_id = "22222222-2222-4222-8222-222222222222"
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        operation_id,
        kind="audio-realtime",
        source_file=audio_cmd.audio_journal_source(
            clip.record_id,
            of=clip.kind,
            target=clip.target,
        ),
        source_sha256=clip.content_fingerprint,
        request_fp=provider.request_fingerprint(
            clip.request_input,
            forced_accent=clip.forced_accent,
        ),
        model=openai_realtime.MODEL,
    )
    operations.OperationJournal.load(config.operations_file).begin_response_capture(operation_id)
    record, record_revision = revision_finish._read_record(plan.record_path)
    revision_finish._reserve_paid_clip(
        config,
        plan.record_path,
        record,
        record_revision,
        audio_application.PaidAudioDispatch(operation_id, clip),
        sentence_provider=sentences,
    )
    deck_before = deck.read_bytes()
    staging_before = staging.read_bytes()
    no_key_sentences = openai_realtime.OpenAiRealtimePool(
        api_key="",
        transport=transport,
        operations_path=config.operations_file,
    )

    with pytest.raises(JankiError, match="OPENAI_API_KEY"):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=no_key_sentences,
        )

    assert deck.read_bytes() == deck_before
    assert staging.read_bytes() == staging_before
    assert not plan.revision.archive_path.exists()
    assert operation_id not in operations.OperationJournal.load(config.operations_file).operations
    released = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert released["state"] == "authorized"
    assert released["paid_clip_reservations"] == [
        {
            "operation_id": operation_id,
            "status": "failed_before_send",
            "target": clip.target,
        }
    ]


def test_failed_reservation_does_not_hide_an_authorized_replacement_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_execute = revision_finish._execute_record_locked

    def stop_after_authority(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("simulated stop after finish authority")

    monkeypatch.setattr(revision_finish, "_execute_record_locked", stop_after_authority)
    with pytest.raises(RuntimeError, match="after finish authority"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    monkeypatch.setattr(revision_finish, "_execute_record_locked", real_execute)

    clip = plan.audio.clips[0]
    failed_id = "33333333-3333-4333-8333-333333333333"
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    record["paid_clip_reservations"] = [
        {
            "target": clip.target,
            "operation_id": failed_id,
            "status": "failed_before_send",
        }
    ]
    plan.record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    orphan_id = "44444444-4444-4444-8444-444444444444"
    _authorize_clip_operation(config, clip, sentences, orphan_id)

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count
    assert orphan_id not in operations.OperationJournal.load(config.operations_file).operations
    final = json.loads(plan.record_path.read_text(encoding="utf-8"))
    replacement = [
        item for item in final["paid_clip_reservations"] if item["target"] == clip.target
    ]
    assert len(replacement) == 1
    assert replacement[0]["operation_id"] not in {failed_id, orphan_id}


def test_finish_retries_interrupted_cleanup_for_an_orphaned_unsent_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    clip = plan.audio.clips[0]
    operation_id = "33333333-3333-4333-8333-333333333333"
    _authorize_clip_operation(config, clip, sentences, operation_id)
    operations.OperationJournal.load(config.operations_file).advance(
        operation_id,
        "failed_before_send",
        detail="proven unsent before finish authority was recorded",
    )
    real_retire = operations._retire_response_spool

    def interrupt_cleanup(_receipt: operations.ResponseSpoolReceipt) -> None:
        raise operations.OperationError("simulated response-spool cleanup failure")

    monkeypatch.setattr(operations, "_retire_response_spool", interrupt_cleanup)
    with pytest.raises(operations.OperationError, match="cleanup remains recorded"):
        operations.OperationJournal.load(config.operations_file).forget([operation_id])

    held = operations.OperationJournal.load(config.operations_file).operations[operation_id]
    assert held.state == "failed_before_send"
    assert held.cleanup is not None
    monkeypatch.setattr(operations, "_retire_response_spool", real_retire)

    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count
    assert operation_id not in operations.OperationJournal.load(config.operations_file).operations


def test_finish_retries_interrupted_cleanup_after_audio_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_retire = operations._retire_response_spool

    def interrupt_cleanup(_receipt: operations.ResponseSpoolReceipt) -> None:
        raise operations.OperationError("simulated committed cleanup failure")

    monkeypatch.setattr(operations, "_retire_response_spool", interrupt_cleanup)
    partial = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert partial.state == "revision_applied"
    assert partial.audio is not None and partial.audio.pending_recovery
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert len(record["paid_clip_reservations"]) == 1
    operation_id = record["paid_clip_reservations"][0]["operation_id"]
    held = operations.OperationJournal.load(config.operations_file).operations[operation_id]
    assert held.state == "committed"
    assert held.cleanup is not None
    fresh = audio_application.plan_deck_audio(
        config,
        plan.revision.deck_path,
        word_provider=words,
        sentence_provider=sentences,
    )
    recovered = [clip for clip in fresh.clips if clip.state == "recoverable"]
    assert [clip.target for clip in recovered] == [record["paid_clip_reservations"][0]["target"]]
    calls_before_resume = calls
    monkeypatch.setattr(operations, "_retire_response_spool", real_retire)
    real_execute_audio = audio_application.execute_deck_audio_locked

    def execute_after_cleanup(*args: Any, **kwargs: Any) -> Any:
        assert (
            operation_id not in operations.OperationJournal.load(config.operations_file).operations
        )
        return real_execute_audio(*args, **kwargs)

    monkeypatch.setattr(
        audio_application,
        "execute_deck_audio_locked",
        execute_after_cleanup,
    )

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count
    assert calls_before_resume == 1
    assert operation_id not in operations.OperationJournal.load(config.operations_file).operations


def test_finish_releases_canceled_before_send_and_retries_the_exact_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_advance = operations.OperationJournal.advance

    def crash_before_dispatch(
        journal: operations.OperationJournal,
        operation_id: str,
        state: str,
        *,
        detail: str = "",
    ) -> operations.Operation:
        if state == "dispatching":
            raise KeyboardInterrupt("simulated stop before dispatch")
        return real_advance(journal, operation_id, state, detail=detail)

    monkeypatch.setattr(
        operations.OperationJournal,
        "advance",
        crash_before_dispatch,
    )
    with pytest.raises(KeyboardInterrupt, match="before dispatch"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    monkeypatch.setattr(operations.OperationJournal, "advance", real_advance)

    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    first = dict(record["paid_clip_reservations"][0])
    ended = operations.OperationJournal.load(config.operations_file).end(first["operation_id"])
    assert ended.state == "canceled_before_send"
    assert calls == 0

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert calls == plan.provider_required_count
    assert (
        first["operation_id"]
        not in operations.OperationJournal.load(config.operations_file).operations
    )
    final = json.loads(plan.record_path.read_text(encoding="utf-8"))
    replacement = [
        item for item in final["paid_clip_reservations"] if item["target"] == first["target"]
    ]
    assert len(replacement) == 1
    assert replacement[0]["operation_id"] != first["operation_id"]


def test_completed_finish_rebuilds_missing_package_after_tts_config_changes(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return _events(request)

    words, sentences = _finish_providers(config, transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert completed.succeeded
    calls_before_resume = calls
    plan.build.output_path.unlink()
    with (tmp_path / "janki.toml").open("a", encoding="utf-8") as config_file:
        config_file.write(
            '\n[tts]\nprovider = "not-a-provider"\nsentence_provider = "not-a-sentence-provider"\n'
        )
    changed_config = type(config).load(tmp_path)

    rebuilt = revision_finish.resume_revision_finish(
        changed_config,
        plan.fingerprint,
    )

    assert rebuilt.succeeded
    assert calls == calls_before_resume
    assert plan.build.output_path.is_file()
    assert rebuilt.package_sha256 == hashlib.sha256(plan.build.output_path.read_bytes()).hexdigest()
