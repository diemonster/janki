"""Operation-journal contract for paid Realtime audio; no real transport."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import audio_cmd, ledger, operations
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.tts import TtsError
from japanese_anki.tts.openai_realtime import (
    MODEL,
    OpenAiRealtimeProvider,
    voice_for_record,
)

PCM = b"\x01\x00" * 240
SOURCE = "word:止む:やむ"
SOURCE_SHA = "a" * 64
TEXT = "雨、まだ止まないの？"


def _events(*, delta: str | None = None) -> list[dict[str, Any]]:
    return [
        {"type": "session.created", "session": {"model": MODEL}},
        {"type": "response.output_audio.delta", "delta": delta or base64.b64encode(PCM).decode()},
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "voice": voice_for_record(SOURCE),
                    }
                },
            },
        },
    ]


def _only(path: Path) -> operations.Operation:
    held = operations.OperationJournal.load(path).operations
    assert len(held) == 1
    return next(iter(held.values()))


class Transport:
    def __init__(self, path: Path, events: list[dict[str, Any]]) -> None:
        self.path, self.events, self.calls = path, events, 0

    def __call__(self, _endpoint: str, _headers: dict[str, str], _request: dict[str, Any]):
        self.calls += 1
        assert _only(self.path).state in {"dispatching", "running"}
        return self.events


def _provider(tmp_path: Path, events: list[dict[str, Any]] | None = None):
    path = tmp_path / "operations.json"
    transport = Transport(path, events or _events())
    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=transport,
        operations_path=path,
    )
    return provider, transport, path


def _synthesize(provider: OpenAiRealtimeProvider, persist):
    return provider.synthesize_journaled(
        TEXT,
        forced_accent=False,
        source_file=SOURCE,
        source_sha256=SOURCE_SHA,
        persist=persist,
    )


def _seed_realtime_operation(
    provider: OpenAiRealtimeProvider,
    path: Path,
) -> operations.OperationJournal:
    journal = operations.OperationJournal.load(path)
    journal.authorize(
        "recover-terminal",
        kind="audio-realtime",
        source_file=SOURCE,
        source_sha256=SOURCE_SHA,
        request_fp=provider.request_fingerprint(TEXT, forced_accent=False),
        model=MODEL,
    )
    journal.begin_response_capture("recover-terminal")
    journal.advance("recover-terminal", "dispatching")
    journal.advance("recover-terminal", "running")
    return journal


def _event_wire(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def _record() -> VocabularyRecord:
    return VocabularyRecord(
        id=SOURCE,
        expression="止む",
        reading="やむ",
        meanings=["to stop"],
        examples=[ExampleSentence(japanese=TEXT)],
        source=SourceReference(type="manual", imported_from="lesson"),
    )


def _generate(tmp_path: Path, provider: OpenAiRealtimeProvider, book: ledger.Ledger):
    return audio_cmd.generate_audio(
        [_record()],
        provider=provider,
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        stage_only=True,
        persist_pending=lambda current, _key: current.save(),
    )


def test_authority_precedes_send_capture_precedes_persist_and_persist_precedes_cleanup(
    tmp_path: Path,
) -> None:
    provider, transport, path = _provider(tmp_path)
    saved: list[bytes] = []

    def persist(wav: bytes) -> str:
        held = _only(path)
        assert held.state == "result_captured"
        assert held.artifact is not None
        assert b"response.done" in (path.parent / held.artifact.relative_name).read_bytes()
        saved.append(wav)
        return "pending-key"

    _synthesize(provider, persist)
    assert transport.calls == 1
    assert saved and saved[0].startswith(b"RIFF")
    assert operations.OperationJournal.load(path).operations == {}


def test_running_state_is_durable_before_transport_can_receive_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        _request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        assert _only(path).state == "running"
        return _events()

    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=transport,
        operations_path=path,
    )
    _synthesize(provider, lambda _wav: "pending-key")

    assert calls == 1
    assert operations.OperationJournal.load(path).operations == {}


def test_plain_synthesis_refuses_before_transport(tmp_path: Path) -> None:
    provider, transport, path = _provider(tmp_path)

    with pytest.raises(TtsError, match="journal|staging"):
        provider.synthesize(TEXT, forced_accent=False)

    assert transport.calls == 0
    assert not path.exists()


def test_terminal_envelope_is_captured_before_decode_refuses(tmp_path: Path) -> None:
    provider, transport, path = _provider(tmp_path, _events(delta="not-base64"))
    with pytest.raises(TtsError, match="base64|audio"):
        _synthesize(provider, lambda _wav: pytest.fail("invalid audio persisted"))
    assert transport.calls == 1
    held = _only(path)
    assert held.state == "result_captured"
    assert b"not-base64" in operations.OperationJournal.load(path).read_reply(held.operation_id)


def test_malformed_received_frame_is_durable_before_json_refusal(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        _request: dict[str, Any],
    ) -> list[str]:
        nonlocal calls
        calls += 1
        return ['{"type":"session.created","session":{"model":"gpt-realtime-1.5"}}', "{not-json"]

    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=transport,
        operations_path=path,
    )

    with pytest.raises(TtsError, match="JSON|malformed"):
        _synthesize(provider, lambda _wav: pytest.fail("malformed reply persisted"))

    assert calls == 1
    held = _only(path)
    assert held.state == "result_captured"
    assert b"{not-json" in operations.OperationJournal.load(path).read_reply(held.operation_id)


def test_wrong_handshake_is_known_before_send_and_retires_its_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    trace: list[str] = []

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        _request: dict[str, Any],
    ):
        def stream():
            trace.append("received-handshake")
            yield json.dumps(
                {"type": "session.created", "session": {"model": "wrong-model"}},
                separators=(",", ":"),
            )
            trace.append("sent-response-create")
            yield from _events()[1:]

        return stream()

    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=transport,
        operations_path=path,
    )

    with pytest.raises(TtsError, match="model|handshake|session"):
        _synthesize(provider, lambda _wav: pytest.fail("nothing was dispatched"))

    assert trace == ["received-handshake"]
    assert operations.OperationJournal.load(path).operations == {}
    assert list((tmp_path / ".pending").glob("*")) == []


def test_connection_failure_before_the_handshake_is_known_before_send(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    calls = 0

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        _request: dict[str, Any],
    ):
        nonlocal calls
        calls += 1
        raise OSError("connection refused before handshake")

    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=transport,
        operations_path=path,
    )

    with pytest.raises(TtsError, match="connection|Realtime"):
        _synthesize(provider, lambda _wav: pytest.fail("nothing was dispatched"))

    assert calls == 1
    assert operations.OperationJournal.load(path).operations == {}
    assert list((tmp_path / ".pending").glob("*")) == []


def test_terminal_frame_append_interruption_resumes_without_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, first_transport, path = _provider(tmp_path)
    real_append = operations.OperationJournal.append_response_frame

    def append_then_interrupt(
        journal: operations.OperationJournal,
        operation_id: str,
        payload: str,
    ) -> None:
        real_append(journal, operation_id, payload)
        if json.loads(payload).get("type") == "response.done":
            raise KeyboardInterrupt("injected death after durable terminal frame")

    monkeypatch.setattr(
        operations.OperationJournal,
        "append_response_frame",
        append_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="durable terminal"):
        _synthesize(provider, lambda _wav: pytest.fail("interrupted before decode"))

    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"

    monkeypatch.setattr(
        operations.OperationJournal,
        "append_response_frame",
        real_append,
    )
    resumed, resumed_transport, _ = _provider(tmp_path)
    saved: list[bytes] = []
    _synthesize(resumed, lambda wav: saved.append(wav) or "pending-key")

    assert resumed_transport.calls == 0
    assert saved and saved[0].startswith(b"RIFF")
    assert operations.OperationJournal.load(path).operations == {}


def test_result_captured_resumes_locally_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, first_transport, path = _provider(tmp_path)
    real_decode = OpenAiRealtimeProvider.decode_response
    monkeypatch.setattr(
        OpenAiRealtimeProvider,
        "decode_response",
        lambda *_: (_ for _ in ()).throw(TtsError("local decode interrupted")),
    )
    with pytest.raises(TtsError, match="local decode interrupted"):
        _synthesize(provider, lambda _wav: pytest.fail("decode did not finish"))
    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"

    monkeypatch.setattr(OpenAiRealtimeProvider, "decode_response", real_decode)
    resumed, resumed_transport, _ = _provider(tmp_path)
    saved: list[bytes] = []
    _synthesize(resumed, lambda wav: saved.append(wav) or "pending-key")
    assert resumed_transport.calls == 0
    assert saved
    assert operations.OperationJournal.load(path).operations == {}


def test_end_then_rerun_seals_fully_journaled_terminal_spool_without_transport(
    tmp_path: Path,
) -> None:
    provider, transport, path = _provider(tmp_path)
    journal = _seed_realtime_operation(provider, path)
    for event in _events():
        journal.append_response_frame("recover-terminal", _event_wire(event))

    ended = operations.OperationJournal.load(path).end("recover-terminal")
    assert ended.state == "outcome_unknown"
    before_calls = transport.calls
    saved: list[bytes] = []

    _synthesize(provider, lambda wav: saved.append(wav) or "pending-key")

    assert transport.calls == before_calls == 0
    assert saved and saved[0].startswith(b"RIFF")
    with wave.open(io.BytesIO(saved[0]), "rb") as rendered:
        assert rendered.readframes(rendered.getnframes()) == PCM
    assert operations.OperationJournal.load(path).operations == {}


def test_end_adopts_terminal_crash_extension_then_rerun_recovers_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, transport, path = _provider(tmp_path)
    journal = _seed_realtime_operation(provider, path)
    events = _events()
    for event in events[:-1]:
        journal.append_response_frame("recover-terminal", _event_wire(event))
    real_write = operations.OperationJournal._write

    def fail_terminal_head(current: operations.OperationJournal) -> None:
        held = current.operations["recover-terminal"]
        receipt = held.response_spool
        if receipt is not None and receipt.frame_count == len(events):
            raise operations.OperationError("injected terminal head failure")
        real_write(current)

    monkeypatch.setattr(operations.OperationJournal, "_write", fail_terminal_head)
    with pytest.raises(operations.OperationError, match="terminal head failure"):
        journal.append_response_frame("recover-terminal", _event_wire(events[-1]))
    monkeypatch.setattr(operations.OperationJournal, "_write", real_write)
    stale = _only(path)
    assert stale.response_spool is not None
    assert stale.response_spool.frame_count == len(events) - 1

    ended = operations.OperationJournal.load(path).end("recover-terminal")
    assert ended.state == "outcome_unknown"
    assert ended.response_spool is not None
    assert ended.response_spool.frame_count == len(events)
    saved: list[bytes] = []

    _synthesize(provider, lambda wav: saved.append(wav) or "pending-key")

    assert transport.calls == 0
    assert saved and saved[0].startswith(b"RIFF")
    with wave.open(io.BytesIO(saved[0]), "rb") as rendered:
        assert rendered.readframes(rendered.getnframes()) == PCM
    assert operations.OperationJournal.load(path).operations == {}


@pytest.mark.parametrize("state", ["running", "outcome_unknown"])
def test_running_and_outcome_unknown_never_redispatch(tmp_path: Path, state: str) -> None:
    provider, transport, path = _provider(tmp_path)
    journal = operations.OperationJournal.load(path)
    journal.authorize(
        "uncertain",
        kind="audio-realtime",
        source_file=SOURCE,
        source_sha256=SOURCE_SHA,
        request_fp=provider.request_fingerprint(TEXT, forced_accent=False),
        model=MODEL,
    )
    journal.advance("uncertain", "dispatching")
    journal.advance("uncertain", "running")
    if state == "outcome_unknown":
        journal.end("uncertain")
    with pytest.raises(Exception, match="running|unknown|settle|operation"):
        _synthesize(provider, lambda _wav: pytest.fail("nothing persisted"))
    assert transport.calls == 0


def test_reconcile_accepts_recovered_stage_hash_without_transport(tmp_path: Path) -> None:
    provider, first_transport, path = _provider(tmp_path)
    staged: list[bytes] = []

    def stage_then_interrupt(wav: bytes) -> str:
        staged.append(wav)
        raise OSError("ledger interrupted after stage")

    with pytest.raises(OSError, match="ledger interrupted"):
        _synthesize(provider, stage_then_interrupt)
    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"

    resumed, transport, _ = _provider(tmp_path)
    resumed.reconcile_journaled(
        TEXT,
        forced_accent=False,
        source_file=SOURCE,
        source_sha256=SOURCE_SHA,
        audio_sha256=hashlib.sha256(staged[0]).hexdigest(),
    )
    assert transport.calls == 0
    assert operations.OperationJournal.load(path).operations == {}


def test_reconcile_refuses_a_different_stage_hash_and_keeps_reply(tmp_path: Path) -> None:
    provider, first_transport, path = _provider(tmp_path)

    with pytest.raises(OSError, match="ledger interrupted"):
        _synthesize(
            provider,
            lambda _wav: (_ for _ in ()).throw(OSError("ledger interrupted")),
        )
    assert first_transport.calls == 1

    resumed, transport, _ = _provider(tmp_path)
    with pytest.raises(TtsError, match="does not match|refusing"):
        resumed.reconcile_journaled(
            TEXT,
            forced_accent=False,
            source_file=SOURCE,
            source_sha256=SOURCE_SHA,
            audio_sha256="0" * 64,
        )

    assert transport.calls == 0
    assert _only(path).state == "result_captured"


def test_generate_audio_leaves_exact_realtime_wav_in_pending_wal(tmp_path: Path) -> None:
    provider, transport, path = _provider(tmp_path)
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    result = _generate(tmp_path, provider, book)

    assert transport.calls == 1
    assert result.stopped_by == ""
    assert len(result.pending_keys) == 1
    key = result.pending_keys[0]
    entry = ledger.load(book.path).pending_audio[key]
    staged = tmp_path / "media" / "audio" / str(entry["staged_file"])
    wav_bytes = staged.read_bytes()
    assert hashlib.sha256(wav_bytes).hexdigest() == entry["staged_sha256"]
    with wave.open(io.BytesIO(wav_bytes), "rb") as rendered:
        assert rendered.getnchannels() == 1
        assert rendered.getsampwidth() == 2
        assert rendered.getframerate() == 24_000
        assert rendered.getnframes() == len(PCM) // 2
        assert rendered.readframes(rendered.getnframes()) == PCM
    assert operations.OperationJournal.load(path).operations == {}


def test_generate_audio_adopts_orphan_stage_without_second_transport(
    tmp_path: Path,
) -> None:
    provider, first_transport, path = _provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    def fail_wal(_current: ledger.Ledger, _key: str) -> None:
        raise OSError("injected WAL persistence failure")

    with pytest.raises(OSError, match="WAL persistence failure"):
        audio_cmd.generate_audio(
            [_record()],
            provider=provider,
            sentence_provider=provider,
            book=ledger.Ledger(path=ledger_path),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            stage_only=True,
            persist_pending=fail_wal,
        )
    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"
    pending_dir = tmp_path / "media" / "audio" / ".pending"
    assert len(list(pending_dir.glob("*.stage"))) == 1

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert result.stopped_by == ""
    assert len(result.pending_keys) == 1
    assert len(ledger.load(ledger_path).pending_audio) == 1
    assert operations.OperationJournal.load(path).operations == {}
