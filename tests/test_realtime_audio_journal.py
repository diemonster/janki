"""Operation-journal contract for paid Realtime audio; no real transport."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import audio_cmd, ledger, operations
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.tts import PaidAttempt, TtsError
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


def _merge_wal(
    current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
) -> None:
    """The durable seam `application.audio` wires for an unforced run.

    Every node in this file used to substitute `current.save()`, a whole-ledger
    write with no compare-and-swap in it. That made recovery's second write to
    the same row — the one that adds the witness — unobservable here, so the
    boundaries below were passing against a collaborator that could not fail.
    """
    current.merge_pending_audio(
        [key], expected=None if expected is None else {key: expected}
    )


def _force_merge_wal(
    current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
) -> None:
    """The same seam for `--force`, which authorizes replacing what is there."""
    current.merge_pending_audio(
        [key], replace=True, expected=None if expected is None else {key: expected}
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
        persist_pending=_merge_wal,
    )


def test_authority_precedes_send_capture_precedes_persist_and_persist_precedes_cleanup(
    tmp_path: Path,
) -> None:
    provider, transport, path = _provider(tmp_path)
    saved: list[bytes] = []
    witnessed: list[PaidAttempt] = []

    def persist(wav: bytes, attempt: PaidAttempt) -> str:
        held = _only(path)
        assert held.state == "result_captured"
        assert held.artifact is not None
        assert b"response.done" in (path.parent / held.artifact.relative_name).read_bytes()
        # The attempt arrives with the bytes, inside the commit and therefore
        # before both the `committed` write and the forget that follows it.
        assert attempt.operation_id == held.operation_id
        assert attempt.request_fp == held.request_fp
        assert attempt.model == MODEL
        assert attempt.audio_sha256 == hashlib.sha256(wav).hexdigest()
        saved.append(wav)
        witnessed.append(attempt)
        return "pending-key"

    _synthesize(provider, persist)
    assert transport.calls == 1
    assert saved and saved[0].startswith(b"RIFF")
    assert len(witnessed) == 1
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
    _synthesize(provider, lambda _wav, _attempt: "pending-key")

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
        _synthesize(
            provider,
            lambda _wav, _attempt: pytest.fail("invalid audio persisted"),
        )
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
        _synthesize(
            provider,
            lambda _wav, _attempt: pytest.fail("malformed reply persisted"),
        )

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
        _synthesize(provider, lambda _wav, _attempt: pytest.fail("nothing was dispatched"))

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
        _synthesize(provider, lambda _wav, _attempt: pytest.fail("nothing was dispatched"))

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
        _synthesize(provider, lambda _wav, _attempt: pytest.fail("interrupted before decode"))

    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"

    monkeypatch.setattr(
        operations.OperationJournal,
        "append_response_frame",
        real_append,
    )
    resumed, resumed_transport, _ = _provider(tmp_path)
    saved: list[bytes] = []
    _synthesize(resumed, lambda wav, _attempt: saved.append(wav) or "pending-key")

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
        _synthesize(provider, lambda _wav, _attempt: pytest.fail("decode did not finish"))
    assert first_transport.calls == 1
    assert _only(path).state == "result_captured"

    monkeypatch.setattr(OpenAiRealtimeProvider, "decode_response", real_decode)
    resumed, resumed_transport, _ = _provider(tmp_path)
    saved: list[bytes] = []
    _synthesize(resumed, lambda wav, _attempt: saved.append(wav) or "pending-key")
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

    _synthesize(provider, lambda wav, _attempt: saved.append(wav) or "pending-key")

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

    _synthesize(provider, lambda wav, _attempt: saved.append(wav) or "pending-key")

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
        _synthesize(
            provider,
            lambda _wav, _attempt: pytest.fail("nothing persisted"),
        )
    assert transport.calls == 0


def test_reconcile_accepts_recovered_stage_hash_without_transport(tmp_path: Path) -> None:
    provider, first_transport, path = _provider(tmp_path)
    staged: list[bytes] = []

    def stage_then_interrupt(wav: bytes, _attempt: PaidAttempt) -> str:
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
            lambda _wav, _attempt: (_ for _ in ()).throw(
                OSError("ledger interrupted")
            ),
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

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
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


# --- writer-owned attempt provenance, and the five durable boundaries --------
#
# A successful journaled call is forgotten, so after a clean run the journal
# can no longer say which call produced a clip — while any other authorized run
# may legitimately voice the same identity-addressed target. These nodes pin
# the witness the writer records instead: minted from the reply, bound to the
# exact bytes, durable before the forget, and never inherited by a render it
# did not produce.


def _witness(book: ledger.Ledger, key: str) -> dict[str, Any]:
    entry = book.pending_audio[key]
    return dict(entry["details"][audio_cmd.PAID_ATTEMPT])


def _capturing_provider(tmp_path: Path, events: list[dict[str, Any]] | None = None):
    """A provider whose transport records the operation id it dispatched under.

    The id is the one fact these tests cannot read afterwards: a successful
    call is forgotten, so the journal is empty by the time the assertions run.
    """

    path = tmp_path / "operations.json"
    inner = Transport(path, events or _events())
    dispatched: list[str] = []

    def watching(endpoint: str, headers: dict[str, str], request: dict[str, Any]):
        dispatched.append(_only(path).operation_id)
        return inner(endpoint, headers, request)

    provider = OpenAiRealtimeProvider(
        record_id=SOURCE,
        api_key="not-a-real-key",
        transport=watching,
        operations_path=path,
    )
    return provider, inner, path, dispatched


def test_the_paid_attempt_is_durable_in_the_wal_row_before_its_call_is_forgotten(
    tmp_path: Path,
) -> None:
    """The witness names the billed call, its request, its model and its bytes.

    Mutation: mint the attempt from the caller's staged bytes alone, dropping
    the operation id the journal entry carries.
    """

    provider, transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    result = _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))

    assert transport.calls == 1 and len(dispatched) == 1
    book = ledger.load(ledger_path)
    key = result.pending_keys[0]
    entry = book.pending_audio[key]
    staged = tmp_path / "media" / "audio" / str(entry["staged_file"])
    assert _witness(book, key) == {
        "operation_id": dispatched[0],
        "request_fp": provider.request_fingerprint(TEXT, forced_accent=False),
        "model": MODEL,
        "audio_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
    }
    # Recorded while the entry existed; the entry is gone now, which is what
    # makes the record the only surviving answer.
    assert operations.OperationJournal.load(path).operations == {}


def test_the_committed_audio_entry_keeps_the_attempt_after_the_journal_is_gone(
    tmp_path: Path,
) -> None:
    """Commit carries the witness into canonical state, unchanged.

    Mutation: drop the WAL row's details when committing it.
    """

    provider, _transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    result = _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    book = ledger.load(ledger_path)
    staged_witness = _witness(book, result.pending_keys[0])

    book.commit_pending_audio(result.pending_keys[0])

    [entry] = book.records[SOURCE]["audio"]
    assert entry[audio_cmd.PAID_ATTEMPT] == staged_witness
    assert entry["file"] == str(
        ledger.load(ledger_path).pending_audio[result.pending_keys[0]]["target"]
    )
    assert book.pending_audio == {}
    assert operations.OperationJournal.load(path).operations == {}


def test_a_stage_written_before_its_wal_row_recovers_its_original_attempt(
    tmp_path: Path,
) -> None:
    """Boundary: after stage bytes, before the WAL persist.

    The row is written by the *adopting* run, which knows nothing about the
    call; only the still-live entry does, so recovery takes the witness from
    there and never from the caller.

    Mutation: skip `attach` during reconciliation and let the adopted row keep
    whatever details the caller supplied.
    """

    provider, first_transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
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
    assert not ledger_path.exists()

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    book = ledger.load(ledger_path)
    assert _witness(book, result.pending_keys[0])["operation_id"] == dispatched[0]
    assert operations.OperationJournal.load(path).operations == {}


def test_an_orphan_stage_recovers_through_the_production_wal_merge(
    tmp_path: Path,
) -> None:
    """Boundary 1 composed with the callback the writer actually runs with.

    The other nodes here substitute a whole-ledger `save`, which cannot notice
    that recovery writes the row twice: once to adopt the orphan stage, once to
    add the witness the still-live entry supplies. Production merges each row
    under the ledger's own lock and refuses a durable row that differs from the
    one being written — including the row this same run wrote a moment earlier.
    So it is the *composition*, not either write, that has to be proven, and an
    ordinary unforced rerun is the only remedy §7.10 promises for this state.

    Mutation: persist the attribution without the row it was read from.
    """

    provider, first_transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
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
    assert not ledger_path.exists()

    resumed, resumed_transport, _ = _provider(tmp_path)
    persisted: list[tuple[str, dict[str, Any] | None]] = []

    def recording(
        current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
    ) -> None:
        persisted.append((key, None if expected is None else dict(expected)))
        _merge_wal(current, key, expected)

    result = audio_cmd.generate_audio(
        [_record()],
        provider=resumed,
        sentence_provider=resumed,
        book=ledger.load(ledger_path),
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        stage_only=True,
        persist_pending=recording,
    )

    assert resumed_transport.calls == 0
    assert result.stopped_by == ""
    key = result.pending_keys[0]
    settled = ledger.load(ledger_path)
    # Durable, not merely in memory: the merge is what this node is about.
    assert _witness(settled, key)["operation_id"] == dispatched[0]
    assert operations.OperationJournal.load(path).operations == {}

    # Two writes to one row: adoption adds it with nothing to compare against,
    # and the attribution carries the exact row it read. The durable result is
    # that row plus the witness and nothing else — the recovered render itself
    # was not rewritten.
    (first_key, adding), (second_key, updating) = persisted
    assert (first_key, adding, second_key) == (key, None, key)
    assert updating is not None
    assert audio_cmd.PAID_ATTEMPT not in updating["details"]
    assert settled.pending_audio[key] == {
        **updating,
        "details": {
            **updating["details"],
            audio_cmd.PAID_ATTEMPT: _witness(settled, key),
        },
    }

    # And the finished state is still resumable: running the exact command
    # again neither pays, rewrites the witness, nor refuses.
    final, final_transport, _ = _provider(tmp_path)
    again = audio_cmd.generate_audio(
        [_record()],
        provider=final,
        sentence_provider=final,
        book=ledger.load(ledger_path),
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        stage_only=True,
        persist_pending=_merge_wal,
    )
    assert final_transport.calls == 0
    assert again.pending_keys == [key]
    assert ledger.load(ledger_path).pending_audio == settled.pending_audio


def test_a_durable_unwitnessed_row_takes_its_still_live_reply_s_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other supported shape: a valid WAL row that has no witness yet.

    A row written before this field existed, or by an interrupted writer, is
    complete and adoptable — it simply cannot say which call paid for it. While
    its entry is still live the reply can, and recording that is again an update
    to a durable row rather than an addition, through the same merge.

    Mutation: persist the attribution against the row being written.
    """

    provider, transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    real_move = operations.OperationJournal._move_under_lock

    def die_before_committed(self, current, operation_id, state, **kwargs):
        if state == "committed":
            raise KeyboardInterrupt("simulated death before the committed write")
        return real_move(self, current, operation_id, state, **kwargs)

    monkeypatch.setattr(
        operations.OperationJournal, "_move_under_lock", die_before_committed
    )
    with pytest.raises(KeyboardInterrupt, match="before the committed write"):
        _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    monkeypatch.setattr(operations.OperationJournal, "_move_under_lock", real_move)

    aged = ledger.load(ledger_path)
    [key] = list(aged.pending_audio)
    # Exactly the durable shape a pre-correction writer left behind, with the
    # captured reply still there to account for it.
    del aged.pending_audio[key]["details"][audio_cmd.PAID_ATTEMPT]
    aged.save()
    unwitnessed = ledger.load(ledger_path).pending_audio[key]
    assert _only(path).state == "result_captured"

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert result.pending_keys == [key]
    settled = ledger.load(ledger_path)
    assert _witness(settled, key)["operation_id"] == dispatched[0]
    assert settled.pending_audio[key] == {
        **unwitnessed,
        "details": {
            **unwitnessed["details"],
            audio_cmd.PAID_ATTEMPT: _witness(settled, key),
        },
    }
    assert operations.OperationJournal.load(path).operations == {}


def test_a_row_changed_under_an_attribution_refuses_and_keeps_what_is_there(
    tmp_path: Path,
) -> None:
    """The protection the swap must not trade away to stop refusing itself.

    Another audio command may legitimately hold the same request key — a forced
    run adopting a different valid orphan stage, say — and its render is not
    this one's to overwrite on the way past. So a row that is no longer the row
    the attribution read refuses, leaves that row and everything beside it
    exactly as the other writer left it, and keeps this call's own evidence
    where the next rerun will find it.

    Mutation: swap against the row being written, or against nothing at all.
    """

    provider, first_transport, path, _dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
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
    staged_before = sorted((tmp_path / "media" / "audio" / ".pending").glob("*.stage"))
    assert len(staged_before) == 1

    resumed, resumed_transport, _ = _provider(tmp_path)
    raced: list[str] = []

    def race_the_attribution(
        current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
    ) -> None:
        if expected is not None and not raced:
            # Between the row being read and its attribution being merged.
            other = ledger.load(ledger_path)
            elsewhere = hashlib.sha256(b"another writer's render").hexdigest()
            other.pending_audio[key] = {
                **dict(other.pending_audio[key]),
                "staged_file": f".pending/{key}-{elsewhere}.stage",
                "staged_sha256": elsewhere,
                "details": {},
            }
            other.record_added("word:読む:よむ")
            other.save()
            raced.append(ledger_path.read_text(encoding="utf-8"))
        _merge_wal(current, key, expected)

    with pytest.raises(ledger.LedgerError, match="no longer the row this update read"):
        audio_cmd.generate_audio(
            [_record()],
            provider=resumed,
            sentence_provider=resumed,
            book=ledger.load(ledger_path),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            stage_only=True,
            persist_pending=race_the_attribution,
        )

    assert resumed_transport.calls == 0
    # Not one byte of the other writer's ledger: its render, and the unrelated
    # record it added in the same breath, are both still exactly as it left them.
    assert ledger_path.read_text(encoding="utf-8") == raced[0]
    assert "word:読む:よむ" in ledger.load(ledger_path).records
    # And this call's own evidence is still recoverable by an ordinary rerun.
    assert _only(path).state == "result_captured"
    assert (
        sorted((tmp_path / "media" / "audio" / ".pending").glob("*.stage"))
        == staged_before
    )


def test_a_wal_row_written_before_the_committed_write_keeps_its_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary: after the WAL persist, before `committed`.

    Mutation: record the attempt after the committed write rather than inside
    the commit.
    """

    provider, transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    real_move = operations.OperationJournal._move_under_lock

    def die_before_committed(self, current, operation_id, state, **kwargs):
        if state == "committed":
            raise KeyboardInterrupt("simulated death before the committed write")
        return real_move(self, current, operation_id, state, **kwargs)

    monkeypatch.setattr(
        operations.OperationJournal, "_move_under_lock", die_before_committed
    )
    with pytest.raises(KeyboardInterrupt, match="before the committed write"):
        _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    monkeypatch.setattr(operations.OperationJournal, "_move_under_lock", real_move)

    assert _only(path).state == "result_captured"
    interrupted = ledger.load(ledger_path)
    [key] = list(interrupted.pending_audio)
    assert _witness(interrupted, key)["operation_id"] == dispatched[0]

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert result.pending_keys == [key]
    assert _witness(ledger.load(ledger_path), key)["operation_id"] == dispatched[0]
    assert operations.OperationJournal.load(path).operations == {}


def test_a_committed_call_not_yet_forgotten_keeps_its_attempt_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary: after `committed`, before `forget`.

    Mutation: attach only when the entry is still `result_captured`.
    """

    provider, transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    real_forget = operations.OperationJournal.forget

    def die_before_forget(self, operation_ids, **kwargs):
        raise KeyboardInterrupt("simulated death before the forget")

    monkeypatch.setattr(operations.OperationJournal, "forget", die_before_forget)
    with pytest.raises(KeyboardInterrupt, match="before the forget"):
        _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    monkeypatch.setattr(operations.OperationJournal, "forget", real_forget)

    assert _only(path).state == "committed"
    [key] = list(ledger.load(ledger_path).pending_audio)

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert result.pending_keys == [key]
    assert _witness(ledger.load(ledger_path), key)["operation_id"] == dispatched[0]
    assert operations.OperationJournal.load(path).operations == {}


def test_after_the_forget_the_wal_row_is_the_only_carrier_and_still_carries_it(
    tmp_path: Path,
) -> None:
    """Boundary: after `forget`, before the records compare-and-swap.

    Reconciliation returns immediately — there is nothing left to settle — so a
    rerun must neither lose the attribution nor invent a replacement for it.

    Mutation: re-derive the witness during recovery instead of preserving the
    row's own.
    """

    provider, transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    first = _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    assert operations.OperationJournal.load(path).operations == {}
    before = _witness(ledger.load(ledger_path), first.pending_keys[0])

    resumed, resumed_transport, _ = _provider(tmp_path)
    again = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert again.pending_keys == first.pending_keys
    assert _witness(ledger.load(ledger_path), first.pending_keys[0]) == before
    assert before["operation_id"] == dispatched[0]


def test_a_replacement_render_never_inherits_the_previous_attempt(
    tmp_path: Path,
) -> None:
    """One request key, two real renders: the witness follows the bytes.

    A request fingerprint says what was asked, not what came back. `--force`
    recovery may legitimately adopt a *different* valid orphan stage at the
    same key, and attributing those bytes to the earlier call would name a
    call that never produced them.

    Mutation: copy the replaced row's `paid_attempt` onto the adopted stage.
    """

    ledger_path = tmp_path / "ledger.json"
    first, _first_transport, path, first_ids = _capturing_provider(tmp_path)
    landed = _generate(tmp_path, first, ledger.Ledger(path=ledger_path))
    key = landed.pending_keys[0]
    book = ledger.load(ledger_path)
    original = _witness(book, key)
    staged_dir = tmp_path / "media" / "audio" / ".pending"
    # The recorded stage's bytes are lost; the row and its attribution are not.
    (staged_dir / Path(str(book.pending_audio[key]["staged_file"])).name).unlink()

    other_pcm = b"\x02\x00" * 240
    second, second_transport, _path, second_ids = _capturing_provider(
        tmp_path, _events(delta=base64.b64encode(other_pcm).decode())
    )

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
        raise OSError("injected WAL persistence failure")

    with pytest.raises(OSError, match="WAL persistence failure"):
        audio_cmd.generate_audio(
            [_record()],
            provider=second,
            sentence_provider=second,
            book=ledger.load(ledger_path),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            force=True,
            stage_only=True,
            persist_pending=fail_wal,
        )
    assert second_transport.calls == 1
    assert second_ids[0] != first_ids[0]
    assert _only(path).state == "result_captured"
    assert _witness(ledger.load(ledger_path), key) == original

    third, third_transport, _path, _ids = _capturing_provider(tmp_path)
    adopted = audio_cmd.generate_audio(
        [_record()],
        provider=third,
        sentence_provider=third,
        book=ledger.load(ledger_path),
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        force=True,
        stage_only=True,
        persist_pending=_force_merge_wal,
    )

    assert third_transport.calls == 0
    assert adopted.pending_keys == [key]
    settled = ledger.load(ledger_path)
    adopted_stage = staged_dir / Path(str(settled.pending_audio[key]["staged_file"])).name
    witness = _witness(settled, key)
    assert witness["operation_id"] == second_ids[0]
    assert witness["operation_id"] != original["operation_id"]
    assert witness["audio_sha256"] == hashlib.sha256(adopted_stage.read_bytes()).hexdigest()
    assert witness["audio_sha256"] != original["audio_sha256"]
    assert operations.OperationJournal.load(path).operations == {}


def test_an_unpersistable_attribution_keeps_the_call_and_its_recovered_bytes(
    tmp_path: Path,
) -> None:
    """Attribution that cannot be saved refuses; it never forgets and hopes.

    Mutation: settle and forget the entry before persisting the attempt.
    """

    provider, first_transport, path, dispatched = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
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
    assert _only(path).state == "result_captured"
    staged_before = sorted(
        (tmp_path / "media" / "audio" / ".pending").glob("*.stage")
    )
    assert len(staged_before) == 1

    resumed, resumed_transport, _ = _provider(tmp_path)
    saves = 0

    def fail_the_attribution(
        current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
    ) -> None:
        nonlocal saves
        saves += 1
        if saves > 1:
            raise OSError("injected attribution persistence failure")
        _merge_wal(current, key, expected)

    with pytest.raises(OSError, match="attribution persistence failure"):
        audio_cmd.generate_audio(
            [_record()],
            provider=resumed,
            sentence_provider=resumed,
            book=ledger.load(ledger_path),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            stage_only=True,
            persist_pending=fail_the_attribution,
        )

    assert resumed_transport.calls == 0
    assert _only(path).state == "result_captured"
    assert (
        sorted((tmp_path / "media" / "audio" / ".pending").glob("*.stage"))
        == staged_before
    )

    final, final_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, final, ledger.load(ledger_path))

    assert final_transport.calls == 0
    assert _witness(ledger.load(ledger_path), result.pending_keys[0])[
        "operation_id"
    ] == dispatched[0]
    assert operations.OperationJournal.load(path).operations == {}


def test_an_old_recoverable_clip_without_any_attempt_still_recovers(
    tmp_path: Path,
) -> None:
    """A row written before this field existed is recovered, not rewritten.

    Mutation: require `paid_attempt` on every adopted pending row.
    """

    provider, transport, path, _ids = _capturing_provider(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    landed = _generate(tmp_path, provider, ledger.Ledger(path=ledger_path))
    key = landed.pending_keys[0]
    aged = ledger.load(ledger_path)
    # Exactly the shape a pre-correction ledger holds: everything else intact.
    del aged.pending_audio[key]["details"][audio_cmd.PAID_ATTEMPT]
    aged.save()

    resumed, resumed_transport, _ = _provider(tmp_path)
    result = _generate(tmp_path, resumed, ledger.load(ledger_path))

    assert resumed_transport.calls == 0
    assert result.pending_keys == [key]
    reloaded = ledger.load(ledger_path)
    assert audio_cmd.PAID_ATTEMPT not in reloaded.pending_audio[key]["details"]
    reloaded.commit_pending_audio(key)
    [entry] = reloaded.records[SOURCE]["audio"]
    assert audio_cmd.PAID_ATTEMPT not in entry


def test_an_attempt_describing_other_bytes_is_refused_at_the_write(
    tmp_path: Path,
) -> None:
    """Structural validation at the two seams that write the attribution.

    Both callers derive the hash from the same bytes today, so these refusals
    are guards on the seam rather than paths an ordinary run reaches. They are
    exercised directly because a witness that named other bytes would be worse
    than none: a reader comparing it against the published file would find a
    self-consistent pair of fields and believe the wrong call produced the clip.
    """

    book = ledger.Ledger(path=tmp_path / "ledger.json")
    audio_dir = tmp_path / "media" / "audio"
    audio_dir.mkdir(parents=True)
    data = b"RIFF-not-really-a-wav"
    arguments: dict[str, Any] = {
        "book": book,
        "record_id": SOURCE,
        "of": "example",
        "target": "janki-paid.wav",
        "request_input": TEXT,
        "forced_accent": False,
        "content_fp": SOURCE_SHA,
        "provider": OpenAiRealtimeProvider(record_id=SOURCE),
        "audio_dir": audio_dir,
    }
    foreign = PaidAttempt(
        operation_id="5f2a",
        request_fp="e" * 64,
        model=MODEL,
        audio_sha256=hashlib.sha256(b"some other render").hexdigest(),
    )

    with pytest.raises(audio_cmd.SynthesisError, match="describes other audio"):
        audio_cmd._stage_audio(data, foreign, **arguments)
    assert book.pending_audio == {}

    key = audio_cmd._stage_audio(
        data,
        PaidAttempt(
            operation_id="5f2a",
            request_fp="e" * 64,
            model=MODEL,
            audio_sha256=hashlib.sha256(data).hexdigest(),
        ),
        **arguments,
    )
    attach = audio_cmd._attach_paid_attempt(
        book,
        SOURCE,
        key=key,
        arguments={
            name: value
            for name, value in (
                ("of", "example"),
                ("target", "janki-paid.wav"),
                ("request_input", TEXT),
                ("forced_accent", False),
                ("content_fp", SOURCE_SHA),
                ("provider", "openai-realtime"),
                ("voice", voice_for_record(SOURCE)),
                ("speed", 1.0),
                ("settings", dict(OpenAiRealtimeProvider(record_id=SOURCE).settings)),
            )
        },
        staged_file=str(book.pending_audio[key]["staged_file"]),
        staged_sha256="c" * 64,
        persist_pending=None,
    )

    with pytest.raises(audio_cmd.SynthesisError, match="no longer stages"):
        attach(foreign)


def test_a_replacement_adopted_with_no_journal_left_carries_no_attribution(
    tmp_path: Path,
) -> None:
    """Discarded evidence leaves the bytes unattributed, never misattributed.

    Not every orphan stage still has a live journal entry behind it: an owner
    may have made the prescribed `operations --forget --force` decision about a
    call whose outcome was unknown. The replaced row's attempt describes other
    bytes entirely, so adopting these without a witness is the honest outcome —
    a paid clip with nothing to attribute simply cannot be proven as newly
    dispatched work.

    Mutation: carry the replaced row's `paid_attempt` onto the adopted stage.
    """

    ledger_path = tmp_path / "ledger.json"
    first, _first_transport, path, first_ids = _capturing_provider(tmp_path)
    landed = _generate(tmp_path, first, ledger.Ledger(path=ledger_path))
    key = landed.pending_keys[0]
    book = ledger.load(ledger_path)
    original = _witness(book, key)
    staged_dir = tmp_path / "media" / "audio" / ".pending"
    (staged_dir / Path(str(book.pending_audio[key]["staged_file"])).name).unlink()

    second, second_transport, _path, second_ids = _capturing_provider(
        tmp_path, _events(delta=base64.b64encode(b"\x02\x00" * 240).decode())
    )

    def fail_wal(
        _current: ledger.Ledger, _key: str, _expected: Mapping[str, Any] | None
    ) -> None:
        raise OSError("injected WAL persistence failure")

    with pytest.raises(OSError, match="WAL persistence failure"):
        audio_cmd.generate_audio(
            [_record()],
            provider=second,
            sentence_provider=second,
            book=ledger.load(ledger_path),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            force=True,
            stage_only=True,
            persist_pending=fail_wal,
        )
    assert second_transport.calls == 1
    # The owner's prescribed decision about a charge whose output never landed.
    operations.OperationJournal.load(path).forget([second_ids[0]], force=True)
    assert operations.OperationJournal.load(path).operations == {}

    third, third_transport, _path, _ids = _capturing_provider(tmp_path)
    adopted = audio_cmd.generate_audio(
        [_record()],
        provider=third,
        sentence_provider=third,
        book=ledger.load(ledger_path),
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        force=True,
        stage_only=True,
        persist_pending=_force_merge_wal,
    )

    assert third_transport.calls == 0
    assert adopted.pending_keys == [key]
    settled = ledger.load(ledger_path)
    entry = settled.pending_audio[key]
    assert str(entry["staged_sha256"]) != original["audio_sha256"]
    assert audio_cmd.PAID_ATTEMPT not in entry["details"]
