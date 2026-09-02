"""OpenAI Realtime sentence speech, captured before it is decoded.

Realtime is a streaming API, but janki needs a finite Anki media file and a
durable answer to a paid call. The transport therefore collects the exact text
frames into one operation-bound envelope. Journaled synthesis persists that
envelope before any audio delta is trusted, then turns the PCM into a finite
WAV through the existing paid-audio write-ahead transaction.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import wave
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from japanese_anki.operations import (
    Operation,
    OperationError,
    OperationJournal,
    capture_artifact,
    prepare_artifact_store,
)
from japanese_anki.tts import TtsError

__all__ = [
    "ENDPOINT",
    "INSTRUCTIONS",
    "KEY_HINT",
    "MODEL",
    "SAMPLE_RATE",
    "VOICES",
    "OpenAiRealtimePool",
    "OpenAiRealtimeProvider",
    "voice_for_record",
]

MODEL = "gpt-realtime-1.5"
VOICES = ("cedar", "ash", "verse", "marin")
SAMPLE_RATE = 24_000
ENDPOINT = f"wss://api.openai.com/v1/realtime?model={MODEL}"
INSTRUCTIONS = (
    "Read the user's text aloud exactly once as a native speaker of standard "
    "Tokyo Japanese. Do not add, omit, translate, answer, or explain anything. "
    "Speak clearly at a slower, learner-friendly pace, approximately 75% of "
    "normal conversational speed, while preserving natural Japanese intonation "
    "and connected phrasing."
)
KEY_HINT = (
    "no OPENAI_API_KEY. Set it in your environment and try again — it is never "
    "read from janki.toml."
)

_ENVELOPE_VERSION = 1
_SELECTOR = "framed-sha256-v1"
_AUDIO_FORMAT = "pcm16-24000-mono"
_JOURNAL_KIND = "audio-realtime"
_RESUMABLE_STATES = frozenset(
    {
        "authorized",
        "dispatching",
        "running",
        "result_captured",
        "committed",
        "outcome_unknown",
    }
)

# Fakes may yield decoded dictionaries. Production yields exact text frames so
# the captured envelope retains the provider's bytes, not a re-serialization.
Transport = Callable[[str, dict[str, str], dict[str, Any]], Iterable[object]]
_Persisted = TypeVar("_Persisted")


class _BeforeDispatchError(TtsError):
    """A Realtime refusal proven to precede ``response.create``."""


def voice_for_record(record_id: str) -> str:
    """Return the frozen v1 voice assignment for one stable record identity."""
    raw = str(record_id).encode("utf-8")
    framed = b"janki:realtime-voice\0v1\0" + len(raw).to_bytes(8, "big") + raw
    value = int.from_bytes(hashlib.sha256(framed).digest()[:8], "big")
    return VOICES[value % len(VOICES)]


def _json_bytes(value: object) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{rendered}\n".encode()


def _event(message: object) -> dict[str, Any]:
    if not isinstance(message, str):
        raise TtsError("OpenAI Realtime returned a non-text event frame.")
    try:
        value = json.loads(message)
    except (UnicodeError, ValueError) as exc:
        raise TtsError("OpenAI Realtime returned invalid JSON.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise TtsError("OpenAI Realtime returned a malformed event.")
    return value


def _envelope_bytes(
    request: Mapping[str, Any],
    payloads: Iterable[str],
) -> bytes:
    return _json_bytes(
        {
            "version": _ENVELOPE_VERSION,
            "endpoint": ENDPOINT,
            "request": dict(request),
            "frames": [
                {"encoding": "utf-8", "payload": payload}
                for payload in payloads
            ],
        }
    )


def _default_transport(
    endpoint: str,
    headers: dict[str, str],
    request: dict[str, Any],
) -> Iterable[object]:
    """Send one response request over one WebSocket, without retrying."""
    try:
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover - packaging catches this
        raise TtsError("OpenAI Realtime requires the websockets package.") from exc
    try:
        with connect(
            endpoint,
            additional_headers=headers,
            open_timeout=30,
            close_timeout=10,
            max_size=None,
        ) as socket:
            # A Realtime connection starts with session.created. Keep its raw
            # text, then submit exactly one isolated response.create request.
            first = socket.recv(timeout=30)
            yield first
            socket.send(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
            while True:
                message = socket.recv(timeout=120)
                yield message
    except TtsError:
        raise
    except Exception as exc:
        # Never retry here. Once connection setup begins, the caller cannot
        # prove that a request did not cross the wire.
        raise TtsError(
            f"Could not complete the OpenAI Realtime connection: {exc}"
        ) from exc


class OpenAiRealtimeProvider:
    """One concrete record-bound Realtime voice profile."""

    def __init__(
        self,
        *,
        record_id: str,
        api_key: str | None = None,
        transport: Transport | None = None,
        operations_path: Path | None = None,
    ) -> None:
        self.record_id = str(record_id)
        self._voice = voice_for_record(self.record_id)
        self._api_key = api_key
        self._transport = transport or _default_transport
        self.operations_path = (
            Path(operations_path) if operations_path is not None else None
        )

    @property
    def name(self) -> str:
        return "openai-realtime"

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def speed(self) -> float:
        # Pace is in the reviewed prose, not an unreviewed numeric transform.
        return 1.0

    @property
    def suffix(self) -> str:
        return ".wav"

    @property
    def launch_hint(self) -> str:
        return KEY_HINT

    @property
    def settings(self) -> dict[str, str]:
        return {
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "audio_format": _AUDIO_FORMAT,
        }

    def _key(self) -> str:
        key = (
            self._api_key
            if self._api_key is not None
            else os.environ.get("OPENAI_API_KEY", "")
        )
        if not key.strip():
            raise TtsError(KEY_HINT[0].upper() + KEY_HINT[1:])
        return key.strip()

    def available(self) -> bool:
        """Whether a fresh call has a credential; this never spends a call."""
        try:
            self._key()
        except TtsError:
            return False
        return True

    def validate_utterance(self, text: str) -> None:
        spoken = str(text).strip()
        if not spoken:
            raise TtsError("Nothing to speak: the sentence was empty.")
        try:
            spoken.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TtsError("OpenAI Realtime input must be valid UTF-8 text.") from exc

    def _request(self, text: str, *, forced_accent: bool) -> dict[str, Any]:
        if forced_accent:
            raise TtsError(
                "OpenAI Realtime is a sentence provider and cannot force a word accent."
            )
        self.validate_utterance(text)
        return {
            "type": "response.create",
            "response": {
                "conversation": "none",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": str(text).strip()}
                        ],
                    }
                ],
                "instructions": INSTRUCTIONS,
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "voice": self.voice,
                    }
                },
                "tool_choice": "none",
            },
        }

    def request_fingerprint(self, text: str, *, forced_accent: bool) -> str:
        request = self._request(text, forced_accent=forced_accent)
        wire = {
            "schema": "janki-openai-realtime-request-v1",
            "endpoint": ENDPOINT,
            "request": request,
        }
        return hashlib.sha256(_json_bytes(wire)).hexdigest()

    def _capture_response(
        self,
        text: str,
        *,
        forced_accent: bool,
        capture_frame: Callable[[str], None] | None,
    ) -> bytes:
        """Perform one call and return exact response frames, still undecoded.

        The journaled production caller supplies ``capture_frame``. It must
        return only after the payload is durable; this method invokes it before
        the first JSON inspection of every received text frame.
        """
        if capture_frame is None:
            raise TtsError(
                "OpenAI Realtime receive requires a durable journal frame capture."
            )
        request = self._request(text, forced_accent=forced_accent)
        frames: list[dict[str, str]] = []
        terminal = False
        try:
            stream = self._transport(
                ENDPOINT,
                {"Authorization": f"Bearer {self._key()}"},
                request,
            )
            try:
                for raw in stream:
                    if isinstance(raw, Mapping):
                        event = dict(raw)
                        payload = json.dumps(
                            event,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    elif isinstance(raw, str):
                        payload = raw
                    else:
                        refusal = TtsError(
                            "OpenAI Realtime returned a non-text event frame."
                        )
                        if not frames:
                            raise _BeforeDispatchError(str(refusal)) from refusal
                        raise refusal
                    try:
                        capture_frame(payload)
                    except BaseException as exc:
                        if not frames:
                            raise _BeforeDispatchError(
                                "OpenAI Realtime frame capture failed before "
                                "response.create was sent."
                            ) from exc
                        raise
                    if isinstance(raw, str):
                        try:
                            event = _event(payload)
                        except TtsError as exc:
                            if not frames:
                                raise _BeforeDispatchError(str(exc)) from exc
                            raise
                    if not isinstance(event.get("type"), str):
                        raise TtsError("OpenAI Realtime returned a malformed event.")
                    if not frames:
                        session = event.get("session")
                        if (
                            event["type"] != "session.created"
                            or not isinstance(session, Mapping)
                            or session.get("model") != MODEL
                        ):
                            raise _BeforeDispatchError(
                                "OpenAI Realtime returned the wrong session handshake "
                                f"for model {MODEL}."
                            )
                    frames.append({"encoding": "utf-8", "payload": payload})
                    if event["type"] in {"response.done", "error"}:
                        terminal = True
                        break
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except _BeforeDispatchError:
            raise
        except TtsError as exc:
            if not frames:
                raise _BeforeDispatchError(str(exc)) from exc
            raise
        except Exception as exc:
            detail = f"Could not complete the OpenAI Realtime connection: {exc}"
            if not frames:
                raise _BeforeDispatchError(detail) from exc
            raise TtsError(detail) from exc
        if not terminal:
            if not frames:
                raise _BeforeDispatchError(
                    "OpenAI Realtime closed before its session handshake; "
                    "response.create was not sent."
                )
            raise TtsError(
                "OpenAI Realtime closed before a terminal response; its outcome "
                "is unknown and this request will not be retried."
            )
        return _envelope_bytes(
            request,
            (frame["payload"] for frame in frames),
        )

    def _envelope_events(self, envelope: bytes) -> list[dict[str, Any]]:
        try:
            value = json.loads(envelope.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError) as exc:
            raise TtsError(
                "OpenAI Realtime response envelope is invalid JSON."
            ) from exc
        if (
            not isinstance(value, Mapping)
            or value.get("version") != _ENVELOPE_VERSION
            or value.get("endpoint") != ENDPOINT
            or not isinstance(value.get("request"), Mapping)
        ):
            raise TtsError("OpenAI Realtime response envelope is malformed.")
        frames = value.get("frames")
        if not isinstance(frames, list) or not frames:
            raise TtsError("OpenAI Realtime response was incomplete.")
        events: list[dict[str, Any]] = []
        for frame in frames:
            if (
                not isinstance(frame, Mapping)
                or frame.get("encoding") != "utf-8"
                or not isinstance(frame.get("payload"), str)
            ):
                raise TtsError(
                    "OpenAI Realtime response envelope has malformed frames."
                )
            events.append(_event(frame["payload"]))
        return events

    def decode_response(self, envelope: bytes) -> bytes:
        """Purely decode one captured envelope into a finite PCM WAV."""
        events = self._envelope_events(envelope)
        first = events[0]
        session = first.get("session") if isinstance(first, Mapping) else None
        if (
            first.get("type") != "session.created"
            or not isinstance(session, Mapping)
            or session.get("model") != MODEL
        ):
            raise TtsError("OpenAI Realtime session used the wrong model.")

        pcm = bytearray()
        terminal: Mapping[str, Any] | None = None
        for event in events[1:]:
            event_type = event.get("type")
            if event_type == "error":
                detail = event.get("error")
                message = (
                    detail.get("message")
                    if isinstance(detail, Mapping)
                    else "provider error"
                )
                raise TtsError(
                    f"OpenAI Realtime provider refused the request: {message}"
                )
            if event_type == "response.output_audio.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise TtsError("OpenAI Realtime audio delta is malformed.")
                try:
                    pcm.extend(base64.b64decode(delta, validate=True))
                except binascii.Error as exc:
                    raise TtsError(
                        "OpenAI Realtime audio delta has invalid base64."
                    ) from exc
            if event_type == "response.done":
                terminal = event
                break
        if terminal is None:
            raise TtsError("OpenAI Realtime response was incomplete.")
        response = terminal.get("response")
        if (
            not isinstance(response, Mapping)
            or response.get("status") != "completed"
            or response.get("output_modalities") != ["audio"]
        ):
            raise TtsError(
                "OpenAI Realtime response was incomplete or not completed."
            )
        audio = response.get("audio")
        output = audio.get("output") if isinstance(audio, Mapping) else None
        audio_format = output.get("format") if isinstance(output, Mapping) else None
        if (
            not isinstance(output, Mapping)
            or output.get("voice") != self.voice
            or not isinstance(audio_format, Mapping)
            or audio_format.get("type") != "audio/pcm"
            or audio_format.get("rate") != SAMPLE_RATE
        ):
            raise TtsError(
                "OpenAI Realtime response used the wrong audio voice or format."
            )
        if not pcm or len(pcm) % 2:
            raise TtsError("OpenAI Realtime returned empty or malformed audio.")

        result = io.BytesIO()
        with wave.open(result, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return result.getvalue()

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        raise TtsError(
            "OpenAI Realtime synthesis must use janki's journaled audio staging "
            "transaction."
        )

    def _operations_path(self, override: Path | None = None) -> Path:
        selected = override if override is not None else self.operations_path
        if selected is None:
            raise TtsError("Realtime journaled synthesis needs operations_path.")
        return Path(selected)

    def _matching_operation(
        self,
        path: Path,
        *,
        source_file: str,
        source_sha256: str,
        request_fp: str,
    ) -> Operation | None:
        matches = [
            operation
            for operation in OperationJournal.load(path).operations.values()
            if operation.kind == _JOURNAL_KIND
            and operation.source_file == source_file
            and operation.source_sha256 == source_sha256
            and operation.request_fp == request_fp
            and operation.model == MODEL
            and (
                operation.state in _RESUMABLE_STATES
                or operation.cleanup is not None
            )
        ]
        if len(matches) > 1:
            raise TtsError(
                "Multiple unfinished Realtime operations match this exact request."
            )
        return matches[0] if matches else None

    def _adopt_interrupted_capture(
        self,
        path: Path,
        held: Operation,
        request: Mapping[str, Any],
    ) -> Operation:
        """Bind a complete artifact WAL or durably terminal frame spool."""
        if held.state not in {"dispatching", "running", "outcome_unknown"}:
            return held
        journal = OperationJournal.load(path)
        try:
            payload = journal.read_reply(held.operation_id)
        except OperationError:
            return self._capture_spooled_result(path, held, request)
        journal.capture_result(
            held.operation_id,
            lambda: capture_artifact(path, held.operation_id, payload),
        )
        return OperationJournal.load(path).operations[held.operation_id]

    def _capture_spooled_result(
        self,
        path: Path,
        held: Operation,
        request: Mapping[str, Any],
    ) -> Operation:
        """Seal exact durable frames once terminal or structurally malformed."""
        if held.response_spool is None:
            return held
        payloads = OperationJournal.load(path).read_response_frames(
            held.operation_id
        )
        capturable = False
        for payload in payloads:
            try:
                event = _event(payload)
            except TtsError:
                # A malformed paid reply must survive the parse refusal just
                # as faithfully as a well-formed one.
                capturable = True
                break
            if event["type"] in {"response.done", "error"}:
                capturable = True
                break
        if not capturable:
            return held
        envelope = _envelope_bytes(request, payloads)
        OperationJournal.load(path).capture_result(
            held.operation_id,
            lambda: capture_artifact(path, held.operation_id, envelope),
        )
        return OperationJournal.load(path).operations[held.operation_id]

    def available_for(
        self,
        text: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
    ) -> bool:
        """Whether this exact clip can use a key or an already captured reply."""
        request_fp = self.request_fingerprint(text, forced_accent=forced_accent)
        if self.operations_path is not None:
            held = self._matching_operation(
                self.operations_path,
                source_file=source_file,
                source_sha256=source_sha256,
                request_fp=request_fp,
            )
            if held is not None and held.cleanup is None:
                # Every matching non-cleanup state is a local settlement or
                # refusal. None may dispatch again, so no fresh credential is
                # needed merely to reach its exact recovery path.
                return True
        return self.available()

    def _capture_new(
        self,
        path: Path,
        held: Operation,
        text: str,
        *,
        forced_accent: bool,
        before_dispatch: Callable[[str], None] | None,
    ) -> Operation:
        # The key is knowable before send. Retire a missing-key authority as a
        # proven before-send failure rather than inventing an unknown charge.
        try:
            self._key()
        except TtsError as exc:
            journal = OperationJournal.load(path)
            journal.advance(
                held.operation_id,
                "failed_before_send",
                detail=str(exc),
            )
            journal.forget([held.operation_id])
            raise

        request = self._request(text, forced_accent=forced_accent)
        try:
            OperationJournal.load(path).begin_response_capture(
                held.operation_id
            )
        except BaseException as exc:
            # The transport has not been opened. A spool-preparation failure is
            # therefore known to be before send; even the dispatching state is
            # written only after this durable capture binding succeeds.
            journal = OperationJournal.load(path)
            journal.advance(
                held.operation_id,
                "failed_before_send",
                detail=str(exc),
            )
            journal.forget([held.operation_id])
            raise

        if before_dispatch is not None:
            try:
                before_dispatch(held.operation_id)
            except BaseException as exc:
                journal = OperationJournal.load(path)
                journal.advance(
                    held.operation_id,
                    "failed_before_send",
                    detail=f"Dispatch authority callback refused: {exc}",
                )
                raise

        OperationJournal.load(path).advance(held.operation_id, "dispatching")
        OperationJournal.load(path).advance(held.operation_id, "running")
        try:
            self._capture_response(
                text,
                forced_accent=forced_accent,
                capture_frame=lambda payload: OperationJournal.load(
                    path
                ).append_response_frame(held.operation_id, payload),
            )
        except _BeforeDispatchError as exc:
            journal = OperationJournal.load(path)
            journal.advance(
                held.operation_id,
                "failed_before_send",
                detail=str(exc),
            )
            if before_dispatch is None:
                journal.forget([held.operation_id])
            raise
        except BaseException as exc:
            # The frame callback fsyncs before this path can parse. A malformed
            # or just-received terminal frame can therefore be sealed locally;
            # only a genuinely nonterminal stream has an unknown outcome.
            current = OperationJournal.load(path).operations[held.operation_id]
            recovered = self._capture_spooled_result(path, current, request)
            if recovered.state != "result_captured":
                OperationJournal.load(path).end(
                    held.operation_id,
                    detail=str(exc),
                )
            raise
        current = OperationJournal.load(path).operations[held.operation_id]
        captured = self._capture_spooled_result(path, current, request)
        if captured.state != "result_captured":  # pragma: no cover - transport contract
            raise TtsError(
                "OpenAI Realtime returned no durable terminal response."
            )
        return captured

    def synthesize_journaled(
        self,
        text: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
        persist: Callable[[bytes], _Persisted],
        operations_path: Path | None = None,
        before_dispatch: Callable[[str], None] | None = None,
    ) -> _Persisted:
        """Capture before decode, then commit only through ``persist``."""
        path = self._operations_path(operations_path)
        request = self._request(text, forced_accent=forced_accent)
        request_fp = self.request_fingerprint(text, forced_accent=forced_accent)
        prepare_artifact_store(path)
        held = self._matching_operation(
            path,
            source_file=source_file,
            source_sha256=source_sha256,
            request_fp=request_fp,
        )
        created = held is None
        if held is None:
            operation_id = str(uuid4())
            OperationJournal.load(path).authorize(
                operation_id,
                kind=_JOURNAL_KIND,
                source_file=source_file,
                source_sha256=source_sha256,
                request_fp=request_fp,
                model=MODEL,
            )
            held = OperationJournal.load(path).operations[operation_id]

        if held.cleanup is not None:
            raise TtsError(
                f"Realtime operation {held.operation_id} has unfinished cleanup."
            )
        held = self._adopt_interrupted_capture(path, held, request)
        if created:
            held = self._capture_new(
                path,
                held,
                text,
                forced_accent=forced_accent,
                before_dispatch=before_dispatch,
            )
        elif held.state == "authorized":
            raise TtsError(
                f"Realtime operation {held.operation_id} is authorized but was "
                "left by another run; refusing to dispatch it again."
            )

        if held.state in {"dispatching", "running", "outcome_unknown"}:
            raise TtsError(
                f"Realtime operation {held.operation_id} is {held.state}; "
                "refusing redispatch."
            )
        if held.state == "committed":
            raise TtsError(
                f"Realtime operation {held.operation_id} is committed but its "
                "exact audio recovery was not found; refusing to bill or persist "
                "a second copy."
            )
        if held.state != "result_captured":
            raise TtsError(
                f"Realtime operation {held.operation_id} is {held.state}."
            )

        envelope = OperationJournal.load(path).read_reply(held.operation_id)
        audio = self.decode_response(envelope)
        persisted: list[_Persisted] = []

        def _persist() -> None:
            persisted.append(persist(audio))

        OperationJournal.load(path).commit_result(held.operation_id, _persist)
        OperationJournal.load(path).forget([held.operation_id])
        return persisted[0]

    def reconcile_journaled(
        self,
        text: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
        audio_sha256: str,
        operations_path: Path | None = None,
    ) -> None:
        """Settle an exact captured call after its audio WAL was recovered."""
        path = self._operations_path(operations_path)
        request = self._request(text, forced_accent=forced_accent)
        request_fp = self.request_fingerprint(text, forced_accent=forced_accent)
        held = self._matching_operation(
            path,
            source_file=source_file,
            source_sha256=source_sha256,
            request_fp=request_fp,
        )
        if held is None:
            return
        if held.cleanup is not None:
            if held.state != "committed":
                raise TtsError(
                    f"Realtime operation {held.operation_id} is being discarded "
                    "before its result was committed."
                )
            OperationJournal.load(path).forget([held.operation_id])
            return

        held = self._adopt_interrupted_capture(path, held, request)
        if held.state not in {"result_captured", "committed"}:
            raise TtsError(
                f"Realtime operation {held.operation_id} is {held.state}; an "
                "audio stage cannot settle an uncertain provider call."
            )
        envelope = OperationJournal.load(path).read_reply(held.operation_id)
        audio = self.decode_response(envelope)
        actual = hashlib.sha256(audio).hexdigest()
        if actual != audio_sha256:
            raise TtsError(
                f"Recovered audio does not match Realtime operation "
                f"{held.operation_id}; refusing to settle either artifact."
            )
        if held.state == "result_captured":
            OperationJournal.load(path).commit_result(
                held.operation_id,
                lambda: None,
            )
        OperationJournal.load(path).forget([held.operation_id])


class OpenAiRealtimePool:
    """Offline selector for the accepted equal-weight per-record voice pool."""

    name = "openai-realtime"
    voice = "per-record"
    speed = 1.0
    suffix = ".wav"
    launch_hint = KEY_HINT
    settings = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "audio_format": _AUDIO_FORMAT,
        "selector": _SELECTOR,
        "voice_pool": ",".join(VOICES),
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
        operations_path: Path | None = None,
    ) -> None:
        self._api_key = api_key
        self._transport = transport
        self._operations_path = operations_path

    def available(self) -> bool:
        return OpenAiRealtimeProvider(
            record_id="",
            api_key=self._api_key,
            transport=self._transport,
        ).available()

    def validate_utterance(self, text: str) -> None:
        OpenAiRealtimeProvider(record_id="").validate_utterance(text)

    def profile_for(self, record_id: str) -> OpenAiRealtimeProvider:
        return OpenAiRealtimeProvider(
            record_id=record_id,
            api_key=self._api_key,
            transport=self._transport,
            operations_path=self._operations_path,
        )

    for_record = profile_for
