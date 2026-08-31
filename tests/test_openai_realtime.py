"""Contract for deterministic OpenAI Realtime sentence audio.

These tests intentionally precede the provider.  Every transport is fake: this
module must never open a socket or spend money.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import struct
import wave
from collections.abc import Iterable
from typing import Any

import pytest

from japanese_anki.tts import TtsError
from japanese_anki.tts.openai_realtime import (
    INSTRUCTIONS,
    MODEL,
    SAMPLE_RATE,
    VOICES,
    OpenAiRealtimeProvider,
    voice_for_record,
)

EXPECTED_INSTRUCTIONS = (
    "Read the user's text aloud exactly once as a native speaker of standard "
    "Tokyo Japanese. Do not add, omit, translate, answer, or explain anything. "
    "Speak clearly at a slower, learner-friendly pace, approximately 75% of "
    "normal conversational speed, while preserving natural Japanese intonation "
    "and connected phrasing."
)


def _expected_voice(record_id: str) -> str:
    """Independent statement of the frozen v1 framed selector."""
    raw = record_id.encode("utf-8")
    framed = b"janki:realtime-voice\0v1\0" + len(raw).to_bytes(8, "big") + raw
    digest = hashlib.sha256(framed).digest()
    return ("cedar", "ash", "verse", "marin")[
        int.from_bytes(digest[:8], "big") % 4
    ]


def _pcm() -> bytes:
    samples = [
        round(5000 * math.sin(2 * math.pi * 220 * frame / 24_000))
        for frame in range(2400)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _completed_events(*, voice: str, pcm: bytes | None = None) -> list[dict[str, Any]]:
    audio = _pcm() if pcm is None else pcm
    midpoint = len(audio) // 2
    return [
        {"type": "session.created", "session": {"model": MODEL}},
        {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(audio[:midpoint]).decode("ascii"),
        },
        {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(audio[midpoint:]).decode("ascii"),
        },
        {
            "type": "response.output_audio_transcript.done",
            "transcript": "橋を渡る。",
        },
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "voice": voice,
                    }
                },
                "usage": {"input_tokens": 5, "output_tokens": 8},
            },
        },
    ]


class FakeTransport:
    def __init__(self, events: Iterable[dict[str, Any]] | BaseException) -> None:
        self.events = events
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, endpoint: str, headers: dict[str, str], request: dict[str, Any]
    ) -> Iterable[dict[str, Any]]:
        self.calls.append((endpoint, headers, request))
        if isinstance(self.events, BaseException):
            raise self.events
        return self.events


def _provider(
    record_id: str = "word:橋:はし",
    *,
    events: Iterable[dict[str, Any]] | BaseException | None = None,
    api_key: str = "super-secret-key",
) -> tuple[OpenAiRealtimeProvider, FakeTransport]:
    voice = voice_for_record(record_id)
    transport = FakeTransport(_completed_events(voice=voice) if events is None else events)
    return (
        OpenAiRealtimeProvider(
            record_id=record_id,
            api_key=api_key,
            transport=transport,
        ),
        transport,
    )


def _capture_and_decode(
    provider: OpenAiRealtimeProvider,
    text: str = "橋を渡る。",
    *,
    forced_accent: bool = False,
) -> bytes:
    """Exercise fake transport and pure decoding without the production router."""
    captured: list[str] = []
    envelope = provider._capture_response(
        text,
        forced_accent=forced_accent,
        capture_frame=captured.append,
    )
    return provider.decode_response(envelope)


def test_profile_and_prompt_are_the_reviewed_audition_contract() -> None:
    provider, _transport = _provider()

    assert MODEL == "gpt-realtime-1.5"
    assert VOICES == ("cedar", "ash", "verse", "marin")
    assert SAMPLE_RATE == 24_000
    assert INSTRUCTIONS == EXPECTED_INSTRUCTIONS
    assert provider.name == "openai-realtime"
    assert provider.suffix == ".wav"
    # Pace is requested in the reviewed prose; no numeric speed field is sent.
    assert provider.speed == 1.0
    assert provider.settings == {
        "model": MODEL,
        "instructions": EXPECTED_INSTRUCTIONS,
        "audio_format": "pcm16-24000-mono",
    }


@pytest.mark.parametrize(
    "record_id",
    ["word:橋:はし", "word:止む:やむ", "word:食べる:たべる", "é\x00日本語"],
)
def test_voice_selection_is_frozen_versioned_framed_sha256(record_id: str) -> None:
    assert voice_for_record(record_id) == _expected_voice(record_id)


def test_all_examples_of_one_record_share_one_voice() -> None:
    first, _ = _provider("word:橋:はし")
    second, _ = _provider("word:橋:はし")

    assert first.voice == second.voice == voice_for_record("word:橋:はし")
    assert first.voice in VOICES


def test_request_is_one_exact_sentence_only_response_create() -> None:
    provider, transport = _provider()

    _capture_and_decode(provider)

    assert len(transport.calls) == 1
    endpoint, headers, request = transport.calls[0]
    assert endpoint == f"wss://api.openai.com/v1/realtime?model={MODEL}"
    assert headers == {"Authorization": "Bearer super-secret-key"}
    assert request == {
        "type": "response.create",
        "response": {
            "conversation": "none",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "橋を渡る。"}],
                }
            ],
            "instructions": EXPECTED_INSTRUCTIONS,
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24_000},
                    "voice": provider.voice,
                }
            },
            "tool_choice": "none",
        },
    }
    assert "super-secret-key" not in json.dumps(request)


def test_completed_audio_deltas_become_a_finite_pcm_wav() -> None:
    provider, _transport = _provider()

    result = _capture_and_decode(provider)

    with wave.open(io.BytesIO(result), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24_000
        assert wav.getcomptype() == "NONE"
        assert wav.getnframes() == len(_pcm()) // 2
        assert wav.readframes(wav.getnframes()) == _pcm()
    assert result[4:8] == (len(result) - 8).to_bytes(4, "little")


def test_realtime_provider_is_sentence_only() -> None:
    provider, transport = _provider()

    with pytest.raises(TtsError, match="sentence|accent"):
        _capture_and_decode(provider, "ハシ", forced_accent=True)

    assert transport.calls == []


def test_receive_without_a_durable_frame_capture_is_refused_before_transport() -> None:
    provider, transport = _provider()

    assert not hasattr(provider, "capture_response")
    with pytest.raises(TtsError, match="durable|journal|capture"):
        provider._capture_response(
            "橋を渡る。",
            forced_accent=False,
            capture_frame=None,
        )

    assert transport.calls == []


@pytest.mark.parametrize(
    "first",
    [
        {"type": "session.updated", "session": {"model": MODEL}},
        {"type": "session.created"},
        {"type": "session.created", "session": {"model": "wrong-model"}},
    ],
)
def test_invalid_handshake_is_captured_but_never_resumes_transport_to_send(
    first: dict[str, Any],
) -> None:
    trace: list[str] = []
    captured: list[str] = []

    def transport(
        _endpoint: str,
        _headers: dict[str, str],
        _request: dict[str, Any],
    ) -> Iterable[object]:
        def stream() -> Iterable[object]:
            trace.append("received-handshake")
            yield json.dumps(first, separators=(",", ":"))
            trace.append("sent-response-create")
            yield from _completed_events(voice=voice_for_record("word:橋:はし"))[1:]

        return stream()

    provider = OpenAiRealtimeProvider(
        record_id="word:橋:はし",
        api_key="not-a-real-key",
        transport=transport,
    )

    with pytest.raises(TtsError, match="session|handshake|model"):
        provider._capture_response(
            "橋を渡る。",
            forced_accent=False,
            capture_frame=captured.append,
        )

    assert trace == ["received-handshake"]
    assert captured == [json.dumps(first, separators=(",", ":"))]


def _refuses(events: list[dict[str, Any]], match: str) -> None:
    provider, transport = _provider(events=events)
    with pytest.raises(TtsError, match=match):
        _capture_and_decode(provider)
    assert len(transport.calls) == 1, "one synthesize call never retries a paid request"


def test_provider_error_is_refused() -> None:
    _refuses(
        [
            {"type": "session.created", "session": {"model": MODEL}},
            {"type": "error", "error": {"message": "refused"}},
        ],
        "refused|provider",
    )


def test_incomplete_response_is_refused() -> None:
    events = _completed_events(voice=voice_for_record("word:橋:はし"))
    events[-1]["response"]["status"] = "incomplete"
    _refuses(events, "incomplete|completed")


def test_completed_response_without_audio_is_refused() -> None:
    voice = voice_for_record("word:橋:はし")
    events = [event for event in _completed_events(voice=voice, pcm=b"")]
    _refuses(events, "audio|empty")


def test_bad_audio_base64_is_refused() -> None:
    events = _completed_events(voice=voice_for_record("word:橋:はし"))
    events[1]["delta"] = "not base64!"
    _refuses(events, "base64|audio")


def test_mismatched_session_model_is_refused() -> None:
    events = _completed_events(voice=voice_for_record("word:橋:はし"))
    events[0]["session"]["model"] = "different-model"
    _refuses(events, "model")


def test_mismatched_response_voice_is_refused() -> None:
    chosen = voice_for_record("word:橋:はし")
    wrong = next(voice for voice in VOICES if voice != chosen)
    _refuses(_completed_events(voice=wrong), "voice")


def test_transport_failure_is_not_retried() -> None:
    provider, transport = _provider(events=OSError("connection lost"))

    with pytest.raises(TtsError, match="connection|Realtime|OpenAI"):
        _capture_and_decode(provider)

    assert len(transport.calls) == 1
