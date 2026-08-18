"""The OpenAI speech provider — sentences only.

The load-bearing test in this file is the one asserting it *refuses* a word. The
provider cannot force a pitch accent, and one that accepted the flag and ignored
it would return plausible audio for 橋 that says 箸 — an error arriving as a
working clip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli
from japanese_anki.config import ProjectConfig
from japanese_anki.tts import TtsError, openai_tts
from japanese_anki.tts.openai_tts import OpenAiSpeechProvider

MP3 = b"ID3\x03\x00\x00\x00fake-mp3"


class FakeApi:
    """Records every call, answers with whatever it was told to."""

    def __init__(self, *, status: int = 200, payload: bytes = MP3) -> None:
        self.calls: list[tuple[str, str, Any, dict[str, str]]] = []
        self.status = status
        self.payload = payload

    def __call__(
        self, method: str, url: str, body: Any = None, headers: dict[str, str] | None = None
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, body, headers or {}))
        return self.status, self.payload

    @property
    def body(self) -> dict[str, Any]:
        assert self.calls, "nothing was sent"
        return self.calls[-1][2]


def provider(api: FakeApi, **kwargs: Any) -> OpenAiSpeechProvider:
    kwargs.setdefault("api_key", "test-key")
    return OpenAiSpeechProvider(transport=api, **kwargs)


# --- it must not voice a word ----------------------------------------------


def test_a_word_is_refused_rather_than_guessed_at() -> None:
    """The whole reason VOICEVOX stays in the project. This API has no accent
    control, so 橋 and 箸 would come back identical — verified live against a
    real engine for the unforced case. A provider that accepted `forced_accent`
    and ignored it would turn that into a clip that sounds fine and is wrong."""
    api = FakeApi()

    with pytest.raises(TtsError, match="cannot force a pitch accent"):
        provider(api).synthesize("ハシ'", forced_accent=True)

    assert api.calls == [], "and it spends nothing finding out"


def test_a_sentence_is_read() -> None:
    api = FakeApi()

    audio = provider(api).synthesize("毎日、妻と日本語で話します。", forced_accent=False)

    assert audio == MP3
    assert api.body["input"] == "毎日、妻と日本語で話します。"


def test_an_empty_sentence_is_refused() -> None:
    api = FakeApi()

    with pytest.raises(TtsError, match="Nothing to speak"):
        provider(api).synthesize("   ", forced_accent=False)


def test_a_blank_model_is_refused_before_any_request() -> None:
    with pytest.raises(TtsError, match="model.*cannot be blank"):
        provider(FakeApi(), model="   ")


# --- the request shape ------------------------------------------------------


def test_the_instructions_carry_the_pace() -> None:
    """Instructions are the pace control janki exposes for sentence audio.

    The API also supports a numeric speed field, but janki leaves that at its
    default. If instructions stop being sent, the learner-specific delivery
    silently disappears and nothing reports it.
    """
    api = FakeApi()

    provider(api, instructions="Speak very slowly.").synthesize("テスト。", forced_accent=False)

    assert api.body["instructions"] == "Speak very slowly."


def test_empty_instructions_are_left_out_entirely() -> None:
    # Not sent as "", which is a request for a style rather than the absence of
    # one, and which a model is entitled to interpret.
    api = FakeApi()

    provider(api, instructions="   ").synthesize("テスト。", forced_accent=False)

    assert "instructions" not in api.body


@pytest.mark.parametrize(
    ("global_instructions", "clip_instructions", "expected"),
    [
        ("Global.", "Clip.", "Global.\n\nClip."),
        ("Global.", "   ", "Global."),
        ("   ", "Clip.", "Clip."),
        ("   ", "   ", None),
    ],
)
def test_clip_instructions_append_to_the_global_instructions(
    global_instructions: str,
    clip_instructions: str,
    expected: str | None,
) -> None:
    """A clip hint augments the collection-wide pace/style contract; it does
    not replace it. The prepared profile is also the exact ledger profile."""
    api = FakeApi()
    base = provider(api, instructions=global_instructions)

    prepared = base.for_clip(clip_instructions)
    prepared.synthesize("テスト。", forced_accent=False)

    if expected is None:
        assert "instructions" not in api.body
        assert prepared.settings == {
            "model": openai_tts.DEFAULT_MODEL,
            "instructions": "",
        }
    else:
        assert api.body["instructions"] == expected
        assert prepared.settings["instructions"] == expected
    assert api.body["input"] == "テスト。", (
        "steering belongs in instructions, never in the spoken sentence"
    )
    assert base.instructions == global_instructions.strip(), (
        "preparing one clip cannot mutate another"
    )


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd"])
def test_the_models_that_ignore_instructions_are_refused_before_a_request(
    model: str,
) -> None:
    """The API documents that ``tts-1`` and ``tts-1-hd`` ignore this field.
    Recording a hint as honored while the model ignored it would make a known-
    wrong clip look current forever."""
    api = FakeApi()

    with pytest.raises(TtsError, match="does not support instructions"):
        provider(api, model=model, instructions="Read slowly.").synthesize(
            "テスト。", forced_accent=False
        )

    assert api.calls == []


def test_instruction_length_is_checked_at_the_documented_boundary() -> None:
    accepted = FakeApi()
    provider(accepted, instructions="x" * 4096).synthesize(
        "テスト。", forced_accent=False
    )
    assert len(accepted.body["instructions"]) == 4096

    refused = FakeApi()
    with pytest.raises(TtsError, match="4096-character limit"):
        provider(refused, instructions="x" * 4097).synthesize(
            "テスト。", forced_accent=False
        )
    assert refused.calls == []


def test_sentence_length_is_checked_at_the_documented_boundary() -> None:
    accepted = FakeApi()
    provider(accepted).synthesize("x" * 4096, forced_accent=False)
    assert len(accepted.body["input"]) == 4096

    refused = FakeApi()
    with pytest.raises(TtsError, match="4096-character limit"):
        provider(refused).synthesize("x" * 4097, forced_accent=False)
    assert refused.calls == []


@pytest.mark.parametrize(
    ("field", "build"),
    [
        ("input", lambda: provider(FakeApi()).validate_utterance("\ud800")),
        ("instructions", lambda: provider(FakeApi()).for_clip("\ud800")),
    ],
)
def test_request_text_must_be_utf8_before_transport(
    field: str, build: Any
) -> None:
    with pytest.raises(TtsError, match=f"{field}.*valid UTF-8"):
        build()


def test_tts_1_remains_usable_when_no_instructions_are_claimed() -> None:
    api = FakeApi()

    provider(api, model="tts-1", instructions="").synthesize(
        "テスト。", forced_accent=False
    )

    assert "instructions" not in api.body


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd"])
def test_legacy_models_refuse_voices_the_endpoint_does_not_offer(model: str) -> None:
    with pytest.raises(TtsError, match="does not support voice 'ballad'"):
        provider(FakeApi(), model=model, voice="ballad", instructions="")


def test_a_supported_legacy_model_voice_remains_usable() -> None:
    api = FakeApi()
    provider(api, model="tts-1", voice="onyx", instructions="").synthesize(
        "テスト。", forced_accent=False
    )
    assert api.body["voice"] == "onyx"


def test_prepared_clip_profiles_are_independent() -> None:
    api = FakeApi()
    base = provider(api, instructions="Global.")
    first = base.for_clip("First.")
    second = base.for_clip("Second.")

    first.synthesize("一。", forced_accent=False)
    first_body = dict(api.body)
    second.synthesize("二。", forced_accent=False)

    assert first_body["instructions"] == "Global.\n\nFirst."
    assert api.body["instructions"] == "Global.\n\nSecond."
    assert base.settings["instructions"] == "Global."


def test_a_prepared_clip_preserves_the_configured_voice_model_and_profile() -> None:
    api = FakeApi()
    base = provider(
        api,
        voice="ash",
        model="gpt-4o-mini-tts-2025-12-15",
        instructions="Global.",
    )

    prepared = base.for_clip("Clip.")
    prepared.synthesize("一。", forced_accent=False)

    assert api.body["voice"] == "ash"
    assert api.body["model"] == "gpt-4o-mini-tts-2025-12-15"
    assert api.body["instructions"] == "Global.\n\nClip."
    assert "speed" not in api.body
    assert prepared.speed == 1.0
    assert prepared.settings == {
        "model": "gpt-4o-mini-tts-2025-12-15",
        "instructions": "Global.\n\nClip.",
    }


def test_mp3_is_requested_not_wav() -> None:
    """The API's WAV is a *streaming* WAV: RIFF and data chunk sizes are
    0xFFFFFFFF placeholders, so the header claims 89,478 seconds for a
    4.5-second clip (measured). Players reading to EOF cope; anything trusting
    the header does not."""
    api = FakeApi()

    sent = provider(api)
    sent.synthesize("テスト。", forced_accent=False)

    assert api.body["response_format"] == "mp3"
    assert sent.suffix == ".mp3", "and the file is named for what it holds"


def test_the_key_travels_as_a_bearer_header_and_not_in_the_body() -> None:
    api = FakeApi()

    provider(api, api_key="sk-not-a-real-key").synthesize("テスト。", forced_accent=False)

    _, _, body, headers = api.calls[-1]
    assert headers["Authorization"] == "Bearer sk-not-a-real-key"
    assert "sk-not-a-real-key" not in json.dumps(body), "never in the payload"


def test_the_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never from janki.toml, like every other credential here."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-env")
    api = FakeApi()

    OpenAiSpeechProvider(transport=api).synthesize("テスト。", forced_accent=False)

    assert api.calls[-1][3]["Authorization"] == "Bearer sk-from-the-env"


def test_a_missing_key_is_not_available_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`available()` is asked before a run spends anything, and a missing key is
    an ordinary state — the caller decides, and its message carries the hint."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api = FakeApi()

    engine = OpenAiSpeechProvider(transport=api)

    assert engine.available() is False
    assert api.calls == [], "and it does not spend a request to find out"
    with pytest.raises(TtsError, match="OPENAI_API_KEY"):
        engine.synthesize("テスト。", forced_accent=False)


def test_available_spends_nothing_when_a_key_is_present() -> None:
    api = FakeApi()

    assert provider(api).available() is True
    assert api.calls == []


# --- what it records --------------------------------------------------------


def test_the_voice_is_recorded_by_name() -> None:
    assert provider(FakeApi(), voice="ash").voice == "ash"
    assert provider(FakeApi()).voice == "onyx", "the default matches the word voice's register"


def test_an_unknown_voice_is_refused_at_construction() -> None:
    """Including the ChatGPT app's voices, which are a different list — checked
    against the API's own 400 on 2026-08-09."""
    with pytest.raises(TtsError, match="Unknown OpenAI voice 'cove'") as caught:
        provider(FakeApi(), voice="cove")

    assert "onyx" in str(caught.value), "and says what is available"


def test_janki_leaves_openai_speed_at_the_api_default() -> None:
    """OpenAI supports speed, but janki has no sentence-speed setting.

    It deliberately omits the request field and records the API's 1.0 default;
    the VOICEVOX word-speed knob must not change sentence currency.
    """
    api = FakeApi()

    engine = provider(api)
    engine.synthesize("テスト。", forced_accent=False)

    assert engine.speed == 1.0
    assert "speed" not in api.body


# --- failures ---------------------------------------------------------------


def test_an_api_error_carries_the_api_s_own_message() -> None:
    api = FakeApi(
        status=400,
        payload=json.dumps({"error": {"message": "Invalid value: 'cove'."}}).encode(),
    )

    with pytest.raises(TtsError, match="Invalid value: 'cove'"):
        provider(api).synthesize("テスト。", forced_accent=False)


def test_a_non_json_error_body_is_still_reported() -> None:
    api = FakeApi(status=502, payload=b"<html>bad gateway</html>")

    with pytest.raises(TtsError, match="502"):
        provider(api).synthesize("テスト。", forced_accent=False)


def test_an_empty_200_is_not_treated_as_audio() -> None:
    """An empty file would be written, named, and recorded as current — a silent
    card that nothing ever revisits."""
    api = FakeApi(payload=b"")

    with pytest.raises(TtsError, match="returned no audio"):
        provider(api).synthesize("テスト。", forced_accent=False)


# --- selection --------------------------------------------------------------


def _config(tmp_path: Path, tts: str) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(f"[tts]\n{tts}\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def test_selecting_openai_gives_sentences_a_different_engine(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        'voicevox_speaker = 13\nsentence_provider = "openai"\nopenai_voice = "ash"',
    )

    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)

    assert (words.name, words.voice) == ("voicevox", 13), "words stay on VOICEVOX"
    assert (sentences.name, sentences.voice) == ("openai", "ash")


def test_the_configured_instructions_reach_the_provider(tmp_path: Path) -> None:
    config = _config(
        tmp_path, 'sentence_provider = "openai"\nopenai_instructions = "Very slowly."'
    )

    sentences = cli._sentence_provider(config, None, object())

    assert sentences.instructions == "Very slowly."


def test_the_default_instructions_ask_for_a_slower_pace(tmp_path: Path) -> None:
    config = _config(tmp_path, 'sentence_provider = "openai"')

    sentences = cli._sentence_provider(config, None, object())

    assert "slower" in sentences.instructions.lower()


def test_an_unknown_sentence_provider_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path, 'sentence_provider = "unknown-engine"')

    with pytest.raises(cli.AudioError, match="Unknown \\[tts\\] sentence_provider"):
        cli._sentence_provider(config, None, object())


def test_the_voice_list_matches_what_the_api_accepts() -> None:
    """Recorded from the API's own 400 so a typo fails at construction rather
    than mid-run on the first sentence of a long collection."""
    assert "cove" not in openai_tts.VOICES, "a ChatGPT app voice, not an API one"
    assert {"onyx", "ash", "nova", "cedar", "marin"} <= set(openai_tts.VOICES)


# --- the transport itself ---------------------------------------------------


def test_a_truncated_response_arrives_as_a_janki_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`http.client.IncompleteRead` is neither an OSError nor a ValueError, so
    it escaped every handler up to `main` as a traceback — taking the run down
    *without saving*, which leaves the clips already written with no record
    reference for the next `--prune` to delete. That is the loss the
    stop-and-keep design exists to prevent."""
    import http.client
    import urllib.request

    def truncated(*_args: Any, **_kwargs: Any) -> Any:
        raise http.client.IncompleteRead(b"half a body")

    monkeypatch.setattr(urllib.request, "urlopen", truncated)

    with pytest.raises(TtsError, match="Could not reach"):
        openai_tts.urllib_transport("POST", "https://api.openai.com/v1/audio/speech", {"a": 1}, {})


def test_a_surrogate_in_a_sentence_is_a_janki_error_not_a_traceback() -> None:
    """Encoding happens inside the guard too: a lone surrogate is a
    UnicodeEncodeError, which is a ValueError, and it fires before any I/O —
    outside the try it took the same run-losing path."""
    with pytest.raises(TtsError, match="Could not reach"):
        openai_tts.urllib_transport(
            "POST", "https://api.openai.com/v1/audio/speech", {"input": "\ud800"}, {}
        )
