"""The VOICEVOX provider, and the endpoint choice that is the whole feature.

The failure this guards against does not look like a failure. ``is_kana=true``
is declared by ``/accent_phrases`` and by nothing else; sent to ``/audio_query``
it is silently dropped, the request succeeds, and the audio comes back accented
however the engine guessed. So the tests here are mostly about *which request
went where*, recorded by a fake transport — an assertion on the returned bytes
would pass just as happily against the broken flow.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pytest

from japanese_anki.tts import TtsError
from japanese_anki.tts.voicevox import LAUNCH_HINT, VoicevoxProvider

#: A value that exists only to be recognised on the other side. If this reaches
#: /synthesis, the forced accent phrases did.
FORCED = [{"marker": "forced-accent-phrases", "moras": []}]

#: What /audio_query answers: an envelope with the engine's *own* guess in it,
#: which must be replaced, and other keys, which must not be touched.
GUESSED = {
    "accent_phrases": [{"marker": "the engine's guess", "moras": []}],
    "speedScale": 1.0,
    "outputSamplingRate": 24000,
    "a_key_a_later_engine_added": "keep me",
}

WAV = b"RIFF....WAVEfake"


class FakeEngine:
    """Records every call, and answers per endpoint."""

    def __init__(
        self,
        *,
        version_status: int = 200,
        accent_phrases: Any = None,
        audio_query: Any = None,
        statuses: dict[str, int] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.version_status = version_status
        self.accent_phrases = FORCED if accent_phrases is None else accent_phrases
        self.audio_query = dict(GUESSED) if audio_query is None else audio_query
        self.statuses = statuses or {}

    def __call__(self, method: str, url: str, body: Any = None) -> tuple[int, bytes]:
        self.calls.append((method, url, body))
        path = urllib.parse.urlparse(url).path
        status = self.statuses.get(path, 200)
        if status != 200:
            return status, b"engine said no"
        if path == "/version":
            return self.version_status, b'"0.14.0"'
        if path == "/accent_phrases":
            return 200, json.dumps(self.accent_phrases).encode()
        if path == "/audio_query":
            return 200, json.dumps(self.audio_query).encode()
        if path == "/synthesis":
            return 200, WAV
        raise AssertionError(f"unexpected endpoint {path}")

    # -- helpers the assertions read through ------------------------------

    def paths(self) -> list[str]:
        return [urllib.parse.urlparse(url).path for _, url, _ in self.calls]

    def query(self, path: str) -> dict[str, list[str]]:
        for _, url, _ in self.calls:
            parsed = urllib.parse.urlparse(url)
            if parsed.path == path:
                return urllib.parse.parse_qs(parsed.query)
        raise AssertionError(f"{path} was never called")

    def body(self, path: str) -> Any:
        for _, url, body in self.calls:
            if urllib.parse.urlparse(url).path == path:
                return body
        raise AssertionError(f"{path} was never called")


def provider(engine: FakeEngine, **kwargs: Any) -> VoicevoxProvider:
    return VoicevoxProvider(
        base_url=kwargs.pop("base_url", "http://localhost:50021"),
        speaker=kwargs.pop("speaker", 46),
        transport=engine,
    )


# ---------------------------------------------------------------------------
# The endpoint choice
# ---------------------------------------------------------------------------


def test_is_kana_goes_to_accent_phrases_and_only_there() -> None:
    """`/audio_query` does not declare `is_kana`, and FastAPI drops parameters a
    route does not declare — so sending it there succeeds and returns audio the
    engine accented by guesswork. There is no error to notice."""
    engine = FakeEngine()

    provider(engine).synthesize("ハシ'", forced_accent=True)

    assert engine.query("/accent_phrases")["is_kana"] == ["true"]
    assert "is_kana" not in engine.query("/audio_query")
    assert "is_kana" not in engine.query("/synthesis")


def test_the_forced_accent_phrases_are_what_reaches_synthesis() -> None:
    """The round trip the whole provider exists for: what `/accent_phrases`
    returned for the *kana*, not what `/audio_query` guessed from the text."""
    engine = FakeEngine()

    provider(engine).synthesize("ハシ'", forced_accent=True)

    assert engine.body("/synthesis")["accent_phrases"] == FORCED


def test_the_rest_of_the_query_is_the_engines_own() -> None:
    """Only `accent_phrases` is replaced. The envelope is engine-versioned, so
    hand-constructing one breaks quietly on an upgrade that adds a field —
    which is why it is fetched rather than built."""
    engine = FakeEngine()

    provider(engine).synthesize("ハシ'", forced_accent=True)

    submitted = engine.body("/synthesis")
    assert submitted["speedScale"] == 1.0
    assert submitted["outputSamplingRate"] == 24000
    assert submitted["a_key_a_later_engine_added"] == "keep me"


def test_the_accent_mark_is_stripped_before_audio_query() -> None:
    """`/audio_query` does not read kana notation, so a mark left in is text to
    be pronounced. Only step one means anything by it."""
    engine = FakeEngine()

    provider(engine).synthesize("ハシ'", forced_accent=True)

    assert engine.query("/accent_phrases")["text"] == ["ハシ'"]
    assert engine.query("/audio_query")["text"] == ["ハシ"]


def test_the_calls_go_in_the_order_the_flow_requires() -> None:
    engine = FakeEngine()

    provider(engine).synthesize("ハシ'", forced_accent=True)

    assert engine.paths() == ["/accent_phrases", "/audio_query", "/synthesis"]


def test_the_speaker_goes_to_every_endpoint() -> None:
    engine = FakeEngine()

    provider(engine, speaker=3).synthesize("ハシ'", forced_accent=True)

    for path in ("/accent_phrases", "/audio_query", "/synthesis"):
        assert engine.query(path)["speaker"] == ["3"]


def test_the_wav_comes_back() -> None:
    engine = FakeEngine()

    assert provider(engine).synthesize("ハシ'", forced_accent=True) == WAV


# ---------------------------------------------------------------------------
# A sentence is a different request
# ---------------------------------------------------------------------------


def test_a_sentence_is_read_naturally_rather_than_forced() -> None:
    """Forcing an accent across a whole sentence would need accent data janki
    does not have for one, so the accent step is skipped entirely."""
    engine = FakeEngine()

    provider(engine).synthesize("毎日日本語を話します。", forced_accent=False)

    assert engine.paths() == ["/audio_query", "/synthesis"]
    assert engine.query("/audio_query")["text"] == ["毎日日本語を話します。"]


def test_a_sentence_keeps_the_engines_own_accent_phrases() -> None:
    engine = FakeEngine()

    provider(engine).synthesize("話します。", forced_accent=False)

    assert engine.body("/synthesis")["accent_phrases"] == GUESSED["accent_phrases"]


def test_an_apostrophe_in_a_sentence_is_not_stripped() -> None:
    """The mark only means an accent in kana notation. In ordinary text it is
    punctuation, and removing it would change what is spoken."""
    engine = FakeEngine()

    provider(engine).synthesize("It's fine.", forced_accent=False)

    assert engine.query("/audio_query")["text"] == ["It's fine."]


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


def test_available_asks_for_the_version() -> None:
    engine = FakeEngine()

    assert provider(engine).available() is True
    assert engine.paths() == ["/version"]


def test_an_engine_that_is_not_running_is_a_no_rather_than_a_crash() -> None:
    """The ordinary state of a local engine, not an exceptional one."""

    def refused(method: str, url: str, body: Any = None) -> tuple[int, bytes]:
        raise TtsError("Could not reach VOICEVOX: connection refused")

    assert VoicevoxProvider(transport=refused).available() is False


def test_an_error_status_is_also_a_no() -> None:
    engine = FakeEngine(statuses={"/version": 500})

    assert provider(engine).available() is False


def test_the_launch_hint_says_how_to_start_it() -> None:
    assert "VOICEVOX" in LAUNCH_HINT
    assert "localhost:50021" in LAUNCH_HINT
    assert VoicevoxProvider().launch_hint == LAUNCH_HINT


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/accent_phrases", "/audio_query", "/synthesis"])
def test_an_error_at_any_step_names_the_step(path: str) -> None:
    engine = FakeEngine(statuses={path: 422})

    with pytest.raises(TtsError) as caught:
        provider(engine).synthesize("ハシ'", forced_accent=True)

    assert path in str(caught.value)
    assert "422" in str(caught.value)


def test_an_audio_query_that_is_not_an_object_is_refused() -> None:
    # Rather than building an AudioQuery to replace it: the schema is the
    # engine's, and a guessed one is the quiet failure this provider avoids.
    engine = FakeEngine(audio_query=["not", "an", "object"])

    with pytest.raises(TtsError) as caught:
        provider(engine).synthesize("ハシ'", forced_accent=True)

    assert "not an object" in str(caught.value)


def test_a_non_json_answer_where_json_was_expected_is_refused() -> None:
    class Garbage(FakeEngine):
        def __call__(self, method: str, url: str, body: Any = None) -> tuple[int, bytes]:
            self.calls.append((method, url, body))
            return 200, b"<html>not json</html>"

    with pytest.raises(TtsError) as caught:
        provider(Garbage()).synthesize("ハシ'", forced_accent=True)

    assert "not JSON" in str(caught.value)


def test_nothing_to_say_is_refused_before_any_call() -> None:
    engine = FakeEngine()

    with pytest.raises(TtsError):
        provider(engine).synthesize("   ", forced_accent=True)

    assert engine.calls == []


def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    engine = FakeEngine()

    provider(engine, base_url="http://localhost:50021/").synthesize(
        "ハシ'", forced_accent=True
    )

    assert all("//version" not in url and "50021//" not in url for _, url, _ in engine.calls)
