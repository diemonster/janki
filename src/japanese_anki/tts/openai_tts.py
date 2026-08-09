"""OpenAI speech, for sentences only — it cannot force an accent.

**This provider must never voice a word.** Every janki word clip is synthesized
from AquesTalk kana with its pitch accent forced, because 橋 and 箸 are the pair
a card exists to distinguish and an engine left to guess renders them alike —
verified against a live engine: unforced, 橋/箸/端 all come out accent 1. This
API has no accent control, no SSML, and no phoneme override, so it cannot do
that job at all. It refuses ``forced_accent=True`` rather than quietly returning
a guess, which is the failure mode this whole milestone was built around.

For a *sentence* nothing is forced anyway — janki has no accent data for a whole
sentence and says so — so the objection does not apply, and the question becomes
which engine reads Japanese better. That one was settled by listening.

Named ``openai_tts`` rather than ``openai`` deliberately: nothing here imports
the ``openai`` SDK (this is the same raw-HTTP transport shape as ``voicevox``),
and a module named for a package janki does not depend on is a question every
future reader has to answer twice.

**Format is mp3, not wav.** The API's WAV is a *streaming* WAV: the RIFF and
data chunk sizes are ``0xFFFFFFFF`` placeholders, so the header claims 89,478
seconds for a 4.5-second clip. Players that read to EOF cope; anything that
trusts the header does not. mp3 has no such trap and is a word of configuration
rather than a header rewrite janki would have to maintain.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from japanese_anki.tts import TtsError

__all__ = [
    "API_URL",
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_VOICE",
    "KEY_HINT",
    "OpenAiSpeechProvider",
    "VOICES",
]

API_URL = "https://api.openai.com/v1/audio/speech"

#: Generous: a sentence is one request, but this is a network round trip to a
#: model that generates audio, not a local engine answering in milliseconds.
DEFAULT_TIMEOUT = 120.0

DEFAULT_MODEL = "gpt-4o-mini-tts"

#: Deep and male, the nearest match to the VOICEVOX speaker chosen for words —
#: a card that says the word in one register and the sentence in another for no
#: reason is a distraction, and the point of a separate sentence voice is that
#: it is a *different reader*, not a different species.
DEFAULT_VOICE = "onyx"

#: What the API accepts, recorded here so a typo fails at config load rather
#: than mid-run on the first sentence. Taken verbatim from the API's own 400
#: (checked 2026-08-09); ``cove`` and the other ChatGPT app voices are not here,
#: because Advanced Voice Mode and this endpoint do not share a voice list.
VOICES: tuple[str, ...] = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "marin", "nova", "onyx", "sage", "shimmer", "verse",
)

#: The only rate control this API has. There is no ``speed`` parameter on
#: ``gpt-4o-mini-tts``, so pace is asked for in prose or not at all — which is
#: why ``speed`` on this provider is recorded but never sent.
DEFAULT_INSTRUCTIONS = (
    "Read this as a native speaker of standard Tokyo Japanese, for someone "
    "learning the language. Speak noticeably slower than conversational pace, "
    "clearly and calmly, with natural pitch accent, and pause briefly at each "
    "comma. Do not sound hurried."
)

KEY_HINT = (
    "Set OPENAI_API_KEY in your environment and try again — it is never read "
    "from janki.toml. Put it in ~/.zshenv rather than ~/.zshrc so non-login "
    "shells (and this tool) can see it."
)

#: ``(method, url, body, headers) -> (status, bytes)``. Same seam as the
#: VOICEVOX transport, so tests swap a fake in rather than a network.
Transport = Callable[[str, str, Any, dict[str, str]], tuple[int, bytes]]


def urllib_transport(
    method: str,
    url: str,
    body: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, bytes]:
    """One HTTPS call, returning the raw body. Status is returned, not raised."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        # Reading the error body is socket I/O on a connection that already
        # misbehaved, and it happens inside this handler where the clause below
        # cannot reach it. The status is the part worth having.
        try:
            detail = exc.read()
        except OSError:
            detail = b""
        return exc.code, detail
    except (OSError, ValueError) as exc:
        # ValueError as well as OSError: a malformed URL raises before any I/O.
        raise TtsError(f"Could not reach the OpenAI speech API: {exc}") from exc


class OpenAiSpeechProvider:
    """OpenAI text-to-speech, for example sentences."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        instructions: str = DEFAULT_INSTRUCTIONS,
        api_key: str | None = None,
        transport: Transport | None = None,
        speed: float = 1.0,
    ) -> None:
        chosen = str(voice).strip().lower()
        if chosen not in VOICES:
            raise TtsError(
                f"Unknown OpenAI voice {voice!r}. Available: {', '.join(VOICES)}. "
                "(The ChatGPT app's voices — Cove, Juniper, Breeze — are a "
                "different set and are not offered by this API.)"
            )
        self._voice = chosen
        self._model = str(model)
        self._instructions = str(instructions)
        self._speed = float(speed)
        self._api_key = api_key
        self._transport: Transport = transport or urllib_transport

    @property
    def name(self) -> str:
        return "openai"

    @property
    def voice(self) -> str:
        """The voice name, as the ledger records it.

        A string where VOICEVOX gives an int, because that is what this API
        calls its voices. Mapping ``onyx`` onto an index would put a number in
        the ledger that means nothing outside janki and would silently change
        meaning the day the voice list grows.
        """
        return self._voice

    @property
    def speed(self) -> float:
        """Recorded, never sent: this API has no rate parameter.

        It is still part of what makes a clip what it is, and the ledger
        compares it, so a provider that reported nothing would make a
        configuration change undetectable. Pace is asked for in
        ``instructions`` instead.
        """
        return self._speed

    @property
    def suffix(self) -> str:
        return ".mp3"

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def launch_hint(self) -> str:
        return KEY_HINT

    def _key(self) -> str:
        key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise TtsError(f"No OPENAI_API_KEY. {KEY_HINT}")
        return key.strip()

    def available(self) -> bool:
        """Is there a key to try with?

        Deliberately does *not* spend a request to find out. Never raises: a
        missing key is an ordinary state the caller decides about, and the
        caller's message already carries :attr:`launch_hint`.
        """
        try:
            self._key()
        except TtsError:
            return False
        return True

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        """Audio for one sentence. Refuses a word.

        The refusal is the point. This API cannot force an accent, and a
        provider that accepted the flag and ignored it would return plausible
        audio for 橋 that says 箸 — an error arriving as a working clip, which
        is the worst way for it to arrive.
        """
        if forced_accent:
            raise TtsError(
                "The OpenAI speech API cannot force a pitch accent, so it "
                "cannot voice a word: 橋 and 箸 would come out identical. Use "
                "it for example sentences (tts.sentence_provider) and leave "
                "words to VOICEVOX."
            )
        spoken = text_or_kana.strip()
        if not spoken:
            raise TtsError("Nothing to speak: the sentence was empty.")

        body: dict[str, Any] = {
            "model": self._model,
            "voice": self._voice,
            "input": spoken,
            "response_format": "mp3",
        }
        if self._instructions.strip():
            body["instructions"] = self._instructions
        status, payload = self._transport(
            "POST",
            API_URL,
            body,
            {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"},
        )
        if status != 200:
            raise TtsError(f"OpenAI speech API answered {status}: {_detail(payload)}")
        if not payload:
            raise TtsError("The OpenAI speech API returned no audio.")
        return payload


def _detail(payload: bytes) -> str:
    """The API's own message, when it sent one JSON body's worth."""
    try:
        message = json.loads(payload).get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return str(message) or payload[:200].decode("utf-8", errors="replace")
