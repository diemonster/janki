"""OpenAI speech, for sentences only — it cannot force an accent.

**This provider must never voice a word.** Every janki word clip is synthesized
from AquesTalk kana with its pitch accent forced, because 橋 and 箸 are the pair
a card exists to distinguish and an engine left to guess renders them alike —
verified against a live engine: unforced, 橋/箸/端 all come out accent 1. This
API has no structured, guaranteed lexical pitch-accent control, no SSML, and no
phoneme override, so it cannot do that job at all. It refuses
``forced_accent=True`` rather than quietly returning a guess, which is the
failure mode this whole milestone was built around.

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

import http.client
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

# The API accepts these models but explicitly does not apply ``instructions``
# to them. Keep this a capability refusal, not an allowlist of model names: new
# instruction-capable models remain usable without a janki release.
_MODELS_WITHOUT_INSTRUCTIONS = frozenset({"tts-1", "tts-1-hd"})
_LEGACY_MODEL_VOICES = frozenset(
    {"alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}
)
_MAX_INSTRUCTIONS_LENGTH = 4096
_MAX_INPUT_LENGTH = 4096

#: Deep and male, the nearest match to the VOICEVOX speaker chosen for words —
#: a card that says the word in one register and the sentence in another for no
#: reason is a distraction, and the point of a separate sentence voice is that
#: it is a *different reader*, not a different species.
DEFAULT_VOICE = "onyx"

#: The built-in voices janki supports, recorded here so a typo fails at config
#: load rather than mid-run on the first sentence. Custom voice objects are a
#: separate API surface and intentionally are not configuration values here.
VOICES: tuple[str, ...] = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "marin", "nova", "onyx", "sage", "shimmer", "verse",
)

#: The learner-specific pace control janki exposes. The endpoint also accepts
#: numeric ``speed``; janki deliberately leaves it at the API's 1.0 default
#: rather than coupling sentence delivery to the VOICEVOX word-speed setting.
DEFAULT_INSTRUCTIONS = (
    "Read this as a native speaker of standard Tokyo Japanese, for someone "
    "learning the language. Speak noticeably slower than conversational pace, "
    "clearly and calmly, with natural pitch accent, and pause briefly at each "
    "comma. Do not sound hurried."
)

KEY_HINT = (
    "no OPENAI_API_KEY. Set it in your environment and try again — it is never read "
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
    try:
        # Encoding and `Request()` are *inside* the try. Both can raise before
        # any I/O — a lone surrogate in a sentence is a UnicodeEncodeError, a
        # malformed URL a ValueError — and anything escaping this function
        # escapes `synthesize`, `_example_audio`, `generate_audio` and `main`
        # as a traceback, taking down the run *without saving* the clips
        # already written. Those files then have no record reference and the
        # next `--prune` deletes them, which is the loss the whole
        # stop-and-keep design exists to prevent.
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = urllib.request.Request(
            url, data=data, headers=headers or {}, method=method
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        # Reading the error body is socket I/O on a connection that already
        # misbehaved, and it happens inside this handler where the clauses
        # below cannot reach it. The status is the part worth having.
        try:
            detail = exc.read()
        except (OSError, http.client.HTTPException):
            detail = b""
        return exc.code, detail
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # HTTPException is neither of the other two: a truncated response body
        # is `http.client.IncompleteRead`, and `BadStatusLine` and
        # `LineTooLong` arrive the same way. The VOICEVOX transport guards the
        # same three, and this claims to be the same seam.
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
    ) -> None:
        chosen = str(voice).strip().lower()
        if chosen not in VOICES:
            raise TtsError(
                f"Unknown OpenAI voice {voice!r}. Available: {', '.join(VOICES)}. "
                "(The ChatGPT app's voices — Cove, Juniper, Breeze — are a "
                "different set and are not offered by this API.)"
            )
        chosen_model = str(model).strip()
        if not chosen_model:
            raise TtsError("OpenAI speech model cannot be blank.")
        if (
            chosen_model in _MODELS_WITHOUT_INSTRUCTIONS
            and chosen not in _LEGACY_MODEL_VOICES
        ):
            raise TtsError(
                f"OpenAI model {chosen_model!r} does not support voice {chosen!r}. "
                f"Available for that model: {', '.join(sorted(_LEGACY_MODEL_VOICES))}."
            )
        self._voice = chosen
        self._model = chosen_model
        self._instructions = str(instructions).strip()
        self._api_key = api_key
        self._transport: Transport = transport or urllib_transport
        self._validate_instructions(self._instructions)

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
        """The API default janki deliberately records and does not override.

        OpenAI accepts a separate numeric speed, but janki exposes no sentence-
        speed setting and omits the request field. Tying an OpenAI clip to the
        VOICEVOX word-speed knob would re-bill every sentence when only the word
        pace changed. Learner-specific pace lives in ``instructions``.
        """
        return 1.0

    @property
    def suffix(self) -> str:
        return ".mp3"

    @property
    def settings(self) -> dict[str, str]:
        """What else decides how this clip sounds, for the ledger to compare.

        ``instructions`` is the pace control janki sends and ``model`` changes
        the voice outright, yet neither is in the content fingerprint — which
        covers what was said — nor in ``voice``. Without them recorded, changing
        the instructions left every clip reporting current and nothing re-voiced.
        """
        # Keep the empty key too. That was the pre-M8.1 ledger shape, so an
        # explicitly blank global prompt remains current and does not trigger a
        # collection-wide no-op regeneration merely because clip hints exist.
        return {"model": self._model, "instructions": self._instructions}

    @property
    def instructions(self) -> str:
        return self._instructions

    def for_clip(self, instructions: str) -> OpenAiSpeechProvider:
        """An independent provider carrying one clip's effective prompt."""
        parts = [
            part.strip()
            for part in (self._instructions, str(instructions or ""))
            if part.strip()
        ]
        effective = "\n\n".join(parts)
        self._validate_instructions(effective)
        if effective == self._instructions:
            return self
        return OpenAiSpeechProvider(
            voice=self._voice,
            model=self._model,
            instructions=effective,
            api_key=self._api_key,
            transport=self._transport,
        )

    def _validate_instructions(self, instructions: str) -> None:
        _require_utf8(instructions, field="instructions")
        if instructions and self._model in _MODELS_WITHOUT_INSTRUCTIONS:
            raise TtsError(
                f"OpenAI model {self._model!r} does not support instructions. "
                "Use gpt-4o-mini-tts (or another instruction-capable speech "
                "model), or clear the instructions."
            )
        if len(instructions) > _MAX_INSTRUCTIONS_LENGTH:
            raise TtsError(
                "OpenAI speech instructions are longer than the API's "
                f"{_MAX_INSTRUCTIONS_LENGTH}-character limit."
            )

    def validate_utterance(self, text: str) -> None:
        """Refuse request text the endpoint cannot accept, without I/O."""
        spoken = str(text).strip()
        if not spoken:
            raise TtsError("Nothing to speak: the sentence was empty.")
        _require_utf8(spoken, field="input")
        if len(spoken) > _MAX_INPUT_LENGTH:
            raise TtsError(
                "OpenAI speech input is longer than the API's "
                f"{_MAX_INPUT_LENGTH}-character limit."
            )

    @property
    def launch_hint(self) -> str:
        return KEY_HINT

    def _key(self) -> str:
        key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise TtsError(KEY_HINT[0].upper() + KEY_HINT[1:])
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
        self.validate_utterance(text_or_kana)
        spoken = text_or_kana.strip()
        self._validate_instructions(self._instructions)

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


def _require_utf8(value: str, *, field: str) -> None:
    """Refuse JSON text Python can represent but the HTTP body cannot encode."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TtsError(f"OpenAI speech {field} must be valid UTF-8 text.") from exc


def _detail(payload: bytes) -> str:
    """The API's own message, when it sent one JSON body's worth."""
    try:
        message = json.loads(payload).get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return str(message) or payload[:200].decode("utf-8", errors="replace")
