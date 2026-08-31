"""What janki needs from a speech engine, and nothing else.

Two providers ship: VOICEVOX for single words, where forcing the pitch accent is
the entire point, and OpenAI for example sentences, where a natural adult voice
matters more. They have nothing in common internally, so the protocols expose
only the narrow sets of facts their callers actually need.

``name`` and ``voice`` are on the protocol rather than looked up per class
because M5.3 writes them into the ledger for every clip it records, and a
caller that has to ask *which* provider this is in order to describe it has an
``isinstance`` ladder that grows with every provider. Same reasoning for
``launch_hint``: a locally-hosted engine that is not running is the most common
failure a user will hit, and the CLI has to be able to say how to start it
without knowing which engine it is holding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

from japanese_anki.errors import JankiError

__all__ = [
    "RenderProfile",
    "JournaledSpeechProvider",
    "SentenceProfileSelector",
    "SpeechProvider",
    "TtsError",
    "sentence_profile_for",
    "validate_utterance",
]

_Persisted = TypeVar("_Persisted")


class TtsError(JankiError):
    """A speech engine refused, or could not be reached."""


@runtime_checkable
class RenderProfile(Protocol):
    """The durable synthesis choices that decide how a clip sounds."""

    @property
    def name(self) -> str:
        """Which engine this is, as the ledger records it."""

    @property
    def voice(self) -> int | str:
        """Which voice, as the ledger records it.

        Whatever the engine calls it: VOICEVOX numbers its speakers, OpenAI
        names them. The ledger stores what was used rather than what it meant,
        so ``onyx`` is stored as ``onyx`` — mapping it onto an index would put
        a number in a committed file that means nothing outside janki and would
        silently change meaning the day the voice list grows.
        """

    @property
    def speed(self) -> float:
        """How fast, as the ledger records it. 1.0 is the engine's own pace.

        Beside ``voice`` for the same reason, and recorded for a sharper one:
        rate is audible and is not part of a clip's content fingerprint, so
        without it in the ledger a re-voice that stops half way leaves a
        collection speaking at two speeds that nothing can detect afterwards.
        """

    @property
    def settings(self) -> dict[str, str]:
        """Anything else that decides how a clip sounds, for the ledger.

        Empty for an engine whose voice and rate say it all. An engine with a
        prose style prompt, or a choice of model, puts them here: they are
        audible, they are not in the content fingerprint, and a clip whose
        ledger entry cannot describe them is one nothing can tell is stale.
        """


@runtime_checkable
class SpeechProvider(RenderProfile, Protocol):
    """One speech engine, as ``janki audio`` sees it."""

    @property
    def suffix(self) -> str:
        """The file extension this engine's audio needs, including the dot.

        On the provider because the format is the engine's choice, not the
        caller's. VOICEVOX returns WAV; Realtime returns raw PCM that its
        provider wraps as a finite WAV. A future encoded stream may choose a
        different suffix without downstream code lying about its contents.
        """

    @property
    def launch_hint(self) -> str:
        """What to tell someone whose engine is not answering."""

    def available(self) -> bool:
        """Is the engine reachable right now?

        Never raises: "not running" is the ordinary state of a local engine, not
        an error, and the caller decides whether to skip or complain.
        """
        ...

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        """Audio bytes for one utterance.

        ``forced_accent`` says which of the two things the caller is holding: an
        AquesTalk kana string whose accent must be honoured exactly (a word), or
        ordinary Japanese to be read naturally (a sentence). It is a flag rather
        than two methods because it is one question — *may the engine decide the
        accent?* — and a provider that cannot force one refuses the flag.
        """
        ...


@runtime_checkable
class SentenceProfileSelector(RenderProfile, Protocol):
    """Choose one concrete sentence renderer from a stable record identity."""

    def profile_for(self, record_id: str) -> SpeechProvider:
        """Return the exact profile every example on ``record_id`` will use."""
        ...


@runtime_checkable
class JournaledSpeechProvider(SpeechProvider, Protocol):
    """A paid stream whose exact reply must become durable before decoding."""

    def synthesize_journaled(
        self,
        text_or_kana: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
        persist: Callable[[bytes], _Persisted],
    ) -> _Persisted:
        """Capture, decode, and persist one reply under its operation entry."""
        ...

    def reconcile_journaled(
        self,
        text_or_kana: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
        audio_sha256: str,
    ) -> None:
        """Settle a captured operation whose exact audio WAL already exists."""
        ...


def sentence_profile_for(
    provider: SpeechProvider | SentenceProfileSelector,
    record_id: str,
) -> SpeechProvider:
    """Resolve a selector once per record; fixed providers pass through."""
    select = getattr(provider, "profile_for", None)
    if not callable(select):
        return provider
    selected = select(record_id)
    if not isinstance(selected, SpeechProvider):
        raise TtsError(
            f"Sentence profile selector returned no speech provider for {record_id!r}."
        )
    return selected


def validate_utterance(provider: RenderProfile, text: str) -> None:
    """Run a provider's side-effect-free per-utterance request checks.

    Most engines have no local text constraint. OpenAI does, and checking every
    selected sentence before synthesis keeps a deterministic length refusal
    from landing earlier paid clips in the same run.
    """
    validate = getattr(provider, "validate_utterance", None)
    if callable(validate):
        validate(text)
