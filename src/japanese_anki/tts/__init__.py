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
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from japanese_anki.errors import JankiError

__all__ = [
    "RenderProfile",
    "JournaledSpeechProvider",
    "PaidAttempt",
    "SentenceProfileSelector",
    "SpeechProvider",
    "TtsError",
    "sentence_profile_for",
    "validate_utterance",
]

_Persisted = TypeVar("_Persisted")


class TtsError(JankiError):
    """A speech engine refused, or could not be reached."""


@dataclass(frozen=True, slots=True)
class PaidAttempt:
    """Which billed call produced exactly these audio bytes.

    Minted by the paid writer from the reply it is about to persist, never by
    a caller and never from a reservation: a reservation says a call was
    *authorized*, and the whole point of this record is to say which call
    actually rendered the bytes now on their way to the ledger.

    It exists because a successful journaled operation is *forgotten* — that
    durable forget is how the repository says the money was dealt with — so
    after a clean run the journal holds nothing to point at. Without a witness
    written before that forget, an old reservation for an attempt that
    produced nothing is indistinguishable from one that produced the clip, and
    a later proof would attribute another run's bytes to it.

    ``audio_sha256`` is over the decoded audio the writer is persisting, so the
    binding survives to any reader that can hash the published file. It is the
    field that keeps a witness from travelling onto a replacement render at the
    same request key: the same request may legitimately be voiced twice, and
    the second reply's bytes are not the first attempt's result.
    """

    operation_id: str
    request_fp: str
    model: str
    audio_sha256: str


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
        persist: Callable[[bytes, PaidAttempt], _Persisted],
        before_dispatch: Callable[[str], None] | None = None,
    ) -> _Persisted:
        """Capture, decode, and persist one reply under its operation entry.

        ``before_dispatch`` receives the durable operation id after response
        capture is prepared but before the provider transport may be opened.

        ``persist`` receives the decoded audio together with the
        :class:`PaidAttempt` that produced it, and is called inside the
        operation's commit — before the ``committed`` write and therefore
        before the successful ``forget`` — so the writer's own attribution
        reaches durable storage while the journal entry still exists.
        """
        ...

    def reconcile_journaled(
        self,
        text_or_kana: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
        audio_sha256: str,
        attach: Callable[[PaidAttempt], None] | None = None,
    ) -> None:
        """Settle a captured operation whose exact audio WAL already exists.

        ``attach`` is offered the attempt only once the still-live reply is
        proven to decode to ``audio_sha256``, and always before the entry is
        settled or forgotten: recovery that adopts bytes must be able to make
        their attribution durable while the journal can still supply it. A
        raising ``attach`` leaves the entry and its evidence in place.
        """
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
