"""What janki needs from a speech engine, and nothing else.

Two providers ship: VOICEVOX for single words, where forcing the pitch accent is
the entire point, and Azure for example sentences, where a natural adult voice
matters more. They have nothing in common internally, so the protocol is the
narrow set of things ``janki audio`` actually asks of one.

``name`` and ``voice`` are on the protocol rather than looked up per class
because M5.3 writes them into the ledger for every clip it records, and a
caller that has to ask *which* provider this is in order to describe it has an
``isinstance`` ladder that grows with every provider. Same reasoning for
``launch_hint``: a locally-hosted engine that is not running is the most common
failure a user will hit, and the CLI has to be able to say how to start it
without knowing which engine it is holding.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from japanese_anki.errors import JankiError

__all__ = ["SpeechProvider", "TtsError"]


class TtsError(JankiError):
    """A speech engine refused, or could not be reached."""


@runtime_checkable
class SpeechProvider(Protocol):
    """One speech engine, as ``janki audio`` sees it."""

    @property
    def name(self) -> str:
        """Which engine this is, as the ledger records it."""

    @property
    def voice(self) -> int:
        """Which voice, as the ledger records it.

        An int because both engines identify voices that way — VOICEVOX by
        speaker id, Azure by an index into its configured voice — and the ledger
        stores what was used rather than what it meant.
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
        """WAV bytes for one utterance.

        ``forced_accent`` says which of the two things the caller is holding: an
        AquesTalk kana string whose accent must be honoured exactly (a word), or
        ordinary Japanese to be read naturally (a sentence). It is a flag rather
        than two methods because it is one question — *may the engine decide the
        accent?* — and a provider that cannot force one answers it by ignoring
        the flag, not by lacking the method.
        """
        ...
