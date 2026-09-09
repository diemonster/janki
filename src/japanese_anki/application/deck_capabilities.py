"""What each deck kind owns, synthesizes and packages — stated once.

Every media decision used to be a separate ``if kind == …`` at the point of
use: the audio census, the deck-scoped clip scope, the package planner.  Each
had to be found and edited when a kind was added, and a kind nobody edited for
simply fell through as an ordinary word deck.  This module is the one fixed
table those callers ask instead.  There is no plugin framework and no way to
register a row at runtime: the kinds are :data:`anki.KNOWN_DECK_KINDS`, and a
kind that reaches this table without a row is **refused**.

That refusal matters more than it looks.  The tempting default — treat an
unrecognized kind as owning nothing — reads as "no media to worry about" and
means the opposite: an empty owner set makes every clip the deck keeps alive
look unreferenced, so the next prune deletes bytes somebody paid for.  So a
missing row stops the census, the publication, the packaging and the prune
before any of them runs, rather than answering them with a guess.

Durable ownership and synthesis/packaging are deliberately separate columns.
A ``pattern`` deck parses its ``source:`` as vocabulary — so a record version
it declares still keeps that record's media alive — while synthesizing and
packaging no media of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import anki, pattern_cards
from japanese_anki.io import RecordsRevision, load_structured
from japanese_anki.models import VocabularyRecord

__all__ = [
    "DeckCapabilityError",
    "DeckMediaCapability",
    "capability",
    "deck_capability",
    "declared_media_owners",
    "drill_media_owners",
    "durable_media_owners",
]


class DeckCapabilityError(JankiError):
    """A deck kind has no media capability, so nothing may act on its media."""


@dataclass(frozen=True, slots=True)
class DeckMediaCapability:
    """One deck kind's complete media contract.

    ``parses_source_as_vocabulary`` is about the deck's ``source:``: a
    ``kanji`` deck's source is the curated character store, whose notes are
    ``kanji:理`` identities rather than vocabulary records, so resolving it as
    a word list answers a question nobody asked and refuses every census in a
    repository that studies characters.
    """

    kind: str
    parses_source_as_vocabulary: bool
    owns_drill_examples: bool
    synthesizes_word_audio: bool
    synthesizes_example_audio: bool
    packages_media: bool


_TABLE: dict[str, DeckMediaCapability] = {
    entry.kind: entry
    for entry in (
        # `""` is the ordinary word deck; `vocabulary` is the same deck for
        # anyone who prefers writing the kind down, and the two rows agree.
        DeckMediaCapability(
            kind="",
            parses_source_as_vocabulary=True,
            owns_drill_examples=False,
            synthesizes_word_audio=True,
            synthesizes_example_audio=True,
            packages_media=True,
        ),
        DeckMediaCapability(
            kind="vocabulary",
            parses_source_as_vocabulary=True,
            owns_drill_examples=False,
            synthesizes_word_audio=True,
            synthesizes_example_audio=True,
            packages_media=True,
        ),
        # A rule deck writes its own small notetype from the reviewed pattern
        # store. It voices nothing and packages no media — but a record
        # version it declares is still durable YAML naming a media file.
        DeckMediaCapability(
            kind="pattern",
            parses_source_as_vocabulary=True,
            owns_drill_examples=False,
            synthesizes_word_audio=False,
            synthesizes_example_audio=False,
            packages_media=False,
        ),
        # Lesson drills own authored polite/casual example sentences under
        # synthetic `drill-audio:` owners. Their words are the canonical
        # vocabulary records', not the deck's.
        DeckMediaCapability(
            kind="conjugation",
            parses_source_as_vocabulary=True,
            owns_drill_examples=True,
            synthesizes_word_audio=False,
            synthesizes_example_audio=True,
            packages_media=True,
        ),
        # Dictionary-only by design: a bare character has no audio, and
        # `kanji_cards` reports `media_count=0`.
        DeckMediaCapability(
            kind="kanji",
            parses_source_as_vocabulary=False,
            owns_drill_examples=False,
            synthesizes_word_audio=False,
            synthesizes_example_audio=False,
            packages_media=False,
        ),
    )
}


def capability(kind: str) -> DeckMediaCapability:
    """The fixed row for one deck kind, or a refusal.

    ``kind`` is normalized exactly as :func:`anki.deck_kind` normalizes what it
    reads out of a deck file, so the two can never disagree about which string
    names which kind.
    """
    normalized = str(kind).strip().lower()
    entry = _TABLE.get(normalized)
    if entry is not None:
        return entry
    if normalized in anki.KNOWN_DECK_KINDS:
        raise DeckCapabilityError(
            f"Deck kind {normalized!r} has no media capability row. Nothing may "
            "census, publish, package or prune its media until "
            "application/deck_capabilities.py states what it owns — answering "
            "with an empty owner set would make its clips look unreferenced."
        )
    # The same refusal `deck_kind` gives, for a caller holding a kind string
    # rather than a deck file.
    raise DeckCapabilityError(
        f"Unknown deck kind {normalized!r}. Valid kinds: "
        + ", ".join(
            k or "(omitted — an ordinary word deck)" for k in anki.KNOWN_DECK_KINDS
        )
    )


def deck_capability(
    deck_path: Path,
    *,
    revision: RecordsRevision | None = None,
) -> DeckMediaCapability:
    """One deck's media contract, classified from its own bytes.

    A supplied ``revision`` is classified by the bytes it carries rather than
    by whatever the file on disk currently says, so a projection of a proposed
    deck answers for the deck it proposes.
    """
    kind = (
        anki.deck_kind(deck_path)
        if revision is None
        else anki.deck_kind_from_revision(deck_path, revision)
    )
    return capability(kind)


def declared_media_owners(
    deck_path: Path,
    *,
    revision: RecordsRevision | None = None,
) -> list[VocabularyRecord]:
    """Every persisted record version this deck file itself can keep alive."""
    if not deck_capability(deck_path, revision=revision).parses_source_as_vocabulary:
        return []
    if revision is not None:
        return anki.deck_declared_record_versions_from_revision(deck_path, revision)
    return anki.deck_declared_record_versions(deck_path)


def drill_media_owners(
    config: ProjectConfig,
    deck_path: Path,
    *,
    revision: RecordsRevision | None = None,
) -> list[VocabularyRecord]:
    """This deck's synthetic audio owners, if its kind owns drill examples."""
    if not deck_capability(deck_path, revision=revision).owns_drill_examples:
        return []
    if revision is not None:
        return pattern_cards.drill_audio_records_from_revision(
            deck_path, config, revision
        )
    raw = load_structured(deck_path)
    section = raw.get("deck") or {} if isinstance(raw, Mapping) else {}
    if not isinstance(section, Mapping) or section.get("drill_examples") is None:
        return []
    return pattern_cards.drill_audio_records(deck_path, config)


def durable_media_owners(
    config: ProjectConfig,
    deck_path: Path,
    *,
    revision: RecordsRevision | None = None,
) -> list[VocabularyRecord]:
    """Every record-shaped media owner one deck keeps alive.

    The corpus-level census in :mod:`japanese_anki.application.audio` consumes
    the two halves separately rather than calling this: its ``drill-audio:``
    collision refusal has to know every deck's *declared* ids before it
    examines any deck's drill owners, or a synthetic owner colliding with a
    record version declared by a later deck would go unnoticed.
    """
    return [
        *declared_media_owners(deck_path, revision=revision),
        *drill_media_owners(config, deck_path, revision=revision),
    ]
