"""Where a card would end up, asked before it is anywhere.

`WORKBENCH_PLAN.md` W1.1b. The dashboard can say a source is ready to add; the
question a person actually has next is *which deck will these words be in*, and
until now the only way to find out was to promote them and look.

That question cannot be answered by filtering the collection, because a staged
record is not in it yet. So this evaluates each deck's own declared rule
against the record directly — through `exporters.anki.DeckSelection`, the same
value `resolve_deck_records` filters with, so a preview cannot drift from the
build it predicts.

**Conjugation decks are included, and that is the point.** `teform-drill`
selects every record it can build a te-form for — no tags, no ids — and puts
a meaning on every card. A membership answer that only understood tag rules
would have called it "no deck", which is exactly the blind spot that let 14
corrected glosses sit in a drill package nobody thought to rebuild. Its rule is
different, so it is asked differently, but it is asked.

Nothing here writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import deck_selection, resolve_deck_records
from japanese_anki.models import VocabularyRecord

__all__ = ["DeckMembership", "deck_membership"]


@dataclass(frozen=True, slots=True)
class DeckMembership:
    """One deck, and whether it would ship this card."""

    stem: str
    path: Path
    #: The deck's display name, which is what a person recognizes — the file
    #: stem is janki's handle for it and means nothing in Anki.
    name: str
    #: ``vocabulary``, ``conjugation`` or ``pattern``.
    kind: str
    takes: bool
    #: Why not, in a learner's words. ``None`` when it does take the card.
    refusal: str | None
    #: True when a vocabulary deck declares no filter and so claims every
    #: record. Worth surfacing: adding a word to the collection puts it in
    #: this deck whether or not anyone meant it to.
    unfiltered: bool = False
    #: Set when the deck file could not be read. Membership is then unknown
    #: rather than false — the same distinction `promote` draws when a deck
    #: will not parse and ids cannot be proved absent.
    unreadable: str = ""


def _deck_files(config: ProjectConfig) -> list[Path]:
    if not config.deck_dir.is_dir():
        return []
    return sorted(
        path
        for path in config.deck_dir.iterdir()
        if path.suffix.lower() in (".yaml", ".yml") and not path.name.startswith(".")
    )


def _conjugation_takes(
    deck_config: dict[str, Any], record: VocabularyRecord
) -> tuple[bool, str | None]:
    """Whether a drill deck can build its form for this record.

    Membership is what `drill_cards` can conjugate, not what part of speech the
    record claims — the deck file says so itself, and a record carrying
    ``verb_group: suru`` still produces nothing unless its expression ends in
    する. Asking the real builder is the only answer that stays true.
    """
    form = str(deck_config.get("form") or "")
    if not form:
        return False, "this drill deck names no form to conjugate into"
    if pattern_cards.drill_cards([record], form):
        return True, None
    return False, (
        f"janki cannot build the {form.replace('_', '-')} of this word, so the "
        "drill deck has no card to make"
    )


def deck_membership(
    config: ProjectConfig, record: VocabularyRecord
) -> list[DeckMembership]:
    """Every deck, and whether promoting this record would put it there.

    Every deck rather than only the matching ones: "it lands in none of them"
    is an answer a person needs, and it is unreadable as an empty list.
    """
    found: list[DeckMembership] = []
    for path in _deck_files(config):
        stem = path.stem
        try:
            deck_config, _records = resolve_deck_records(path)
        except JankiError as exc:
            found.append(
                DeckMembership(
                    stem=stem,
                    path=path,
                    name=stem,
                    kind="vocabulary",
                    takes=False,
                    refusal=None,
                    unreadable=str(exc),
                )
            )
            continue
        kind = str(deck_config.get("kind") or "vocabulary")
        name = str(deck_config.get("name") or stem)
        if kind == "conjugation":
            takes, refusal = _conjugation_takes(deck_config, record)
            found.append(
                DeckMembership(
                    stem=stem, path=path, name=name, kind=kind,
                    takes=takes, refusal=refusal,
                )
            )
            continue
        if kind == "pattern":
            # A pattern deck ships grammar entries from `patterns.json`. It has
            # no vocabulary records to hold, so this is "not applicable" rather
            # than a rule the card failed.
            found.append(
                DeckMembership(
                    stem=stem, path=path, name=name, kind=kind, takes=False,
                    refusal="this deck holds grammar patterns, not word cards",
                )
            )
            continue
        selection = deck_selection(deck_config, path)
        found.append(
            DeckMembership(
                stem=stem,
                path=path,
                name=name,
                kind=kind,
                takes=selection.includes(record),
                refusal=selection.refusal(record),
                unfiltered=selection.takes_everything,
            )
        )
    return found
