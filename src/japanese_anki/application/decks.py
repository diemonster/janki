"""Where a card would end up, asked before it is anywhere.

`WORKBENCH_PLAN.md` W1.1b. The dashboard can say a source is ready to add; the
question a person has next is *which deck will these words be in*, and until
now the only way to find out was to promote them and look.

That cannot be answered by filtering the collection, because a staged record is
not in it yet. So each deck's own rule is asked about the record directly — and
"its own rule" has to mean the function the build actually uses, or the preview
becomes a confident second opinion. Every answer here therefore routes through
an existing authority:

* which files are decks at all — `status.deck_files`, the enumerator
  `build --all` and `validate` use, which walks subdirectories and refuses two
  decks sharing a stem;
* what kind of deck a file is — `anki.deck_kind`, which strips and lowercases
  and refuses a typo outright, because a `kind: patern` that reads as an
  ordinary word deck is how a phantom notetype mismatch got reported once;
* which records a word deck claims — `anki.DeckSelection`, the value
  `resolve_deck_records` filters with;
* which records a drill deck claims — `pattern_cards.shipping_records`, which
  applies the deck's `include_ids`/`exclude_ids` and defaults a missing `form:`
  to `te_form` exactly as the build does.

Asking a simplified version of any of those is not a smaller answer, it is a
wrong one: a drill deck's `exclude_ids` previewed as "yes" is a card someone is
told to expect in a package that will never hold it.

Nothing here writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import (
    deck_kind,
    deck_selection,
    resolve_deck_records,
)
from japanese_anki.io import load_structured
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
    #: Why not, in a learner's words. ``None`` when it takes the card, and also
    #: when `unreadable` is set — an unopened deck has no rule to have failed.
    refusal: str | None
    #: True when a vocabulary deck declares no filter and so claims every
    #: record. Worth surfacing: adding a word to the collection puts it in
    #: this deck whether or not anyone meant it to.
    unfiltered: bool = False
    #: Set when the deck file could not be read or its kind is not one janki
    #: dispatches. Membership is then unknown rather than false — the same
    #: distinction `promote` draws when a deck will not parse and ids cannot
    #: be proved absent.
    unreadable: str = ""


def _display_name(path: Path, fallback: str) -> str:
    raw = load_structured(path)
    section = raw.get("deck") if isinstance(raw, dict) else None
    if isinstance(section, dict):
        return str(section.get("name") or fallback)
    return fallback


def _conjugation(
    path: Path, record: VocabularyRecord
) -> tuple[bool, str | None]:
    """Whether a drill deck would ship this record, and why not.

    The verdict comes from `shipping_records` — the same call the build makes,
    so a deck's `exclude_ids` and its defaulted `form:` are honoured here
    without this module knowing either rule. Only the *sentence* is derived
    separately, because the build has no reason to explain itself.
    """
    takes = bool(pattern_cards.shipping_records(path, [record]))
    if takes:
        return True, None
    section = load_structured(path).get("deck") or {}
    include = {str(item) for item in (section.get("include_ids") or [])}
    exclude = {str(item) for item in (section.get("exclude_ids") or [])}
    if include and record.id not in include:
        return False, "this deck drills an exact list of cards, and this is not one"
    if record.id in exclude:
        return False, "this deck excludes this card by name"
    form = str(section.get("form") or "te_form").strip().replace("_", "-")
    return False, (
        f"janki cannot build the {form} of this word, so the drill deck has "
        "no card to make"
    )


def deck_membership(
    config: ProjectConfig, record: VocabularyRecord
) -> list[DeckMembership]:
    """Every deck, and whether promoting this record would put it there.

    Every deck rather than only the matching ones: "it lands in none of them"
    is an answer a person needs, and it is unreadable as an empty list.
    """
    try:
        paths = status_module.deck_files(config)
    except JankiError:
        # Two decks sharing a stem, or an unreadable deck directory. Nothing
        # can be said about membership when the deck set itself is in question.
        return []

    found: list[DeckMembership] = []
    for path in paths:
        stem = path.stem
        try:
            kind = deck_kind(path) or "vocabulary"
            name = _display_name(path, stem)
        except JankiError as exc:
            # Includes a `kind:` janki does not dispatch. `build` refuses such
            # a file outright, so reporting `takes=False` would quietly claim
            # its rule had been consulted.
            found.append(
                DeckMembership(
                    stem=stem, path=path, name=stem, kind="vocabulary",
                    takes=False, refusal=None, unreadable=str(exc),
                )
            )
            continue

        try:
            if kind == "conjugation":
                takes, refusal = _conjugation(path, record)
                found.append(
                    DeckMembership(
                        stem=stem, path=path, name=name, kind=kind,
                        takes=takes, refusal=refusal,
                    )
                )
                continue
            if kind == "pattern":
                # A pattern deck ships grammar entries from `patterns.json`. It
                # has no vocabulary records to hold, so this is "not
                # applicable" rather than a rule the card failed.
                found.append(
                    DeckMembership(
                        stem=stem, path=path, name=name, kind=kind, takes=False,
                        refusal="this deck holds grammar patterns, not word cards",
                    )
                )
                continue
            if kind == "kanji":
                # Likewise for a character deck: its notes are characters from
                # the curated character store, and a word never lands in one.
                found.append(
                    DeckMembership(
                        stem=stem, path=path, name=name, kind=kind, takes=False,
                        refusal="this deck holds character notes, not word cards",
                    )
                )
                continue
            # Vocabulary only: `resolve_deck_records` validates a *word* deck,
            # and running it over a drill deck refuses shapes that deck builds
            # from perfectly well.
            deck_config, _records = resolve_deck_records(path)
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
        except JankiError as exc:
            found.append(
                DeckMembership(
                    stem=stem, path=path, name=name, kind=kind,
                    takes=False, refusal=None, unreadable=str(exc),
                )
            )
    return found
