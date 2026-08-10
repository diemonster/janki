"""Cards for a rule, from a document that teaches one.

A conjugation chart is a deck in itself. `janki patterns` already reads one and
checks its worked examples against `conjugation.conjugate`; this turns the rules
it holds into cards, and it ships **only what that check passed**.

That last part is the whole design. Everything on these cards is either
transcribed from the document by a model — which this project never trusts on
its own — or computed by janki. A rule whose worked example janki disagrees
with is a rule janki has reason to think was mis-transcribed, and putting it on
a card would drill the error. A rule with nothing checkable at all (``く → いて``
names an ending, not a verb) still ships: there is nothing to disagree with, and
the rule is the thing being taught.

**Its own notetype, not new fields on the vocabulary one.** A pattern card has
no reading, no pitch, no audio; it shares nothing with a word card but the deck
it may sit beside. Measured against `anki` 26.8.1: adding a notetype leaves the
collection's ``scm`` mark alone, while appending a field bumps it — so this
costs no forced one-directional AnkiWeb sync, which a new field on the existing
notetype would.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by the import guard in `build_pattern_deck`
    import genanki
except ImportError:  # pragma: no cover
    genanki = None  # type: ignore[assignment]

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError, load_structured
from japanese_anki.patterns import (
    CHECKABLE_KINDS,
    LIST_SEPARATORS,
    PatternSet,
    check_pattern_rules,
)

__all__ = [
    "PatternCard",
    "PatternDeckError",
    "build_pattern_deck",
    "cards_for",
]


class PatternDeckError(JankiError):
    """A pattern deck could not be built."""


#: The arrows a chart writes a rule with.
_ARROW = re.compile(r"\s*(?:⇨|→|->|=>)\s*")

#: Every separator a chart puts between items, from `patterns` so the two cannot
#: drift. Whether a given one divides *rules* or lists the triggers of one rule
#: is decided per template — see `_split_rules`.
_SEPARATOR = re.compile(
    f"\\s*[{re.escape(''.join(LIST_SEPARATORS))}]\\s*"
)

#: Field order. Appended-only, like the vocabulary notetype's, for the same
#: reason: a note's values are positional.
FIELDS: tuple[str, ...] = ("Trigger", "Result", "Gloss", "Examples", "Source")


@dataclass(frozen=True, slots=True)
class PatternCard:
    """One rule, as a card asks it."""

    #: What the learner is shown: ``う・つ・る``, ``くる``.
    trigger: str
    #: The answer: ``って``, ``きて``. Empty when the rule has no arrow, in which
    #: case the gloss carries it and the card is a plain statement.
    result: str
    gloss: str = ""
    #: ``verb ⇨ form`` pairs janki checked and agreed with. Never the document's
    #: unverified ones.
    examples: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        """What the note's GUID is derived from.

        The trigger and the source document, not the result: a chart corrected
        from って to んで is the *same card* with a fixed answer, and a GUID that
        moved would orphan its review history.
        """
        return f"{self.trigger}\x1f{self.gloss}"


def cards_for(
    entry: PatternSet, groups: Mapping[str, str] | None = None
) -> list[PatternCard]:
    """The cards a document's rules make, with only verified examples.

    One card per rule, and a rule stating several — ``くる → きて / する → して``
    — becomes one card each, which is how they are drilled anyway.

    ``groups`` is the verb classes to check the worked examples against, from
    the collection. Without it every example is held back — `check_pattern_rules`
    will not guess a class — and the cards ship with none, which is correct but
    thin: the rule is still the thing being taught.
    """
    if entry.kind not in CHECKABLE_KINDS:
        raise PatternDeckError(
            f"{entry.source} is a {entry.kind} document, and only "
            f"{'/'.join(CHECKABLE_KINDS)} documents state rules to make cards "
            f"from. A lesson deck's grammar steers `enrich --ai` instead."
        )
    if not entry.reviewed:
        raise PatternDeckError(
            f"{entry.source} has not been reviewed. Its rules are a model's "
            f"reading of a page, and nothing here ships unread: "
            f"janki patterns --review {entry.source!r}"
        )

    agreed: dict[str, list[tuple[str, str]]] = {}
    for check in check_pattern_rules(entry, groups):
        if check.agrees:
            agreed.setdefault(check.template, []).append(
                (check.verb, f"{check.verb} ⇨ {check.claimed}")
            )

    cards: list[PatternCard] = []
    for pattern in entry.patterns:
        checked = agreed.get(pattern.template, [])
        for trigger, result in _split_rules(pattern.template):
            # A row stating several rules — `くる → きて / する → して` — has
            # worked examples for each, and putting both on both cards asks
            # about くる while showing する. Where the trigger *is* the example's
            # verb, keep only its own; where it is an ending (`う・つ・る`) no
            # example names it, so the row's whole set belongs to the card.
            mine = [text for verb, text in checked if verb == trigger]
            examples = tuple(mine) if mine else tuple(
                text for verb, text in checked
                if not any(verb == other for other, _ in
                           [(t, r) for t, r in _split_rules(pattern.template)])
            )
            cards.append(
                PatternCard(
                    trigger=trigger,
                    result=result,
                    gloss=pattern.gloss,
                    examples=examples,
                )
            )
    return cards


def _split_rules(template: str) -> list[tuple[str, str]]:
    """``A → B / C → D`` into two pairs, ``う/つ/る → って`` into one.

    The same character does both jobs. `patterns.INSTRUCTIONS` asks the model to
    write a rule's triggers as ``う/つ/る → って``, and a chart also puts two
    whole rules on one line as ``くる → きて / する → して`` — so a separator
    divides *rules* only when every piece it produces carries an arrow of its
    own. Splitting unconditionally turned the first into three cards, one of
    them drilling ``る → って``, which is the ichidan ending and takes て. A card
    teaching an error is the thing this module exists not to ship.

    A template with no arrow is a rule stated in prose rather than as a
    transformation, and becomes a single card with no answer half — the gloss is
    the answer. Guessing where to cut it would invent a question the document
    does not ask.
    """
    pieces = [piece.strip() for piece in _SEPARATOR.split(template) if piece.strip()]
    if len(pieces) > 1 and all(_ARROW.search(piece) for piece in pieces):
        parts_of = pieces
    else:
        parts_of = [template.strip()]

    found: list[tuple[str, str]] = []
    for piece in parts_of:
        parts = _ARROW.split(piece)
        if len(parts) == 2 and all(part.strip() for part in parts):
            found.append((parts[0].strip(), parts[1].strip()))
        else:
            found.append((piece, ""))
    return found or [(template.strip(), "")]


def _notetype(model_id: int, model_name: str, template_dir: Path) -> Any:
    front = _read(template_dir / "pattern-front.html")
    back = _read(template_dir / "pattern-back.html")
    return genanki.Model(
        model_id,
        model_name,
        fields=[{"name": name} for name in FIELDS],
        templates=[{"name": "Rule", "qfmt": front, "afmt": back}],
        css=_read(template_dir / "style.css"),
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatternDeckError(f"Missing template file: {path}") from exc


def build_pattern_deck(
    deck_path: Path,
    project_config: ProjectConfig,
    store: dict[str, PatternSet],
    output_path: Path | None = None,
    groups: Mapping[str, str] | None = None,
) -> tuple[Path, int]:
    """Build one pattern deck. Returns ``(package path, card count)``."""
    if genanki is None:
        raise PatternDeckError(
            "genanki is not installed. Run: python -m pip install -e '.[dev]'"
        )
    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    deck_config = raw.get("deck") or {}
    if not isinstance(deck_config, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")

    document = str(deck_config.get("document") or "").strip()
    if not document:
        raise PatternDeckError(
            f"{deck_path}: a pattern deck needs 'document:' naming a document "
            f"`janki patterns` has read."
        )
    entry = store.get(document)
    if entry is None:
        known = ", ".join(sorted(store)) or "none"
        raise PatternDeckError(
            f"{deck_path}: no document has been read under {document!r}. "
            f"Known: {known}"
        )

    cards = cards_for(entry, groups)
    if not cards:
        raise PatternDeckError(f"{document} states no rules to make cards from.")

    deck_id = deck_config.get("deck_id")
    if deck_id is None or isinstance(deck_id, bool) or not isinstance(deck_id, int):
        raise DataError(
            f"deck.deck_id must be an integer, got {deck_id!r}: {deck_path}"
        )
    model_id = deck_config.get("model_id")
    if model_id is None or isinstance(model_id, bool) or not isinstance(model_id, int):
        raise DataError(
            f"deck.model_id must be an integer, got {model_id!r}: {deck_path}"
        )

    model = _notetype(
        model_id,
        str(deck_config.get("model_name") or "Japanese Pattern"),
        project_config.template_dir,
    )
    deck = genanki.Deck(deck_id, str(deck_config.get("name") or deck_path.stem))
    for card in cards:
        deck.add_note(
            genanki.Note(
                model=model,
                fields=[
                    html.escape(card.trigger),
                    html.escape(card.result),
                    html.escape(card.gloss),
                    "<br>".join(html.escape(item) for item in card.examples),
                    html.escape(document),
                ],
                guid=genanki.guid_for(f"pattern:{document}:{card.identity}"),
            )
        )

    filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
    target = output_path or (project_config.dist_dir / filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(target))
    return target, len(cards)
