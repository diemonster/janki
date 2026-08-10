"""Cards for a rule, and cards for practising it.

Two decks, one notetype, because both ask the same shape of question: here is a
trigger, what does it become. A chart states the rules (`build_pattern_deck`);
the collection supplies verbs to run them on (`build_conjugation_deck`).

The rule deck:


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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by the import guard in `build_pattern_deck`
    import genanki
except ImportError:  # pragma: no cover
    genanki = None  # type: ignore[assignment]

from japanese_anki.config import ProjectConfig
from japanese_anki.conjugation import CONJUGATION_FORMS
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import _deck_string_set
from japanese_anki.io import DataError, load_structured
from japanese_anki.models import VocabularyRecord
from japanese_anki.patterns import (
    CHECKABLE_KINDS,
    LIST_SEPARATORS,
    PatternSet,
    check_pattern_rules,
)

__all__ = [
    "PatternCard",
    "PatternDeckError",
    "build_conjugation_deck",
    "build_pattern_deck",
    "cards_for",
    "collection_for",
    "deck_problems",
    "drill_cards",
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
FIELDS: tuple[str, ...] = (
    "Trigger", "Result", "Gloss", "Examples", "Source", "Kind",
)


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
        """What the note's GUID is derived from: the trigger, and nothing else.

        The caller scopes it by document, so the trigger is enough to name a
        rule. Neither the result nor the gloss belongs here — a chart corrected
        from って to んで is the *same card* with a fixed answer, and the gloss
        is model-extracted prose that any re-extraction rewrites and a reviewer
        routinely shortens by hand. Either in the identity means a rebuild
        duplicates the card and strands its review history, which is the failure
        AGENTS.md's deterministic-GUID rule exists to prevent.
        """
        return self.trigger


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
    parts_of = [_tidy(template)]
    # One separator at a time. Testing the whole set at once had no branch for a
    # line that mixes both roles — `う・つ・る → って / く → いて`, which is how a
    # chart cell compresses two rows — and sent it down the *prose* path: one
    # card whose question was the entire line, both answers included, with an
    # empty back, and the second rule never made a card at all.
    for separator in sorted(LIST_SEPARATORS):
        pieces = [
            _tidy(piece) for piece in template.split(separator) if _tidy(piece)
        ]
        if len(pieces) > 1 and all(len(_ARROW.split(piece)) == 2 for piece in pieces):
            parts_of = pieces
            break

    found: list[tuple[str, str]] = []
    for piece in parts_of:
        parts = _ARROW.split(piece)
        if len(parts) == 2 and all(_tidy(part) for part in parts):
            found.append((_tidy(parts[0]), _tidy(parts[1])))
        else:
            found.append((piece, ""))
    return found or [(_tidy(template), "")]


def _tidy(text: str) -> str:
    """Trim whitespace *and* a dangling separator.

    `str.strip()` alone left the `/` on a truncated row like `く → いて /`, so a
    card's answer read "いて /".
    """
    return text.strip().strip("".join(LIST_SEPARATORS)).strip()


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


def _deck_section(deck_path: Path) -> dict[str, Any]:
    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    section = raw.get("deck") or {}
    if not isinstance(section, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    return section


def _identifier(deck_config: dict[str, Any], key: str, deck_path: Path) -> int:
    """A deck or model id, refusing anything that only looks like one.

    YAML 1.1 again: `deck_id: yes` is `True` and `int(True)` is 1 — a deck that
    quietly merges into whatever owns deck 1.
    """
    value = deck_config.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"deck.{key} must be an integer, got {value!r}: {deck_path}")
    return value


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatternDeckError(f"Missing template file: {path}") from exc


def collection_for(deck_path: Path, project_config: ProjectConfig) -> Path:
    """The records file a conjugation deck drills, honouring its own ``source:``.

    A vocabulary deck resolves `source:` against its own directory and both
    shipped ones use it; this path read `config.normalized_file` and ignored the
    key entirely, so a drill deck naming another collection silently drilled the
    wrong one. Missing is reported here rather than as an empty record list,
    which `build_conjugation_deck` could only describe as a missing verb class.
    """
    section = _deck_section(deck_path)
    source = section.get("source")
    if source is None:
        path = project_config.normalized_file.resolve()
    else:
        if not isinstance(source, str):
            raise DataError(
                f"deck.source must be a path, got {type(source).__name__}: {deck_path}"
            )
        path = (deck_path.parent / source).resolve()
    if not path.exists():
        raise PatternDeckError(
            f"{deck_path.name}: no collection at {path}. A conjugation deck "
            f"drills the records janki holds; import some first."
        )
    return path


def deck_problems(
    deck_path: Path, store: Mapping[str, PatternSet] | None = None
) -> list[str]:
    """What is wrong with a pattern or conjugation deck file, for `validate`.

    The build refuses all of these, but `janki validate` is the command whose
    job is catching a broken deck *before* a build — and it read a pattern deck
    as an ordinary one, found no records, and called it clean.
    """
    problems: list[str] = []
    try:
        deck_config = _deck_section(deck_path)
    except DataError as exc:
        return [str(exc)]

    for key in ("deck_id", "model_id"):
        try:
            _identifier(deck_config, key, deck_path)
        except DataError as exc:
            problems.append(str(exc))

    kind = str(deck_config.get("kind") or "").strip().lower()
    if kind == "conjugation":
        form = str(deck_config.get("form") or "te_form").strip()
        if form not in CONJUGATION_FORMS:
            problems.append(
                f"deck.form must be one of {', '.join(CONJUGATION_FORMS)}, "
                f"got {form!r}"
            )
        return problems

    document = str(deck_config.get("document") or "").strip()
    if not document:
        problems.append("a pattern deck needs 'document:' naming a document janki has read")
        return problems
    if store is None:
        return problems
    entry = store.get(document)
    if entry is None:
        known = ", ".join(sorted(store)) or "none"
        problems.append(f"no document has been read under {document!r}. Known: {known}")
    elif entry.kind not in CHECKABLE_KINDS:
        problems.append(f"{document} is a {entry.kind} document, which states no rules")
    elif not entry.reviewed:
        problems.append(f"{document} has not been reviewed")
    return problems


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
    deck_config = _deck_section(deck_path)

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
    # Raised rather than absorbed into the identity. Two rules sharing a trigger
    # inside one document would otherwise collide on a GUID and silently drop a
    # card, and adding prose to tell them apart is what made the GUID unstable.
    seen: dict[str, int] = {}
    for card, *_ in ((card,) for card in cards):
        seen[card.identity] = seen.get(card.identity, 0) + 1
    clashing = sorted(name for name, count in seen.items() if count > 1)
    if clashing:
        raise PatternDeckError(
            f"{document} states more than one rule for {', '.join(clashing)}. "
            f"Each rule needs its own trigger; give them distinct templates in "
            f"data/patterns.json."
        )

    deck_id = _identifier(deck_config, "deck_id", deck_path)
    model_id = _identifier(deck_config, "model_id", deck_path)

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
                    "Rule",
                ],
                guid=genanki.guid_for(f"pattern:{document}:{card.identity}"),
            )
        )

    filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
    target = output_path or (project_config.dist_dir / filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(target))
    return target, len(cards)


def drill_cards(
    records: Sequence[VocabularyRecord], form: str
) -> list[tuple[PatternCard, str]]:
    """One card per verb janki can conjugate, with the answer it computes.

    Returns ``(card, record id)`` pairs so the caller can key a GUID on the
    record rather than on the text, which moves when a reading is corrected.

    **Nothing here is inferred.** The answer comes from
    `conjugation.conjugate`, the same rules every vocabulary card is built with,
    so a drill card and its word card can never disagree. A verb janki declines
    — ゆく, whose て-form is genuinely contested, or a record whose `verb_group`
    is a class name janki does not know — produces no card at all rather than a
    guess: this deck's whole value is that its answers are right.
    """
    from japanese_anki.conjugation import conjugate

    drilled: list[tuple[PatternCard, str]] = []
    for record in records:
        # `verb_group or part_of_speech`, like every other `conjugate` caller:
        # jpdb has no verb class for an い-adjective, so 高い carries its class
        # in `part_of_speech` and `conjugate` resolves that alias. Without the
        # fallback 高い got no drill card while its word card rendered 高くて —
        # the two disagreeing by omission, which this module says cannot happen.
        answer = conjugate(
            record.expression,
            record.reading,
            record.verb_group or record.part_of_speech,
        ).get(form, "")
        if not answer:
            continue
        # The reading rides along on the front when it adds something: a kanji
        # verb cannot be conjugated without it, and hiding it would make the
        # card test the reading instead of the form.
        shown = record.expression
        if record.reading and record.reading != record.expression:
            shown = f"{record.expression}（{record.reading}）"
        drilled.append((
            PatternCard(
                trigger=shown,
                result=answer,
                # One sense. A drill card tests the form, and the meaning is
            # there so you know which word it is — nineteen senses of する is
            # the wall this project already caps on a vocabulary card.
            gloss=record.meanings[0] if record.meanings else "",
                examples=(record.verb_group,) if record.verb_group else (),
            ),
            record.id,
        ))
    return drilled


def build_conjugation_deck(
    deck_path: Path,
    project_config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    output_path: Path | None = None,
) -> tuple[Path, int]:
    """Build one conjugation drill deck. Returns ``(package path, card count)``."""
    if genanki is None:
        raise PatternDeckError(
            "genanki is not installed. Run: python -m pip install -e '.[dev]'"
        )
    deck_config = _deck_section(deck_path)

    form = str(deck_config.get("form") or "te_form").strip()
    if form not in CONJUGATION_FORMS:
        raise PatternDeckError(
            f"{deck_path}: form must be one of {', '.join(CONJUGATION_FORMS)}, "
            f"got {form!r}"
        )

    # The vocabulary path's own reader, so a drill deck cannot disagree with a
    # word deck about what a filter means. Written by hand, `exclude_ids:` as a
    # bare string became a set of single characters and excluded nothing — the
    # record the user held back shipped — while `include_ids: []` filtered
    # everything out and blamed a missing verb_group.
    include = _deck_string_set(deck_config, "include_ids", deck_path)
    exclude = _deck_string_set(deck_config, "exclude_ids", deck_path)
    chosen = [
        record
        for record in records
        if (not include or record.id in include) and record.id not in exclude
    ]
    cards = drill_cards(chosen, form)
    if not cards:
        raise PatternDeckError(
            f"{deck_path}: no record janki can conjugate into a {form}. A verb "
            f"needs a verb_group janki knows — `janki enrich --jpdb` records one."
        )

    deck_id = _identifier(deck_config, "deck_id", deck_path)
    model_id = _identifier(deck_config, "model_id", deck_path)
    model = _notetype(
        model_id,
        str(deck_config.get("model_name") or "Japanese Pattern"),
        project_config.template_dir,
    )
    deck = genanki.Deck(deck_id, str(deck_config.get("name") or deck_path.stem))
    label = form.replace("_", " ")
    for card, record_id in cards:
        deck.add_note(
            genanki.Note(
                model=model,
                fields=[
                    html.escape(card.trigger),
                    html.escape(card.result),
                    html.escape(card.gloss),
                    html.escape(", ".join(card.examples)),
                    html.escape("computed by janki"),
                    html.escape(label),
                ],
                # Keyed on the record and the form, never the text: correcting a
                # reading rewrites the front, and a GUID that moved with it would
                # orphan the card's review history.
                guid=genanki.guid_for(f"drill:{form}:{record_id}"),
            )
        )

    filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
    target = output_path or (project_config.dist_dir / filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(target))
    return target, len(cards)
