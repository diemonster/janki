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
from japanese_anki.identifiers import normalize_identity_part
from japanese_anki.io import DataError, load_records, load_structured
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
    "shipping_records",
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
_SEPARATOR_CHARS = "".join(sorted(LIST_SEPARATORS))
_SEPARATOR_CLASS = f"[{re.escape(_SEPARATOR_CHARS)}]"

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
        rules = _split_rules(pattern.template)
        # Matched on the trigger without its parenthetical, because
        # `check_pattern_rules` reads the same row with parentheticals removed:
        # `check.verb` for `かう (exception) → かって` is the bare かう. Compared
        # against the displayed trigger, the example matched no card, and the
        # fallback below — which keeps what belongs to no *other* rule — then
        # put it on every card on the row.
        bare = [_unannotated(trigger) for trigger, _, _ in rules]
        for (trigger, result, annotation), verb_of in zip(rules, bare, strict=True):
            # A row stating several rules — `くる → きて / する → して` — has
            # worked examples for each, and putting both on both cards asks
            # about くる while showing する. Where the trigger *is* the example's
            # verb, keep only its own; where it is an ending (`う・つ・る`) no
            # example names it, so the row's whole set belongs to the card.
            mine = [text for verb, text in checked if verb == verb_of]
            examples = tuple(mine) if mine else tuple(
                text for verb, text in checked if verb not in bare
            )
            cards.append(
                PatternCard(
                    trigger=trigger,
                    result=result,
                    # The annotation the answer gave up, beside the model's own
                    # gloss rather than dropped: `(voiced → で)` is a rule
                    # explanation transcribed from the page, and a rebuild would
                    # otherwise overwrite it in a collection that has it.
                    gloss=" ".join(part for part in (pattern.gloss, annotation) if part),
                    examples=examples,
                )
            )
    return cards


#: Stands in for a parenthetical while the line's shape is read. One character
#: per character, so every offset still points at the same place in the template
#: — the text itself is never rewritten, only made invisible to the parser.
_MASK = "\uf8ff"


#: Separators that join the items of one statement and never divide two. The
#: nakaguro is a joiner in Japanese typography, which is why `う・つ・る` is a
#: trigger list and never two rules — and taking it as a divider cut
#: `くる → きて・きた / する → して` after きて, losing きた and asking `きた / する`.
#:
#: Anything not named here is treated as a possible divider, so a separator
#: added to `LIST_SEPARATORS` needs no edit here and cannot fail on a chart row.
_JOINERS = frozenset("・･")


def _masked(template: str) -> str:
    """The template with parentheticals blanked out, character for character.

    `ぐ → いで (voiced → で)` carries an arrow inside a gloss, and counting it
    made the whole line look like a multi-rule row.
    """
    return re.sub(
        r"[(（][^)）]*[)）]", lambda m: _MASK * len(m.group(0)), template
    )


def _split_rules(template: str) -> list[tuple[str, str, str]]:
    """``A → B / C → D`` into two pairs, ``う/つ/る → って`` into one.

    The same characters do both jobs: `patterns.INSTRUCTIONS` asks for a rule's
    triggers as ``う/つ/る → って``, and a chart also puts two whole rules on one
    line as ``くる → きて / する → して`` — sometimes with different separators
    for each job, since the model is told to write it the way the page does.

    The arrows say how many rules the line states; the question is only where to
    cut. So the divider is *found* rather than assumed: a separator that divides
    rules cuts the line into exactly that many pieces with exactly one arrow
    each, and a separator that lists triggers or results does not. On the
    canonical row

        う・つ・る → って / む・ぶ・ぬ → んで / く → いて / ぐ → いで / す → して

    `/` cuts five one-arrow pieces and `・` cuts fifteen pieces, most with no
    arrow at all — so `/` divides and `・` lists, without either being named in
    advance. Two dividers on one line (``… / … 、 …``) are found by cutting on
    every separator at once, which is the same test applied to their union.

    More than one cut can fit by coincidence: in ``くる → きて・きた / する → して``
    both `・` and `/` yield two one-arrow pieces, and cutting at `・` drops きた
    from the first answer and asks ``きた / する``. They are told apart by how the
    characters are used rather than by where they appear — `・` joins the items
    of a list and never divides two statements, a solidus divides, and the
    commas fall in between.

    When no cut fits at all, the line uses one character for both jobs.
    Arrow-less pieces before the first rule are its trigger list and are rejoined
    to it — ``う/つ/る → って / く → いて``. Arrow-less pieces *after* a rule are
    genuinely undecidable, since ``って/った / く → いて`` reads equally as a
    two-item result list or as the next rule's trigger, and the line ships as one
    prose card rather than as a rule teaching the wrong trigger — which is also
    the note's GUID.

    A template with no arrow is a rule stated in prose and becomes a single card
    with no answer half; the gloss is the answer. Guessing where to cut it would
    invent a question the document does not ask.

    Throughout, a parenthetical is invisible to the parser and untouched in the
    output: it is masked to read the line's shape, and every card is sliced out
    of the original text. Cutting it away instead emptied `（〜てもいい）` into a
    blank card with the guid `pattern:<document>:`, and shortened
    `行く (exception)` — a trigger that is also the GUID, so the next build of a
    shipped deck adds a second note and strands the first one's review history.
    """
    masked = _masked(template)
    arrows = len(_ARROW.findall(masked))
    if not arrows:
        return [(_tidy(template), "", "")]
    if arrows == 1:
        return [_rule_pair(template, masked)]
    spans = _rule_spans(masked, arrows)
    if spans is None:
        return [(_tidy(template), "", "")]
    return [_rule_pair(template[a:b], masked[a:b]) for a, b in spans]


def _cut_at(masked: str, cuts: Sequence[int]) -> list[tuple[int, int]]:
    """The spans between the given separator positions, blank ones dropped."""
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in (*cuts, len(masked)):
        # All whitespace, not the ASCII space alone: U+3000 is ordinary in
        # Japanese source text, and a trailing `/　` left a span that fit
        # nothing, so a two-rule line shipped as one prose card.
        if masked[start:cut].strip().strip(_SEPARATOR_CHARS).strip():
            spans.append((start, cut))
        start = cut + 1
    return spans


def _fitting_cut(
    masked: str, chars: str, arrows: int
) -> list[tuple[int, int]] | None:
    """The spans this cut makes, if it makes one rule out of each."""
    spans = _cut_at(masked, [i for i, char in enumerate(masked) if char in chars])
    if len(spans) == arrows and all(
        len(_ARROW.findall(masked[a:b])) == 1 for a, b in spans
    ):
        return spans
    return None


def _stray_dividers(masked: str, spans: Sequence[tuple[int, int]]) -> int:
    """Dividers left inside the trigger halves of a cut.

    Joiners are not counted: `う・つ → って` is one rule with a trigger list, and
    the nakaguro there says nothing about where the line divides.
    """
    strays = 0
    for a, b in spans:
        arrow = _ARROW.search(masked[a:b])
        trigger = masked[a:b][: arrow.start()] if arrow else masked[a:b]
        strays += sum(
            1
            for char in trigger
            if char in _SEPARATOR_CHARS and char not in _JOINERS
        )
    return strays


def _rule_spans(masked: str, arrows: int) -> list[tuple[int, int]] | None:
    """One span per rule, or ``None`` when the line cannot be cut safely."""
    # Each separator the line uses, in the order it first appears, plus their
    # union — so the choice never depends on a set's iteration order.
    candidates: list[str] = []
    for char in masked:
        if char in _SEPARATOR_CHARS and char not in _JOINERS and char not in candidates:
            candidates.append(char)

    # Every candidate is tried, not just the first that fits: two can fit the
    # same line by coincidence of counts.
    fitting = [
        spans
        for candidate in candidates
        if (spans := _fitting_cut(masked, candidate, arrows))
    ]
    if fitting:
        agreed = {tuple(spans) for spans in fitting}
        if len(agreed) == 1:
            return fitting[0]
        # Two dividers that fit and disagree. No fixed order of the characters
        # can settle it — `/` lists triggers in the shape
        # `patterns.INSTRUCTIONS` asks for *and* divides rules, and so does each
        # comma — but the line still says which cut is real: a wrong one leaves
        # a divider it did not cut on inside a trigger. In
        # `する → して, した / くる → きて` cutting at `,` asks `した / くる`, while
        # `/` leaves `する` and `くる`; the mirror row resolves the same way.
        # Joiners are not counted, since `う・つ` is an ordinary trigger list.
        ranked = sorted(fitting, key=lambda spans: _stray_dividers(masked, spans))
        if _stray_dividers(masked, ranked[0]) < _stray_dividers(masked, ranked[1]):
            return ranked[0]
        # Nothing left to tell them apart. One prose card carrying the whole
        # line, as for `う/つ/る → って/った / く → いて` — the trigger is the
        # note's GUID, and a guessed one is worse than a rule left undrilled.
        return None
    # Two dividers on one line (`… / … 、 …`): neither cuts the line alone, and
    # their union is the same test over both.
    union = _fitting_cut(masked, _SEPARATOR_CHARS, arrows)
    if union:
        return union

    # One character doing both jobs. Arrow-less pieces before the first rule are
    # its trigger list; one after a rule has ended could be that rule's result
    # list or the next rule's triggers, and nothing in the line says which.
    spans = []
    cuts = [i for i, char in enumerate(masked) if char in _SEPARATOR_CHARS]
    start: int | None = None
    for a, b in _cut_at(masked, cuts):
        if start is None:
            start = a
        if _ARROW.search(masked[a:b]):
            spans.append((start, b))
            start = None
        elif spans:
            return None
    return spans if len(spans) == arrows else None


def _rule_pair(text: str, masked: str) -> tuple[str, str, str]:
    """The question and answer halves of one rule, and the answer's annotation.

    The trigger keeps whatever the document wrote, parenthetical included: it is
    the question *and* `PatternCard.identity`, so an annotation dropped from it
    is a note GUID that moves. The answer does not — `Result` is documented as
    the answer alone (`って`, `きて`), and `って (godan)` in an answer slot is a
    classification the learner is being asked to produce — so what is taken out
    of it is handed back for the gloss. Nothing the document wrote leaves the
    pipeline.

    Unless the annotation *is* the answer: `く → （いて）` is a well-formed rule
    written in full-width brackets, and stripping it left an empty answer, which
    collapsed the rule into a prose card whose question contained its own answer
    — and whose trigger, the GUID, became the whole line.
    """
    arrow = _ARROW.search(masked)
    if arrow:
        trigger = text[: arrow.start()]
        raw = text[arrow.end():]
        stripped = _tidy(masked[arrow.end():].replace(_MASK, ""))
        annotation = (
            " ".join(_tidy(part) for part in re.findall(r"[(（][^)）]*[)）]", raw))
            if stripped
            else ""
        )
        result = stripped or _tidy(raw)
        if _tidy(trigger) and result:
            return (_tidy(trigger), result, annotation)
    return (_tidy(text), "", "")


def _unannotated(text: str) -> str:
    """The text with its parentheticals gone — what the rule checker reads.

    Normalized like the checker's own reading, not only stripped: a chart
    extracted on macOS arrives with decomposed kana, so an unnormalized ぐ is
    く plus U+3099 and matches nothing the checker returns.
    """
    bare = _masked(text).replace(_MASK, "")
    return _tidy(normalize_identity_part(bare) or bare)


def _tidy(text: str) -> str:
    """Trim whitespace *and* a dangling separator.

    `str.strip()` alone left the `/` on a truncated row like `く → いて /`, so a
    card's answer read "いて /".
    """
    return text.strip().strip(_SEPARATOR_CHARS).strip()


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


def shipping_records(
    deck_path: Path, records: Sequence[VocabularyRecord], form: str = ""
) -> list[VocabularyRecord]:
    """The records this deck will put on a card: filtered, and conjugable.

    What the review gate has to ask about. Gating the whole collection instead
    refused a build over records the deck structurally cannot ship — a noun has
    no verb class, so no drill card could ever carry it — and over records the
    deck's own `exclude_ids` had deliberately held back.
    """
    section = _deck_section(deck_path)
    wanted = str(form or section.get("form") or "te_form").strip()
    include = _deck_string_set(section, "include_ids", deck_path)
    exclude = _deck_string_set(section, "exclude_ids", deck_path)
    chosen = [
        record
        for record in records
        if (not include or record.id in include) and record.id not in exclude
    ]
    keep = {record_id for _card, record_id in drill_cards(chosen, wanted)}
    return [record for record in chosen if record.id in keep]


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
    deck_path: Path,
    store: Mapping[str, PatternSet] | None = None,
    project_config: ProjectConfig | None = None,
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
        # The refusals the build gained. Without these `janki validate` passed a
        # deck `janki build` then rejected, which is the one thing this function
        # exists to prevent.
        for key in ("include_ids", "exclude_ids"):
            try:
                _deck_string_set(deck_config, key, deck_path)
            except DataError as exc:
                problems.append(str(exc))
        if project_config is not None:
            try:
                path = collection_for(deck_path, project_config)
            except JankiError as exc:
                problems.append(str(exc))
            else:
                # Whether the deck makes a card at all — the same question the
                # pattern branch asks below. Right after a plain Shirabe import
                # and before `janki enrich --jpdb` no record carries a
                # `verb_group`, so `validate` passed a deck the very next
                # `build` refused: the one drift this function exists to close.
                problems.extend(_drill_set_problems(deck_path, path, form))
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
    else:
        # The two refusals a *well-formed* deck can still hit. `groups` only
        # filters examples, never triggers, so the card set here is the one the
        # build produces and validate can answer for it.
        try:
            cards_for(entry)
        except PatternDeckError as exc:
            problems.append(str(exc))
        else:
            problems.extend(_card_set_problems(entry, document))
    return problems


def _drill_set_problems(deck_path: Path, collection: Path, form: str) -> list[str]:
    """The build's own emptiness refusal, asked of the same inputs.

    A collection this deck names and janki cannot read *is* this deck's problem.
    Nothing else in a `validate` sweep opens it — a drill deck's `source:` may
    name a file no other deck references — so swallowing that error left the one
    command whose job is catching a broken deck first saying "0 error(s)" over a
    file the next build refuses.

    A filter error is swallowed, because the `include_ids`/`exclude_ids` loop
    above has already reported it and saying it twice helps nobody.
    """
    if form not in CONJUGATION_FORMS:
        return []
    try:
        records = load_records(collection)
    except JankiError as exc:
        return [str(exc)]
    try:
        cards = drill_cards(shipping_records(deck_path, records, form), form)
    except JankiError:
        return []
    if not cards:
        return [
            f"no record janki can conjugate into a {form}. A verb needs a "
            f"verb_group janki knows — `janki enrich --jpdb` records one."
        ]
    return []


def _card_set_problems(entry: PatternSet, document: str) -> list[str]:
    cards = cards_for(entry)
    if not cards:
        return [f"{document} states no rules to make cards from"]
    seen: dict[str, int] = {}
    for card in cards:
        seen[card.identity] = seen.get(card.identity, 0) + 1
    clashing = sorted(name for name, count in seen.items() if count > 1)
    if clashing:
        return [
            f"{document} states more than one rule for {', '.join(clashing)}"
        ]
    return []


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
    cards = drill_cards(shipping_records(deck_path, records, form), form)
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
