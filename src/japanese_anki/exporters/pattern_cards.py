"""Cards for a rule, and cards for practising it.

Two decks, one notetype, because both ask the same shape of question: here is a
trigger, what does it become. A chart states the rules (`build_pattern_deck`);
the collection supplies verbs to run them on (`build_conjugation_deck`).

The rule deck:


A conjugation chart is a deck in itself: this turns the rules it holds into
cards, worked examples included, exactly as the chart states them. The gate is
the human ``reviewed:`` mark — M8.3 deleted the checker that used to adjudicate
each worked example against janki's own conjugation tables before letting it
ship, because janki's logic enriches the card and never audits the model
(DESIGN.md). A chart a person has reviewed is trusted.

**Its own notetype, not new fields on the vocabulary one.** A rule card has no
word reading, pitch, or audio; a lesson-specific drill may embed an explicitly
authored furigana example and the source record's usage note in the existing
``Examples`` field. Neither shares the vocabulary notetype. Measured against
`anki` 26.8.1: adding a notetype leaves the collection's ``scm`` mark alone,
while appending a field bumps it — so this costs no forced one-directional
AnkiWeb sync, which a new field on the existing notetype would.
"""

from __future__ import annotations

import copy
import hashlib
import html
import io
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, ScalarString
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

try:  # pragma: no cover - exercised by the import guard in `build_pattern_deck`
    import genanki
except ImportError:  # pragma: no cover
    genanki = None  # type: ignore[assignment]

from japanese_anki.config import ProjectConfig
from japanese_anki.conjugation import CONJUGATION_FORMS
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import (
    _deck_string_set,
    _refuse_conjugation_only_content,
    _resolve_media,
    _write_package_atomic,
)
from japanese_anki.identifiers import normalize_identity_part
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    atomic_write_text_bound,
    load_records,
    load_structured,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.patterns import (
    CHECKABLE_KINDS,
    LIST_SEPARATORS,
    PatternSet,
    verb_pairs_in,
    worked_examples_in,
)
from japanese_anki.validation import (
    field_separator_fault,
    has_errors,
    refusal_text,
    validate_records,
)

__all__ = [
    "PATTERN_TEMPLATE_FILENAMES",
    "PatternCard",
    "PatternDeckError",
    "build_conjugation_deck",
    "build_pattern_deck",
    "cards_for",
    "collection_for_section",
    "collection_for",
    "conjugation_media_paths_for_section",
    "conjugation_deck_section",
    "shipping_records",
    "shipping_records_for_section",
    "deck_problems",
    "drill_cards",
    "declared_drill_audio_owner_ids",
    "drill_audio_owner_id",
    "drill_audio_records",
    "drill_audio_records_from_revision",
    "DrillDeckContent",
    "read_drill_deck_content",
    "render_drill_deck_content",
    "render_drill_audio_records",
    "pattern_template_paths",
    "save_drill_deck_content",
    "save_drill_audio_records",
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

#: Explicit Anki furigana notation in deck-authored drill examples. This reads
#: markup, not Japanese: the author has already named both the displayed run
#: and its reading. Rendering it here is necessary because Anki does not apply
#: a ``furigana:`` filter recursively to HTML stored inside another field.
_FURIGANA = re.compile(r" ?([^>\s\[\]]+?)\[([^\[\]\r\n]+?)\]")

#: A conjugation deck's authored examples use the same sentence/audio shape as
#: vocabulary examples, except for romaji (which no card renders here). Audio is
#: filled by the shared paid-operation/WAL transaction; ``spoken_japanese`` is
#: the same sparse human escape hatch used after listening to a vocabulary clip.
_DRILL_EXAMPLE_FIELDS = frozenset({
    "japanese",
    "furigana",
    "english",
    "register",
    "audio",
    "spoken_japanese",
})

_DRILL_YAML_WIDTH = 1 << 30

# The complete template set consumed by both rule and conjugation packages.
# Build authority imports this tuple rather than maintaining a second list, so
# adding a new exporter input cannot leave it outside the confirmed plan.
PATTERN_TEMPLATE_FILENAMES: tuple[str, ...] = (
    "pattern-front.html",
    "pattern-back.html",
    "style.css",
)


class _UniqueDeckKeyLoader(yaml.SafeLoader):
    """SafeLoader that cannot silently replace earlier deck-file content."""


def _unique_deck_mapping(
    loader: _UniqueDeckKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, but found {node.id}",
            node.start_mark,
        )
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a deck mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a deck mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueDeckKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_deck_mapping,
)


def _editable_drill_document(text: str, deck_path: Path) -> tuple[YAML, MutableMapping[str, Any]]:
    """Parse one curated deck for a comment/quote-preserving edit."""
    parser = YAML()
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.indent(mapping=2, sequence=4, offset=2)
    parser.width = _DRILL_YAML_WIDTH
    try:
        document = parser.load(io.StringIO(text))
    except YAMLError as exc:
        raise DataError(f"Could not read {deck_path} for rewriting: {exc}") from exc
    if not isinstance(document, MutableMapping):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    section = document.get("deck")
    if not isinstance(section, MutableMapping):
        raise DataError(f"Deck file must contain a deck mapping: {deck_path}")
    examples = section.get("drill_examples")
    if not isinstance(examples, MutableMapping):
        raise DataError(
            f"deck.drill_examples must be a mapping keyed by record id: {deck_path}"
        )
    return parser, document


def _drill_furigana_malformed(value: str) -> bool:
    """Whether authored drill furigana has malformed bracket structure.

    This is only a structural artifact check. It does not decide which text is
    Japanese or whether a displayed run has the right reading.
    """
    inside_reading = False
    has_base = False
    has_reading = False
    for character in value:
        if character == "[":
            if inside_reading or not has_base:
                return True
            inside_reading = True
            has_reading = False
        elif character == "]":
            if not inside_reading or not has_reading:
                return True
            inside_reading = False
            has_base = False
        elif inside_reading:
            if character in "\r\n":
                return True
            if not character.isspace():
                has_reading = True
        elif character == ">" or character.isspace():
            has_base = False
        else:
            has_base = True
    return inside_reading

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
    #: ``verb ⇨ form`` pairs the reviewed chart states.
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

        Composed, because a chart read on one machine arrives with decomposed
        kana and on another with composed, and `patterns.json` stores the
        template with only `.strip()` — so nothing upstream settles the form and
        the same rule minted two GUIDs, the second build adding a note beside
        the first instead of updating it. The displayed `Trigger` keeps the
        document's own bytes; only what the GUID is derived from is folded.

        NFC, and deliberately not the NFKC that `normalize_identity_part`
        applies to a *record* identity. NFKC folds compatibility characters, and
        a chart cell is full of them: `（〜てもいい）` in full-width brackets
        would fold to ASCII parens, moving the GUID of a card already shipped
        and stranding its review history — which is the very thing this identity
        exists to hold still. Canonical composition settles the kana question
        and touches nothing the document chose.
        """
        return unicodedata.normalize("NFC", self.trigger).strip() or self.trigger


def cards_for(entry: PatternSet) -> list[PatternCard]:
    """The cards a reviewed document's rules make.

    One card per rule, and a rule stating several — ``くる → きて / する → して``
    — becomes one card each, which is how they are drilled anyway.
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

    # The chart's own worked examples, as it states them. M8.3 deleted the
    # checker that used to adjudicate each pair against janki's conjugation
    # tables before letting it onto a card — a reviewed chart is trusted, and
    # the human `reviewed:` gate above is the approval.
    agreed = worked_examples_in(entry)

    cards: list[PatternCard] = []
    for pattern in entry.patterns:
        checked = agreed.get(pattern.template, [])
        rules = _split_rules(pattern.template)
        # Matched on the trigger without its parenthetical, because
        # `worked_examples_in` reads the row with parentheticals removed:
        # the scanned verb for `かう (exception) → かって` is the bare かう. Compared
        # against the displayed trigger, the example matched no card, and the
        # fallback below — which keeps what belongs to no *other* rule — then
        # put it on every card on the row.
        bare = [_unannotated(trigger) for trigger, _, _ in rules]
        # Plus the verbs named *inside* a piece that states no rule. A row can
        # hold a chain beside a rule — `くる → きて → きた / く → いて` — and the
        # chain's card is its whole text, so its verb appears in no trigger.
        # Unclaimed, the fallback below (which keeps what belongs to no *other*
        # rule) put くる ⇨ きて on the `く → いて` card, showing an irregular's
        # て-form as a worked example of the く rule.
        owned = set(bare)
        for trigger, result, _annotation in rules:
            if not result:
                owned.update(verb for verb, _ in verb_pairs_in(_unannotated(trigger)))
        for (trigger, result, annotation), verb_of in zip(rules, bare, strict=True):
            # A row stating several rules — `くる → きて / する → して` — has
            # worked examples for each, and putting both on both cards asks
            # about くる while showing する. Where the trigger *is* the example's
            # verb, keep only its own; where it is an ending (`う・つ・る`) no
            # example names it, so the row's whole set belongs to the card.
            # A card that states no rule is its whole text, so the verbs it
            # claims are the ones named *in* it rather than its trigger.
            claims = (
                {verb_of}
                if result
                else {verb_of, *(verb for verb, _ in verb_pairs_in(verb_of))}
            )
            mine = [text for verb, text in checked if verb in claims]
            examples = tuple(mine) if mine else tuple(
                text for verb, text in checked if verb not in owned
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

    The same characters do both jobs: source extraction can return a rule's
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
    from the first answer and asks ``きた / する``. Only the nakaguro is decided
    in advance — it joins the items of a list and never divides two statements.
    Every other separator is a peer, and two peers that fit and disagree are
    refused rather than ranked: ``する → して, した / くる → きて`` and
    ``く → いて / う、つ → って`` are the same string up to which character plays
    which role.

    When no cut fits at all, the line uses one character for both jobs.
    Arrow-less pieces before the first rule are its trigger list and are rejoined
    to it — ``う/つ/る → って / く → いて``. Arrow-less pieces *after* a rule are
    genuinely undecidable, since ``って/った / く → いて`` reads equally as a
    two-item result list or as the next rule's trigger.

    A line whose cut is *ambiguous* is refused, naming the template: janki does
    not know what the document says there, and the row is hand-fixable in
    `patterns.json` by putting each rule on its own line. A line — or a piece of
    one — that simply states no cut, like the chain ``〜て → 〜ている → 〜てる``,
    is prose and becomes one card.

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
    return _cards_in(template, _masked(template))


def _cards_in(template: str, masked: str) -> list[tuple[str, str, str]]:
    """The cards one piece of a line makes — the whole line, or one cut of it.

    Applied again to each piece, because a cut can leave a piece that is itself
    more than one rule's worth of arrows: `〜て → 〜ている → 〜てる / 〜ておく → 〜とく`
    divides unambiguously at the `/`, and only the first half is a chain. Judged
    once for the whole line, the well-formed rule beside it never became a card,
    and the single note that shipped carried the entire line as its trigger —
    which is also its GUID.
    """
    arrows = len(_ARROW.findall(masked))
    if not arrows:
        return [(_tidy(template), "", "")]
    if arrows == 1:
        return [_rule_pair(template, masked)]
    spans = _rule_spans(masked, arrows, template)
    if spans is None:
        # No cut, and no ambiguity either: `〜て → 〜ている → 〜てる` is a
        # progression chain with no separator in it at all. Prose is what it is.
        return [(_tidy(template), "", "")]
    return [
        card
        for a, b in spans
        for card in _cards_in(template[a:b], masked[a:b])
    ]


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


def _rule_spans(
    masked: str, arrows: int, template: str
) -> list[tuple[int, int]] | None:
    """One span per rule.

    ``None`` when the line states no cut to make — a chain like
    ``〜て → 〜ている → 〜てる`` has no separator in it — which the caller reads as
    prose. A span may hold more than one arrow and still be a real cut; the
    caller asks again about each piece.

    An *ambiguous* line raises instead, saying which ambiguity it is: janki does
    not know what the document states there, and shipping it as one prose card
    put both answers on the card's own front.
    """
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
        # Two dividers that fit and disagree, and nothing in the line says
        # which is real. `する → して, した / くる → きて` — a result list, then a
        # rule — and `く → いて / う、つ → って` — a rule, then a trigger list —
        # are the same string up to which character plays which role, so any
        # score that prefers one is only a fixed preference for the earlier or
        # the later cut. Scoring the trigger halves read the second row as the
        # first: it drilled `いて / う` as an answer and minted a GUID for the
        # trigger `つ`, which the document never wrote.
        #
        # So: refused by name. The row is hand-fixable in `patterns.json`, and
        # saying so beats a card that teaches nothing.
        raise PatternDeckError(
            f"janki cannot tell where {template!r} divides: two separators cut "
            f"it into rules equally well, and the wrong one drills a trigger "
            f"the document never wrote. Put each rule on its own row."
        )
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
            # One character doing both jobs, and a run after a rule has ended:
            # in `う/つ/る → って/った / く → いて`, `った` is either the rest of
            # that rule's answer or the next rule's trigger, and the line does
            # not say which. Named as its own ambiguity — telling someone to
            # look for a second separator that is not there helps nobody.
            raise PatternDeckError(
                f"janki cannot tell where {template!r} divides: it uses one "
                f"character for both listing and dividing, so the run after the "
                f"first rule could belong to that rule's answer or to the next "
                f"rule's trigger. Put each rule on its own row."
            )
    # More than one span is a real cut, even when a span holds more than one
    # arrow: the caller looks at each piece again, so a chain beside a rule
    # keeps the rule. One span is no cut at all — the chain by itself.
    return spans if len(spans) > 1 else None


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


def pattern_template_paths(template_dir: Path) -> tuple[Path, ...]:
    """Return every exact filesystem input consumed by the pattern notetype."""

    return tuple(
        (template_dir / filename).resolve()
        for filename in PATTERN_TEMPLATE_FILENAMES
    )


def _notetype(model_id: int, model_name: str, template_dir: Path) -> Any:
    front_path, back_path, style_path = pattern_template_paths(template_dir)
    front = _read(front_path)
    back = _read(back_path)
    return genanki.Model(
        model_id,
        model_name,
        fields=[{"name": name} for name in FIELDS],
        templates=[{"name": "Rule", "qfmt": front, "afmt": back}],
        css=_read(style_path),
    )


def _deck_section(deck_path: Path) -> dict[str, Any]:
    raw = load_structured(deck_path, yaml_loader=_UniqueDeckKeyLoader)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    section = raw.get("deck") or {}
    if not isinstance(section, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    _refuse_conjugation_only_content(section, deck_path)
    return section


def _drill_form_note(deck_config: dict[str, Any], deck_path: Path) -> str:
    """The deck-wide explanation shown beneath each computed answer."""
    value = deck_config.get("form_note")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DataError(
            f"deck.form_note must be text, got {type(value).__name__}: {deck_path}"
        )
    note = value.strip()
    if note and not _deck_string_set(deck_config, "include_ids", deck_path):
        raise DataError(
            "deck.form_note needs a nonempty deck.include_ids list so its "
            f"lesson explanation cannot grow to new records: {deck_path}"
        )
    if note and deck_config.get("drill_examples") is None:
        raise DataError(
            "deck.form_note requires deck.drill_examples so every lesson card "
            f"carries an example of the form it teaches: {deck_path}"
        )
    return note


def _drill_examples(
    deck_config: dict[str, Any],
    records: Sequence[VocabularyRecord],
    deck_path: Path,
    form: str,
) -> dict[str, tuple[ExampleSentence, ...]]:
    """Read explicitly authored, form-specific examples from a drill deck.

    These cannot come from ``record.examples``: those sentences teach the base
    word and mechanically replacing their verb would be Japanese-writing logic.
    Keeping the examples beside the drill deck also leaves the vocabulary
    cards' reviewed polite/casual pair untouched.
    """
    raw = deck_config.get("drill_examples")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise DataError(
            f"deck.drill_examples must be a mapping keyed by record id: {deck_path}"
        )

    include_ids = _deck_string_set(deck_config, "include_ids", deck_path)
    if not include_ids:
        raise DataError(
            "deck.drill_examples needs a nonempty deck.include_ids list so its "
            f"reviewed lesson scope cannot grow when the collection grows: {deck_path}"
        )
    exclude_ids = _deck_string_set(deck_config, "exclude_ids", deck_path)
    overlap = sorted(include_ids & exclude_ids)
    if overlap:
        raise DataError(
            "deck.drill_examples has record(s) listed in both deck.include_ids "
            f"and deck.exclude_ids: {', '.join(overlap)}. Remove each id from "
            f"one list: {deck_path}"
        )
    shipping_ids = {record.id for record in records}
    parsed: dict[str, tuple[ExampleSentence, ...]] = {}
    for raw_id, raw_examples in raw.items():
        if not isinstance(raw_id, str):
            raise DataError(
                f"deck.drill_examples keys must be record ids, got "
                f"{type(raw_id).__name__}: {deck_path}"
            )
        record_id = raw_id
        if record_id not in include_ids:
            raise DataError(
                f"deck.drill_examples key {record_id!r} is not listed in "
                f"deck.include_ids: {deck_path}"
            )
        if not isinstance(raw_examples, list):
            raise DataError(
                f"deck.drill_examples[{record_id!r}] must be a list: {deck_path}"
            )
        if not raw_examples:
            raise DataError(
                f"deck.drill_examples[{record_id!r}] needs at least one example: "
                f"{deck_path}"
            )

        examples: list[ExampleSentence] = []
        for index, raw_example in enumerate(raw_examples):
            if not isinstance(raw_example, dict):
                raise DataError(
                    f"deck.drill_examples[{record_id!r}][{index}] must be a "
                    f"mapping: {deck_path}"
                )
            unknown = sorted(
                str(key) for key in raw_example if key not in _DRILL_EXAMPLE_FIELDS
            )
            if unknown:
                raise DataError(
                    f"deck.drill_examples[{record_id!r}][{index}] has unknown "
                    f"field(s): {', '.join(str(key) for key in unknown)}: {deck_path}"
                )
            for field, value in raw_example.items():
                if not isinstance(value, str):
                    raise DataError(
                        f"deck.drill_examples[{record_id!r}][{index}] {field} "
                        f"must be text, got {type(value).__name__}: {deck_path}"
                    )
            example = ExampleSentence.from_dict(raw_example, position=index)
            if example.audio.startswith("[sound:"):
                raise DataError(
                    f"deck.drill_examples[{record_id!r}][{index}] audio must be "
                    f"a packaged media path, not a verbatim sound tag: {deck_path}"
                )
            if _drill_furigana_malformed(example.furigana):
                raise DataError(
                    f"deck.drill_examples[{record_id!r}][{index}] furigana "
                    f"brackets are unbalanced: {deck_path}"
                )
            if (
                not example.japanese
                or not example.english
                or example.register not in {"polite", "casual"}
            ):
                raise DataError(
                    f"deck.drill_examples[{record_id!r}][{index}] needs "
                    f"japanese, english, and register polite/casual: {deck_path}"
                )
            examples.append(example)
        registers = Counter(example.register for example in examples)
        polite = registers["polite"]
        casual = registers["casual"]
        if len(examples) != 2 or polite != 1 or casual != 1:
            raise DataError(
                f"deck.drill_examples[{record_id!r}] needs exactly one polite "
                f"and one casual example; found {polite} polite and {casual} "
                f"casual: {deck_path}"
            )
        parsed[record_id] = tuple(examples)
    missing = sorted(include_ids - set(parsed))
    if missing:
        raise DataError(
            "deck.drill_examples opts into rich examples and needs an entry "
            f"for every included record; missing: {', '.join(missing)}: {deck_path}"
        )
    unavailable = sorted(include_ids - shipping_ids)
    if unavailable:
        raise DataError(
            "deck.drill_examples has included record(s) missing from the "
            f"collection or cannot be conjugated into {form}: "
            f"{', '.join(unavailable)}. Restore their verb_group/part_of_speech, "
            "or remove each id from both deck.include_ids and deck.drill_examples: "
            f"{deck_path}"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class DrillDeckContent:
    """The exact editable content and whole-file revision of one drill deck."""

    revision: RecordsRevision
    form: str
    record_ids: tuple[str, ...]
    form_note: str
    drill_examples: Mapping[str, tuple[ExampleSentence, ...]]


def _revision_target(deck_path: Path, expected: RecordsRevision) -> Path:
    """Bind a content snapshot to the lexical deck path it was read from."""
    target = deck_path.absolute()
    if expected.path.absolute() != target:
        raise DataError(
            f"Drill revision for {expected.path} cannot guard a write to {target}."
        )
    if expected.text is None:
        raise DataError(f"Drill deck {target} no longer exists.")
    return target


def _strict_drill_document(
    text: str,
    deck_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the exact snapshot through the deck's duplicate-key contract."""
    try:
        document = yaml.load(text, Loader=_UniqueDeckKeyLoader)
    except yaml.YAMLError as exc:
        raise DataError(f"Could not read drill deck {deck_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    section = document.get("deck")
    if not isinstance(section, dict):
        raise DataError(f"Deck file must contain a deck mapping: {deck_path}")
    _refuse_conjugation_only_content(section, deck_path)
    if str(section.get("kind") or "").strip().lower() != "conjugation":
        raise DataError(f"Drill revision requires a conjugation deck: {deck_path}")
    return document, section


def conjugation_deck_section(deck_path: Path, text: str) -> dict[str, Any]:
    """Parse one exact in-memory conjugation-deck snapshot."""

    _document, section = _strict_drill_document(text, deck_path)
    return section


def _ordered_drill_scope(
    deck_config: Mapping[str, Any],
    deck_path: Path,
) -> tuple[str, ...]:
    """The explicit, ordered IDs a rich drill revision is allowed to replace."""
    value = deck_config.get("include_ids")
    if not isinstance(value, list):
        raise DataError(
            f"deck.include_ids must be a nonempty list of record ids: {deck_path}"
        )
    identifiers: list[str] = []
    for position, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise DataError(
                f"deck.include_ids[{position}] must be a nonblank record id: "
                f"{deck_path}"
            )
        identifiers.append(item)
    if not identifiers:
        raise DataError(
            f"deck.include_ids must be a nonempty list of record ids: {deck_path}"
        )
    if len(identifiers) != len(set(identifiers)):
        raise DataError(f"deck.include_ids must not repeat record ids: {deck_path}")
    return tuple(identifiers)


def _drill_content_from_revision(
    deck_path: Path,
    expected: RecordsRevision,
) -> DrillDeckContent:
    """Validate rich content from one already-captured whole-file snapshot."""
    target = _revision_target(deck_path, expected)
    assert expected.text is not None  # established by `_revision_target`
    _editable_drill_document(expected.text, target)
    _document, section = _strict_drill_document(expected.text, target)
    form = str(section.get("form") or "te_form").strip()
    if form not in CONJUGATION_FORMS:
        raise DataError(
            f"{target}: form must be one of {', '.join(CONJUGATION_FORMS)}, "
            f"got {form!r}"
        )
    record_ids = _ordered_drill_scope(section, target)
    # `_drill_examples` needs only the shipping identities for its scope check.
    # Canonical vocabulary facts and conjugation belong to the application plan,
    # while this primitive owns only the exact deck artifact it will rewrite.
    scoped_records = [
        VocabularyRecord(id=record_id, expression=record_id)
        for record_id in record_ids
    ]
    note = _drill_form_note(section, target)
    examples = _drill_examples(section, scoped_records, target, form)
    if not examples:
        raise DataError(
            f"{target}: deck.drill_examples is required for a drill revision"
        )
    return DrillDeckContent(
        revision=expected,
        form=form,
        record_ids=record_ids,
        form_note=note,
        drill_examples={record_id: examples[record_id] for record_id in record_ids},
    )


def read_drill_deck_content(
    deck_path: Path,
) -> DrillDeckContent:
    """Read one exact, full-scope rich-drill snapshot for a later CAS."""
    target = deck_path.absolute()
    return _drill_content_from_revision(
        target,
        records_revision(target),
    )


def _replacement_examples(
    drill_examples: Mapping[str, Sequence[ExampleSentence]],
    record_ids: tuple[str, ...],
    deck_path: Path,
) -> dict[str, dict[str, ExampleSentence]]:
    """Index one ordered, nonempty subset by record and card register."""
    if not isinstance(drill_examples, Mapping):
        raise DataError(f"Drill revision examples must be a mapping: {deck_path}")
    selected = tuple(drill_examples)
    if not selected:
        raise DataError(
            f"Drill revision examples must select at least one record: {deck_path}"
        )
    if any(not isinstance(record_id, str) or not record_id.strip() for record_id in selected):
        raise DataError(
            f"Drill revision record ids must be nonblank text: {deck_path}"
        )
    current_ids = set(record_ids)
    extra = [record_id for record_id in selected if record_id not in current_ids]
    ordered_subset = tuple(
        record_id for record_id in record_ids if record_id in set(selected)
    )
    if extra or selected != ordered_subset:
        detail = (
            f"unexpected {', '.join(extra)}"
            if extra
            else "selected ids are not in deck.include_ids order"
        )
        raise DataError(
            "Drill revision examples must be an exact ordered subset of "
            f"deck.include_ids ({detail}): {deck_path}"
        )
    indexed: dict[str, dict[str, ExampleSentence]] = {}
    for record_id in selected:
        values = drill_examples[record_id]
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise DataError(
                f"Drill revision examples for {record_id!r} must be a sequence: "
                f"{deck_path}"
            )
        examples = list(values)
        if any(not isinstance(example, ExampleSentence) for example in examples):
            raise DataError(
                f"Drill revision examples for {record_id!r} must contain "
                f"ExampleSentence values: {deck_path}"
            )
        registers = Counter(example.register for example in examples)
        if len(examples) != 2 or registers["polite"] != 1 or registers["casual"] != 1:
            raise DataError(
                f"Drill revision examples for {record_id!r} need exactly one "
                f"polite and one casual ExampleSentence: {deck_path}"
            )
        indexed[record_id] = {example.register: example for example in examples}
    return indexed


_DRILL_REVISION_FIELDS = ("japanese", "furigana", "english", "register")


def _styled_replacement(current: Any, value: str) -> str:
    """Keep an authored scalar's quote/block style when replacing its value."""
    if current == value:
        return current
    if isinstance(current, ScalarString):
        return type(current)(value)
    # A new plain scalar such as ``yes`` is a string to ruamel's YAML 1.2
    # reader but a boolean to the project's YAML 1.1 strict loader. Quoting all
    # changed plain values makes the semantic verification below stable.
    return DoubleQuotedScalarString(value)


def _set_revision_scalar(
    mapping: MutableMapping[str, Any],
    field: str,
    value: str,
) -> None:
    """Set one target scalar without manufacturing an absent empty field."""
    if field not in mapping and value == "":
        return
    current = mapping.get(field)
    if current != value:
        mapping[field] = _styled_replacement(current, value)


def _semantic_without_drill_content(document: Mapping[str, Any]) -> dict[str, Any]:
    """Copy every semantic value this primitive has no authority to change."""
    outside = copy.deepcopy(dict(document))
    section = outside.get("deck")
    if isinstance(section, dict):
        section.pop("form_note", None)
        section.pop("drill_examples", None)
    return outside


def _unselected_drill_content(
    document: Mapping[str, Any],
    selected: set[str],
) -> dict[Any, Any]:
    """Copy the rich examples a partial proposal has no authority to touch."""
    section = document.get("deck")
    if not isinstance(section, Mapping):
        return {}
    examples = section.get("drill_examples")
    if not isinstance(examples, Mapping):
        return {}
    return copy.deepcopy({
        record_id: value
        for record_id, value in examples.items()
        if record_id not in selected
    })


def render_drill_deck_content(
    deck_path: Path,
    *,
    expected: RecordsRevision,
    form_note: str,
    drill_examples: Mapping[str, Sequence[ExampleSentence]],
) -> str:
    """Render only ``form_note`` and ``drill_examples`` from an exact snapshot.

    Provider output has no authority over media. Existing ``audio`` and the
    human ``spoken_japanese`` override remain byte-for-byte only while that
    register's displayed Japanese is byte-identical; changing the sentence
    clears both so the ordinary audio transaction can plan a fresh clip.
    """
    target = _revision_target(deck_path, expected)
    current = _drill_content_from_revision(
        target,
        expected,
    )
    replacements = _replacement_examples(
        drill_examples,
        current.record_ids,
        target,
    )
    if not isinstance(form_note, str):
        raise DataError(
            f"Drill revision form_note must be text, got "
            f"{type(form_note).__name__}: {target}"
        )
    assert expected.text is not None  # established by `_revision_target`
    raw, raw_section = _strict_drill_document(expected.text, target)
    intended = copy.deepcopy(raw)
    intended_section = intended["deck"]
    raw_examples = raw_section["drill_examples"]
    intended_examples = intended_section["drill_examples"]
    if "form_note" in intended_section or form_note:
        intended_section["form_note"] = form_note

    for record_id in replacements:
        authored = raw_examples[record_id]
        intended_authored = intended_examples[record_id]
        for position, raw_example in enumerate(authored):
            authored_example = ExampleSentence.from_dict(
                raw_example,
                position=position,
            )
            incoming = replacements[record_id][authored_example.register]
            intended_example = intended_authored[position]
            japanese_unchanged = (
                isinstance(incoming.japanese, str)
                and raw_example["japanese"].encode("utf-8")
                == incoming.japanese.encode("utf-8")
            )
            for field in _DRILL_REVISION_FIELDS:
                value = getattr(incoming, field)
                if field not in intended_example and value == "":
                    continue
                intended_example[field] = value
            if not japanese_unchanged:
                intended_example.pop("audio", None)
                intended_example.pop("spoken_japanese", None)

    # Use the same semantic readers as build/audio before touching the
    # round-trip tree. This validates field types, furigana brackets, exact
    # polite/casual cardinality, and the complete include-id scope.
    scoped_records = [
        VocabularyRecord(id=record_id, expression=record_id)
        for record_id in current.record_ids
    ]
    _drill_form_note(intended_section, target)
    _drill_examples(intended_section, scoped_records, target, current.form)

    parser, editable = _editable_drill_document(expected.text, target)
    editable_section = editable["deck"]
    if "form_note" in editable_section or form_note:
        _set_revision_scalar(editable_section, "form_note", form_note)
    editable_examples = editable_section["drill_examples"]
    for record_id in replacements:
        authored = raw_examples[record_id]
        editable_authored = editable_examples[record_id]
        for position, raw_example in enumerate(authored):
            authored_example = ExampleSentence.from_dict(
                raw_example,
                position=position,
            )
            incoming = replacements[record_id][authored_example.register]
            editable_example = editable_authored[position]
            japanese_unchanged = (
                raw_example["japanese"].encode("utf-8")
                == incoming.japanese.encode("utf-8")
            )
            for field in _DRILL_REVISION_FIELDS:
                _set_revision_scalar(
                    editable_example,
                    field,
                    getattr(incoming, field),
                )
            if not japanese_unchanged:
                editable_example.pop("audio", None)
                editable_example.pop("spoken_japanese", None)

    buffer = io.StringIO()
    parser.dump(editable, buffer)
    rendered = buffer.getvalue()
    try:
        reparsed = yaml.load(rendered, Loader=_UniqueDeckKeyLoader)
    except yaml.YAMLError as exc:
        raise DataError(
            f"Could not verify the drill content rendered for {target}: {exc}"
        ) from exc
    if _semantic_without_drill_content(reparsed) != _semantic_without_drill_content(raw):
        raise DataError(
            f"Refused a drill revision that would change other deck content: {target}"
        )
    selected = set(replacements)
    if _unselected_drill_content(reparsed, selected) != _unselected_drill_content(
        raw, selected
    ):
        raise DataError(
            f"Refused a drill revision that would change an unselected card: {target}"
        )
    if reparsed != intended:
        raise DataError(
            f"Refused a drill revision whose rendered content changed shape: {target}"
        )
    return rendered


def save_drill_deck_content(
    deck_path: Path,
    *,
    expected: RecordsRevision,
    form_note: str,
    drill_examples: Mapping[str, Sequence[ExampleSentence]],
) -> RecordsRevision:
    """CAS one rendered rich-drill revision over its exact whole-file base."""
    target = _revision_target(deck_path, expected)
    rendered = render_drill_deck_content(
        target,
        expected=expected,
        form_note=form_note,
        drill_examples=drill_examples,
    )
    assert expected.text is not None  # established by `_revision_target`
    atomic_write_text_bound(
        target,
        rendered,
        expected_revision=hashlib.sha256(expected.text.encode("utf-8")).hexdigest(),
    )
    return RecordsRevision(target, rendered)


def drill_audio_owner_id(deck_id: int, form: str, record_id: str) -> str:
    """The ledger/WAL owner for one drill card's sentence clips.

    It is deliberately distinct from both the vocabulary record id and the
    Anki note GUID. Two decks may teach the same form with different authored
    sentences; their audio transactions must not retire each other's ledger
    rows, while the existing note identity remains stable for Anki review
    history.
    """
    return f"drill-audio:{deck_id}:{form}:{record_id}"


def declared_drill_audio_owner_ids(deck_path: Path) -> frozenset[str]:
    """Synthetic owners a rich drill deck claims before content projection.

    Pending paid bytes must survive a temporarily invalid example or missing
    source record.  The owner identity itself is structural: the pinned deck
    id, form, and authored record scope.  Take the conservative union of the
    two scope declarations so a mismatch between them cannot make an existing
    WAL row look deleted.  This does not inspect or judge Japanese content.
    """
    deck_config = _deck_section(deck_path)
    kind = str(deck_config.get("kind") or "").strip().lower()
    raw_examples = deck_config.get("drill_examples")
    if kind != "conjugation" or raw_examples is None:
        return frozenset()

    deck_id = _identifier(deck_config, "deck_id", deck_path)
    form = str(deck_config.get("form") or "te_form").strip()
    if form not in CONJUGATION_FORMS:
        return frozenset()

    record_ids: set[str] = set()
    raw_include_ids = deck_config.get("include_ids")
    if isinstance(raw_include_ids, list | tuple):
        record_ids.update(
            record_id
            for value in raw_include_ids
            if (record_id := str(value))
        )
    if isinstance(raw_examples, Mapping):
        record_ids.update(
            record_id
            for value in raw_examples
            if isinstance(value, str) and (record_id := value)
        )
    return frozenset(
        drill_audio_owner_id(deck_id, form, record_id)
        for record_id in record_ids
    )


def drill_audio_records(
    deck_path: Path,
    project_config: ProjectConfig,
    records: Sequence[VocabularyRecord] | None = None,
) -> list[VocabularyRecord]:
    """Project a rich drill deck onto the ordinary sentence-audio contract.

    These are operational audio owners, not vocabulary records and not a new
    synthesis path. The existing audio service sees the same
    :class:`ExampleSentence` values it already journals, stages, recovers and
    ledgers for vocabulary cards.
    """
    return drill_audio_records_from_revision(
        deck_path,
        project_config,
        records_revision(deck_path),
        records,
    )


def drill_audio_records_from_revision(
    deck_path: Path,
    project_config: ProjectConfig,
    revision: RecordsRevision,
    records: Sequence[VocabularyRecord] | None = None,
) -> list[VocabularyRecord]:
    """Project sentence-audio owners from exact proposed deck bytes."""
    target = deck_path.resolve()
    if revision.path.resolve() != target:
        raise DataError(
            f"Drill revision for {revision.path} cannot project audio for {target}."
        )
    if revision.text is None:
        raise DataError(f"Deck file no longer exists: {target}")
    try:
        raw = yaml.load(revision.text, Loader=_UniqueDeckKeyLoader)
    except yaml.YAMLError as exc:
        raise DataError(f"Could not parse {target}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {target}")
    deck_config = raw.get("deck") or {}
    if not isinstance(deck_config, dict):
        raise DataError(f"The deck section must be a mapping: {target}")
    _refuse_conjugation_only_content(deck_config, target)
    kind = str(deck_config.get("kind") or "").strip().lower()
    if kind != "conjugation":
        raise DataError(
            f"Deck audio examples require a conjugation deck, got {kind or 'vocabulary'}: "
            f"{target}"
        )
    if deck_config.get("drill_examples") is None:
        raise DataError(
            f"{target}: deck.drill_examples is required for deck example audio"
        )
    # A static rewrite failure is knowable before a paid sentence call. The
    # writer uses this same round-trip parser so comments, quoting and anchors
    # survive when only generated audio scalars are inserted.
    _editable_drill_document(revision.text, target)
    form = str(deck_config.get("form") or "te_form").strip()
    if form not in CONJUGATION_FORMS:
        raise DataError(
            f"{target}: form must be one of {', '.join(CONJUGATION_FORMS)}, "
            f"got {form!r}"
        )
    source_records = (
        list(records)
        if records is not None
        else load_records(collection_for_section(target, project_config, deck_config))
    )
    shipping = shipping_records_for_section(target, deck_config, source_records, form)
    examples_by_id = _drill_examples(deck_config, shipping, target, form)
    if not examples_by_id:
        raise DataError(
            f"{target}: deck.drill_examples is required for deck example audio"
        )
    deck_id = _identifier(deck_config, "deck_id", target)
    projected: list[VocabularyRecord] = []
    for record in shipping:
        examples = examples_by_id.get(record.id)
        if examples is None:
            continue
        projected.append(
            replace(
                record,
                id=drill_audio_owner_id(deck_id, form, record.id),
                audio="",
                image="",
                examples=list(examples),
            )
        )
    return projected


def render_drill_audio_records(
    deck_path: Path,
    records: Sequence[VocabularyRecord],
    *,
    expected: RecordsRevision,
) -> str:
    """Render only generated audio references from an exact drill snapshot.

    This pure projection changes only generated audio scalars, preserving
    comments, quoting, anchors and unknown keys. A semantic reparse proves the
    rendered document differs in no other value, and the duplicate-key loader
    prevents silently choosing one of two authored entries.
    """
    target = deck_path.resolve()
    if expected.path.resolve() != target:
        raise DataError(
            f"Drill revision for {expected.path} cannot guard a write to {target}."
        )
    if expected.text is None:
        raise DataError(f"Drill deck {target} no longer exists.")
    raw = yaml.load(expected.text, Loader=_UniqueDeckKeyLoader)
    if not isinstance(raw, dict) or not isinstance(raw.get("deck"), dict):
        raise DataError(f"Deck file must contain a deck mapping: {target}")
    parser, document = _editable_drill_document(expected.text, target)
    section = raw["deck"]
    editable_section = document["deck"]
    form = str(section.get("form") or "te_form").strip()
    deck_id = _identifier(section, "deck_id", target)
    raw_examples = section.get("drill_examples")
    editable_examples = editable_section.get("drill_examples")
    if not isinstance(raw_examples, dict):
        raise DataError(
            f"deck.drill_examples must be a mapping keyed by record id: {target}"
        )
    if not isinstance(editable_examples, MutableMapping):
        raise DataError(
            f"deck.drill_examples must be a mapping keyed by record id: {target}"
        )
    by_owner = {record.id: record for record in records}
    expected_owners = {
        drill_audio_owner_id(deck_id, form, str(record_id))
        for record_id in raw_examples
    }
    if (
        len(records) != len(by_owner)
        or len(by_owner) != len(expected_owners)
        or set(by_owner) != expected_owners
    ):
        raise DataError(
            f"Drill audio result no longer matches every authored card in {target}."
        )
    for record_id, authored in raw_examples.items():
        if not isinstance(authored, list):
            raise DataError(
                f"deck.drill_examples[{record_id!r}] must be a list: {target}"
            )
        owner = drill_audio_owner_id(deck_id, form, str(record_id))
        updated = by_owner[owner]
        editable_authored = editable_examples.get(record_id)
        if not isinstance(editable_authored, list):
            raise DataError(
                f"deck.drill_examples[{record_id!r}] must be a list: {target}"
            )
        if len(authored) != len(updated.examples):
            raise DataError(
                f"Drill audio result changed the example count for {record_id!r}: "
                f"{target}"
            )
        if len(editable_authored) != len(updated.examples):
            raise DataError(
                f"Drill audio result changed the example count for {record_id!r}: "
                f"{target}"
            )
        for position, (raw_example, editable_example, example) in enumerate(
            zip(authored, editable_authored, updated.examples, strict=True)
        ):
            if not isinstance(raw_example, dict) or not isinstance(
                editable_example, MutableMapping
            ):
                raise DataError(
                    f"deck.drill_examples[{record_id!r}] must contain mappings: "
                    f"{target}"
                )
            authored_example = ExampleSentence.from_dict(
                raw_example, position=position
            )
            if (
                example.japanese != authored_example.japanese
                or example.spoken_japanese != authored_example.spoken_japanese
            ):
                raise DataError(
                    f"Drill audio result changed the spoken identity of "
                    f"{record_id!r} example {position + 1}: {target}"
                )
            if example.audio:
                raw_example["audio"] = example.audio
                editable_example["audio"] = example.audio
            else:
                raw_example.pop("audio", None)
                editable_example.pop("audio", None)
    buffer = io.StringIO()
    parser.dump(document, buffer)
    rendered = buffer.getvalue()
    try:
        reparsed = yaml.load(rendered, Loader=_UniqueDeckKeyLoader)
    except yaml.YAMLError as exc:
        raise DataError(
            f"Could not verify the audio update rendered for {target}: {exc}"
        ) from exc
    if reparsed != raw:
        raise DataError(
            f"Refused an audio update that would change other deck content: {target}"
        )
    return rendered


def save_drill_audio_records(
    deck_path: Path,
    records: Sequence[VocabularyRecord],
    *,
    expected: RecordsRevision,
) -> None:
    """CAS one pure drill-audio rendering while its owner lock is held."""
    target = deck_path.resolve()
    if expected.path.resolve() != target:
        raise DataError(
            f"Drill revision for {expected.path} cannot guard a write to {target}."
        )
    current = records_revision(target)
    if current.text != expected.text:
        raise DataError(
            f"Drill deck {target} changed on disk since it was read; saving now "
            "would discard those changes. Re-run the audio command."
        )
    if expected.text is None:
        raise DataError(f"Drill deck {target} no longer exists.")
    rendered = render_drill_audio_records(
        target,
        records,
        expected=expected,
    )
    atomic_write_text_bound(
        target,
        rendered,
        expected_revision=hashlib.sha256(expected.text.encode("utf-8")).hexdigest(),
    )


def _furigana_html(value: str) -> str:
    """Render explicit ``word[reading]`` notation as safe ruby HTML.

    Spaces immediately before an annotated run are Anki's boundary markers,
    not Japanese spaces, so they are consumed the same way the field filter
    consumes them. Everything outside those structural brackets is escaped.
    """
    parts: list[str] = []
    cursor = 0
    for match in _FURIGANA.finditer(value):
        parts.append(_escaped_lines(value[cursor:match.start()]))
        parts.append(
            "<ruby><rb>"
            f"{html.escape(match.group(1))}</rb><rt>"
            f"{html.escape(match.group(2))}</rt></ruby>"
        )
        cursor = match.end()
    parts.append(_escaped_lines(value[cursor:]))
    return "".join(parts)


def _escaped_lines(value: str) -> str:
    """Escape authored prose while preserving its explicit line breaks."""
    escaped = html.escape(value)
    return (
        escaped.replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def _drill_support_html(
    record: VocabularyRecord,
    card: PatternCard,
    form_note: str,
    examples: Sequence[ExampleSentence],
    audio_fields: Sequence[str] = (),
) -> str:
    """The rich answer context stored inside the existing Examples field."""
    parts = ['<section class="drill-support">']
    if form_note:
        parts.extend([
            '<section class="drill-form-note">',
            '<div class="drill-section-label">Using this form</div>',
            f"<div>{_escaped_lines(form_note)}</div>",
            "</section>",
        ])
    if card.examples:
        parts.append(
            '<div class="drill-meta"><span class="drill-pill">'
            f"{html.escape(', '.join(card.examples))}</span></div>"
        )
    if audio_fields and len(audio_fields) != len(examples):
        raise PatternDeckError("Drill example audio fields no longer match the examples")
    rendered_audio = tuple(audio_fields) if audio_fields else ("",) * len(examples)
    for example, audio_field in zip(examples, rendered_audio, strict=True):
        register = "Polite" if example.register == "polite" else "Casual"
        japanese = (
            _furigana_html(example.furigana)
            if example.furigana
            else _escaped_lines(example.japanese)
        )
        parts.extend([
            '<article class="drill-example">',
            f'<div class="drill-example-label">{register} example</div>',
            f'<div class="drill-example-japanese" lang="ja">{japanese}</div>',
            f'<div class="drill-example-english">{_escaped_lines(example.english)}</div>',
            *(
                [f'<div class="example-audio">{audio_field}</div>']
                if audio_field
                else []
            ),
            "</article>",
        ])
    if record.usage_notes:
        parts.extend([
            '<details class="drill-usage">',
            "<summary>About this word</summary>",
            f"<div>{_escaped_lines(record.usage_notes)}</div>",
            "</details>",
        ])
    parts.append("</section>")
    return "".join(parts)


def _drill_note_values(
    record: VocabularyRecord,
    card: PatternCard,
    form_note: str,
    examples: Sequence[ExampleSentence],
    label: str,
    deck_path: Path,
    audio_fields: Sequence[str] = (),
) -> list[str]:
    """Render and structurally guard one drill note's positional fields."""
    rich = bool(form_note or examples)
    support = (
        _drill_support_html(record, card, form_note, examples, audio_fields)
        if rich
        else html.escape(", ".join(card.examples))
    )
    source = (
        "form computed by janki; context supplied by deck and record"
        if rich
        else "computed by janki"
    )
    values = [
        html.escape(card.trigger),
        html.escape(card.result),
        html.escape(card.gloss),
        support,
        html.escape(source),
        html.escape(label),
    ]
    # Rich support carries deck-authored text, which record validation cannot
    # see. A separator there would shift every later positional field in the
    # installed note, so both validate and build refuse the same rendered row.
    fault = field_separator_fault(FIELDS, values)
    if fault is not None:
        raise PatternDeckError(
            f"{record.id}: the {fault} field contains U+001F, the separator "
            "Anki joins a note's fields with. Writing it would shift every "
            f"later field out of place: {deck_path}"
        )
    return values


def _resolve_drill_audio_fields(
    examples: Sequence[ExampleSentence],
    *,
    project_config: ProjectConfig,
    deck_path: Path,
    record_id: str,
    media_files: list[str],
    claimed_media: dict[str, tuple[str, str]],
    allowed_missing_media: frozenset[Path] = frozenset(),
) -> list[str]:
    """Resolve and claim every authored clip exactly as the package will."""
    audio_fields: list[str] = []
    warnings: list[str] = []
    for example in examples:
        if not example.audio:
            audio_fields.append("")
            continue
        found = _resolve_media(
            example.audio,
            media_dir=project_config.media_dir,
            deck_dir=deck_path.parent,
            record_id=record_id,
            label=f"{example.register.capitalize()} drill example audio",
            media_files=media_files,
            warnings=warnings,
            claimed=claimed_media,
            allowed_missing=allowed_missing_media,
        )
        audio_fields.append(f"[sound:{found.name}]" if found is not None else "")
    if warnings:
        raise PatternDeckError("; ".join(warnings))
    return audio_fields


def conjugation_media_paths_for_section(
    deck_path: Path,
    project_config: ProjectConfig,
    section: Mapping[str, Any],
    records: Sequence[VocabularyRecord],
    form: str,
    *,
    allowed_missing_media: frozenset[Path] = frozenset(),
) -> tuple[Path, ...]:
    """Resolve every file a conjugation package will consume.

    A finish plan may name provider output that does not exist yet.  Its exact
    future path is passed in ``allowed_missing_media``; ordinary builds pass no
    exception and therefore retain the exporter's missing-media refusal.  The
    resolution and basename-collision rules are the same ones used while
    rendering the package.
    """

    target = deck_path.resolve()
    shipping = shipping_records_for_section(target, section, records, form)
    examples_by_id = _drill_examples(section, shipping, target, form)
    cards = drill_cards(shipping, form)
    media_files: list[str] = []
    claimed_media: dict[str, tuple[str, str]] = {}
    for _card, record_id in cards:
        _resolve_drill_audio_fields(
            examples_by_id.get(record_id, ()),
            project_config=project_config,
            deck_path=target,
            record_id=record_id,
            media_files=media_files,
            claimed_media=claimed_media,
            allowed_missing_media=allowed_missing_media,
        )
    return tuple(sorted({Path(path) for path in media_files}, key=str))


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

    What the build's gates get to judge. Validating the whole collection
    instead refused a build over records the deck structurally cannot ship — a
    noun has no verb class, so no drill card could ever carry it — and over
    records the deck's own `exclude_ids` had deliberately held back.
    """
    return shipping_records_for_section(
        deck_path,
        _deck_section(deck_path),
        records,
        form,
    )


def shipping_records_for_section(
    deck_path: Path,
    section: Mapping[str, Any],
    records: Sequence[VocabularyRecord],
    form: str = "",
) -> list[VocabularyRecord]:
    """Filter one collection using an already-parsed exact deck snapshot."""

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
    return collection_for_section(deck_path, project_config, _deck_section(deck_path))


def collection_for_section(
    deck_path: Path,
    project_config: ProjectConfig,
    section: Mapping[str, Any],
) -> Path:
    """Resolve the collection named by an already-parsed exact deck snapshot."""

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
        filters_valid = True
        for key in ("include_ids", "exclude_ids"):
            try:
                _deck_string_set(deck_config, key, deck_path)
            except DataError as exc:
                filters_valid = False
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
                if filters_valid:
                    problems.extend(
                        _drill_set_problems(
                            deck_path, path, form, project_config
                        )
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
    else:
        # The refusals a *well-formed* deck can still hit, so the card set
        # here is the one the build produces and validate can answer for it.
        try:
            cards_for(entry)
        except PatternDeckError as exc:
            problems.append(str(exc))
        else:
            problems.extend(_card_set_problems(entry, document))
    return problems


def _drill_set_problems(
    deck_path: Path,
    collection: Path,
    form: str,
    project_config: ProjectConfig,
) -> list[str]:
    """The build's own emptiness refusal, asked of the same inputs.

    A collection this deck names and janki cannot read *is* this deck's problem.
    Nothing else in a `validate` sweep opens it — a drill deck's `source:` may
    name a file no other deck references — so swallowing that error left the one
    command whose job is catching a broken deck first saying "0 error(s)" over a
    file the next build refuses.

    ``deck_problems`` calls this only after parsing the filters itself. That
    structurally prevents duplicate filter messages without guessing which
    error occurred from words that may also appear in a rich-drill refusal.
    """
    if form not in CONJUGATION_FORMS:
        return []
    try:
        records = load_records(collection)
    except JankiError as exc:
        return [str(exc)]
    try:
        shipping = shipping_records(deck_path, records, form)
        deck_config = _deck_section(deck_path)
        form_note = _drill_form_note(deck_config, deck_path)
        examples_by_id = _drill_examples(deck_config, shipping, deck_path, form)
        cards = drill_cards(shipping, form)
        record_by_id = {record.id: record for record in shipping}
        label = form.replace("_", " ")
        media_files: list[str] = []
        claimed_media: dict[str, tuple[str, str]] = {}
        for card, record_id in cards:
            examples = examples_by_id.get(record_id, ())
            audio_fields = _resolve_drill_audio_fields(
                examples,
                project_config=project_config,
                deck_path=deck_path,
                record_id=record_id,
                media_files=media_files,
                claimed_media=claimed_media,
            )
            _drill_note_values(
                record_by_id[record_id],
                card,
                form_note,
                examples,
                label,
                deck_path,
                audio_fields,
            )
    except JankiError as exc:
        return [str(exc)]
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
    *,
    output_expected_revision: str | None = None,
    output_expected_identity: tuple[int, int] | None = None,
    output_expected_absent: bool = False,
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

    cards = cards_for(entry)
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
        values = [
            html.escape(card.trigger),
            html.escape(card.result),
            html.escape(card.gloss),
            "<br>".join(html.escape(item) for item in card.examples),
            html.escape(document),
            "Rule",
        ]
        # A rule card's values come from `data/patterns.json` and the deck file
        # — never from a record — so no record-level validation sees them.
        fault = field_separator_fault(FIELDS, values)
        if fault is not None:
            raise PatternDeckError(
                f"{document}: {card.identity}'s {fault} field contains "
                "U+001F, the separator Anki joins a note's fields with. "
                "Writing it would shift every later field out of place."
            )
        deck.add_note(
            genanki.Note(
                model=model,
                fields=values,
                guid=genanki.guid_for(f"pattern:{document}:{card.identity}"),
            )
        )

    filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
    target = output_path or (project_config.dist_dir / filename)
    target = Path(os.path.abspath(os.fspath(target)))
    _write_package_atomic(
        genanki.Package(deck),
        target,
        expected_revision=output_expected_revision,
        expected_identity=output_expected_identity,
        expected_absent=output_expected_absent,
    )
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
    *,
    output_expected_revision: str | None = None,
    output_expected_identity: tuple[int, int] | None = None,
    output_expected_absent: bool = False,
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
    shipping = shipping_records(deck_path, records, form)
    # The same internal backstop build_deck has (exporters/anki.py): a drill
    # card carries the expression, reading and a meaning straight off the
    # record, and without this an `--output` build — which skips the CLI's
    # shipping gates by design — packaged records that fail validation with
    # zero checks anywhere.
    issues = validate_records(list(shipping), deck_path)
    if has_errors(issues):
        raise PatternDeckError(refusal_text(deck_path.name, issues))
    form_note = _drill_form_note(deck_config, deck_path)
    examples_by_id = _drill_examples(deck_config, shipping, deck_path, form)
    cards = drill_cards(shipping, form)
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
    record_by_id = {record.id: record for record in shipping}
    media_files: list[str] = []
    claimed_media: dict[str, tuple[str, str]] = {}
    for card, record_id in cards:
        examples = examples_by_id.get(record_id, ())
        audio_fields = _resolve_drill_audio_fields(
            examples,
            project_config=project_config,
            deck_path=deck_path,
            record_id=record_id,
            media_files=media_files,
            claimed_media=claimed_media,
        )
        values = _drill_note_values(
            record_by_id[record_id],
            card,
            form_note,
            examples,
            label,
            deck_path,
            audio_fields,
        )
        deck.add_note(
            genanki.Note(
                model=model,
                fields=values,
                # Keyed on the record and the form, never the text: correcting a
                # reading rewrites the front, and a GUID that moved with it would
                # orphan the card's review history.
                guid=genanki.guid_for(f"drill:{form}:{record_id}"),
            )
        )

    filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
    target = output_path or (project_config.dist_dir / filename)
    package = genanki.Package(deck)
    package.media_files = sorted(set(media_files))
    target = Path(os.path.abspath(os.fspath(target)))
    _write_package_atomic(
        package,
        target,
        expected_revision=output_expected_revision,
        expected_identity=output_expected_identity,
        expected_absent=output_expected_absent,
    )
    return target, len(cards)
