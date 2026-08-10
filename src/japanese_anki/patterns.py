"""What a document is *teaching*, as opposed to which words it contains.

`janki extract` asks a page "which vocabulary is here". That is the wrong
question for half of what a learner is handed. A te-form chart contains almost
no vocabulary and is entirely about a *form*; a week's lecture slides contain
sixty words nobody glossed and are really about 〜んだ and つもり. Reading either
for its word list throws away the thing it was written to convey.

So this asks a different question — **what does this teach, and what is the
shape of it?** — and the answer is a set of patterns: a template a learner can
recognise (``〜てもいいですか``), what it does, and a sentence from the document
showing it.

**A model is the only thing that can read this.** There is no vocabulary slide
to parse, no heading convention, no consistent furigana; the intent is in prose,
in the ordering of slides, and in what the examples have in common. That is
inference, not extraction — so nothing here is trusted the way a dictionary
lookup is. Every pattern lands ``reviewed: false`` and is ignored until a human
says otherwise, the same rule this project applies to every other written-rather-
than-looked-up thing.

Two uses, and they are different:

* a **pattern document** — the te-form chart — is a deck in itself, one card per
  rule. Where janki can compute the answer (``conjugation.conjugate`` already
  produces te-forms), the extracted rule is *checked* against it rather than
  believed, which is the same dictionary-checks-writer shape the furigana path
  uses.
* a **lesson document** — a week's slides — is not cards. It says which grammar
  the learner is being taught right now, so `enrich --ai` can write examples in
  the patterns they are actually studying instead of whatever the model reaches
  for.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki import claude_client
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import han_character_class, normalize_identity_part
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import atomic_write_text

__all__ = [
    "DOCUMENT_KINDS",
    "Pattern",
    "PatternError",
    "PatternSet",
    "extract_patterns",
    "load_store",
    "save_store",
]


class PatternError(JankiError):
    """A document could not be read for what it teaches."""


#: What a document turns out to be. The model chooses; the caller does
#: different things with each, so a wrong guess is visible rather than silent.
DOCUMENT_KINDS: tuple[str, ...] = ("pattern", "lesson", "vocabulary", "unknown")


@dataclass(frozen=True, slots=True)
class Pattern:
    """One thing a document teaches."""

    #: The shape a learner recognises: ``〜てもいいですか``, ``う/つ/る → って``.
    template: str
    #: What it does, in a clause.
    gloss: str = ""
    #: Sentences from the document showing it, verbatim.
    examples: tuple[str, ...] = ()
    #: Where in the document it was found, for someone checking.
    where: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "gloss": self.gloss,
            "examples": list(self.examples),
            "where": self.where,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Pattern:
        # `examples` must be a list. A bare string iterates one character at a
        # time, so `"どうしたの?"` became ten single-character "examples" that
        # each looked like a sentence from the document.
        examples = raw.get("examples") or []
        if not isinstance(examples, list):
            raise PatternError(
                f"examples must be a list of sentences, got {type(examples).__name__}"
            )
        return cls(
            template=str(raw.get("template") or "").strip(),
            gloss=str(raw.get("gloss") or "").strip(),
            examples=tuple(str(e).strip() for e in examples if str(e).strip()),
            where=str(raw.get("where") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class PatternSet:
    """Everything one document teaches."""

    source: str
    kind: str = "unknown"
    title: str = ""
    patterns: tuple[Pattern, ...] = ()
    #: Inferred, not looked up. Until a human has read them, nothing uses them.
    reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "reviewed": self.reviewed,
            "patterns": [pattern.to_dict() for pattern in self.patterns],
        }

    @classmethod
    def from_dict(cls, source: str, raw: dict[str, Any]) -> PatternSet:
        listed = raw.get("patterns") or []
        if not isinstance(listed, list):
            raise PatternError(
                f"{source}: patterns must be a list, got {type(listed).__name__}"
            )
        for item in listed:
            # `str.get` raises AttributeError, which the CLI does not catch and
            # cannot format — a traceback instead of a message naming the file.
            if not isinstance(item, dict):
                raise PatternError(
                    f"{source}: each pattern must be an object, got "
                    f"{type(item).__name__}"
                )
        try:
            built = tuple(Pattern.from_dict(item) for item in listed)
        except PatternError as exc:
            raise PatternError(f"{source}: {exc}") from exc
        return cls(
            source=source,
            kind=str(raw.get("kind") or "unknown"),
            title=str(raw.get("title") or ""),
            patterns=built,
            reviewed=bool(raw.get("reviewed", False)),
        )


INSTRUCTIONS = """\
You read Japanese teaching material and report what it *teaches*, not which
words it contains.

Decide first what the document is:

* "pattern" — it is about one grammatical form, and its content is the rule for
  producing that form (a te-form chart, a conjugation table).
* "lesson" — a class handout or slide deck that teaches several grammar points
  and uses vocabulary in passing.
* "vocabulary" — mostly a word list. Report no patterns; another command reads
  those.
* "unknown" — anything else. Report no patterns rather than inventing some.

Then list the patterns it teaches. For each, give:

* template — the shape a learner would recognise, with 〜 where a word goes:
  〜てもいいですか, ない form + つもり, う/つ/る → って. Write it the way the
  document writes it.
* gloss — what the pattern does, in one clause.
* examples — sentences **from the document, verbatim**. Do not compose new ones;
  an example you cannot find on the page is one nobody can check.
* where — the slide or page it is taught on.

Report only what the document actually teaches. A pattern it merely uses in
passing is not a pattern it teaches, and a list padded with those is worse than
a short one: it will be used to write example sentences for a learner who has
not met them."""


@functools.cache
def pattern_schema() -> Any:
    """The shape a pattern extraction must take."""
    from pydantic import BaseModel, Field

    class ExtractedPattern(BaseModel):
        template: str = Field(description="The shape, with 〜 where a word goes.")
        gloss: str = Field(default="", description="What it does, in one clause.")
        examples: list[str] = Field(
            default_factory=list,
            description="Sentences from the document, verbatim.",
        )
        where: str = Field(default="", description="Slide or page number.")

    class Extraction(BaseModel):
        kind: str = Field(description="pattern, lesson, vocabulary, or unknown.")
        title: str = Field(default="", description="What the document calls itself.")
        patterns: list[ExtractedPattern] = Field(default_factory=list)

    return Extraction


def extract_patterns(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str = "",
    client: Any | None = None,
) -> PatternSet:
    """Read one document for what it teaches.

    Both incomplete outcomes are refused rather than salvaged, on the same
    reasoning `extract` uses: a truncated answer looks like a complete one with
    fewer patterns, and nothing downstream could tell the difference.
    """
    blocks = (
        claude_client.system_blocks(style_guide, INSTRUCTIONS)
        if style_guide
        else claude_client.system_blocks(INSTRUCTIONS)
    )
    call = claude_client.parse_call(
        model,
        blocks,
        [
            prepared.content_block(),
            {"type": "text", "text": (
                f"This is {prepared.origin_path.name}. What does it teach?"
            )},
        ],
        pattern_schema(),
        client,
        max_tokens=4000,
    )
    if call.stop_reason == "refusal":
        raise PatternError(
            f"The model declined to read {prepared.origin_path.name}"
            + (f": {call.refusal}" if call.refusal else "")
        )
    if call.parsed is None:
        raise PatternError(
            f"No usable answer for {prepared.origin_path.name} "
            f"(stopped: {call.stop_reason})"
        )

    kind = str(getattr(call.parsed, "kind", "") or "unknown").strip().lower()
    if kind not in DOCUMENT_KINDS:
        kind = "unknown"
    patterns = tuple(
        Pattern(
            template=str(getattr(item, "template", "") or "").strip(),
            gloss=str(getattr(item, "gloss", "") or "").strip(),
            examples=tuple(
                str(e).strip() for e in (getattr(item, "examples", []) or []) if str(e).strip()
            ),
            where=str(getattr(item, "where", "") or "").strip(),
        )
        for item in (getattr(call.parsed, "patterns", []) or [])
    )
    return PatternSet(
        source=prepared.origin_path.name,
        kind=kind,
        title=str(getattr(call.parsed, "title", "") or "").strip(),
        patterns=tuple(p for p in patterns if p.template),
    )


#: The kana a dictionary-form verb can end in. Used only to tell a verb pair
#: (``かう ⇨ かって``) from a rule shape (``う・つ・る → って``), never to decide
#: a conjugation — that is `conjugation.conjugate`'s job and it needs the group.
_DICTIONARY_ENDINGS = frozenset("うくぐすつぬぶむる")

#: ``A ⇨ B``, ``A → B``, ``A -> B``. The chart uses all three. Kanji as well as
#: kana: a chart is written 買う ⇨ 買って far more often than かう ⇨ かって, and
#: a hiragana-only class made the feature a no-op on its ordinary input — 行く,
#: the row a reader is most likely to have copied down wrong, included.
#: The kanji half comes from `identifiers.han_character_class`, which owns that
#: definition. A hand-written `一-龥` here missed 𠮟 — a word real exports carry
#: — and every Extension-A ideograph, so those rows matched nothing at all.
_WORD = rf"[ぁ-ゖーァ-ヺ{han_character_class()}]{{2,}}"
_PAIR = re.compile(rf"({_WORD})\s*(?:⇨|→|->|=>)\s*({_WORD})")

#: A row listing several verbs against several results — ``かう・まつ・とる ⇨
#: かって・まって・とって`` — cannot be paired positionally without guessing
#: which result belongs to which verb, and `_PAIR` would pair the last item
#: before the arrow with the first after it (とる ⇨ かって). Skipped, for the
#: same reason rule shapes are.
_LIST_SEPARATORS = frozenset("・、,，/／")

#: Groups `conjugate` knows. Tried in turn because the chart states its rule in
#: prose ("godan verbs ending in う, つ, or る"), and parsing that prose to pick
#: a group would be a guess about what the model wrote.
_GROUPS: tuple[str, ...] = ("godan", "ichidan", "kuru", "suru")

#: Which form a claim is *about*, read from how it ends, longest suffix first.
#: Only to make a disagreement message name the form the chart meant; agreement
#: is decided against the whole table, so a chart teaching た-forms, ない-forms
#: or anything else is checked rather than contradicted on every correct row.
_FORM_BY_ENDING: tuple[tuple[str, str], ...] = (
    ("なかった", "past_negative"),
    ("ない", "negative"),
    # `CONJUGATION_FORMS` has seven entries, and leaving these two out meant a
    # claim janki could disprove — `食べる ⇨ 食べれる`, ら抜き — was excused as
    # "janki computes no form with this ending", which is false and reads as
    # reassurance on a row the checker could have failed.
    ("られる", "potential"),
    ("れる", "passive"),
    ("える", "potential"),
    ("て", "te_form"),
    ("で", "te_form"),
    ("た", "past"),
    ("だ", "past"),
)


def _pairs_in(text: str) -> list[tuple[str, str]]:
    """Every complete ``verb ⇨ form`` claim on one line, or nothing.

    Split on list separators first, then match inside a segment, so a match can
    never span a separator. The rule is then simply **every segment must carry
    an arrow**:

    * ``くる ⇨ きて / する ⇨ して / いく ⇨ いって`` — three segments, three
      arrows. Three complete claims, all checkable.
    * ``かう・まつ・とる ⇨ かって・まって・とって`` — six segments, one arrow.
      The arrow-bearing segment spans a list boundary, so which result belongs
      to which verb is a guess. Nothing here is checkable.
    * ``かう・まつ ⇨ かって`` — the asymmetric row a model produces when it drops
      a result or a line breaks. Two segments, one arrow: also a guess, and
      pairing まつ with かって would be janki's error, not the chart's.

    Written as a segment rule after two attempts at looking only at the
    characters adjacent to a match. Requiring a separator on *either* side threw
    away complete pairs written `A ⇨ B、C ⇨ D`; requiring one on *both* sides
    threw away every interior pair of a three-pair line, and let the asymmetric
    row through as a false disagreement. Neither is a property of one character.
    """
    segments = [
        segment
        for segment in re.split(f"[{re.escape(''.join(_LIST_SEPARATORS))}]", text)
        if segment.strip()
    ]
    found: list[tuple[str, str]] = []
    for segment in segments:
        matches = _PAIR.findall(segment)
        if matches:
            found.extend(matches)
            continue
        # An arrow-less segment only makes the line ambiguous if it could have
        # been one of the paired items. `う` and `つ` in `う・つ・る → って` are
        # candidates; the prose in `う, つ, る verbs: かう ⇨ かいて` and the gloss
        # in `買う ⇨ 買って, to buy` are not, and refusing on those threw away a
        # garbled claim entirely — reported as "all 1 agree", which is the
        # vanishing-under-an-all-clear this whole function exists to prevent.
        if re.search(_WORD, segment):
            return []
    return found


#: Endings janki computes no form for. `CONJUGATION_FORMS` has no polite family
#: at all, and reading `食べました` as a *past* because it ends in た reported the
#: commonest polite chart there is as wrong. Checked before `_FORM_BY_ENDING`,
#: longest first, so ませんでした is not read as ました.
_NO_TABLE_ENDINGS: tuple[str, ...] = (
    "ませんでした", "ましょう", "ませんか", "ましたら",
    "ません", "ました", "ます", "まして",
    "たかった", "たくない", "たい",
    "れば", "けれ", "たら", "なら",
)


def _claimed_form(claimed: str, verb: str = "") -> str:
    """Which form a claim is about, or ``""`` when janki computes no such form.

    The polite endings are matched against the claim's tail *beyond the verb's
    own stem*, not against the whole string. Matched against the whole string
    they collided with verbs whose stem ends the same way: `だます ⇨ だしまして`
    — a plausible transcription garble of だまして, which janki can disprove —
    ends in まして and was excused, and so was every ます-verb's past against
    ました (済ます, 冷ます, 覚ます, 励ます).
    """
    stem = verb[:-1] if verb else ""
    tail = claimed[len(stem):] if stem and claimed.startswith(stem) else ""
    if tail and any(tail.endswith(ending) for ending in _NO_TABLE_ENDINGS):
        return ""
    for ending, form in _FORM_BY_ENDING:
        if claimed.endswith(ending):
            return form
    return ""


#: The kinds whose worked examples are checked. A lesson deck's "examples" are
#: sentences from a slide, not conjugation claims.
CHECKABLE_KINDS: tuple[str, ...] = ("pattern",)


@dataclass(frozen=True, slots=True)
class RuleCheck:
    """One ``verb ⇨ inflected form`` claim from a document, against janki."""

    template: str
    #: The dictionary form the document gave.
    verb: str
    #: The inflected form the document claims for it.
    claimed: str
    #: The verb group under which `conjugate` agrees, or ``""`` if none does.
    group: str = ""
    #: Which form it turned out to be — ``te_form``, ``past``, … — when agreed.
    form: str = ""
    #: What `conjugate` produces instead, per group, when nothing agreed.
    computed: tuple[str, ...] = ()
    #: Why this row was found but not examined, or ``""`` if it was. A row janki
    #: has no opinion about is neither agreement nor disagreement, and dropping
    #: it silently let a claim disappear under an all-clear.
    held_back: str = ""

    @property
    def agrees(self) -> bool:
        return bool(self.group)

    @property
    def examined(self) -> bool:
        return not self.held_back


def check_pattern_rules(entry: PatternSet) -> tuple[RuleCheck, ...]:
    """Check a document's worked examples against :func:`conjugation.conjugate`.

    A pattern document is inference like everything else here, so its rules are
    *checked* rather than believed — the same dictionary-checks-writer shape the
    furigana path uses. janki already conjugates, so where the chart shows its
    work (``買う ⇨ 買って``, ``くる ⇨ きて``) there is something to check it
    against, and a rule the model garbled shows up as a disagreement instead of
    becoming a card.

    The claim is checked against the **whole** conjugation table, not against
    ``te_form``. A chart teaching the た-form repeats the て-form chart's rule
    shapes verbatim (``う・つ・る → った``), so a checker that assumed て would
    have contradicted every correct row of it and exited non-zero.

    Four things are deliberately not checked, each because checking would mean
    guessing:

    * **Rule shapes.** ``う・つ・る → って`` is the shape of a rule and
      ``く → いて`` names an ending; inventing a verb to test them with would be
      checking janki against itself.
    * **Multi-verb rows.** ``かう・まつ・とる ⇨ かって・まって・とって`` cannot
      be paired positionally without deciding which result belongs to which
      verb.
    * **Words janki refuses.** ``conjugate`` returns nothing for ゆく because
      both ゆいて and 行って are attested — janki has no opinion, which is not
      the same as disagreeing, and reporting it as one would fail a chart that
      is right.
    * **The verb group.** The chart states it in English prose; reading that
      would be a guess. Every group is tried, and agreement under any of them is
      the claim — weaker than "and it is godan", on purpose.

    That last one has a cost worth naming, because it looks like a bug from the
    outside: a form that is wrong *for the verb's real group* can be right for
    another, and this agrees with it. ``食べる ⇨ 食べれる`` is ら抜き and janki
    computes 食べられる — but 食べる run through the godan rules gives 食べれる,
    so it passes. Narrowing that needs the group, which only the collection
    knows and the chart does not say; the alternative is guessing which class a
    chart's example verb belongs to, and a checker that guesses is worse than
    one with a stated blind spot.
    """
    from japanese_anki.conjugation import conjugate

    if entry.kind not in CHECKABLE_KINDS:
        return ()

    checks: list[RuleCheck] = []
    for pattern in entry.patterns:
        seen: set[tuple[str, str]] = set()
        for text in (pattern.template, *pattern.examples):
            # Parentheticals carry the English gloss — "(to buy)", "(う ending
            # changes to って)" — and the second of those holds a → of its own.
            stripped = re.sub(r"[(（][^)）]*[)）]", " ", text)
            # Normalized like every other Japanese-text path here. A decomposed
            # ぐ is く plus U+3099, which is outside the word class, so
            # およぐ ⇨ およいで matched nothing at all.
            stripped = normalize_identity_part(stripped) or stripped
            for match in _pairs_in(stripped):
                verb, claimed = match
                if verb[-1] not in _DICTIONARY_ENDINGS:
                    continue
                if (verb, claimed) in seen:
                    continue
                seen.add((verb, claimed))
                tables = [
                    (group, table)
                    for group in _GROUPS
                    if (table := conjugate(verb, verb, group))
                ]
                if not tables:
                    # janki declines this word entirely — ゆく, or a compound of
                    # an irregular. Recorded rather than dropped: a row nobody
                    # examined must not vanish under an all-clear.
                    checks.append(RuleCheck(
                        template=pattern.template, verb=verb, claimed=claimed,
                        held_back="janki has no conjugation for this word",
                    ))
                    continue
                agreed = form = ""
                for group, table in tables:
                    for name, value in table.items():
                        if value == claimed:
                            agreed, form = group, name
                            break
                    if agreed:
                        break
                wanted = _claimed_form(claimed, verb)
                # A claim about a form janki has no table for is no opinion, the
                # same as a word `conjugate` refuses. `CONJUGATION_FORMS` stops
                # at seven, so a ます / たい / ば / volitional chart — a `pattern`
                # document by the extractor's own definition — matched nothing
                # in the table and was reported as *wrong*, with a te-form
                # offered as the correction.
                if not agreed and not wanted:
                    checks.append(RuleCheck(
                        template=pattern.template, verb=verb, claimed=claimed,
                        held_back="janki computes no form with this ending",
                    ))
                    continue
                checks.append(
                    RuleCheck(
                        template=pattern.template,
                        verb=verb,
                        claimed=claimed,
                        group=agreed,
                        form=form,
                        computed=() if agreed else tuple(
                            f"{group}: {table[wanted]}"
                            for group, table in tables
                            if wanted in table
                        ),
                    )
                )
    return tuple(checks)


def load_store(path: Path) -> dict[str, PatternSet]:
    """Every document janki has read, keyed by its file name."""
    file = Path(path)
    if not file.exists():
        return {}
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PatternError(f"Could not read {file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PatternError(f"{file} must hold a JSON object keyed by document name")
    # Refused, not skipped. `save_store` rewrites the whole file from what was
    # loaded, so dropping an entry it could not read would erase that document
    # from the committed store on the next command that writes — silently, and
    # reported as success. `kanji.load_store` refuses for the same reason.
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise PatternError(
                f"{file}: entry for {str(name)!r} must be an object, got "
                f"{type(value).__name__}"
            )
    return {str(name): PatternSet.from_dict(str(name), value) for name, value in raw.items()}


def save_store(path: Path, store: dict[str, PatternSet]) -> None:
    """Write the pattern file, sorted, so a re-read produces no diff by itself."""
    payload = {name: store[name].to_dict() for name in sorted(store)}
    atomic_write_text(
        Path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


#: The only kind that steers a sentence. A ``pattern`` document is a conjugation
#: chart — its rows are production rules, not sentence patterns — and a
#: ``vocabulary`` document teaches no grammar at all.
STEERING_KINDS: tuple[str, ...] = ("lesson",)


def reviewed_patterns(
    store: dict[str, PatternSet],
    sources: Iterable[str] = (),
    kinds: Iterable[str] = STEERING_KINDS,
) -> list[Pattern]:
    """Patterns a human has signed off, optionally from named documents only.

    Only lesson documents, which is the split this module is built around: a
    lesson says which grammar the learner is being taught this week, and that is
    a sensible thing to prefer in a sentence. A ``pattern`` document is the
    te-form chart, whose rows are ``く → いて`` and ``む・ぶ・ぬ → んで`` — asking
    a model to "prefer ``く → いて`` when a sentence can use one naturally" is
    not a coherent instruction, and eight such rows drowned the six real ones
    while biasing every generated example toward the て-form.

    Unreviewed sets are skipped rather than warned about here: this is asked on
    the way into writing a sentence, and a warning per record would drown the
    diff it is trying to help someone read.
    """
    wanted = {str(name) for name in sources}
    steering = {str(kind) for kind in kinds}
    found: list[Pattern] = []
    for name, entry in sorted(store.items()):
        if not entry.reviewed:
            continue
        if entry.kind not in steering:
            continue
        if wanted and name not in wanted:
            continue
        found.extend(entry.patterns)
    return found


def format_patterns(patterns: Sequence[Pattern]) -> str:
    """The block `enrich --ai` puts in a prompt, or ``""``."""
    if not patterns:
        return ""
    lines = [
        "The learner is currently studying these patterns. Prefer them when a "
        "sentence can use one naturally; never force one that does not fit:",
    ]
    for pattern in patterns:
        gloss = f" — {pattern.gloss}" if pattern.gloss else ""
        lines.append(f"* {pattern.template}{gloss}")
    return "\n".join(lines)


@dataclass(slots=True)
class ExtractionSummary:
    """What one `janki patterns` run did, for the caller to report."""

    read: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
