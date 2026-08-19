"""What a source document teaches, alongside the words extracted from it.

`janki extract` asks a page "which vocabulary is here". That is the wrong
question for half of what a learner is handed. A te-form chart contains almost
no vocabulary and is entirely about a *form*; a week's lecture slides contain
sixty words nobody glossed and are really about 〜んだ and つもり. Reading either
for its word list throws away the thing it was written to convey.

The rich source response asks both questions once.  This module owns the
durable pattern data, its human review state, and the structural helpers that
turn reviewed patterns into cards or labeled context for bare-word enrichment.
It does not make a second paid model call.

**A model is the only thing that can read this.** There is no vocabulary slide
to parse, no heading convention, no consistent furigana; the intent is in prose,
in the ordering of slides, and in what the examples have in common. That is
inference, not extraction — so nothing here is trusted the way a dictionary
lookup is. Every pattern lands ``reviewed: false`` and is ignored until a human
says otherwise, the same rule this project applies to every other written-rather-
than-looked-up thing.

Two uses, and they are different:

* a **pattern document** — the te-form chart — is a deck in itself, one card per
  rule, gated by the human ``reviewed:`` mark. Since M8.3 nothing re-checks a
  reviewed chart: its rules and worked examples ship as it states them
  (DESIGN.md — janki's logic enriches the card, it never audits the model).
* a **lesson document** — a week's slides — is not cards. It says which grammar
  the learner is being taught right now, so `enrich --ai` can write examples in
  the patterns they are actually studying instead of whatever the model reaches
  for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import han_character_class, normalize_identity_part
from japanese_anki.io import atomic_write_text

__all__ = [
    "CHECKABLE_KINDS",
    "DOCUMENT_KINDS",
    "LIST_SEPARATORS",
    "Pattern",
    "PatternError",
    "PatternSet",
    "format_patterns",
    "load_store",
    "reviewed_patterns",
    "save_store",
    "verb_pairs_in",
    "with_prompt_provenance",
    "worked_examples_in",
]


class PatternError(JankiError):
    """Durable source-pattern data could not be read or written."""


#: What a document turns out to be. The model chooses; the caller does
#: different things with each, so a wrong guess is visible rather than silent.
DOCUMENT_KINDS: tuple[str, ...] = ("pattern", "lesson", "vocabulary", "unknown")


def _review_run_id(source: str, raw: dict[str, Any]) -> str | None:
    """Read one optional canonical UUIDv4 lineage marker from a pattern set."""
    if "review_run_id" not in raw:
        return None
    value = raw.get("review_run_id")
    if not isinstance(value, str):
        raise PatternError(
            f"{source}: [review-run-id-invalid] review_run_id must be canonical "
            "lowercase UUIDv4 text"
        )
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise PatternError(
            f"{source}: [review-run-id-invalid] review_run_id must be canonical "
            "lowercase UUIDv4 text"
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise PatternError(
            f"{source}: [review-run-id-invalid] review_run_id must be canonical "
            "lowercase UUIDv4 text"
        )
    return value


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
    #: Exact prompt inputs that produced this inference. Historical entries
    #: predate prompt provenance and therefore carry an empty mapping.
    prompt_provenance: dict[str, Any] = field(default_factory=dict)
    #: One paid answer's durable identity. Historical entries predate it.
    review_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "title": self.title,
            "reviewed": self.reviewed,
            "patterns": [pattern.to_dict() for pattern in self.patterns],
        }
        if self.prompt_provenance:
            value["prompt_provenance"] = dict(self.prompt_provenance)
        if self.review_run_id is not None:
            value["review_run_id"] = self.review_run_id
        return value

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
        provenance = raw.get("prompt_provenance") or {}
        if not isinstance(provenance, dict):
            raise PatternError(
                f"{source}: prompt_provenance must be an object, got "
                f"{type(provenance).__name__}"
            )
        return cls(
            source=source,
            kind=str(raw.get("kind") or "unknown"),
            title=str(raw.get("title") or ""),
            patterns=built,
            reviewed=bool(raw.get("reviewed", False)),
            prompt_provenance={str(key): value for key, value in provenance.items()},
            review_run_id=_review_run_id(source, raw),
        )


def with_prompt_provenance(
    pattern_set: PatternSet, provenance: dict[str, Any]
) -> PatternSet:
    """Bind a source inference to the exact request that produced it."""
    return PatternSet(
        source=pattern_set.source,
        kind=pattern_set.kind,
        title=pattern_set.title,
        patterns=pattern_set.patterns,
        reviewed=pattern_set.reviewed,
        prompt_provenance=dict(provenance),
        review_run_id=pattern_set.review_run_id,
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
LIST_SEPARATORS = frozenset("・、,，/／")

def verb_pairs_in(text: str) -> list[tuple[str, str]]:
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
        for segment in re.split(f"[{re.escape(''.join(LIST_SEPARATORS))}]", text)
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


#: The kinds whose worked examples are checked. A lesson deck's "examples" are
#: sentences from a slide, not conjugation claims.
CHECKABLE_KINDS: tuple[str, ...] = ("pattern",)


def worked_examples_in(entry: PatternSet) -> dict[str, list[tuple[str, str]]]:
    """Each row's worked examples, as the chart states them.

    ``template -> [(verb, "verb ⇨ claimed")]``, scanned with the same
    stripping every other reader of a chart applies: parentheticals carry the
    English gloss, and the text is normalized so a decomposed dakuten still
    reads as its verb.

    This is notation reading, nothing more. M8.3 deleted the checker that used
    to adjudicate these pairs against janki's own conjugation tables — the
    dictionary-checks-writer shape M7.6V retired, rebuilt for charts. A chart
    reaches cards only through the human ``reviewed:`` gate, and a reviewed
    chart's worked examples ship as it states them.
    """
    if entry.kind not in CHECKABLE_KINDS:
        return {}
    by_template: dict[str, list[tuple[str, str]]] = {}
    for pattern in entry.patterns:
        seen: set[tuple[str, str]] = set()
        for text in (pattern.template, *pattern.examples):
            stripped = re.sub(r"[(（][^)）]*[)）]", " ", text)
            stripped = normalize_identity_part(stripped) or stripped
            for verb, claimed in verb_pairs_in(stripped):
                if verb[-1] not in _DICTIONARY_ENDINGS:
                    continue
                if (verb, claimed) in seen:
                    continue
                seen.add((verb, claimed))
                by_template.setdefault(pattern.template, []).append(
                    (verb, f"{verb} ⇨ {claimed}")
                )
    return by_template

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
    """Reviewed lesson-pattern data for a bare-word user turn, or ``""``."""
    if not patterns:
        return ""
    lines = ["Reviewed lesson patterns:"]
    for pattern in patterns:
        gloss = f" — {pattern.gloss}" if pattern.gloss else ""
        lines.append(f"* {pattern.template}{gloss}")
    return "\n".join(lines)
