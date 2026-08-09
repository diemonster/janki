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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki import claude_client
from japanese_anki.errors import JankiError
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
        return cls(
            template=str(raw.get("template") or "").strip(),
            gloss=str(raw.get("gloss") or "").strip(),
            examples=tuple(str(e).strip() for e in (raw.get("examples") or []) if str(e).strip()),
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
        return cls(
            source=source,
            kind=str(raw.get("kind") or "unknown"),
            title=str(raw.get("title") or ""),
            patterns=tuple(Pattern.from_dict(p) for p in (raw.get("patterns") or [])),
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
    return {
        str(name): PatternSet.from_dict(str(name), value)
        for name, value in raw.items()
        if isinstance(value, dict)
    }


def save_store(path: Path, store: dict[str, PatternSet]) -> None:
    """Write the pattern file, sorted, so a re-read produces no diff by itself."""
    payload = {name: store[name].to_dict() for name in sorted(store)}
    atomic_write_text(
        Path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def reviewed_patterns(store: dict[str, PatternSet], sources: Iterable[str] = ()) -> list[Pattern]:
    """Patterns a human has signed off, optionally from named documents only.

    Unreviewed sets are skipped rather than warned about here: this is asked on
    the way into writing a sentence, and a warning per record would drown the
    diff it is trying to help someone read.
    """
    wanted = {str(name) for name in sources}
    found: list[Pattern] = []
    for name, entry in sorted(store.items()):
        if not entry.reviewed:
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
