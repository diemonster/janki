"""Shared structured-output shapes for the two card-writing AI paths.

``extract`` reads source material and ``enrich --ai`` reads a bare vocabulary
record.  The surrounding evidence differs, but the card content they return
must not: meanings, examples, and a usage note have one schema here.  Keeping
the model classes behind functions preserves the project's optional ``ai``
dependency — importing janki for a build must not import Pydantic.

The field descriptions are intentionally terse.  They are serialized into the
JSON schema and therefore reach the model, but the substantive instructions
belong in ``prompts/`` where a person can read and edit the whole asking.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "generated_example_schema",
    "adapt_rich_card",
    "RichCardContent",
    "rich_card_schema",
    "source_pattern_schema",
]


@dataclass(frozen=True, slots=True)
class RichCardContent:
    """Canonical card content decoded from either rich model response."""

    meanings: tuple[str, ...]
    examples: tuple[Any, ...]
    usage_notes: str
    romaji_rejected: tuple[str, ...]


def adapt_rich_card(value: Any) -> RichCardContent:
    """Apply the one structural decoding contract shared by both AI paths.

    This does not judge Japanese.  It trims wire strings, removes duplicate
    gloss entries, maps the schema's speech-level label to the card field, and
    settles the mechanically derived romaji against the reading stated by the
    answer's furigana.  Source routing and whether an identity is already known
    must not change any of those results.
    """
    from japanese_anki import qc
    from japanese_anki.models import ExampleSentence

    meanings = tuple(
        dict.fromkeys(
            text
            for item in (getattr(value, "meanings", []) or [])
            if (text := str(item or "").strip())
        )
    )
    examples: list[ExampleSentence] = []
    rejected: list[str] = []
    for item in getattr(value, "examples", []) or []:
        japanese = str(getattr(item, "japanese", "") or "").strip()
        if not japanese:
            continue
        example = ExampleSentence(
            japanese=japanese,
            furigana=str(getattr(item, "furigana", "") or "").strip(),
            romaji=str(getattr(item, "romaji", "") or "").strip(),
            english=str(getattr(item, "english", "") or "").strip(),
            register=str(getattr(item, "speech_level", "") or "").strip().lower(),
        )
        settled, warning = qc.settle_example_romaji(example)
        examples.append(settled)
        if warning:
            rejected.append(warning)
    return RichCardContent(
        meanings=meanings,
        examples=tuple(examples),
        usage_notes=str(getattr(value, "usage_notes", "") or "").strip(),
        romaji_rejected=tuple(rejected),
    )


@functools.cache
def generated_example_schema() -> Any:
    """The example shape shared byte-for-byte by both card-writing paths."""
    from pydantic import BaseModel, ConfigDict, Field

    class GeneratedExample(BaseModel):
        model_config = ConfigDict(extra="forbid")

        japanese: str = Field(description="Japanese sentence.")
        speech_level: Literal["polite", "casual"] = Field(
            description="Speech level."
        )
        furigana: str = Field(default="", description="Sentence with Anki furigana.")
        romaji: str = Field(default="", description="Sentence in Hepburn romaji.")
        english: str = Field(default="", description="Natural English translation.")

    return GeneratedExample


@functools.cache
def rich_card_schema() -> Any:
    """Meanings and teaching content common to source and bare-word answers."""
    from pydantic import BaseModel, ConfigDict, Field

    GeneratedExample = generated_example_schema()

    class RichCard(BaseModel):
        model_config = ConfigDict(extra="forbid")

        meanings: list[str] = Field(default_factory=list, description="English glosses.")
        examples: list[GeneratedExample] = Field(default_factory=list)
        usage_notes: str = Field(default="", description="Usage note.")

    return RichCard


@functools.cache
def source_pattern_schema() -> Any:
    """One grammar or usage pattern reported while a source is already open."""
    from pydantic import BaseModel, ConfigDict, Field

    class SourcePattern(BaseModel):
        model_config = ConfigDict(extra="forbid")

        template: str = Field(description="Pattern template.")
        gloss: str = Field(default="", description="English gloss.")
        examples: list[str] = Field(default_factory=list, description="Source examples.")
        where: str = Field(default="", description="Source location.")

    return SourcePattern
