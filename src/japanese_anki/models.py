from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from japanese_anki.identifiers import stable_record_id


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


@dataclass(slots=True)
class ExampleSentence:
    japanese: str = ""
    furigana: str = ""
    romaji: str = ""
    english: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExampleSentence:
        data = data or {}
        return cls(
            japanese=str(data.get("japanese", "")).strip(),
            furigana=str(data.get("furigana", "")).strip(),
            romaji=str(data.get("romaji", "")).strip(),
            english=str(data.get("english", "")).strip(),
        )


@dataclass(slots=True)
class SourceReference:
    type: str = "manual"
    imported_from: str = ""
    row: int | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourceReference:
        data = data or {}
        row_value = data.get("row")
        try:
            row = int(row_value) if row_value not in (None, "") else None
        except (TypeError, ValueError):
            row = None
        raw_fields = {
            str(key): str(value)
            for key, value in (data.get("raw_fields") or {}).items()
        }
        return cls(
            type=str(data.get("type", "manual")).strip() or "manual",
            imported_from=str(data.get("imported_from", "")).strip(),
            row=row,
            raw_fields=raw_fields,
        )


@dataclass(slots=True)
class VocabularyRecord:
    id: str
    expression: str
    reading: str = ""
    furigana: str = ""
    romaji: str = ""
    meanings: list[str] = field(default_factory=list)
    part_of_speech: str = ""
    verb_group: str = ""
    transitivity: str = ""
    examples: list[ExampleSentence] = field(default_factory=list)
    conjugations: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    usage_notes: str = ""
    audio: str = ""
    image: str = ""
    source: SourceReference = field(default_factory=SourceReference)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularyRecord:
        expression = str(data.get("expression", "")).strip()
        reading = str(data.get("reading", "")).strip()
        record_id = str(data.get("id", "")).strip() or stable_record_id(expression, reading)
        examples_value = data.get("examples") or []
        if isinstance(examples_value, dict):
            examples_value = [examples_value]
        conjugations = {
            str(key).strip(): str(value).strip()
            for key, value in (data.get("conjugations") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        return cls(
            id=record_id,
            expression=expression,
            reading=reading,
            furigana=str(data.get("furigana", "")).strip(),
            romaji=str(data.get("romaji", "")).strip(),
            meanings=_string_list(data.get("meanings")),
            part_of_speech=str(data.get("part_of_speech", "")).strip(),
            verb_group=str(data.get("verb_group", "")).strip(),
            transitivity=str(data.get("transitivity", "")).strip(),
            examples=[ExampleSentence.from_dict(item) for item in examples_value],
            conjugations=conjugations,
            tags=_string_list(data.get("tags")),
            usage_notes=str(data.get("usage_notes", "")).strip(),
            audio=str(data.get("audio", "")).strip(),
            image=str(data.get("image", "")).strip(),
            source=SourceReference.from_dict(data.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def first_example(self) -> ExampleSentence:
        return self.examples[0] if self.examples else ExampleSentence()
