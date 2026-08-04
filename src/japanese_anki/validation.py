from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from japanese_anki.models import VocabularyRecord


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: str
    message: str
    record_id: str = ""
    source: str = ""

    def format(self) -> str:
        location = self.source
        if self.record_id:
            location = f"{location}:{self.record_id}" if location else self.record_id
        prefix = f"[{self.level.upper()}]"
        return f"{prefix} {location}: {self.message}" if location else f"{prefix} {self.message}"


def _contains_kanji(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def validate_record(record: VocabularyRecord, source: str = "") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(level: str, message: str) -> None:
        issues.append(
            ValidationIssue(level=level, message=message, record_id=record.id, source=source)
        )

    if not record.id:
        add("error", "missing stable ID")
    if not record.expression:
        add("error", "missing expression")
    if _contains_kanji(record.expression) and not record.reading:
        add("error", "expression contains kanji but reading is missing")
    if not record.meanings:
        add("error", "at least one English meaning is required")
    if record.furigana and record.furigana.count("[") != record.furigana.count("]"):
        add("error", "furigana brackets are unbalanced")
    if record.furigana and not record.reading:
        add("warning", "furigana is present but the plain reading is empty")
    if record.verb_group and not record.part_of_speech:
        add("warning", "verb group is present but part of speech is empty")
    for index, example in enumerate(record.examples, start=1):
        if example.japanese and not example.english:
            add("warning", f"example {index} has Japanese but no English translation")
        if example.english and not example.japanese:
            add("warning", f"example {index} has English but no Japanese sentence")
        if example.furigana and example.furigana.count("[") != example.furigana.count("]"):
            add("error", f"example {index} has unbalanced furigana brackets")
    return issues


def validate_records(
    records: list[VocabularyRecord], source: str | Path = ""
) -> list[ValidationIssue]:
    source_text = str(source)
    issues: list[ValidationIssue] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        issues.extend(validate_record(record, source_text))
        if record.id in seen:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"duplicate ID also seen at record {seen[record.id]}",
                    record_id=record.id,
                    source=source_text,
                )
            )
        else:
            seen[record.id] = index
    if not records:
        issues.append(
            ValidationIssue(level="warning", message="no records found", source=source_text)
        )
    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.level == "error" for issue in issues)
