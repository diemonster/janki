from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from japanese_anki.identifiers import contains_kanji
from japanese_anki.models import VocabularyRecord

# An ID minted from an expression with no reading: ``word:話す:``. The reading is
# part of the ID, so this one cannot be repaired in place once Anki has seen it.
_READINGLESS_ID = re.compile(r"^word:(?P<expression>.*):$")

# Self-contained on purpose: the remedy has to be readable from the error, not
# from a document. Pointing a reviewer at the review they just did is how this
# check stops being actionable.
_STAGING_HINT = (
    "route through data/staging review — fill in the reading and delete the record's "
    "'id:' line so the ID is re-minted from expression + reading"
)


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


def _is_readingless_kanji_id(record_id: str) -> bool:
    """True for ``word:<kanji expression>:``, an ID whose reading slot is empty.

    Caught separately from the empty-``reading`` check: a record whose reading
    was filled in later still carries the malformed ID, and that ID is what
    Anki's GUID derives from.
    """
    match = _READINGLESS_ID.match(record_id)
    return bool(match) and contains_kanji(match.group("expression"))


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
    if contains_kanji(record.expression) and not record.reading:
        add("error", f"expression contains kanji but reading is missing; {_STAGING_HINT}")
    elif _is_readingless_kanji_id(record.id):
        # Reported once the reading is filled in and the ID still is not: while the
        # reading is empty the message above states the same fault, and a staging
        # file under review would carry two errors per row for one fix.
        add(
            "error",
            "ID was minted without a reading (word:<expression>:) and cannot be "
            f"corrected in place without orphaning review history; {_STAGING_HINT}",
        )
    elif contains_kanji(record.reading):
        # The ID looks well formed (word:<expression>:<reading>) and is not: a
        # reading is kana, and this one is the written form copied across. It is
        # what an importer mints when a row supplies only one of the two columns,
        # and it is just as permanent as the empty-reading shape above.
        add(
            "error",
            "reading is written in kanji, so the ID's reading slot holds a spelling "
            "rather than a pronunciation and cannot be corrected in place without "
            f"orphaning review history; {_STAGING_HINT}",
        )
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
