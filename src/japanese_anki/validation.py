from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from japanese_anki.identifiers import contains_kanji
from japanese_anki.models import VocabularyRecord

# An ID minted from an expression with no reading: ``word:話す:``. The reading is
# part of the ID, so this one cannot be repaired in place once Anki has seen it.
_READINGLESS_ID = re.compile(r"^word:(?P<expression>.*):$")

# A jpdb accent pattern: one H or L per kana of the reading, plus one for the
# particle that would follow the word. Anything else is not a pattern at all.
# Case-insensitive to match `pitch._LEVELS`, which reads `h`/`l` deliberately.
# The two disagreeing is not a style question: this one is an *error*, so
# `has_errors` is true and `janki build` refuses the whole deck — over a pattern
# `to_aquestalk` converts correctly and speaks correctly.
_PITCH_PATTERN = re.compile(r"^[HL]+$", re.IGNORECASE)

# Self-contained on purpose: the remedy has to be readable from the error, not
# from a document. Pointing a reviewer at the review they just did is how this
# check stops being actionable. No literal path either — the staging directory
# is configurable ([paths] staging_dir), so a hardcoded one would name a
# directory some projects do not have.
_STAGING_HINT = (
    "route through the staging review directory — fill in the reading and delete the "
    "record's 'id:' line so the ID is re-minted from expression + reading"
)


def _misplaced_furigana(furigana: str) -> list[str]:
    """Ruby groups whose reading will land on the text before them.

    Delegates to :func:`japanese_anki.qc.spilled_furigana_groups` — janki
    should not hold two ideas about what Anki will draw, and the rule is subtler
    than it looks. See that function for the two plausible tests that are wrong:
    position alone (``日[にっ]本[ぽん]`` abuts legitimately) and "the run starts
    with kana" (which flags correct whole-word ruby, and whose advice would
    break it).
    """
    from japanese_anki import qc

    return [text for text, _reading in qc.spilled_furigana_groups(furigana)]


def _kana(reading: str) -> str:
    """``reading`` with combining marks composed, which is what a kana count is.

    ``が`` typed as ``か`` + U+3099 is two codepoints and one kana.
    :func:`japanese_anki.pitch.to_aquestalk` measures it this way, so measuring
    it any other way here would warn about a pattern that converts perfectly
    well — and send someone to "fix" a pattern into the mismatch the warning
    exists to prevent.
    """
    return unicodedata.normalize("NFC", reading)


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


def _accent_patterns(record: VocabularyRecord) -> list[tuple[str, str]]:
    """Every accent pattern on the record, labelled by where it came from.

    ``audio_accent`` is checked by the same rules as ``pitch_accent``: it is the
    pattern audio generation actually uses when set, so a typo there is the one
    that reaches the synthesizer.
    """
    patterns = [
        (f"pitch_accent[{index}]", pattern) for index, pattern in enumerate(record.pitch_accent)
    ]
    if record.audio_accent:
        patterns.append(("audio_accent", record.audio_accent))
    return patterns


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
    elif record.furigana and (spilled := _misplaced_furigana(record.furigana)):
        add(
            "warning",
            "furigana is missing a space before "
            + ", ".join(repr(text) for text in spilled)
            + " — Anki draws a reading over everything back to the previous "
            "space, so it will spill onto the kana before it",
        )
    if record.furigana and not record.reading:
        add("warning", "furigana is present but the plain reading is empty")
    if record.verb_group and not record.part_of_speech:
        add("warning", "verb group is present but part of speech is empty")
    for label, pattern in _accent_patterns(record):
        if not _PITCH_PATTERN.match(pattern):
            add(
                "error",
                f"{label} {pattern!r} is not an accent pattern: expected only "
                "'H' and 'L', one per kana of the reading plus the following particle",
            )
        elif record.reading and len(pattern) != len(_kana(record.reading)) + 1:
            # A warning, not an error: that the pattern covers the particle slot
            # is community-verified rather than documented, so a mismatch means
            # "look at this", not "this file is wrong". Audio generation (M5.1)
            # refuses such a pattern instead of guessing the alignment, so the
            # record simply gets no audio until someone checks it.
            add(
                "warning",
                f"{label} {pattern!r} has {len(pattern)} position(s) for a "
                f"{len(_kana(record.reading))}-kana reading; "
                f"{len(_kana(record.reading)) + 1} were expected (one per kana "
                "plus the following particle) and audio generation will skip "
                "this record rather than guess",
            )
    for index, example in enumerate(record.examples, start=1):
        if example.japanese and not example.english:
            add("warning", f"example {index} has Japanese but no English translation")
        if example.english and not example.japanese:
            add("warning", f"example {index} has English but no Japanese sentence")
        if example.furigana and example.furigana.count("[") != example.furigana.count("]"):
            add("error", f"example {index} has unbalanced furigana brackets")
        elif example.furigana and (spilled := _misplaced_furigana(example.furigana)):
            add(
                "warning",
                f"example {index} furigana is missing a space before "
                + ", ".join(repr(text) for text in spilled)
                + " — Anki draws a reading over everything back to the previous "
                "space, so it will spill onto the kana before it",
            )
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
