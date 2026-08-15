from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
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

# C0 control characters, minus the three that are ordinary text. U+001F is the
# one that corrupts rather than merely looks wrong: Anki stores a note's fields
# as a single `\x1f`-joined string, so one inside a value adds a field and every
# value after it shifts by one position — `UsageNotes` keeps the head, `Audio`
# receives the tail, `Image` receives the audio tag, and so on to the end of the
# notetype. The build succeeds and reports nothing, and `janki status --rebuild`
# reads the collection back through the same split. The rest are refused with
# it because none of them belongs in a Japanese sentence either, and a value
# carrying one is evidence its importer mis-parsed a row.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

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


def _stray_furigana_spaces(furigana: str) -> tuple[str, ...]:
    """Spaces in a furigana field that no ruby group follows.

    Delegates to :func:`japanese_anki.qc.stray_furigana_spaces` for the same
    reason as above: `furigana_reading` decides which spaces are notation, so
    only it can say which are content.
    """
    from japanese_anki import qc

    return qc.stray_furigana_spaces(furigana)


def _content_holds(example: object) -> list[tuple[str, str, str]]:
    """Delegates to :func:`japanese_anki.qc.example_content_holds` (same rule)."""
    from japanese_anki import qc

    return qc.example_content_holds(example)


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
    code: str = ""

    def format(self) -> str:
        location = self.source
        if self.record_id:
            location = f"{location}:{self.record_id}" if location else self.record_id
        identity = f" {self.code}" if self.code else ""
        prefix = f"[{self.level.upper()}{identity}]"
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


def field_separator_fault(names: Sequence[str], values: Sequence[str]) -> str | None:
    """The name of the first field carrying U+001F, or ``None``.

    The name alone. A caller reporting the value would put the separator into
    its own error message, and the field is what a person needs to look at.

    Anki stores a note's fields as a single U+001F-joined string, so one inside
    a value adds a field and shifts every later value into the next slot — on a
    build that succeeds and reports nothing.

    For the values that never pass through a record and so are never seen by
    :func:`validate_record` — a word deck's kanji block, rendered from
    `data/kanji.json`, and a rule card's fields, which come from
    `data/patterns.json` and the deck file. ``html.escape`` leaves control
    characters alone. A drill deck asks nothing here: every one of its values
    derives from a record its builder has already validated.

    Only U+001F, where :func:`validate_record` refuses a wider class. This one
    corrupts the note; the rest merely look wrong, and refusing a build over a
    stray byte in a chart nobody can edit through janki would trade a cosmetic
    fault for an unshippable deck.

    Returns rather than raises, so each exporter reports it as its own error.
    """
    for name, value in zip(names, values, strict=True):
        if "\x1f" in value:
            return name
    return None


def _control_characters(record: VocabularyRecord) -> list[tuple[str, str]]:
    """Every ``(field path, character)`` in the record that is not text.

    Over the record's own serialized shape rather than a hand-written list of
    fields, which would go stale the next time a field is added — but *not*
    over ``source.raw_fields``, which is the opposite kind of value. That is the
    verbatim source row, kept precisely so an unknown column is never silently
    discarded; it reaches no note field (`_source_text` reads `type`,
    `imported_from`, and `row`), so a stray byte in a column janki does not map
    cannot corrupt a note. Refusing it would make an unreadable export refuse
    the *whole deck*, with the only remedies being to delete the provenance the
    repository exists to keep or to edit `data/inbox/`.
    """
    found: dict[tuple[str, str], None] = {}

    def walk(value: object, path: str) -> None:
        if isinstance(value, str):
            for match in _CONTROL_CHARACTERS.finditer(value):
                found[(path or "record", match.group())] = None
        elif isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                if child == "source.raw_fields":
                    continue
                # The key as well as the value. A `conjugations` form name is
                # rendered into the Conjugations field, and `from_dict` only
                # strips it — U+001F is whitespace to `str.strip`, so a leading
                # one is removed and an interior one survives. Without this the
                # record validates clean and the *build* is what refuses it,
                # which is the later and worse error.
                for match in _CONTROL_CHARACTERS.finditer(str(key)):
                    found[(f"{path or 'record'} key {key!r}", match.group())] = None
                walk(item, child)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(record.to_dict(), "")
    return list(found)


def validate_record(record: VocabularyRecord, source: str = "") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append(
            ValidationIssue(
                level=level,
                message=message,
                record_id=record.id,
                source=source,
                code=code,
            )
        )

    for path, character in _control_characters(record):
        add(
            "error",
            "control-character",
            f"{path} contains U+{ord(character):04X}"
            + (
                ", the separator Anki joins a note's fields with — the value "
                "would add a field and shift every later one out of place"
                if character == "\x1f"
                else ", which is not text a card can carry"
            ),
        )
    if not record.id:
        add("error", "missing-id", "missing stable ID")
    if not record.expression:
        add("error", "missing-expression", "missing expression")
    if contains_kanji(record.expression) and not record.reading:
        add(
            "error",
            "missing-reading",
            f"expression contains kanji but reading is missing; {_STAGING_HINT}",
        )
    elif _is_readingless_kanji_id(record.id):
        # Reported once the reading is filled in and the ID still is not: while the
        # reading is empty the message above states the same fault, and a staging
        # file under review would carry two errors per row for one fix.
        add(
            "error",
            "readingless-id",
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
            "kanji-reading",
            "reading is written in kanji, so the ID's reading slot holds a spelling "
            "rather than a pronunciation and cannot be corrected in place without "
            f"orphaning review history; {_STAGING_HINT}",
        )
    if not record.meanings:
        add("error", "missing-meaning", "at least one English meaning is required")
    if record.furigana and record.furigana.count("[") != record.furigana.count("]"):
        add("error", "unbalanced-furigana", "furigana brackets are unbalanced")
    elif record.furigana and (spilled := _misplaced_furigana(record.furigana)):
        add(
            "warning",
            "spilled-furigana",
            "furigana is missing a space before "
            + ", ".join(repr(text) for text in spilled)
            + " — Anki draws a reading over everything back to the previous "
            "space, so it will spill onto the kana before it",
        )
    if record.furigana and not record.reading:
        add(
            "warning",
            "furigana-without-reading",
            "furigana is present but the plain reading is empty",
        )
    if record.verb_group and not record.part_of_speech:
        add(
            "warning",
            "verb-group-without-part-of-speech",
            "verb group is present but part of speech is empty",
        )
    for label, pattern in _accent_patterns(record):
        if not _PITCH_PATTERN.match(pattern):
            add(
                "error",
                "invalid-pitch-accent",
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
                "pitch-accent-length",
                f"{label} {pattern!r} has {len(pattern)} position(s) for a "
                f"{len(_kana(record.reading))}-kana reading; "
                f"{len(_kana(record.reading)) + 1} were expected (one per kana "
                "plus the following particle) and audio generation will skip "
                "this record rather than guess",
            )
    for index, example in enumerate(record.examples, start=1):
        # The teaching-suitability judgment itself lives in
        # :func:`japanese_anki.qc.example_content_holds`, because the audio
        # command applies the same gate before voicing — janki must not hold
        # two ideas about what a teachable example is. Each hold carries its
        # own level: the certain shapes (fragments, false register labels)
        # are errors that stop a build, the camera pilot's exact gap — and
        # they are *visible* here, not only at the audio gate.
        for code, level, why in _content_holds(example):
            add(level, code, f"example {index} {why}")
        if example.japanese and not example.english:
            add(
                "warning",
                "example-missing-english",
                f"example {index} has Japanese but no English translation",
            )
        if example.english and not example.japanese:
            add(
                "warning",
                "example-missing-japanese",
                f"example {index} has English but no Japanese sentence",
            )
        if example.furigana and example.furigana.count("[") != example.furigana.count("]"):
            add(
                "error",
                "example-unbalanced-furigana",
                f"example {index} has unbalanced furigana brackets",
            )
        elif example.furigana and (spilled := _misplaced_furigana(example.furigana)):
            add(
                "warning",
                "example-spilled-furigana",
                f"example {index} furigana is missing a space before "
                + ", ".join(repr(text) for text in spilled)
                + " — Anki draws a reading over everything back to the previous "
                "space, so it will spill onto the kana before it",
            )
        if example.furigana and (stray := _stray_furigana_spaces(example.furigana)):
            add(
                "warning",
                "example-stray-furigana-space",
                f"example {index} furigana has a space before "
                + ", ".join(repr(text) for text in stray)
                + ", which no reading annotates — in a furigana field a space "
                "means 'the next group starts here', so this one survives into "
                "the reading and the romaji, and shows on the card as a gap "
                "the sentence itself does not have",
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
                    code="duplicate-id",
                )
            )
        else:
            seen[record.id] = index
    if not records:
        issues.append(
            ValidationIssue(
                level="warning",
                message="no records found",
                source=source_text,
                code="no-records",
            )
        )
    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.level == "error" for issue in issues)


def refusal_text(deck_name: str, issues: list[ValidationIssue]) -> str:
    """The one refusal a build states when local validation fails.

    Every issue, warnings included: a refused build is the moment the person
    is looking, and a warning hidden here (a missing translation) ships
    silently once the errors are fixed. One formatter for
    every gate — the CLI's shipping pre-gate and the exporters' backstops —
    so the same broken deck cannot report differently depending on which gate
    caught it.
    """
    formatted = "\n".join(issue.format() for issue in issues)
    return (
        f"{deck_name} fails local validation — fix these before any review "
        f"run:\n{formatted}"
    )
