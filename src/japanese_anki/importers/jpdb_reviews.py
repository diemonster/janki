"""jpdb's "Export vocabulary reviews" file, matched against records janki holds.

This importer **creates nothing**. It reads a ``reviews.json`` export and marks
the records janki already has that jpdb is already drilling, so a deck can leave
them out::

    deck:
      exclude_tags: [jpdb-known]

Two things land on a matched record: the ``jpdb-known`` tag, and a review count
in ``source.raw_fields``. The count lives there and **not** in the ledger on
purpose — ``ledger.record_source_seen`` identifies a reference by every key but
``seen_at``, so a count that changes every week would append a near-duplicate
line per record per run, and a weekly pass over 2,000 words would add 2,000
lines a week to a git-tracked file. The ledger gets one detail-free sighting per
matched record instead, which is idempotent forever (IMPLEMENTATION_PLAN M2.7;
DESIGN_V2's jpdb section says the same).

Matching is by ``vid`` where the record carries one — an ``import-jpdb`` record
does — and by expression + reading otherwise, which is how a record that reached
janki from Shirabe or a photographed table is found. Entries that match nothing
are reported, never dropped: they are words in jpdb that janki does not have,
which is a fact worth seeing rather than an error.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import record_scope_id, stable_record_id
from japanese_anki.models import VocabularyRecord

__all__ = [
    "KNOWN_TAG",
    "REVIEW_COUNT_FIELD",
    "JpdbReviewsError",
    "ReviewEntry",
    "ReviewMatch",
    "apply_reviews",
    "read_reviews",
]


class JpdbReviewsError(JankiError):
    pass


#: What a matched record is tagged. The deck filter recipe names this string,
#: so it is API: changing it silently stops every existing deck from excluding
#: anything.
KNOWN_TAG = "jpdb-known"

#: Where the review count is stored on a matched record.
REVIEW_COUNT_FIELD = "jpdb_reviews"

# Top-level keys holding vocabulary cards. jpdb exports one list per card type
# — `cards_vocabulary_jp_en` and `cards_vocabulary_en_jp` are the same words
# drilled in two directions — alongside kanji lists whose entries are characters
# and keywords, not words. Matching on the prefix takes every vocabulary list
# jpdb adds later without a code change; everything else is reported as skipped
# rather than quietly ignored.
VOCABULARY_PREFIX = "cards_vocabulary_"


@dataclass(frozen=True, slots=True)
class ReviewEntry:
    """One word in the export, with how many times it has been reviewed."""

    vid: str
    spelling: str
    reading: str
    reviews: int

    @property
    def label(self) -> str:
        return f"{self.spelling} [{self.reading}]" if self.reading else self.spelling


@dataclass(slots=True)
class ReviewMatch:
    """The outcome of applying an export to a collection."""

    records: list[VocabularyRecord] = field(default_factory=list)
    matched: dict[str, int] = field(default_factory=dict)
    unmatched: list[ReviewEntry] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    skipped_sections: list[str] = field(default_factory=list)


def _entry_count(raw: Mapping[str, Any], where: str) -> int:
    """How many times this card has been reviewed.

    An absent ``reviews`` key counts zero — a card jpdb made but never showed
    you is still a word jpdb knows. A key that is *present* but not a list is a
    different thing entirely: the export shape has drifted, and counting it zero
    would write ``jpdb_reviews: "0"`` into the records as though that were the
    truth.
    """
    if "reviews" not in raw:
        return 0
    reviews = raw["reviews"]
    if isinstance(reviews, Sequence) and not isinstance(reviews, str):
        return len(reviews)
    raise JpdbReviewsError(
        f"{where} has a 'reviews' value that is not a list of reviews, but a "
        f"{type(reviews).__name__}. This command reads jpdb's 'Export vocabulary "
        "reviews' file; counting it as zero would record a number nobody wrote."
    )


def read_reviews(path: Path) -> tuple[list[ReviewEntry], list[str]]:
    """Every vocabulary entry in a reviews export, and the sections skipped.

    One entry per word, with its review counts summed across card types: a word
    drilled in both directions is one word janki knows about, and reporting it
    twice would double every count.
    """
    path = Path(path)
    try:
        # utf-8-sig, like every other foreign input this project reads: a file
        # round-tripped through a Windows editor carries a BOM, and json.loads
        # rejects one as a syntax error on line 1.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise JpdbReviewsError(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JpdbReviewsError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}. "
            "This command reads jpdb's 'Export vocabulary reviews' file."
        ) from exc
    if not isinstance(payload, Mapping):
        raise JpdbReviewsError(
            f"Expected a mapping of card lists in {path}, got {type(payload).__name__}. "
            "This command reads jpdb's 'Export vocabulary reviews' file."
        )

    by_word: dict[tuple[str, str, str], ReviewEntry] = {}
    skipped: list[str] = []
    sections = 0
    for key, value in payload.items():
        name = str(key)
        if not name.startswith(VOCABULARY_PREFIX):
            skipped.append(name)
            continue
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise JpdbReviewsError(f"'{name}' in {path} must be a list of cards")
        sections += 1
        for index, raw in enumerate(value, start=1):
            if not isinstance(raw, Mapping):
                raise JpdbReviewsError(
                    f"Card {index} of '{name}' in {path} must be a mapping, "
                    f"got {type(raw).__name__}"
                )
            vid = "" if raw.get("vid") is None else str(raw.get("vid")).strip()
            spelling = str(raw.get("spelling") or "").strip()
            reading = str(raw.get("reading") or "").strip()
            if not spelling and not vid:
                raise JpdbReviewsError(
                    f"Card {index} of '{name}' in {path} has neither a vid nor a "
                    "spelling, so it cannot be matched to anything"
                )
            key_of = (vid, spelling, reading)
            existing = by_word.get(key_of)
            count = _entry_count(raw, f"Card {index} of '{name}' in {path}") + (
                existing.reviews if existing else 0
            )
            by_word[key_of] = ReviewEntry(
                vid=vid, spelling=spelling, reading=reading, reviews=count
            )

    if not sections:
        raise JpdbReviewsError(
            f"No '{VOCABULARY_PREFIX}*' list in {path}. This command reads jpdb's "
            "'Export vocabulary reviews' file; found: "
            + (", ".join(skipped) or "nothing")
        )
    return list(by_word.values()), skipped


def _index(records: Iterable[VocabularyRecord]) -> tuple[dict[str, str], dict[str, str]]:
    """Record ids indexed by jpdb vid, and by identity id.

    First match wins in both. Two records sharing a vid is a duplicate janki
    should not have, and ``janki status --duplicates`` is the command that says
    so — silently tagging both here would spread the problem rather than
    surface it.

    A deck-scoped copy is not indexed at all. This export is the history of the
    words in jpdb's own collection, and first-match-wins would otherwise let a
    standalone copy that happens to sort first take the review count belonging
    to the shared word — leaving the word this export is about unmatched.
    """
    by_vid: dict[str, str] = {}
    by_identity: dict[str, str] = {}
    for record in records:
        if record_scope_id(record.id):
            continue
        vid = str(record.source.raw_fields.get("vid", "")).strip()
        if vid:
            by_vid.setdefault(vid, record.id)
        by_identity.setdefault(stable_record_id(record.expression, record.reading), record.id)
    return by_vid, by_identity


def apply_reviews(
    records: Sequence[VocabularyRecord], entries: Sequence[ReviewEntry]
) -> ReviewMatch:
    """Tag and count the records an export matches, leaving the rest alone.

    Returns every record, matched or not, so the caller saves one list. A record
    already tagged and already carrying the same count is left untouched and
    absent from ``changed`` — re-running a week later then rewrites only the
    words whose counts actually moved.
    """
    result = ReviewMatch(records=list(records))
    by_vid, by_identity = _index(result.records)
    positions = {record.id: index for index, record in enumerate(result.records)}

    # Resolve everything first, then write once per record. Two entries can
    # land on one record — the same vid with the reading written differently in
    # two card lists, or a vid-bearing entry and a vid-less one for the same
    # word — and their counts have to *add*, the way two card types for one word
    # do. Writing as we went would let the last entry's count win and would
    # report the record as changed twice.
    for entry in entries:
        record_id = by_vid.get(entry.vid) if entry.vid else None
        if record_id is None:
            record_id = by_identity.get(stable_record_id(entry.spelling, entry.reading))
        if record_id is None:
            result.unmatched.append(entry)
            continue
        result.matched[record_id] = result.matched.get(record_id, 0) + entry.reviews

    for record_id, reviews in result.matched.items():
        index = positions[record_id]
        record = result.records[index]
        tags = record.tags if KNOWN_TAG in record.tags else sorted({*record.tags, KNOWN_TAG})
        raw_fields = dict(record.source.raw_fields)
        count = str(reviews)
        if tags == record.tags and raw_fields.get(REVIEW_COUNT_FIELD) == count:
            continue
        raw_fields[REVIEW_COUNT_FIELD] = count
        result.records[index] = replace(
            record,
            tags=tags,
            source=replace(record.source, raw_fields=raw_fields),
        )
        result.changed.append(record_id)
    return result
