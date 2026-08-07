"""What janki knows about the collection: records, ledger state, duplicates.

``janki status`` is the read-only view over the two files that between them
describe the collection — ``vocabulary.json`` (card content) and
``data/ledger.json`` (what the tools did to it). It answers the questions the
other commands are about to act on: what has never been exported, what has no
audio, what has no examples, which audio no longer matches its record, and
which records look like two copies of the same word.

Two things shape this module:

* **The record universe is the normalized file plus inline deck notes.** Deck
  YAML files may still carry notes of their own (M1.7 migrates them), and a
  status report that ignored them would under-count everything. Where an id
  exists in both, the deck-resolved record wins: that is the record the deck
  actually exports today, and the one M1.7 will make authoritative.
* **Nothing here writes, except ``rebuild``.** A status report must be safe to
  run at any moment, including on a repo that has never run an import — a
  missing ledger file is an empty ledger, and every count simply reads as
  "missing".
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.identifiers import normalize_identity_part
from japanese_anki.io import load_records
from japanese_anki.ledger import (
    Ledger,
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import SourceReference, VocabularyRecord

# Every media file janki generates is named ``janki-<filename fingerprint>``;
# the prefix is what tells a rebuild (and M5.3's --prune) which files are ours.
MEDIA_PREFIX = "janki-"

# A rebuilt audio entry says who made it: the file on disk carries no record of
# which engine or voice spoke it, and guessing from the current configuration
# would put a plausible lie in the ledger.
REBUILT_PROVIDER = "unknown"
REBUILT_VOICE = -1


def display_path(path: Path, root: Path) -> str:
    """Path as typed by a human standing in the project root, when possible."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# The record universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeckView:
    """One deck file and the record ids it currently resolves to."""

    path: Path
    stem: str
    ids: list[str]


@dataclass(frozen=True, slots=True)
class RecordUniverse:
    records: list[VocabularyRecord]
    decks: list[DeckView]
    normalized_count: int
    warnings: list[str] = field(default_factory=list)
    # The normalized file's own source per id, kept even where a deck note
    # shadows the record: the deck-resolved copy is what a deck exports, but
    # its default-manual source is not the record's provenance, and --rebuild
    # writes provenance into a ledger designed to be committed.
    normalized_sources: dict[str, SourceReference] = field(default_factory=dict)

    @property
    def inline_count(self) -> int:
        """Records that exist only inside a deck file."""
        return len(self.records) - self.normalized_count


def deck_files(config: ProjectConfig) -> list[Path]:
    return sorted([*config.deck_dir.glob("*.yaml"), *config.deck_dir.glob("*.yml")])


def collect_records(config: ProjectConfig) -> RecordUniverse:
    """Every record janki manages, from the normalized file and every deck.

    A deck file that cannot be read is reported as a warning and skipped rather
    than aborting the report: status is the command you run to find out what is
    wrong, so it has to survive one broken file. ``janki validate`` is where a
    bad deck is an error.
    """
    warnings: list[str] = []
    by_id: dict[str, VocabularyRecord] = {}

    normalized_ids: set[str] = set()
    normalized_sources: dict[str, SourceReference] = {}
    if config.normalized_file.exists():
        for record in load_records(config.normalized_file):
            by_id[record.id] = record
            normalized_ids.add(record.id)
            normalized_sources[record.id] = record.source

    decks: list[DeckView] = []
    for deck_path in deck_files(config):
        try:
            _, records = resolve_deck_records(deck_path)
        except JankiError as exc:
            warnings.append(f"skipping deck {deck_path}: {exc}")
            continue
        for record in records:
            # The deck's version wins: inline notes override the normalized
            # record for the deck that is actually built from them.
            by_id[record.id] = record
        decks.append(
            DeckView(path=deck_path, stem=deck_path.stem, ids=[record.id for record in records])
        )

    records = sorted(by_id.values(), key=lambda record: record.id)
    return RecordUniverse(
        records=records,
        decks=decks,
        normalized_count=sum(1 for record in records if record.id in normalized_ids),
        warnings=warnings,
        normalized_sources=normalized_sources,
    )


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def pitch_accent_supported() -> bool:
    """Whether records carry pitch accent yet (the field arrives in M2.2)."""
    return any(item.name == "pitch_accent" for item in dataclasses.fields(VocabularyRecord))


@dataclass(frozen=True, slots=True)
class DeckStatus:
    stem: str
    total: int
    unexported_ids: list[str]


@dataclass(frozen=True, slots=True)
class StatusReport:
    root: Path
    normalized_path: Path
    deck_dir: Path
    ledger_path: Path
    ledger_exists: bool
    ledger_entries: int
    record_ids: list[str]
    normalized_count: int
    inline_count: int
    by_source: list[tuple[str, int]]
    decks: list[DeckStatus]
    missing_audio: list[str]
    stale_audio: list[str]
    missing_enrichment: list[str]
    # None means "the schema has no pitch accent yet", which is not the same
    # answer as "no record is missing one".
    missing_pitch_accent: list[str] | None

    @property
    def total(self) -> int:
        return len(self.record_ids)


def build_report(config: ProjectConfig, universe: RecordUniverse, book: Ledger) -> StatusReport:
    records = universe.records
    missing_pitch: list[str] | None = None
    if pitch_accent_supported():
        missing_pitch = [
            record.id for record in records if not getattr(record, "pitch_accent", None)
        ]

    return StatusReport(
        root=config.root,
        normalized_path=config.normalized_file,
        deck_dir=config.deck_dir,
        ledger_path=book.path,
        ledger_exists=Path(book.path).exists(),
        ledger_entries=len(book.records),
        record_ids=[record.id for record in records],
        normalized_count=universe.normalized_count,
        inline_count=universe.inline_count,
        by_source=sorted(
            Counter(record.source.type or "manual" for record in records).items(),
            key=lambda item: (-item[1], item[0]),
        ),
        decks=[
            DeckStatus(
                stem=deck.stem,
                total=len(deck.ids),
                unexported_ids=book.unexported(deck.stem, deck.ids),
            )
            for deck in universe.decks
        ],
        missing_audio=book.missing_audio(records),
        stale_audio=book.stale_audio(records),
        missing_enrichment=Ledger.missing_enrichment(records),
        missing_pitch_accent=missing_pitch,
    )


def format_report(report: StatusReport) -> list[str]:
    root = report.root
    lines = [
        f"Records: {report.total} "
        f"({report.normalized_count} in {display_path(report.normalized_path, root)}, "
        f"{report.inline_count} inline in deck files)"
    ]
    if report.by_source:
        lines.append(
            "By source type: "
            + ", ".join(f"{name} {count}" for name, count in report.by_source)
        )
    entries = report.ledger_entries
    ledger_line = (
        f"Ledger: {display_path(report.ledger_path, root)} — "
        f"{entries} record entr{'y' if entries == 1 else 'ies'}"
    )
    if not report.ledger_exists:
        ledger_line += " (no ledger file yet; everything below reads as missing)"
    lines.append(ledger_line)

    if report.decks:
        lines.append(
            "Never exported: "
            + ", ".join(
                f"{deck.stem} {len(deck.unexported_ids)} of {deck.total}"
                for deck in report.decks
            )
        )
    else:
        lines.append(f"Never exported: no deck files under {display_path(report.deck_dir, root)}")

    lines.append(f"Missing word audio: {len(report.missing_audio)} of {report.total}")
    lines.append(f"Stale audio: {len(report.stale_audio)}")
    lines.append(
        f"Missing enrichment: {len(report.missing_enrichment)} "
        "(no example sentence, or no usage notes)"
    )
    if report.missing_pitch_accent is None:
        lines.append("Missing pitch accent: n/a until the pitch-accent schema lands (M2.2)")
    else:
        lines.append(f"Missing pitch accent: {len(report.missing_pitch_accent)}")
    return lines


def format_unexported(report: StatusReport) -> list[str]:
    pending = [deck for deck in report.decks if deck.unexported_ids]
    if not pending:
        return ["Never exported: nothing — every deck has been built with all of its records."]
    lines: list[str] = []
    for deck in pending:
        lines.append(f"Never exported by {deck.stem} ({len(deck.unexported_ids)}):")
        lines.extend(f"  {record_id}" for record_id in deck.unexported_ids)
    return lines


def format_missing_audio(report: StatusReport) -> list[str]:
    if not report.missing_audio:
        return ["Missing word audio: none."]
    return [
        f"Missing word audio ({len(report.missing_audio)}):",
        *(f"  {record_id}" for record_id in report.missing_audio),
    ]


# ---------------------------------------------------------------------------
# Duplicate candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Records that look like the same word under two ids."""

    kind: str  # "expression", "reading" or "vid"
    key: str
    reason: str
    ids: list[str]


def _jpdb_vid(record: VocabularyRecord) -> str:
    """The jpdb vocabulary id an importer stashed in the source row, if any.

    Compared numerically where possible: a round-tripped file can hold the
    same vid as ``1577980``, ``"1577980"`` or ``1577980.0``, and a typing
    accident must not hide a duplicate.
    """
    raw = str(record.source.raw_fields.get("vid", "")).strip()
    if not raw:
        return ""
    try:
        number = float(raw)
    except ValueError:
        return raw
    return str(int(number)) if number.is_integer() else raw


def find_duplicates(records: Iterable[VocabularyRecord]) -> list[DuplicateGroup]:
    """Every duplicate class, not just the obvious one.

    (a) The same expression under two ids — a word that arrived twice with
    different readings, or once before its reading was known.

    (b) The same reading under two expressions, where one of them is written in
    the kana of that reading (Shirabe's わかる beside jpdb's 分かる) or both
    carry the same jpdb ``vid``. This is the common real case, and the one a
    naive expression-only check never sees.

    (c) The same non-empty jpdb ``vid`` under more than one id, regardless of
    reading: a shared vid is definitionally the same dictionary word, so a
    hand-corrected or empty reading must not hide the pair. A vid group whose
    ids an earlier group already covers is not reported twice.

    Nothing here resolves anything: near-duplicates are reported so a human can
    pick a survivor, because merging would mean re-IDing a record and orphaning
    its Anki review history.
    """
    records = list(records)
    groups: list[DuplicateGroup] = []

    by_expression: dict[str, list[VocabularyRecord]] = {}
    by_reading: dict[str, list[VocabularyRecord]] = {}
    for record in records:
        expression = normalize_identity_part(record.expression)
        if expression:
            by_expression.setdefault(expression, []).append(record)
        reading = normalize_identity_part(record.reading)
        if reading:
            by_reading.setdefault(reading, []).append(record)

    for expression, group in sorted(by_expression.items()):
        ids = sorted({record.id for record in group})
        if len(ids) > 1:
            groups.append(
                DuplicateGroup(
                    kind="expression",
                    key=expression,
                    reason="same expression, different id",
                    ids=ids,
                )
            )

    for reading, group in sorted(by_reading.items()):
        if len(group) < 2:
            continue
        matched: dict[str, None] = {}  # an ordered set of ids
        reasons: set[str] = set()
        for first, second in itertools.combinations(
            sorted(group, key=lambda record: record.id), 2
        ):
            first_expression = normalize_identity_part(first.expression)
            second_expression = normalize_identity_part(second.expression)
            if first_expression == second_expression:
                continue  # same expression as well: the pass above owns this pair
            pair_reasons: list[str] = []
            # Readings are equal here, so "one expression is the other's
            # reading" is the same test as "one expression is this reading".
            if reading in {first_expression, second_expression}:
                pair_reasons.append("one is the kana form")
            vid = _jpdb_vid(first)
            if vid and vid == _jpdb_vid(second):
                pair_reasons.append(f"same jpdb vid {vid}")
            if not pair_reasons:
                continue
            matched[first.id] = None
            matched[second.id] = None
            reasons.update(pair_reasons)
        if matched:
            groups.append(
                DuplicateGroup(
                    kind="reading",
                    key=reading,
                    reason="same reading, different expression: " + "; ".join(sorted(reasons)),
                    ids=list(matched),
                )
            )

    by_vid: dict[str, list[VocabularyRecord]] = {}
    for record in records:
        vid = _jpdb_vid(record)
        if vid:
            by_vid.setdefault(vid, []).append(record)
    covered = [set(group.ids) for group in groups]
    for vid, group in sorted(by_vid.items()):
        ids = sorted({record.id for record in group})
        if len(ids) <= 1:
            continue
        if any(set(ids) <= ids_seen for ids_seen in covered):
            # The pair is already on the report; a second group would make one
            # duplicate look like two.
            continue
        groups.append(
            DuplicateGroup(
                kind="vid",
                key=vid,
                reason=f"same jpdb vid {vid} under more than one id",
                ids=ids,
            )
        )

    return groups


def format_duplicates(groups: Sequence[DuplicateGroup]) -> list[str]:
    if not groups:
        return ["Duplicate candidates: none."]
    lines = [f"Duplicate candidates ({len(groups)} group(s)):"]
    for group in groups:
        lines.append(f"  {group.key} — {group.reason}")
        lines.extend(f"    {record_id}" for record_id in group.ids)
    lines.append("Resolution is manual: pick the id that keeps its review history and")
    lines.append("delete the other from the file it came from. Records are never re-IDed,")
    lines.append("so the loser's review history is the accepted cost.")
    return lines


# ---------------------------------------------------------------------------
# --rebuild
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RebuildSummary:
    sources: int
    word_audio: int
    example_audio: int
    unprovable_audio: int
    unmatched_media: int
    # Files sharing a fingerprint with the file a rebuilt entry claimed: a
    # provider switch's leftover twin (janki-<fp>.mp3 beside janki-<fp>.wav).
    ambiguous_media: int
    media_dir: Path


# When several files claim one fingerprint, the rebuilt entry binds the first
# by this order: .wav is what janki's own generators write, so it is the best
# guess, and the order being documented makes the choice reproducible.
_EXTENSION_PREFERENCE: tuple[str, ...] = (".wav", ".mp3", ".ogg", ".m4a")


def _media_rank(path: Path) -> tuple[int, str]:
    suffix = path.suffix.lower()
    known = suffix in _EXTENSION_PREFERENCE
    return (_EXTENSION_PREFERENCE.index(suffix) if known else len(_EXTENSION_PREFERENCE), str(path))


def _media_by_fingerprint(media_dir: Path) -> dict[str, list[Path]]:
    """Every ``janki-<fingerprint>.*`` file under ``media_dir``, by fingerprint.

    All claimants are kept, best candidate first — dropping a twin here would
    erase it from the rebuild accounting entirely.
    """
    found: dict[str, list[Path]] = {}
    if not media_dir.exists():
        return found
    for path in sorted(media_dir.rglob(f"{MEDIA_PREFIX}*")):
        if path.is_file():
            found.setdefault(path.stem[len(MEDIA_PREFIX) :], []).append(path)
    for paths in found.values():
        paths.sort(key=_media_rank)
    return found


def _first(paths: list[Path] | None) -> Path | None:
    """The preferred claimant of a fingerprint, if any file claims it."""
    return paths[0] if paths else None


def _has_audio_entry(book: Ledger, record_id: str, filename: str) -> bool:
    entry = book.records.get(record_id) or {}
    return any(
        isinstance(item, dict) and item.get("file") == filename
        for item in (entry.get("audio") or [])
    )


def _rebuilt_word_content_fp(record: Any) -> str:
    """The content fingerprint a found word-audio file can be *proved* to hold.

    A word audio filename is addressed by the record id, and the id embeds the
    reading — so finding the file proves it speaks this reading. It proves
    nothing about the accent pattern (``pitch_accent``/``audio_accent``, M2.2),
    which is not part of the address. A record that carries accent data
    therefore gets no fingerprint: ``status`` and ``janki audio`` will call that
    file stale and regenerate it, which is the safe direction to be wrong in.
    """
    if getattr(record, "audio_accent", "") or getattr(record, "pitch_accent", None):
        return ""
    return word_audio_content_fingerprint(record)


def rebuild(
    book: Ledger,
    records: Iterable[VocabularyRecord],
    media_dir: Path,
    sources_by_id: dict[str, SourceReference] | None = None,
) -> RebuildSummary:
    """Reconstruct the ledger entries that records and media files still prove.

    Sources come from each record's own ``source`` — except where
    ``sources_by_id`` (the normalized file's own sources, see
    ``RecordUniverse.normalized_sources``) knows better: a deck note that
    shadows a normalized record carries a default-manual source that is not
    the record's provenance, and the ledger is committed to git. Audio comes
    from files whose names match the filename fingerprints. Neither carries a
    date, so the dates written here are today's — a reconstruction, not
    history. Export state is not reconstructible at all: nothing outside the
    ledger records which build included which note. The caller must say so.

    Existing entries are left alone: a real ``janki audio`` entry knows the
    provider and voice, and must not be replaced by a rebuilt one that does not.
    """
    media = _media_by_fingerprint(media_dir)
    claimed: set[str] = set()
    sources = word_audio = example_audio = unprovable = 0

    for record in records:
        source = (sources_by_id or {}).get(record.id, record.source)
        if book.record_source_seen(record.id, source.type or "manual", source.imported_from):
            sources += 1

        word_file = _first(media.get(word_audio_filename_fingerprint(record)))
        if word_file is not None:
            claimed.add(word_file.name)
            if not _has_audio_entry(book, record.id, word_file.name):
                content_fp = _rebuilt_word_content_fp(record)
                if not content_fp:
                    unprovable += 1
                book.record_audio(
                    record.id,
                    file=word_file.name,
                    of="word",
                    provider=REBUILT_PROVIDER,
                    voice=REBUILT_VOICE,
                    content_fp=content_fp,
                    rebuilt=True,
                )
                word_audio += 1

        for example in record.examples:
            if not example.japanese:
                continue
            example_file = _first(media.get(example_audio_filename_fingerprint(record, example)))
            if example_file is None:
                continue
            claimed.add(example_file.name)
            if _has_audio_entry(book, record.id, example_file.name):
                continue
            # An example audio filename is addressed by the sentence text, so
            # the match proves what this file says.
            book.record_audio(
                record.id,
                file=example_file.name,
                of="example",
                provider=REBUILT_PROVIDER,
                voice=REBUILT_VOICE,
                content_fp=example_audio_content_fingerprint(example),
                rebuilt=True,
            )
            example_audio += 1

    unmatched = ambiguous = 0
    for paths in media.values():
        names = [path.name for path in paths]
        if any(name in claimed for name in names):
            # The fingerprint bound to a record; every unclaimed twin is a
            # competing candidate the user must know exists.
            ambiguous += sum(1 for name in names if name not in claimed)
        else:
            unmatched += len(names)
    return RebuildSummary(
        sources=sources,
        word_audio=word_audio,
        example_audio=example_audio,
        unprovable_audio=unprovable,
        unmatched_media=unmatched,
        ambiguous_media=ambiguous,
        media_dir=media_dir,
    )


def format_rebuild(summary: RebuildSummary, root: Path) -> list[str]:
    lines = [
        "Rebuilt from records and media on disk:",
        f"  {summary.sources} source reference(s) recovered from each record's own source",
        f"  {summary.word_audio} word and {summary.example_audio} example audio file(s) "
        f"matched under {display_path(summary.media_dir, root)}",
    ]
    if summary.unprovable_audio:
        lines.append(
            f"  {summary.unprovable_audio} word file(s) recorded without a content "
            "fingerprint (the filename cannot prove the accent) — they will report "
            "as stale until regenerated"
        )
    if summary.ambiguous_media:
        lines.append(
            f"  {summary.ambiguous_media} {MEDIA_PREFIX}* file(s) share a fingerprint "
            "with a file a rebuilt entry claimed (extension preference: "
            f"{', '.join(_EXTENSION_PREFERENCE)}) — check which file janki actually "
            "generated and delete the other"
        )
    if summary.unmatched_media:
        lines.append(
            f"  {summary.unmatched_media} {MEDIA_PREFIX}* file(s) matched no record; "
            "they belong to records that are gone or renamed"
        )
    lines.append(
        "  dates recorded are today's: neither a record nor a media file remembers when "
        "it was first seen"
    )
    lines.append("Export state is NOT reconstructible: the ledger was the only place that")
    lines.append("ever held it. Nothing is lost — GUIDs are deterministic, so re-exporting a")
    lines.append("note updates it instead of duplicating it. The honest degradation is that")
    lines.append("the next 'janki build --only-new' includes everything.")
    return lines


# ---------------------------------------------------------------------------
# --format ids
# ---------------------------------------------------------------------------


def selected_ids(
    report: StatusReport,
    duplicate_groups: Sequence[DuplicateGroup],
    *,
    unexported: bool = False,
    missing_audio: bool = False,
    duplicates: bool = False,
) -> list[str]:
    """The ids the chosen detail flags name, deduplicated, in report order.

    With no detail flag the answer is every record janki manages — ``janki
    status --format ids`` is then the list of everything, which is what a
    pipeline expects from a bare listing.
    """
    chosen: list[str] = []
    if unexported:
        for deck in report.decks:
            chosen.extend(deck.unexported_ids)
    if missing_audio:
        chosen.extend(report.missing_audio)
    if duplicates:
        for group in duplicate_groups:
            chosen.extend(group.ids)
    if not (unexported or missing_audio or duplicates):
        chosen = list(report.record_ids)
    return list(dict.fromkeys(chosen))
