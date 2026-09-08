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
import re
import shlex
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki import operations, patterns
from japanese_anki.collection import (
    CollectionError,
    clone_suffix_of,
    default_anki_root,
    find_profiles,
    read_notetypes,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import (
    deck_declared_ids,
    deck_declared_record_versions,
    deck_kind,
    resolve_deck_records,
)
from japanese_anki.identifiers import (
    IdentityError,
    normalize_identity_part,
    record_scope_id,
)
from japanese_anki.io import DataError, load_records, load_structured
from japanese_anki.ledger import (
    Ledger,
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import (
    SourceReference,
    VocabularyRecord,
    split_provisional,
)
from japanese_anki.staging import (
    LiveStaging,
    live_staging,
    require_resolved_coverage,
    validate_coverage_facts,
)
from japanese_anki.tts import RenderProfile

# Every media file janki generates is named ``janki-<filename fingerprint>``;
# the prefix is what tells a rebuild (and M5.3's --prune) which files are ours.
MEDIA_PREFIX = "janki-"

# A rebuilt audio entry says who made it: the file on disk carries no record of
# which engine or voice spoke it, and guessing from the current configuration
# would put a plausible lie in the ledger.
REBUILT_PROVIDER = "unknown"
REBUILT_VOICE = -1
# Same reasoning for the rate. Negative so it can never equal a configured
# speed: a rebuilt entry must not let ``janki audio`` call the clip current,
# because nothing knows how it was spoken.
REBUILT_SPEED = -1.0


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
    # Every persisted version for audio currency. Unlike `records`, this does
    # not collapse same-id inline overrides: two cards can durably demand two
    # render profiles while sharing one identity-addressed filename.
    audio_records: list[VocabularyRecord] = field(default_factory=list)
    # Synthetic rich-drill owners live in deck YAML, not vocabulary.json.
    # This map makes pending recovery name the exact deck scope that can write
    # the reference back without treating those owners as vocabulary records.
    drill_audio_decks: dict[str, Path] = field(default_factory=dict)
    # Structurally exact rich-drill owners whose deck cannot currently project
    # a writable record. Their paid WAL remains owned, but repair comes first.
    blocked_drill_audio_decks: dict[str, Path] = field(default_factory=dict)
    # A collision is never assigned to whichever deck happened to sort last:
    # recovery and prune must stop until the owner namespace is unambiguous.
    ambiguous_drill_audio_owners: set[str] = field(default_factory=set)
    # Durable inline-only owners are real, but corpus audio cannot write their
    # deck YAML and drill audio owns only synthetic `drill-audio:` identities.
    inline_audio_decks: dict[str, tuple[Path, ...]] = field(default_factory=dict)
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
    """Return every deck definition under the configured deck tree."""
    paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in config.deck_dir.rglob(pattern)
    )
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        # Package filenames and repository clones can live on a
        # case-insensitive filesystem even when this check runs on Linux.
        by_stem.setdefault(path.stem.casefold(), []).append(path)
    duplicates = {
        stem: matches for stem, matches in by_stem.items() if len(matches) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(str(path) for path in matches)}"
            for stem, matches in sorted(duplicates.items())
        )
        raise JankiError(
            "Deck file stems must be unique under "
            f"{config.deck_dir}; duplicate stems: {details}"
        )
    return paths


def _inline_record_ids(deck_path: Path) -> set[str]:
    """Stable ids physically authored under one already-validated notes list."""
    raw = load_structured(deck_path)
    if not isinstance(raw, Mapping):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    notes = raw.get("notes") or []
    if not isinstance(notes, list):
        raise DataError(f"The notes section must be a list: {deck_path}")
    if not all(isinstance(item, dict) for item in notes):
        raise DataError(f"Each note must be a mapping: {deck_path}")
    return {
        str(item.get("id", "")).strip() or VocabularyRecord.from_dict(item).id
        for item in notes
    }


def collect_records(
    config: ProjectConfig,
    *,
    deck_paths: Sequence[Path] | None = None,
) -> RecordUniverse:
    """Every record janki manages, from the normalized file and every deck.

    A deck file that cannot be read is reported as a warning and skipped rather
    than aborting the report: status is the command you run to find out what is
    wrong, so it has to survive one broken file. ``janki validate`` is where a
    bad deck is an error.  ``deck_paths`` lets a stricter caller supply its own
    already-validated census; the ordinary project status still discovers all
    configured decks when it is omitted.
    """
    warnings: list[str] = []
    by_id: dict[str, VocabularyRecord] = {}

    normalized_ids: set[str] = set()
    normalized_sources: dict[str, SourceReference] = {}
    audio_records: list[VocabularyRecord] = []
    drill_audio_decks: dict[str, Path] = {}
    blocked_drill_audio_decks: dict[str, Path] = {}
    drill_candidates: list[tuple[VocabularyRecord, Path]] = []
    structural_drill_paths: dict[str, set[Path]] = {}
    inline_audio_paths: dict[str, set[Path]] = {}
    if config.normalized_file.exists():
        for record in load_records(config.normalized_file):
            by_id[record.id] = record
            normalized_ids.add(record.id)
            normalized_sources[record.id] = record.source
            audio_records.append(record)

    decks: list[DeckView] = []
    configured_decks = deck_files(config) if deck_paths is None else list(deck_paths)
    for deck_path in configured_decks:
        resolved_path = deck_path.resolve()
        try:
            if deck_kind(deck_path) == "kanji":
                # A character deck holds notes from the curated character
                # store, not vocabulary records. Reading it here would resolve
                # that store as a word list and report the deck as unreadable
                # — a warning about a deck that builds perfectly well.
                continue
        except JankiError as exc:
            warnings.append(f"skipping deck {deck_path}: {exc}")
            continue
        try:
            structural_owners = pattern_cards.declared_drill_audio_owner_ids(
                deck_path
            )
        except JankiError:
            structural_owners = frozenset()
        for owner_id in structural_owners:
            structural_drill_paths.setdefault(owner_id, set()).add(resolved_path)
        try:
            deck_config, records = resolve_deck_records(deck_path)
            declared_versions = deck_declared_record_versions(deck_path)
            audio_records.extend(declared_versions)
            for record_id in _inline_record_ids(deck_path) - normalized_ids:
                inline_audio_paths.setdefault(record_id, set()).add(resolved_path)
            kind = str(deck_config.get("kind") or "").strip().lower()
            if kind == "conjugation" and deck_config.get("drill_examples") is not None:
                drill_records = pattern_cards.drill_audio_records(deck_path, config)
                for record in drill_records:
                    drill_candidates.append((record, resolved_path))
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

    ordinary_audio_ids = {record.id for record in audio_records}
    drill_paths = structural_drill_paths
    for record, deck_path in drill_candidates:
        drill_paths.setdefault(record.id, set()).add(deck_path)
    ambiguous_drill_audio_owners = {
        record_id
        for record_id, paths in drill_paths.items()
        if len(paths) > 1 or record_id in ordinary_audio_ids
    }
    projected = {
        (record.id, deck_path): record for record, deck_path in drill_candidates
    }
    for record_id, paths in drill_paths.items():
        if record_id in ambiguous_drill_audio_owners:
            owners = ", ".join(str(path) for path in sorted(paths, key=str))
            collision = (
                " and a durable vocabulary record"
                if record_id in ordinary_audio_ids
                else ""
            )
            warnings.append(
                f"ambiguous drill audio owner {record_id!r} is claimed by "
                f"{owners}{collision}; give the decks distinct deck_id values "
                "before generating, recovering, rebuilding, or pruning audio"
            )
        else:
            deck_path = next(iter(paths))
            record = projected.get((record_id, deck_path))
            if record is None:
                blocked_drill_audio_decks[record_id] = deck_path
            else:
                drill_audio_decks[record_id] = deck_path
                audio_records.append(record)

    records = sorted(by_id.values(), key=lambda record: record.id)
    return RecordUniverse(
        records=records,
        decks=decks,
        normalized_count=sum(1 for record in records if record.id in normalized_ids),
        audio_records=audio_records,
        drill_audio_decks=drill_audio_decks,
        blocked_drill_audio_decks=blocked_drill_audio_decks,
        ambiguous_drill_audio_owners=ambiguous_drill_audio_owners,
        inline_audio_decks={
            record_id: tuple(sorted(paths, key=str))
            for record_id, paths in inline_audio_paths.items()
        },
        warnings=warnings,
        normalized_sources=normalized_sources,
    )


def surviving_ids(
    config: ProjectConfig, records: Iterable[VocabularyRecord]
) -> tuple[set[str], list[str]]:
    """Every record id the collection still carries, ``records`` being its file.

    The same definition of "the collection" :func:`collect_records` uses — the
    normalized file plus every deck's inline notes — reduced to ids, and asked
    the one question ``--replace`` has to answer before it deletes a ledger
    entry: *is this id really gone?* The records are passed in rather than
    re-read because the caller has just written them.

    A deck that will not resolve is named in the second element instead of being
    skipped. Its ids are unknown, so nothing can be *proved* absent, and a
    ledger entry holds ``added_at`` and ``exports`` that nothing reconstructs.

    Ids are read **before** a deck's include/exclude filters, via
    :func:`deck_declared_ids`. The filters answer "what does this deck build",
    and that is not this question: a note the deck declares and a filter drops
    is still in the file, still carries whatever a human wrote into it, and its
    GUID may already be in Anki. Treating it as absent is how a curated note
    gets a second copy under a new id, or loses its ledger entry.
    """
    ids = {record.id for record in records}
    unreadable: list[str] = []
    for deck_path in deck_files(config):
        try:
            ids.update(deck_declared_ids(deck_path))
        except JankiError as exc:
            unreadable.append(f"{deck_path}: {exc}")
            continue
    return ids, unreadable


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def pitch_accent_supported() -> bool:
    """Whether records carry pitch accent yet (the field arrives in M2.2)."""
    return any(item.name == "pitch_accent" for item in dataclasses.fields(VocabularyRecord))


@dataclass(frozen=True, slots=True)
class StagedFile:
    """One staging file and the ids waiting in it."""

    path: Path
    ids: list[str]
    reasons: dict[str, str]
    is_rich_extraction: bool
    pattern_source: str | None
    pattern_review_state: str
    pattern_review_issue: str


_RICH_EXTRACTION_META_KEYS = frozenset(
    {
        "review_run_id",
        "coverage",
        "prompt_provenance",
        "pattern_set",
        "reviewed_pattern_set",
        "candidate_accounting",
    }
)


def _looks_like_rich_extraction(meta: Mapping[str, Any]) -> bool:
    """Whether empty staging carries (or carried) the rich extraction contract."""
    return bool(_RICH_EXTRACTION_META_KEYS.intersection(meta))


def _pattern_review_state(
    meta: Mapping[str, Any],
    pattern_source: str | None,
    store: Mapping[str, patterns.PatternSet],
    store_issue: str,
) -> tuple[str, str]:
    """Structural readiness of a zero-record rich extraction for archival."""
    provenance = meta.get("prompt_provenance")
    version = (
        provenance.get("response_schema_version")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(meta.get("pattern_set"), Mapping)
        or pattern_source is None
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 3
    ):
        return (
            "staging-invalid",
            "its schema-v3-or-newer pattern_set, top-level source_file, or prompt "
            "provenance is missing",
        )
    try:
        patterns.PatternSet.from_dict(pattern_source, dict(meta["pattern_set"]))
        validate_coverage_facts(meta)
    except JankiError as exc:
        return "staging-invalid", str(exc)
    try:
        require_resolved_coverage(meta)
    except JankiError as exc:
        return "coverage-unresolved", str(exc)
    if store_issue:
        return "store-unreadable", store_issue
    stored = store.get(pattern_source)
    if stored is None:
        return "store-missing", ""
    run_id = meta.get("review_run_id")
    if (
        stored.review_run_id != run_id
        or not isinstance(provenance, Mapping)
        or stored.prompt_provenance != dict(provenance)
    ):
        return "store-stale", ""
    return ("reviewed" if stored.reviewed else "unreviewed"), ""


def collect_staged(
    config: ProjectConfig,
    *,
    parsed: Sequence[LiveStaging] | None = None,
) -> tuple[list[StagedFile], list[str]]:
    """Every row waiting for a human under ``staging_dir``.

    Held rows are the one category of record that is *not* in the collection and
    needs a person, so a report that never mentions them lets a review queue sit
    unnoticed forever. Unreadable files are warnings, not a dead report — the
    same rule the deck scan follows.

    ``parsed`` is every live staging file as some caller has *already* read it
    (`staging.live_staging`).  A half-megabyte staging file costs
    ~90 ms to parse, so a caller that also needs source names or proposal kinds
    reads once and hands the same entries to each answer — which also stops two
    answers about one directory from disagreeing because a file changed between
    their reads.  Unreadable entries still become the warnings above.
    """
    staged: list[StagedFile] = []
    readable: list[tuple[Path, list[VocabularyRecord], dict[str, Any]]] = []
    warnings: list[str] = []
    if not config.staging_dir.is_dir():
        # "No staging directory yet" and "staging_dir points at something that
        # is not a directory" are different answers. The second is a
        # misconfiguration: reporting "none" for it hides the problem until the
        # next import that needs to stage a row dies on it.
        if config.staging_dir.exists() or config.staging_dir.is_symlink():
            warnings.append(
                f"staging_dir {config.staging_dir} is not a directory, so no review "
                "queue could be read"
            )
        return staged, warnings
    if parsed is None:
        parsed = live_staging(config)
    for entry in parsed:
        if entry.records is None or entry.meta is None:
            warnings.append(f"skipping staging file {entry.path}: {entry.error}")
            continue
        readable.append((entry.path, entry.records, entry.meta))

    needs_pattern_store = any(
        not records and _looks_like_rich_extraction(meta)
        for _path, records, meta in readable
    )
    pattern_store: dict[str, patterns.PatternSet] = {}
    pattern_store_issue = ""
    if needs_pattern_store:
        try:
            pattern_store = patterns.load_store(config.patterns_file)
        except JankiError as exc:
            pattern_store_issue = str(exc)
            warnings.append(
                f"could not inspect pattern review state in {config.patterns_file}: {exc}"
            )

    for path, records, meta in readable:
        # Promote uses this exact top-level value as the pattern-store key.
        # Whitespace is tested only for nonblankness, never normalized away;
        # advertising a stripped key could make status's command disagree with
        # the transition it claims is ready.
        pattern_source_value = meta.get("source_file")
        pattern_source = (
            pattern_source_value
            if isinstance(pattern_source_value, str) and pattern_source_value.strip()
            else None
        )
        is_rich_extraction = _looks_like_rich_extraction(meta)
        pattern_review_state = "not-applicable"
        pattern_review_issue = ""
        if not records and is_rich_extraction:
            pattern_review_state, pattern_review_issue = _pattern_review_state(
                meta,
                pattern_source,
                pattern_store,
                pattern_store_issue,
            )
        staged.append(
            StagedFile(
                path=path,
                ids=[record.id for record in records],
                reasons={
                    record.id: str(record.source.raw_fields.get("hold_reason", ""))
                    for record in records
                },
                # Schema v3 introduced the attributed nested pattern answer.
                # Any reserved remnant is evidence to preserve: losing one half
                # must not make the other look like an ordinary empty file.
                is_rich_extraction=is_rich_extraction,
                pattern_source=pattern_source,
                pattern_review_state=pattern_review_state,
                pattern_review_issue=pattern_review_issue,
            )
        )
    return staged, warnings


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
    pending_audio_count: int
    record_ids: list[str]
    normalized_count: int
    inline_count: int
    by_source: list[tuple[str, int]]
    decks: list[DeckStatus]
    missing_audio: list[str]
    #: How many example sentences exist at all — the denominator for the count
    #: below, which is meaningless without one: "188 silent" reads very
    #: differently against 343 than against 190.
    example_count: int
    #: ``(record id, example index)`` per silent example sentence. Its own
    #: field, not folded into `missing_audio`, so the word-level count keeps
    #: meaning "this record cannot be heard at all".
    unvoiced_examples: list[tuple[str, int]]
    stale_audio: list[str]
    missing_enrichment: list[str]
    # None means "the schema has no pitch accent yet", which is not the same
    # answer as "no record is missing one".
    missing_pitch_accent: list[str] | None
    staging_dir: Path
    #: Paid calls whose money is not yet accounted for and therefore block a
    #: new authorization.
    #:
    #: Here because the extract command tells them to be. When a call is sent
    #: and no answer is captured, the message says "Run 'janki status' to see
    #: it" — and until this line `status` did not read the journal at all, so
    #: somebody following that instruction after a call that may have been
    #: billed was shown nothing about it.
    blocking_operations: tuple[operations.Operation, ...] = ()
    #: The complete actionable operations view: blockers plus durable forget
    #: decisions whose exact recovery-data cleanup still needs a retry.
    tracked_operations: tuple[operations.Operation, ...] = ()
    #: ``field name -> record ids`` whose value is still a model's claim: the
    #: mark extraction wrote and nobody who can read the card has settled.
    #:
    #: Here because a mark nobody can see does not do its job. It exists to say
    #: "this is a guess", and until this line it was visible only to the code
    #: that acts on it — so a wrong guess looked exactly like curated content,
    #: and the only way to find one was to already suspect it. That is how a
    #: sheet teaching おたふく = "mumps" shipped reading "homely woman".
    provisional: dict[str, list[str]] = field(default_factory=dict)
    staged: list[StagedFile] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.record_ids)

    @property
    def staged_count(self) -> int:
        return sum(len(item.ids) for item in self.staged)

    @property
    def provisional_ids(self) -> list[str]:
        """Every record with at least one unsettled field, in report order."""
        seen: dict[str, None] = {}
        for ids in self.provisional.values():
            seen.update(dict.fromkeys(ids))
        return list(seen)


def _provisional_by_field(
    records: Sequence[VocabularyRecord],
) -> dict[str, list[str]]:
    """Unsettled fields, grouped by field name, in ``records`` order.

    Only *active* marks. A stale one means the field was edited after
    extraction, so it is curated content wearing a mark the next enrich run
    will clear — reporting it as a model's guess would be the opposite of the
    truth.
    """
    found: dict[str, list[str]] = {}
    for record in records:
        for name in split_provisional(record)[0]:
            found.setdefault(name, []).append(record.id)
    return {name: found[name] for name in sorted(found)}


def build_report(
    config: ProjectConfig,
    universe: RecordUniverse,
    book: Ledger,
    *,
    word_provider: RenderProfile | None,
    example_provider: RenderProfile | None,
    staged: Sequence[StagedFile] = (),
) -> StatusReport:
    records = universe.records
    operation_journal = operations.OperationJournal.load(config.operations_file)
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
        pending_audio_count=len(book.pending_audio),
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
        example_count=sum(
            1 for record in records for example in record.examples if example.japanese
        ),
        unvoiced_examples=book.unvoiced_examples(records),
        stale_audio=book.stale_audio(
            universe.audio_records or records,
            word_provider=word_provider,
            example_provider=example_provider,
        ),
        missing_enrichment=Ledger.missing_enrichment(records),
        missing_pitch_accent=missing_pitch,
        provisional=_provisional_by_field(records),
        # `blocking`, not `needing_attention`: a call killed mid-dispatch is
        # money that may already be gone, and reporting only the states that
        # need a *decision* let `status` print "every call janki made either
        # landed or is recorded as finished" over exactly that entry.
        blocking_operations=tuple(operation_journal.blocking()),
        # Cleanup tombstones are accounted-for money, but a killed cleanup is
        # still an actionable journal entry and must remain discoverable.
        tracked_operations=tuple(operation_journal.tracked()),
        staging_dir=config.staging_dir,
        staged=list(staged),
    )


def resolve_collection(config: ProjectConfig) -> tuple[Path | None, str]:
    """Which collection to inspect, and why not, when there is none.

    Returns ``(path, note)``. A ``note`` without a path is never an error: not
    having Anki installed, or not having imported yet, is an ordinary state for
    a tool that builds packages, and `janki status` must stay useful there.
    """
    if config.anki_collection.strip():
        named = Path(config.anki_collection)   # already absolute, from the config
        if named.is_file():
            return named, ""
        return None, f"[anki] collection is {named}, which does not exist"

    profiles = find_profiles()
    wanted = config.anki_profile.strip()
    if wanted:
        # Read before the empty check: a user who named a profile and got total
        # silence has no way to tell janki looked somewhere else — an Anki
        # started with `-b`, a portable install, a different XDG_DATA_HOME.
        if wanted in profiles:
            return profiles[wanted], ""
        if not profiles:
            root = default_anki_root()
            where = f" under {root}" if root else " (no Anki directory found)"
            return None, f"[anki] profile {wanted!r} not found{where}"
        available = ", ".join(sorted(profiles))
        return None, f"[anki] profile {wanted!r} not found. Available: {available}"
    if not profiles:
        return None, ""
    if len(profiles) == 1:
        return next(iter(profiles.values())), ""
    # Guessing between profiles would report findings about a collection the
    # user never meant, which is worse than reporting nothing.
    return None, (
        "several Anki profiles found (" + ", ".join(sorted(profiles)) + "); set "
        "[anki] profile in janki.toml to check one"
    )


@dataclass(frozen=True, slots=True)
class NotetypeFinding:
    """One thing wrong with how a deck landed in Anki."""

    #: The deck stems that build this notetype, joined. Decks sharing a card set
    #: share a notetype — this repo's own two do — so a finding reported per
    #: deck printed the same problem and the same remedy twice.
    where: str
    message: str


def check_collection(
    collection: Path,
    decks: Iterable[tuple[str, int, str, int]],
) -> tuple[list[NotetypeFinding], list[str]]:
    """Compare what a build would write against what the collection holds.

    ``decks`` is ``(stem, model_id, model_name, field count)`` per deck — taken
    from the exporter rather than recomputed, so a deck that pins ``model_id``
    is read the way it is built.

    Returns findings and warnings separately: a collection janki cannot read is
    not a finding about the user's decks, and must not read like one.
    """
    try:
        notetypes = read_notetypes(collection)
    except CollectionError as exc:
        return [], [str(exc)]

    by_id = {notetype.id: notetype for notetype in notetypes}
    # Grouped by notetype, because decks with the same card set derive the same
    # model id — this repo's own two do — and one problem reported once per deck
    # is the same actionable line buried under copies of itself.
    # Keyed on the whole comparison, not the id: two decks may pin one id and
    # different names, and keeping only the first deck's name dropped the
    # collision finding for the second — the worst one to lose, since importing
    # both in sequence renames the notetype in the live collection.
    grouped: dict[tuple[int, str, int], list[str]] = {}
    for stem, model_id, model_name, fields in decks:
        grouped.setdefault((model_id, model_name, fields), []).append(stem)

    findings: list[NotetypeFinding] = []
    for (model_id, model_name, fields), stems in grouped.items():
        landed = by_id.get(model_id)
        if landed is None:
            # Never imported, or imported under a different id. Not a failure —
            # a deck built and not yet imported is the ordinary state.
            continue
        where = ", ".join(sorted(stems))
        if landed.field_count < fields:
            findings.append(NotetypeFinding(
                where,
                f"'{landed.name}' has {landed.field_count} fields where this "
                f"deck writes {fields}. Re-import with 'Merge Notetypes' ticked; "
                "without it the new fields never reach a note.",
            ))
        for clone in clone_suffix_of(landed.name, notetypes):
            # The documented failure leaves the clone *empty* and the notes on
            # the old notetype without the new fields. Saying "the notes on it
            # are on the wrong notetype" of a clone holding zero notes sends the
            # reader looking for cards that are not there.
            fate = (
                f"your {landed.note_count} note(s) stayed on '{landed.name}' "
                "without the new fields"
                if clone.note_count == 0
                else f"{clone.note_count} note(s) ended up on it instead of "
                f"'{landed.name}'"
            )
            findings.append(NotetypeFinding(
                where,
                f"'{clone.name}' sits beside '{landed.name}' — an import that "
                f"left 'Merge Notetypes' unticked, and {fate}.",
            ))
        if model_name.strip() != landed.name.strip():
            findings.append(NotetypeFinding(
                where,
                f"this deck builds notetype '{model_name}' but id {model_id} is "
                f"named '{landed.name}' in Anki. A rename is harmless; a "
                "collision is not.",
            ))
    return findings, []


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
    lines.append(f"Pending audio recovery: {report.pending_audio_count}")

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
    lines.append(
        f"Missing example audio: {len(report.unvoiced_examples)} of "
        f"{report.example_count} sentence(s)"
    )
    lines.append(f"Stale audio: {len(report.stale_audio)}")
    lines.append(
        f"Missing enrichment: {len(report.missing_enrichment)} "
        "(no meanings or example sentence)"
    )
    if report.blocking_operations:
        lines.append(
            f"Paid calls not accounted for: {len(report.blocking_operations)} "
            "(run 'janki operations' for what each one cost)"
        )
    cleanup_count = sum(
        operation.cleanup is not None
        for operation in report.tracked_operations
    )
    if cleanup_count:
        lines.append(
            f"Paid-call cleanup pending: {cleanup_count} "
            "(run 'janki operations' to resume it)"
        )
    if report.provisional:
        lines.append(
            "Unsettled model claims: "
            + ", ".join(
                f"{name} {len(ids)}" for name, ids in report.provisional.items()
            )
            + " (a model's guess; see --unsettled for what settles each)"
        )
    else:
        lines.append("Unsettled model claims: none")
    if report.missing_pitch_accent is None:
        lines.append("Missing pitch accent: n/a until the pitch-accent schema lands (M2.2)")
    else:
        lines.append(f"Missing pitch accent: {len(report.missing_pitch_accent)}")

    # An ordinary empty staging file is not a review queue: a reviewer who
    # decided no row was worth keeping leaves `records: []` behind. A rich
    # extraction with no card rows is different: its nested pattern answer is
    # attributed evidence that only the zero-record promote path archives safely.
    waiting = [item for item in report.staged if item.ids]
    pattern_only = [
        item for item in report.staged if not item.ids and item.is_rich_extraction
    ]
    empty = [
        item for item in report.staged if not item.ids and not item.is_rich_extraction
    ]
    if waiting:
        lines.append(
            f"Staged for review: {report.staged_count} row(s) in {len(waiting)} "
            f"file(s) under {display_path(report.staging_dir, root)} "
            "(not in the collection until a human resolves them)"
        )
        if pattern_only:
            lines.append(
                f"Pattern-only extraction review: {len(pattern_only)} file(s) under "
                f"{display_path(report.staging_dir, root)} remain live until "
                "promotion or recovery"
            )
    elif pattern_only:
        lines.append(
            f"Staged for review: no rows; {len(pattern_only)} pattern-only extraction "
            f"file(s) under {display_path(report.staging_dir, root)} remain live "
            "until promotion or recovery"
        )
    elif empty:
        lines.append(
            f"Staged for review: none ({len(empty)} tracked empty file(s) under "
            f"{display_path(report.staging_dir, root)}; no rows are waiting; "
            "preserve as repository data)"
        )
    else:
        lines.append("Staged for review: none")
    return lines


def format_staged(report: StatusReport) -> list[str]:
    waiting = [item for item in report.staged if item.ids]
    pattern_only = [
        item for item in report.staged if not item.ids and item.is_rich_extraction
    ]
    lines: list[str] = []
    if waiting:
        lines.append(f"Staged for review ({report.staged_count}):")
        for item in waiting:
            lines.append(f"{display_path(item.path, report.root)} ({len(item.ids)}):")
            for record_id in item.ids:
                reason = item.reasons.get(record_id) or ""
                lines.append(f"  {record_id}{f' — {reason}' if reason else ''}")
        lines.append("Resolve them in place, then run 'janki promote' on each file:")
        lines.append("it re-mints the malformed IDs above, checks every reading, and")
        lines.append("registers what lands. Moving them across by hand skips all three.")
    elif pattern_only:
        lines.append(
            f"Staged for review: no rows; {len(pattern_only)} pattern-only extraction "
            "file(s) remain live until completion."
        )
    else:
        lines.append("Staged for review: none.")
    for item in report.staged:
        if not item.ids:
            shown_path = display_path(item.path, report.root)
            if item.is_rich_extraction:
                lines.append(
                    f"{shown_path} holds no card rows, but carries rich extraction "
                    "evidence."
                )
                if item.pattern_review_state == "staging-invalid":
                    lines.append(
                        "Its extraction metadata is incomplete or invalid: "
                        f"{item.pattern_review_issue}. Restore this exact staging "
                        "artifact from version control before promotion; do not "
                        "delete it by hand."
                    )
                    continue
                if item.pattern_review_state == "store-unreadable":
                    lines.append(
                        "Its matching pattern-store state could not be verified. Fix "
                        "the warning above before review or promotion; do not delete "
                        "this staging file by hand."
                    )
                    continue
                if item.pattern_review_state == "coverage-unresolved":
                    lines.append(
                        "Its coverage block still requires the repository owner's "
                        "resolution before promotion. Status cannot grant that "
                        "approval; preserve this file and do not delete it by hand."
                    )
                    continue
                if item.pattern_review_state == "store-missing":
                    lines.append(
                        f"Its matching {item.pattern_source!r} pattern-store entry is "
                        "missing. There is no automatic recovery command: restore the "
                        "exact entry from history if it exists, or reconstruct it from "
                        "this staging file's pattern_set copy before review or "
                        "promotion. Do not delete this staging file by hand."
                    )
                    continue
                if item.pattern_review_state == "store-stale":
                    lines.append(
                        f"Its matching {item.pattern_source!r} pattern-store entry "
                        "belongs to a different extraction run. Preserve both and "
                        "restore the exact entry from history if present, or reconcile "
                        "it from this staging file's pattern_set copy before promotion. "
                        "There is no automatic recovery command; do not delete this "
                        "staging file by hand."
                    )
                    continue
                command_root = f"janki --root {shlex.quote(str(report.root))}"
                if item.pattern_review_state == "unreviewed":
                    lines.append(
                        "Review its matching pattern set: "
                        f"{command_root} patterns --review "
                        f"{shlex.quote(str(item.pattern_source))}"
                    )
                else:
                    lines.append("Its matching pattern set is already reviewed.")
                lines.append(
                    f"Then run {command_root} promote {shlex.quote(str(item.path))} "
                    "to preserve its extraction evidence in the done archive; do "
                    "not delete it by hand."
                )
                continue
            lines.append(
                f"{shown_path} holds no rows — no review rows are waiting. It "
                "must be preserved as tracked staging data; status has no completion action "
                "for it."
            )
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


#: What actually settles each provisional field, because they differ and the
#: difference costs money to get wrong.
#:
#: `enrich --ai` can only settle a field it writes, and `AI_FIELDS` does not
#: include `part_of_speech` — so piping a pos-marked record into it buys a
#: call that cannot clear the mark. Worse, doing that with
#: `--force-fields meanings` puts a *curated* meaning up for replacement on a
#: record that was only ever marked for its part of speech.
SETTLES = {
    "meanings": "a person editing the card, or 'enrich --ai --force-fields meanings'",
    "part_of_speech": "'enrich --jpdb', which reconciles it against the exact word",
}


def format_operations(
    attention: Sequence[operations.Operation],
    *,
    journal_path: Path,
    noun: str = "needing a person",
) -> list[str]:
    """Every actionable paid call in the given set, and its exact next move.

    A durable cleanup intent takes precedence over the old lifecycle state:
    its money decision is already settled, and only ordinary exact cleanup
    remains. Otherwise the state decides the next move: an answer that arrived
    has bytes to inspect, while a vanished reply makes a retry risk paying
    twice.

    `noun` names the view the caller asked for in the heading.
    """
    if not attention:
        return [
            f"Paid calls {noun}: none — every call janki made either "
            "landed or is recorded as finished."
        ]
    lines = [f"Paid calls {noun} ({len(attention)}):"]
    for op in attention:
        quoted_id = shlex.quote(op.operation_id)
        lines.append(f"  {op.operation_id}  {op.kind} · {op.source_file}")
        if op.cleanup is not None:
            lines.append(
                f"    state: cleanup pending (forget recorded from "
                f"{op.state!r} at {op.updated_at})"
            )
            lines.append(
                "    the forget decision is already recorded; exact "
                "recovery-data cleanup did not finish, and no new --force "
                "decision is needed"
            )
            lines.append(
                "    finish cleanup: "
                f"'janki operations --forget {quoted_id}'"
            )
            continue
        reply = operations.reply_observation(journal_path, op)
        frame_reply = operations.response_spool_observation(journal_path, op)
        lines.append(f"    state: {op.state}, authorized {op.authorized_at}")
        if reply.readable:
            lines.append(
                "    the exact reply is recoverable: "
                f"'janki operations --show-reply {quoted_id}'"
            )
        elif reply.recorded:
            lines.append(
                "    the journal records a captured reply, but its exact "
                "recovery bytes are unavailable"
            )
        elif frame_reply.readable:
            lines.append(
                "    durable response frames remain; inspect the exact "
                "frame-preserving view (the command revalidates their binding): "
                f"'janki operations --show-reply {quoted_id}'"
            )
        elif frame_reply.recovery_pending:
            lines.append(
                "    an exact newly durable response frame awaits recovery"
            )
        elif frame_reply.recorded:
            lines.append("    exact response frames are unavailable")
        elif reply.interrupted:
            lines.append(
                "    answer capture was interrupted; operation-bound recovery "
                "evidence remains, but it does not prove a complete reply"
            )
        elif op.money_may_have_been_spent:
            # Only when nothing came back. A captured reply was billed too, but
            # its answer is on disk — telling someone a retry "risks a second
            # charge" there points them at re-running instead of at the file
            # they already paid for.
            lines.append(
                "    it was sent and no answer came back, so re-running it "
                "risks a second charge"
            )
        if op.detail:
            lines.append(f"    {op.detail}")
        if op.state == "committed":
            lines.append(
                "    action: retire this committed operation with "
                f"'janki operations --forget {quoted_id}'"
            )
        elif reply.readable:
            lines.append(
                "    after reading the saved reply, explicitly discard its "
                "recovery copy: "
                f"'janki operations --forget {quoted_id} --force'"
            )
        elif reply.recorded:
            lines.append(
                "    if you accept that unavailable recovery copy as lost, "
                "explicitly discard its record: "
                f"'janki operations --forget {quoted_id} --force'"
            )
        elif (
            frame_reply.recorded
            and op.state not in operations.IN_FLIGHT
            and not op.money_may_have_been_spent
        ):
            lines.append(
                "    action: retire this before-send operation with "
                f"'janki operations --forget {quoted_id}'"
            )
        elif (
            frame_reply.recovery_pending
            and op.state not in operations.IN_FLIGHT
        ):
            lines.append(
                "    rerun the exact matching provider command to recover the "
                "newly durable frame; otherwise explicitly discard it"
            )
            lines.append(
                "    action: explicitly discard it with "
                f"'janki operations --forget {quoted_id} --force'"
            )
        elif frame_reply.readable and op.state not in operations.IN_FLIGHT:
            lines.append(
                "    rerun the exact matching provider command to recover any "
                "terminal response already on disk; otherwise, after inspection, "
                "explicitly discard the frames"
            )
            lines.append(
                "    action: explicitly discard them with "
                f"'janki operations --forget {quoted_id} --force'"
            )
        elif (
            frame_reply.recorded
            and op.state not in operations.IN_FLIGHT
        ):
            lines.append(
                "    action: explicitly discard their record with "
                f"'janki operations --forget {quoted_id} --force'"
            )
        elif op.state == "authorized":
            lines.append(
                "    clear this unused authority: "
                f"'janki operations --end {quoted_id}'"
            )
        elif op.state in operations.IN_FLIGHT:
            # The one state whose next move depends on something janki cannot
            # see. A live call finishes on its own; a killed one never will,
            # and only the person at the keyboard knows which this is.
            lines.append(
                "    if nothing is actually running, that process is gone: "
                f"'janki operations --end {quoted_id}'"
            )
        elif op.state == "outcome_unknown":
            lines.append(
                "    once you have dealt with what it may have cost: "
                f"'janki operations --forget {quoted_id}'"
            )
    return lines


def format_provisional(report: StatusReport) -> list[str]:
    """Which records are still carrying a guess, and in which field.

    Grouped by field rather than by record because the remedy differs: a
    meaning is settled by reading the card, a part of speech by a dictionary
    that resolved the exact word.
    """
    if not report.provisional:
        return [
            "Unsettled model claims: none — every extracted field has been "
            "confirmed, edited, or settled from a dictionary."
        ]
    lines: list[str] = []
    for name, ids in report.provisional.items():
        lines.append(f"Unsettled {name} ({len(ids)}):")
        lines.append(f"  settled by: {SETTLES[name]}")
        lines.extend(f"  {record_id}" for record_id in ids)
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


# jpdb vocabulary ids are positive integers. Nothing else may be grouped on:
# see :func:`_jpdb_vid`.
_VID_PATTERN = re.compile(r"\d+")


def _jpdb_vid(record: VocabularyRecord) -> str:
    """The jpdb vocabulary id an importer stashed in the source row, if any.

    Compared numerically where possible: a round-tripped file can hold the
    same vid as ``1577980``, ``"1577980"`` or ``1577980.0``, and a typing
    accident must not hide a duplicate.

    Anything that is not a plausible vid answers ``""`` — a jpdb vid is a
    positive integer, and this value *groups records for deletion*. A column of
    placeholders (``-``, ``n/a``, ``unknown``, a JSON ``null`` a loader
    stringified) is shared by every record that has no vid at all, so accepting
    one would collapse the whole file into a single "duplicate" group whose
    printed remedy is to delete the others.
    """
    raw = str(record.source.raw_fields.get("vid", "")).strip()
    if not raw:
        return ""
    try:
        number = float(raw)
    except ValueError:
        candidate = raw
    else:
        candidate = str(int(number)) if number.is_integer() else raw
    return candidate if _VID_PATTERN.fullmatch(candidate) else ""


def find_duplicates(records: Iterable[VocabularyRecord]) -> list[DuplicateGroup]:
    """Every duplicate class, one record scope at a time.

    The universe this reads unions the collection with every deck's records, so
    a standalone deck's copies arrive beside the shared words they were copied
    from. Those pairs are the intended shape of a standalone deck — same
    spelling, same reading, sometimes the same jpdb vid — and reporting them
    would print a remedy ("pick a survivor and delete the other") that is wrong
    for every one of them, burying the real duplicates in noise.

    So the whole report runs per scope, using exactly the same three tests
    below. Two copies of one word inside one standalone deck are still a
    duplicate, and the shared collection's report is unchanged.
    """
    partitions: dict[str, list[VocabularyRecord]] = {}
    for record in records:
        try:
            scope = record_scope_id(record.id)
        except IdentityError:
            # A malformed reserved id belongs to no scope janki can name. It
            # cannot be grouped with anything, so it gets a partition of its
            # own rather than taking the whole report down.
            scope = f"\x1f{record.id}"
        partitions.setdefault(scope, []).append(record)
    return [
        group
        for scope in sorted(partitions)
        for group in _duplicates_in_one_scope(partitions[scope])
    ]


def _duplicates_in_one_scope(
    records: Iterable[VocabularyRecord],
) -> list[DuplicateGroup]:
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
    # Files sharing a fingerprint with the file a rebuilt entry claimed: for
    # example an encoded historical clip beside its current WAV replacement.
    ambiguous_media: int
    media_dir: Path


# When several files claim one fingerprint and *the record names none of them*,
# the rebuilt entry binds the first by this order. It is a last resort: janki
# has used more than one audio format over its history, so the extension alone
# is not evidence. When the record names a file, that beats this outright — see
# `_claim_for`.
_EXTENSION_PREFERENCE: tuple[str, ...] = (".wav", ".mp3", ".ogg", ".m4a")


def _claim_for(candidates: list[Path], named: str) -> Path:
    """Which file a rebuilt entry should bind, preferring the one named.

    The record is the better evidence: a provider or format switch can leave
    two suffixes until a prune, and ranking by extension may otherwise bind the
    stale one while calling the file the card actually plays ambiguous.
    """
    if named:
        wanted = Path(named).name
        for candidate in candidates:
            if candidate.name == wanted:
                return candidate
    return min(candidates, key=_media_rank)


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


def _first(paths: list[Path] | None, named: str = "") -> Path | None:
    """The preferred claimant of a fingerprint, if any file claims it.

    ``named`` is what the record itself plays. It wins over the extension
    ranking, which is only a guess and became a bad one when a second engine
    started writing a second format.
    """
    if not paths:
        return None
    return _claim_for(paths, named)


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
    audio_records: Iterable[VocabularyRecord] | None = None,
) -> RebuildSummary:
    """Reconstruct the ledger entries that records and media files still prove.

    Sources come from each vocabulary record's own ``source`` — except where
    ``sources_by_id`` (the normalized file's own sources, see
    ``RecordUniverse.normalized_sources``) knows better: a deck note that
    shadows a normalized record carries a default-manual source that is not
    the record's provenance, and the ledger is committed to git. ``audio_records``
    may additionally carry synthetic rich-drill owners: their files and ledger
    rows are durable, but they are not vocabulary provenance. Audio comes from
    files whose names match the filename fingerprints. Neither carries a date,
    so the dates written here are today's — a reconstruction, not history.
    Export state is not reconstructible at all: nothing outside the ledger
    records which build included which note. The caller must say so.

    Existing entries are left alone: a real ``janki audio`` entry knows the
    provider and voice, and must not be replaced by a rebuilt one that does not.
    """
    media = _media_by_fingerprint(media_dir)
    # Keyed by path, not by name: `media` is built with rglob, so two files in
    # different sub-directories can share both a fingerprint and a basename, and
    # a name-keyed set would report the loser as claimed — counted as neither
    # ambiguous nor unmatched, and so absent from the accounting entirely.
    claimed: set[Path] = set()
    sources = word_audio = example_audio = unprovable = 0
    source_records = list(records)
    durable_audio_records = (
        list(audio_records) if audio_records is not None else source_records
    )

    for record in source_records:
        # Only a *default* source defers to the normalized file. A deck note
        # that states its own source said something deliberate, into a file that
        # is committed to git; the fallback exists for the note that said
        # nothing and therefore carries `SourceReference()` by construction.
        source = record.source
        if source == SourceReference():
            source = (sources_by_id or {}).get(record.id, source)
        if book.record_source_seen(record.id, source.type or "manual", source.imported_from):
            sources += 1

    for record in durable_audio_records:
        word_file = _first(
            media.get(word_audio_filename_fingerprint(record)), record.audio
        )
        if word_file is not None:
            claimed.add(word_file)
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
                    speed=REBUILT_SPEED,
                    content_fp=content_fp,
                    rebuilt=True,
                )
                word_audio += 1

        for example in record.examples:
            if not example.japanese:
                continue
            example_file = _first(
                media.get(example_audio_filename_fingerprint(record, example)),
                example.audio,
            )
            if example_file is None:
                continue
            claimed.add(example_file)
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
                speed=REBUILT_SPEED,
                content_fp=example_audio_content_fingerprint(example),
                rebuilt=True,
            )
            example_audio += 1

    unmatched = ambiguous = 0
    for paths in media.values():
        if any(path in claimed for path in paths):
            # The fingerprint bound to a record; every unclaimed twin is a
            # competing candidate the user must know exists. The ledger entry
            # stores only `path.name`, so two twins with one basename are still
            # indistinguishable *in the ledger* — a separate, pre-existing
            # ambiguity — but the summary now counts both files.
            ambiguous += sum(1 for path in paths if path not in claimed)
        else:
            unmatched += len(paths)
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
            "with a file a rebuilt entry claimed. The entry binds the file the "
            "record names; failing that, by extension "
            f"({', '.join(_EXTENSION_PREFERENCE)}). Check which file janki "
            "actually generated and delete the other"
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
    staged: bool = False,
    provisional: bool = False,
    provisional_field: str = "",
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
    if staged:
        for item in report.staged:
            chosen.extend(item.ids)
    if provisional:
        # Narrowed to one field when asked, because the remedies differ: a
        # single list piped into the meanings pass spends money on records
        # that pass cannot settle.
        chosen.extend(
            report.provisional.get(provisional_field, [])
            if provisional_field
            else report.provisional_ids
        )
    if not (unexported or missing_audio or duplicates or staged or provisional):
        chosen = list(report.record_ids)
    return list(dict.fromkeys(chosen))
