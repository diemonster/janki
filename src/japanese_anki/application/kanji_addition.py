"""Plan and add kanji reference data for one bounded record scope.

The workbench carries the exact ids returned by promotion.  This service turns
that scope into the distinct characters those records introduce, without
letting an empty or partly unknown request widen into the whole collection.
The ordinary CLI uses the separate corpus planner, then shares the same fetch
and additive store transaction.

Two sources, decided **independently**.  ``data/kanji.json`` holds KANJIDIC's
meanings, on/kun inventory and strokes; ``data/jpdb_readings.json`` holds the
reading percentages jpdb reports and the words jpdb itself binds to each
reading.  A character can be complete in one and absent from the other, so
each store gets its own already-known/to-fetch decision from its own contents.
Having looked a character up before is not a reason to ship a word card whose
character block has no reported readings on it.

Network lookups deliberately run without either store's lock.  Once they have
finished, execution locks each exact file, reloads its latest contents, and
merges only the successful answers.  An unrelated lookup that completed while
this one was on the network therefore survives the save.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from japanese_anki import jpdb_kanji, kanji
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    exclusive_path_lock,
    load_records_snapshot,
    read_text_bound,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "KanjiAdditionError",
    "KanjiAdditionPlan",
    "KanjiAdditionResult",
    "KanjiFetchFailure",
    "execute_kanji_addition",
    "plan_corpus_kanji_addition",
    "plan_kanji_addition",
]


class KanjiAdditionError(JankiError):
    """A requested kanji addition is not one exact canonical record scope."""


@dataclass(frozen=True, slots=True)
class KanjiFetchFailure:
    """One character whose reference sources did not return a usable answer."""

    character: str
    message: str


@dataclass(frozen=True, slots=True)
class KanjiAdditionPlan:
    """The exact records, characters, and store state one run observed."""

    project_root: Path
    canonical_path: Path
    canonical_fingerprint: str
    kanji_path: Path
    record_ids: tuple[str, ...]
    characters: tuple[str, ...]
    already_known: tuple[str, ...]
    to_fetch: tuple[str, ...]
    refresh: bool
    targeted: bool
    fingerprint: str
    #: The provider facts store, and its own scope decision. Separate fields
    #: rather than a merged list because the two stores are separately
    #: complete: `already_known` above says nothing about whether jpdb has
    #: been asked about the same character.
    jpdb_readings_path: Path = Path()
    readings_already_known: tuple[str, ...] = ()
    readings_to_fetch: tuple[str, ...] = ()
    #: The exact facts-store bytes the two tuples above were decided from. A
    #: file that has moved since is a scope nobody confirmed: its saved facts
    #: may already cover what this plan is about to pay to request.
    readings_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class KanjiAdditionResult:
    """Exact fetch and merge facts from executing one plan."""

    plan: KanjiAdditionPlan
    successes: tuple[str, ...]
    failures: tuple[KanjiFetchFailure, ...]
    added: tuple[str, ...]
    replaced: tuple[str, ...]
    preserved_concurrent: tuple[str, ...]
    store_entries: int
    saved: bool
    #: The same six facts for the provider store, kept apart for the same
    #: reason the plan keeps its scope apart: one store can save while the
    #: other refuses every character it asked about.
    reading_successes: tuple[str, ...] = ()
    reading_failures: tuple[KanjiFetchFailure, ...] = ()
    reading_added: tuple[str, ...] = ()
    reading_replaced: tuple[str, ...] = ()
    reading_preserved_concurrent: tuple[str, ...] = ()
    reading_store_entries: int = 0
    readings_saved: bool = False


def _plan_fingerprint(plan: KanjiAdditionPlan) -> str:
    claims = {
        "project_root": str(plan.project_root),
        "canonical_path": str(plan.canonical_path),
        "canonical_fingerprint": plan.canonical_fingerprint,
        "kanji_path": str(plan.kanji_path),
        "record_ids": plan.record_ids,
        "characters": plan.characters,
        "already_known": plan.already_known,
        "to_fetch": plan.to_fetch,
        "refresh": plan.refresh,
        "targeted": plan.targeted,
        "jpdb_readings_path": str(plan.jpdb_readings_path),
        "readings_already_known": plan.readings_already_known,
        "readings_to_fetch": plan.readings_to_fetch,
        "readings_fingerprint": plan.readings_fingerprint,
    }
    encoded = json.dumps(
        claims, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_snapshot(
    config: ProjectConfig,
) -> tuple[list[VocabularyRecord], Path, str]:
    path = config.normalized_file.resolve()
    records, revision = load_records_snapshot(path)
    marker = b"missing\0" if revision.text is None else b"present\0"
    wire = b"" if revision.text is None else revision.text.encode("utf-8")
    return records, path, hashlib.sha256(marker + wire).hexdigest()


def _facts_fingerprint(path: Path) -> str:
    """The exact facts-store bytes a scope decision was read from.

    A missing file is its own state rather than an empty one: "nobody has
    fetched anything" and "the file was emptied" are different histories, and
    a plan that could not tell them apart would treat a deletion as no change.
    """
    try:
        text = read_text_bound(path)
    except FileNotFoundError:
        return hashlib.sha256(b"missing\0").hexdigest()
    except (JankiError, OSError) as exc:
        raise KanjiAdditionError(f"Could not read {path}: {exc}") from exc
    return hashlib.sha256(b"present\0" + text.encode("utf-8")).hexdigest()


def _characters(records: Sequence[VocabularyRecord]) -> tuple[str, ...]:
    found: dict[str, None] = {}
    for record in records:
        found.update(dict.fromkeys(kanji.kanji_in(record.expression)))
    return tuple(found)


def _plan(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    *,
    canonical_path: Path,
    canonical_fingerprint: str,
    record_ids: tuple[str, ...],
    refresh: bool,
    targeted: bool,
) -> KanjiAdditionPlan:
    path = config.kanji_file.resolve()
    readings_path = config.jpdb_readings_file.resolve()
    characters = _characters(records)
    if records:
        store = kanji.load_store(path)
        already_known = tuple(
            character for character in characters if character in store.entries
        )
        to_fetch = characters if refresh else tuple(store.missing(characters))
        # Decided from the facts file's own contents. A character KANJIDIC
        # already covers still needs asking about here if jpdb never was.
        facts = jpdb_kanji.load_readings(readings_path)
        readings_fingerprint = _facts_fingerprint(readings_path)
        readings_already_known = tuple(
            character for character in characters if character in facts
        )
        readings_to_fetch = (
            characters
            if refresh
            else tuple(character for character in characters if character not in facts)
        )
    else:
        # Preserve the CLI's empty-collection fast path: no record means there
        # is no store decision to make, so a stale or unreadable store is
        # irrelevant to the successful no-op.
        already_known = ()
        to_fetch = ()
        readings_fingerprint = ""
        readings_already_known = ()
        readings_to_fetch = ()
    draft = KanjiAdditionPlan(
        project_root=config.root.resolve(),
        canonical_path=canonical_path,
        canonical_fingerprint=canonical_fingerprint,
        kanji_path=path,
        record_ids=record_ids,
        characters=characters,
        already_known=already_known,
        to_fetch=to_fetch,
        refresh=refresh,
        targeted=targeted,
        fingerprint="",
        jpdb_readings_path=readings_path,
        readings_already_known=readings_already_known,
        readings_to_fetch=readings_to_fetch,
        readings_fingerprint=readings_fingerprint,
    )
    return replace(draft, fingerprint=_plan_fingerprint(draft))


def _exact_record_ids(record_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(record_ids, str):
        raise KanjiAdditionError("Record ids must be a nonempty sequence, not text.")
    selected = tuple(record_ids)
    if not selected:
        raise KanjiAdditionError(
            "Targeted kanji addition needs at least one exact promoted record id."
        )
    if any(not isinstance(record_id, str) or not record_id.strip() for record_id in selected):
        raise KanjiAdditionError("Every targeted kanji record id must be nonblank text.")
    repeated = [
        record_id
        for record_id, count in Counter(selected).items()
        if count > 1
    ]
    if repeated:
        raise KanjiAdditionError(
            f"Targeted kanji record id {repeated[0]!r} was supplied more than once."
        )
    return selected


def plan_kanji_addition(
    config: ProjectConfig, record_ids: Sequence[str]
) -> KanjiAdditionPlan:
    """Plan only missing characters for one exact, nonempty canonical id set."""
    selected_ids = _exact_record_ids(record_ids)
    records, canonical_path, canonical_fingerprint = _canonical_snapshot(config)
    counts = Counter(record.id for record in records)
    missing = [record_id for record_id in selected_ids if counts[record_id] == 0]
    if missing:
        raise KanjiAdditionError(
            f"No canonical record has id {missing[0]!r}. Nothing was looked up."
        )
    ambiguous = [record_id for record_id in selected_ids if counts[record_id] > 1]
    if ambiguous:
        raise KanjiAdditionError(
            f"Canonical record id {ambiguous[0]!r} occurs more than once; "
            "targeted kanji addition requires one exact record."
        )
    by_id = {record.id: record for record in records}
    selected = tuple(by_id[record_id] for record_id in selected_ids)
    return _plan(
        config,
        selected,
        canonical_path=canonical_path,
        canonical_fingerprint=canonical_fingerprint,
        record_ids=selected_ids,
        refresh=False,
        targeted=True,
    )


def plan_corpus_kanji_addition(
    config: ProjectConfig, *, refresh: bool = False
) -> KanjiAdditionPlan:
    """Plan the CLI's explicit whole-collection maintenance operation."""
    if not isinstance(refresh, bool):
        raise KanjiAdditionError("Kanji refresh must be true or false.")
    records, canonical_path, canonical_fingerprint = _canonical_snapshot(config)
    return _plan(
        config,
        records,
        canonical_path=canonical_path,
        canonical_fingerprint=canonical_fingerprint,
        record_ids=tuple(record.id for record in records),
        refresh=refresh,
        targeted=False,
    )


def _provider_reader(
    config: ProjectConfig, *, refresh: bool
) -> Callable[[str], jpdb_kanji.CharacterReadings]:
    """The provider lookup, bound to this project's private page cache.

    Cached pages are reused, so a character asked about before costs no
    request; ``refresh`` is the only thing that re-requests one.
    """

    def read(character: str) -> jpdb_kanji.CharacterReadings:
        return jpdb_kanji.fetch_character(
            character, html_cache=config.jpdb_html_cache, refresh=refresh
        )

    return read


def execute_kanji_addition(
    config: ProjectConfig,
    plan: KanjiAdditionPlan,
    *,
    expected_fingerprint: str,
    fetch: Callable[[str], kanji.KanjiInfo] | None = None,
    fetch_readings: Callable[[str], jpdb_kanji.CharacterReadings] | None = None,
) -> KanjiAdditionResult:
    """Fetch outside the store locks, then merge into each store's latest state."""
    expected = (
        config.root.resolve(),
        config.normalized_file.resolve(),
        config.kanji_file.resolve(),
        config.jpdb_readings_file.resolve(),
    )
    actual = (
        plan.project_root,
        plan.canonical_path,
        plan.kanji_path,
        plan.jpdb_readings_path,
    )
    if actual != expected:
        raise KanjiAdditionError(
            "This kanji plan belongs to a different repository configuration."
        )
    if plan.targeted and not plan.record_ids:
        raise KanjiAdditionError("A targeted kanji plan cannot have an empty scope.")
    if (
        not isinstance(expected_fingerprint, str)
        or not isinstance(plan.fingerprint, str)
        or not secrets.compare_digest(expected_fingerprint, plan.fingerprint)
        or not secrets.compare_digest(_plan_fingerprint(plan), plan.fingerprint)
    ):
        raise KanjiAdditionError(
            "[kanji-plan-stale] the exact kanji plan changed after it was prepared. "
            "Nothing was fetched or written; plan it again."
        )

    current_records, current_path, current_fingerprint = _canonical_snapshot(config)
    if (
        current_path != plan.canonical_path
        or not secrets.compare_digest(
            current_fingerprint, plan.canonical_fingerprint
        )
    ):
        raise KanjiAdditionError(
            "[kanji-scope-stale] the canonical records changed after this exact "
            "kanji scope was prepared. Nothing was fetched or written; plan it "
            "again."
        )
    current_by_id = {record.id: record for record in current_records}
    if plan.targeted:
        selected = tuple(current_by_id.get(record_id) for record_id in plan.record_ids)
        if any(record is None for record in selected):
            raise KanjiAdditionError(
                "[kanji-scope-stale] a targeted canonical record disappeared. "
                "Nothing was fetched or written; plan it again."
            )
        scoped_records = tuple(record for record in selected if record is not None)
    else:
        if plan.record_ids != tuple(record.id for record in current_records):
            raise KanjiAdditionError(
                "[kanji-scope-stale] the corpus kanji scope no longer names the "
                "canonical collection. Nothing was fetched or written."
            )
        scoped_records = tuple(current_records)
    if _characters(scoped_records) != plan.characters:
        raise KanjiAdditionError(
            "[kanji-scope-stale] the exact kanji characters no longer match the "
            "canonical records. Nothing was fetched or written; plan it again."
        )

    # Checked before a request leaves, not after: the plan's to-fetch lists
    # were decided from these exact bytes, and a facts file that has moved may
    # already hold the pages this run is about to pay to request again.
    if plan.readings_to_fetch or plan.readings_already_known:
        current_facts = _facts_fingerprint(plan.jpdb_readings_path)
        if not secrets.compare_digest(current_facts, plan.readings_fingerprint):
            raise KanjiAdditionError(
                "[kanji-readings-scope-stale] the saved reading facts changed "
                "after this exact scope was prepared. Nothing was fetched or "
                "written; plan it again."
            )

    fetch_one = fetch or kanji.fetch_kanji
    fetched: list[kanji.KanjiInfo] = []
    failures: list[KanjiFetchFailure] = []
    for character in plan.to_fetch:
        try:
            info = fetch_one(character)
            if info.character != character:
                raise KanjiAdditionError(
                    f"Lookup for {character} returned {info.character!r} instead."
                )
        except JankiError as exc:
            failures.append(KanjiFetchFailure(character, str(exc)))
        else:
            fetched.append(info)

    fetch_reading_one = fetch_readings or _provider_reader(config, refresh=plan.refresh)
    fetched_readings: list[jpdb_kanji.CharacterReadings] = []
    reading_failures: list[KanjiFetchFailure] = []
    for character in plan.readings_to_fetch:
        try:
            entry = fetch_reading_one(character)
            if entry.character != character:
                raise KanjiAdditionError(
                    f"The reading page for {character} answered for "
                    f"{entry.character!r} instead."
                )
        except JankiError as exc:
            reading_failures.append(KanjiFetchFailure(character, str(exc)))
        else:
            fetched_readings.append(entry)

    added: list[str] = []
    replaced: list[str] = []
    preserved: list[str] = []
    reading_added: list[str] = []
    reading_replaced: list[str] = []
    reading_preserved: list[str] = []
    saved = False
    readings_saved = False
    # Keep the canonical lock from the final scope check through both merges.
    # Otherwise a canonical writer can win the seam between this read and a
    # store lock, leaving a successful save attached to stale records.
    with exclusive_path_lock(plan.canonical_path):
        _records_after, path_after, fingerprint_after = _canonical_snapshot(config)
        if path_after != plan.canonical_path or not secrets.compare_digest(
            fingerprint_after, plan.canonical_fingerprint
        ):
            raise KanjiAdditionError(
                "[kanji-scope-stale] the canonical records changed while kanji "
                "facts were fetched. No kanji reference data was written; plan "
                "it again."
            )

        with exclusive_path_lock(plan.kanji_path):
            latest = kanji.load_store(plan.kanji_path)
            missing_known = tuple(
                character
                for character in plan.already_known
                if character not in latest.entries
            )
            if missing_known:
                raise KanjiAdditionError(
                    "[kanji-store-stale] previously available kanji reference "
                    f"data for {missing_known[0]} disappeared while facts were "
                    "fetched. Nothing was written; plan it again."
                )

            if plan.to_fetch:
                for info in fetched:
                    if info.character in latest.entries:
                        if not plan.refresh:
                            preserved.append(info.character)
                            continue
                        replaced.append(info.character)
                    else:
                        added.append(info.character)
                    latest.entries[info.character] = info
                kanji.save_store(plan.kanji_path, latest)
                saved = True
            store_entries = len(latest.entries)

        with exclusive_path_lock(plan.jpdb_readings_path):
            latest_facts = jpdb_kanji.load_readings(plan.jpdb_readings_path)
            missing_facts = tuple(
                character
                for character in plan.readings_already_known
                if character not in latest_facts
            )
            if missing_facts:
                raise KanjiAdditionError(
                    "[kanji-readings-store-stale] previously saved reading facts "
                    f"for {missing_facts[0]} disappeared while facts were "
                    "fetched. No reading facts were written; plan it again."
                )

            if plan.readings_to_fetch:
                for entry in fetched_readings:
                    if entry.character in latest_facts:
                        if not plan.refresh:
                            reading_preserved.append(entry.character)
                            continue
                        reading_replaced.append(entry.character)
                    else:
                        reading_added.append(entry.character)
                    latest_facts[entry.character] = entry
                jpdb_kanji.save_readings(plan.jpdb_readings_path, latest_facts)
                readings_saved = True
            reading_store_entries = len(latest_facts)

    return KanjiAdditionResult(
        plan=plan,
        successes=tuple(info.character for info in fetched),
        failures=tuple(failures),
        added=tuple(added),
        replaced=tuple(replaced),
        preserved_concurrent=tuple(preserved),
        store_entries=store_entries,
        saved=saved,
        reading_successes=tuple(entry.character for entry in fetched_readings),
        reading_failures=tuple(reading_failures),
        reading_added=tuple(reading_added),
        reading_replaced=tuple(reading_replaced),
        reading_preserved_concurrent=tuple(reading_preserved),
        reading_store_entries=reading_store_entries,
        readings_saved=readings_saved,
    )
