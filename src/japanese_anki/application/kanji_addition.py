"""Plan and add kanji reference data for one bounded record scope.

The workbench carries the exact ids returned by promotion.  This service turns
that scope into the distinct characters those records introduce, without
letting an empty or partly unknown request widen into the whole collection.
The ordinary CLI uses the separate corpus planner, then shares the same fetch
and additive store transaction.

Network lookups deliberately run without the kanji-file lock.  Once they have
finished, execution locks that exact file, reloads its latest contents, and
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

from japanese_anki import kanji
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import exclusive_path_lock, load_records_snapshot
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
    characters = _characters(records)
    if records:
        store = kanji.load_store(path)
        already_known = tuple(
            character for character in characters if character in store.entries
        )
        to_fetch = characters if refresh else tuple(store.missing(characters))
    else:
        # Preserve the CLI's empty-collection fast path: no record means there
        # is no kanji-store decision to make, so a stale or unreadable store is
        # irrelevant to the successful no-op.
        already_known = ()
        to_fetch = ()
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


def execute_kanji_addition(
    config: ProjectConfig,
    plan: KanjiAdditionPlan,
    *,
    expected_fingerprint: str,
    fetch: Callable[[str], kanji.KanjiInfo] | None = None,
) -> KanjiAdditionResult:
    """Fetch outside the store lock, then merge successes into its latest state."""
    expected = (
        config.root.resolve(),
        config.normalized_file.resolve(),
        config.kanji_file.resolve(),
    )
    actual = (plan.project_root, plan.canonical_path, plan.kanji_path)
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

    added: list[str] = []
    replaced: list[str] = []
    preserved: list[str] = []
    # Keep the canonical lock from the final scope check through the kanji
    # merge.  Otherwise a canonical writer can win the seam between this read
    # and the kanji lock, leaving a successful save attached to stale records.
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

            if not plan.to_fetch:
                return KanjiAdditionResult(
                    plan=plan,
                    successes=(),
                    failures=(),
                    added=(),
                    replaced=(),
                    preserved_concurrent=(),
                    store_entries=len(latest.entries),
                    saved=False,
                )

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
            store_entries = len(latest.entries)

    return KanjiAdditionResult(
        plan=plan,
        successes=tuple(info.character for info in fetched),
        failures=tuple(failures),
        added=tuple(added),
        replaced=tuple(replaced),
        preserved_concurrent=tuple(preserved),
        store_entries=store_entries,
        saved=True,
    )
