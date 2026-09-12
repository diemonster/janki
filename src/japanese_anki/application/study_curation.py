"""Cross-source curation: one owner decision, written to every occurrence.

A study job's parts each stage their own proposals, and one identity can be
proposed by more than one part with different printed forms. This module lets
the owner settle that once, over the **current edited staged values**, and
writes the result into every occurrence's own staged record.

Four rules shape it.

**The current staged values decide.** Proposals are read from each part's
staging document's editable ``records`` list — what a reviewer has in front of
them right now — joined for display to the immutable ``candidate_accounting``
beside it, which is never edited, never backfilled and never the source of a
choice. Collision groups are computed at read time from the per-part documents;
no combined accounting record is written and no synthetic multi-source staging
document is fabricated. Nothing merges first-wins, and no local logic picks
between two Japanese values: the owner does, or the model's existing extraction
pass already did.

**The decision precedes its first write.** The intent is appended to the job
document — CAS-bound and fsynced — carrying the complete prepared text of every
file it will write, before any of them is touched. A digest cannot be replayed
after a crash, and a resume that had to recompute the payload from a store that
has since moved is not a resume. A decision that was not durably recorded before
its first write does not exist. A decision that was never finished or abandoned
still blocks its successor, and a replan over one that *was* closed carries a
``supersedes`` edge back to it — the original and its evidence are never
deleted or rewritten.

**All locks, then all prechecks, then any write.** The shared staging mutation
coordination lock comes first, before any path lock; then every affected path's
lock in sorted order, so two concurrent curations cannot deadlock; then a
precheck of *every* bound file while holding all of them. Per-file
compare-and-swap alone is insufficient — it permits a partial application whose
remaining files then refuse, leaving a decision recovery cannot finish.

**The writer stays the writer.** This module computes and supplies exact
expected data and reproduces no writer logic: ``staging.prepare_record_update``
renders the round-trip edit and proves it changed nothing else, and
``staging.apply_prepared_update`` performs the bound replacement. Deleting a
cell or a table is an ordinary local curation edit, not an extraction defect,
and it never costs a paid retry.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki import extract, staging
from japanese_anki.application import study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import exclusive_path_lock
from japanese_anki.models import SourceFormsTable, VocabularyRecord

__all__ = [
    "CURATION_FIELDS",
    "CURATION_LOCK_NAME",
    "MISSING_DIGEST",
    "CurationBarrier",
    "CurationChoice",
    "CurationGroup",
    "CurationOccurrence",
    "CurationOutcome",
    "CurationPlan",
    "PreparedFileUpdate",
    "StudyCurationError",
    "abandon_intent",
    "apply_curation",
    "curation_guard",
    "curation_intent_digest",
    "open_curation_barriers",
    "pending_curation_refusal",
    "plan_curation",
    "read_curation_groups",
    "resume_curation",
    "staging_curation_guard",
    "unresolved_curation_predecessors",
]


class StudyCurationError(JankiError):
    """A cross-source curation could not be planned, applied or resumed."""


#: A lock name, not a file janki writes, modelled on the existing
#: ``.janki-audio-operation`` operation lock. ``io.exclusive_path_lock`` keeps
#: its lock files in a private per-user directory, so naming one here creates
#: no untracked artifact under ``data/``.
CURATION_LOCK_NAME = ".janki-curation-lock"

#: The record fields a curation decision may write. One today: the canonical
#: printed-forms table. Everything else on a staged row is either the review
#: editor's work or a writer's own accounting.
CURATION_FIELDS: tuple[str, ...] = ("source_forms",)

#: What an abandonment records for a bound path it could not read at all. It
#: is not a digest and cannot collide with one — every real entry is 64 hex
#: characters — so a snapshot that says this is saying the file was gone, not
#: that it hashed to something.
MISSING_DIGEST = "missing"

_ACTIONS: tuple[str, ...] = ("replace", "remove")


@contextlib.contextmanager
def staging_curation_guard(staging_dir: Path) -> Iterator[None]:
    """The one coordination lock, for a caller that holds paths not a config.

    `workbench.review.ReviewPanel` is opened from three explicit paths rather
    than a `ProjectConfig` — deliberately, so an approval binds the exact files
    it read — and it is nonetheless an entry that writes staged bytes. It takes
    the guard through here. Same lock file, same exclusion, one convention:
    :func:`curation_guard` is this with the staging directory read off the
    configuration.
    """

    with exclusive_path_lock(Path(staging_dir) / CURATION_LOCK_NAME):
        yield


@contextlib.contextmanager
def curation_guard(config: ProjectConfig) -> Iterator[None]:
    """The staging mutation coordination lock, taken before any other lock.

    Every mutation entry that can reach promotion acquires this **before** its
    own staging, canonical, audio-operation or finish-record lock, and keeps
    its existing internal order after it. It is deliberately non-reentrant, so
    an inner promotion call must use its unguarded entry rather than trying to
    take this again under a lock its caller already holds.
    """

    with staging_curation_guard(config.staging_dir):
        yield


# --- reading the current staged values ----------------------------------------


@dataclass(frozen=True, slots=True)
class CurationOccurrence:
    """One part's current staged row for one identity."""

    part_name: str
    staging_path: Path
    row_index: int
    record_id: str
    expression: str
    source_forms: SourceFormsTable | None

    @property
    def forms_wire(self) -> Any:
        return None if self.source_forms is None else self.source_forms.to_dict()


@dataclass(frozen=True, slots=True)
class CurationGroup:
    """Every part that currently stages one identity, and what each proposes."""

    record_id: str
    expression: str
    occurrences: tuple[CurationOccurrence, ...]

    @property
    def conflicting(self) -> bool:
        """Whether the parts disagree about this identity's printed table.

        A structural comparison of the canonical values, never a reading of
        their Japanese: two parts that staged different cells disagree, and the
        owner settles it.
        """

        seen = {
            json.dumps(item.forms_wire, ensure_ascii=False, sort_keys=True)
            for item in self.occurrences
        }
        return len(seen) > 1


def _job_staging_documents(
    config: ProjectConfig, job_id: str
) -> list[tuple[str, Path, list[VocabularyRecord]]]:
    """Each published part's current staging document, in receipt order.

    A part with no staging file yet is not an error: a job curates what its
    parts have actually staged, and a part still waiting for its answer has
    nothing to curate.
    """

    found: list[tuple[str, Path, list[VocabularyRecord]]] = []
    for part_name in study_job.job_published_parts(config, job_id):
        path = extract.staging_path(config.staging_dir, part_name)
        if not path.is_file():
            continue
        try:
            records, _meta = staging.read_staging(path)
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise StudyCurationError(
                f"Could not read the staged proposals in {path.name}: {exc}"
            ) from exc
        found.append((part_name, path, list(records)))
    return found


def read_curation_groups(
    config: ProjectConfig, job_id: str
) -> tuple[CurationGroup, ...]:
    """Every identity this job currently stages, and which parts propose it.

    A read: nothing is written, nothing is merged, and no part's document is
    rewritten to create a combined accounting record. Groups are ordered by
    first sighting so the list is stable between calls.
    """

    order: list[str] = []
    grouped: dict[str, list[CurationOccurrence]] = {}
    for part_name, path, records in _job_staging_documents(config, job_id):
        for index, record in enumerate(records):
            if record.id not in grouped:
                order.append(record.id)
                grouped[record.id] = []
            grouped[record.id].append(
                CurationOccurrence(
                    part_name=part_name,
                    staging_path=path,
                    row_index=index,
                    record_id=record.id,
                    expression=record.expression,
                    source_forms=record.source_forms,
                )
            )
    return tuple(
        CurationGroup(
            record_id=record_id,
            expression=grouped[record_id][0].expression,
            occurrences=tuple(grouped[record_id]),
        )
        for record_id in order
    )


# --- the owner's decision ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CurationChoice:
    """One owner decision about one identity's printed table.

    ``key`` names one cell of that table; empty means the table itself.
    ``replace`` with an empty string is the printed-blank spelling — a declared
    row whose value the source printed empty — and ``remove`` deletes the cell
    or the whole table outright.
    """

    record_id: str
    action: str
    field: str = "source_forms"
    key: str = ""
    value: Any = None

    def describe(self) -> str:
        subject = f"{self.field}[{self.key}]" if self.key else self.field
        if self.action == "remove":
            return f"remove {subject} from {self.record_id}"
        if self.key:
            return f"set {subject} of {self.record_id} to {self.value!r}"
        return f"set {subject} of {self.record_id}"


@dataclass(frozen=True, slots=True)
class PreparedFileUpdate:
    """The complete prepared text of one file this decision will write."""

    staging_path: str
    sha256_before: str
    sha256_after: str
    content_after: str
    record_ids: tuple[str, ...]
    operations: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "staging_path": self.staging_path,
            "sha256_before": self.sha256_before,
            "sha256_after": self.sha256_after,
            "content_after": self.content_after,
            "record_ids": list(self.record_ids),
            "operations": [dict(item) for item in self.operations],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedFileUpdate:
        try:
            return cls(
                staging_path=str(raw["staging_path"]),
                sha256_before=str(raw["sha256_before"]),
                sha256_after=str(raw["sha256_after"]),
                content_after=str(raw["content_after"]),
                record_ids=tuple(str(item) for item in raw.get("record_ids") or ()),
                operations=tuple(
                    dict(item) for item in raw.get("operations") or ()
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StudyCurationError(
                f"A recorded curation intent names an unreadable file update: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CurationPlan:
    """What one decision would write, before anything is recorded or written."""

    job_id: str
    decision: str
    choices: tuple[CurationChoice, ...]
    prepared: tuple[PreparedFileUpdate, ...]
    #: Identities this decision names that no part currently stages.
    absent: tuple[str, ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(item.staging_path for item in self.prepared)


@dataclass(frozen=True, slots=True)
class CurationOutcome:
    """What one applied, resumed or abandoned decision actually did."""

    job_id: str
    intent_id: str
    state: str
    written: tuple[str, ...] = ()
    already_current: tuple[str, ...] = ()
    detail: str = ""
    #: ``(staging_path, sha256)`` as this call measured it with the locks held,
    #: in sorted path order. Evidence about the files, not a restatement of the
    #: intent — an abandonment records nothing else, because nothing else is
    #: true about a decision that was never completed.
    observed: tuple[tuple[str, str], ...] = ()
    #: The closed decision this one was recorded over, read back from the
    #: intent's own ``supersedes`` edge. Empty when it replaces nothing.
    supersedes: str = ""


def _relative(config: ProjectConfig, path: Path) -> str:
    resolved = Path(os.path.realpath(path))
    root = Path(os.path.realpath(config.root))
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise StudyCurationError(
            f"A curated staging file lives in this repository; {path} does not."
        ) from exc


def _checked_choices(choices: Sequence[CurationChoice]) -> tuple[CurationChoice, ...]:
    checked = tuple(choices)
    if not checked:
        raise StudyCurationError("A curation decision names at least one choice.")
    for choice in checked:
        if not isinstance(choice, CurationChoice):
            raise StudyCurationError("A curation decision is made of CurationChoice.")
        if choice.field not in CURATION_FIELDS:
            raise StudyCurationError(
                f"Curation writes {', '.join(CURATION_FIELDS)}, not "
                f"{choice.field!r}."
            )
        if choice.action not in _ACTIONS:
            raise StudyCurationError(
                f"A curation choice is {' or '.join(_ACTIONS)}, not "
                f"{choice.action!r}."
            )
        if not isinstance(choice.record_id, str) or not choice.record_id.strip():
            raise StudyCurationError("A curation choice names one record identity.")
        if choice.action == "replace" and not choice.key:
            # Replacing the whole table means adopting an exact canonical
            # value; it is validated here rather than at the document so a
            # malformed adoption never reaches a prepared text.
            SourceFormsTable.from_dict(choice.value)
        if choice.action == "replace" and choice.key and not isinstance(
            choice.value, str
        ):
            raise StudyCurationError(
                f"The cell {choice.key!r} of {choice.record_id} is text; a "
                "printed blank is the empty string."
            )
    seen: set[tuple[str, str, str]] = set()
    for choice in checked:
        key = (choice.record_id, choice.field, choice.key)
        if key in seen:
            raise StudyCurationError(
                f"This decision names {choice.describe()} twice; one identity's "
                "cell is settled once."
            )
        seen.add(key)
    return checked


def _apply_choice(
    record: VocabularyRecord, choice: CurationChoice
) -> VocabularyRecord:
    """The record this choice makes, and the structural edit it needs.

    Returns the edited record plus the explicit operation that expresses the
    part of it a round-trip diff cannot: the diff writes changed keys and never
    removes one, which is what keeps a hand-written row from acquiring empty
    schema fields.
    """

    table = record.source_forms
    if choice.action == "remove" and not choice.key:
        if table is None:
            raise StudyCurationError(
                f"{record.id} stages no {choice.field}, so there is nothing to "
                "remove."
            )
        return replace(record, source_forms=None)
    if choice.action == "remove":
        if table is None or choice.key not in table.cells:
            raise StudyCurationError(
                f"{record.id} stages no {choice.field} cell {choice.key!r}, so "
                "there is nothing to remove."
            )
        cells = dict(table.cells)
        del cells[choice.key]
        return replace(record, source_forms=replace(table, cells=cells))
    if not choice.key:
        return replace(
            record, source_forms=SourceFormsTable.from_dict(choice.value)
        )
    if table is None or all(
        column.id != choice.key for column in table.columns
    ):
        raise StudyCurationError(
            f"{record.id} declares no {choice.field} column {choice.key!r}, so "
            "it has no cell there to settle. Curation edits the cells a source "
            "printed; it does not invent a column."
        )
    cells = dict(table.cells)
    cells[choice.key] = choice.value
    return replace(record, source_forms=replace(table, cells=cells))


def _operations_for(
    before: VocabularyRecord,
    after: VocabularyRecord,
    *,
    row_index: int,
) -> list[staging.FieldOperation]:
    """The explicit deletions this row's change needs, and only those."""

    operations: list[staging.FieldOperation] = []
    if before.source_forms is not None and after.source_forms is None:
        operations.append(
            staging.FieldOperation(
                row_index=row_index,
                record_id=before.id,
                field="source_forms",
                action="remove",
            )
        )
        return operations
    if before.source_forms is None or after.source_forms is None:
        return operations
    gone = [
        key for key in before.source_forms.cells if key not in after.source_forms.cells
    ]
    for key in gone:
        operations.append(
            staging.FieldOperation(
                row_index=row_index,
                record_id=before.id,
                field="source_forms",
                action="remove",
                key=key,
            )
        )
    return operations


def plan_curation(
    config: ProjectConfig,
    job_id: str,
    choices: Sequence[CurationChoice],
    *,
    decision: str,
) -> CurationPlan:
    """Compute every prepared file update, from one read of each file.

    Nothing is recorded and nothing is written. The chosen value lands in
    **every** occurrence's own staged record for that identity, taken from the
    current editable records, so whichever part promotes last writes the same
    bytes and cannot overwrite the choice.
    """

    if not isinstance(decision, str) or not decision.strip():
        raise StudyCurationError(
            "A curation decision records the owner's own words for what they "
            "chose; janki does not compose one."
        )
    checked = _checked_choices(choices)
    study_job.load_study_job(config, job_id)
    documents = _job_staging_documents(config, job_id)

    touched: set[str] = set()
    prepared: list[PreparedFileUpdate] = []
    for _part_name, path, records in documents:
        after = list(records)
        operations: list[staging.FieldOperation] = []
        changed: list[str] = []
        for index, record in enumerate(records):
            for choice in checked:
                if record.id != choice.record_id:
                    continue
                touched.add(choice.record_id)
                edited = _apply_choice(after[index], choice)
                operations.extend(
                    _operations_for(after[index], edited, row_index=index)
                )
                after[index] = edited
                if record.id not in changed:
                    changed.append(record.id)
        if not changed:
            continue
        update = staging.prepare_record_update(
            path, after, operations=tuple(operations)
        )
        if not update.applied:
            # The file already holds exactly what this decision would write.
            # Recording it would reserve a write with nothing behind it.
            continue
        prepared.append(
            PreparedFileUpdate(
                staging_path=_relative(config, path),
                sha256_before=update.sha256_before,
                sha256_after=update.sha256_after,
                content_after=update.text,
                record_ids=tuple(changed),
                operations=tuple(
                    {
                        "row_index": item.row_index,
                        "record_id": item.record_id,
                        "field": item.field,
                        "action": item.action,
                        "key": item.key or "",
                    }
                    for item in operations
                ),
            )
        )
    if not prepared:
        raise StudyCurationError(
            "This decision changes nothing that is staged: every occurrence it "
            "names already holds exactly these values. Nothing was recorded."
        )
    return CurationPlan(
        job_id=job_id,
        decision=decision.strip(),
        choices=checked,
        # Sorted so two concurrent curations take their locks in one order.
        prepared=tuple(sorted(prepared, key=lambda item: item.staging_path)),
        absent=tuple(
            sorted(choice.record_id for choice in checked if choice.record_id not in touched)
        ),
    )


def _curation_intent(
    plan: CurationPlan, *, supersedes: str = ""
) -> study_job.ActionIntent:
    return study_job.ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="curation",
        decided_at=datetime.now(UTC).isoformat(),
        supersedes=supersedes,
        reserves={
            "prepared": [item.to_dict() for item in plan.prepared],
        },
        bindings={
            "job_id": plan.job_id,
            "decision": plan.decision,
            "choices": [
                {
                    "record_id": choice.record_id,
                    "field": choice.field,
                    "action": choice.action,
                    "key": choice.key,
                    "value": choice.value,
                }
                for choice in plan.choices
            ],
        },
    )


def _plan_subjects(plan: CurationPlan) -> frozenset[tuple[str, str]]:
    """What this decision settles: one ``(record_id, field)`` per choice."""

    return frozenset((choice.record_id, choice.field) for choice in plan.choices)


def _recorded_subjects(intent: study_job.ActionIntent) -> frozenset[tuple[str, str]]:
    """What a recorded decision settled, read from its own bindings.

    Tolerant on purpose: an entry it cannot read is not a subject rather than a
    refusal, because this is used to *link* a replan to what it replaces and to
    widen a refusal, never to admit one. The bindings are written by
    :func:`_curation_intent` and immutable once appended.
    """

    raw = intent.bindings.get("choices")
    if not isinstance(raw, list):
        return frozenset()
    found: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        record_id = str(item.get("record_id") or "")
        field = str(item.get("field") or "")
        if record_id and field:
            found.add((record_id, field))
    return frozenset(found)


def _replanned_predecessor(job: study_job.StudyJob, plan: CurationPlan) -> str:
    """The decision this one settles again, latest first, or ``""``.

    §6.2's replan: an owner who changes their mind about an identity's printed
    cell gets a fresh intent computed against fresh snapshots, and the
    successor carries ``supersedes: intent_id`` — there is no mutable
    ``superseded_by`` to write on the original. A decision about a *different*
    identity is not a replan and gets no edge, because a durable log that
    claimed it replaced an unrelated decision would be saying something untrue.

    State is deliberately not filtered here: :func:`study_job.append_intent` is
    the thing that enforces the predecessor exists and already has an outcome,
    under the job document's own compare-and-swap.
    """

    subjects = _plan_subjects(plan)
    for intent in reversed(job.intents):
        if intent.kind != "curation":
            continue
        if _recorded_subjects(intent) & subjects:
            return intent.intent_id
    return ""


def _prepared_of(intent: study_job.ActionIntent) -> tuple[PreparedFileUpdate, ...]:
    raw = intent.reserves.get("prepared")
    if not isinstance(raw, list) or not raw:
        raise StudyCurationError(
            f"Curation intent {intent.intent_id} records no prepared file "
            "update, so janki cannot say what it was about to write."
        )
    return tuple(PreparedFileUpdate.from_dict(item) for item in raw)


def _digest(path: Path, staging_path: str) -> str:
    """One bound file's digest, over its exact bytes.

    The same primitive every other digest in this transaction uses: the
    prepared snapshot's, the barrier's, and the writer's own compare-and-swap.
    Reading through text I/O would translate a `\\r\\n` away before hashing and
    answer a different question than the one the digests were computed for —
    refusing a file that is unchanged, or accepting one that is not.
    """

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise StudyCurationError(
            f"{staging_path} could not be read for this decision: {exc}. "
            "Nothing was written."
        ) from exc


@contextlib.contextmanager
def _bound_locks(
    config: ProjectConfig, prepared: Sequence[PreparedFileUpdate]
) -> Iterator[tuple[list[Path], list[PreparedFileUpdate]]]:
    """Every bound path's exclusive lock, in sorted repository-relative order.

    Sorted so two concurrent decisions cannot deadlock, and deliberately not
    nested: ``io.exclusive_path_lock`` opens a fresh handle per call, so taking
    one of these inside another one in the same process would block forever.
    Each phase that needs the whole set takes it once and lets it go.
    """

    ordered = sorted(prepared, key=lambda item: item.staging_path)
    paths = [config.root / item.staging_path for item in ordered]
    with contextlib.ExitStack() as locks:
        for path in paths:
            locks.enter_context(exclusive_path_lock(path))
        yield paths, ordered


def _precheck_bound_files(
    paths: Sequence[Path],
    ordered: Sequence[PreparedFileUpdate],
    *,
    consequence: str,
) -> tuple[list[tuple[Path, PreparedFileUpdate]], list[str]]:
    """Every bound file at one of its two recorded digests, or a refusal.

    Splits them into the ones this decision still has to write and the ones
    already at its recorded result. One mismatch refuses the whole set, because
    a per-file check permits a partial application whose remaining files then
    refuse — leaving a decision recovery cannot finish. ``consequence`` says
    what the caller's refusal leaves behind, which differs by phase: before the
    intent is recorded there is nothing to leave open.
    """

    pending: list[tuple[Path, PreparedFileUpdate]] = []
    current: list[str] = []
    for path, item in zip(paths, ordered, strict=True):
        digest = _digest(path, item.staging_path)
        if digest == item.sha256_after:
            current.append(item.staging_path)
            continue
        if digest != item.sha256_before:
            raise StudyCurationError(
                f"{item.staging_path} is at {digest}, which is neither the "
                f"{item.sha256_before} this decision was computed over nor "
                f"the {item.sha256_after} it would leave. One mismatch "
                f"refuses the whole decision: {consequence}"
            )
        pending.append((path, item))
    return pending, current


def _write_prepared(
    config: ProjectConfig, prepared: Sequence[PreparedFileUpdate]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """All locks, then all prechecks, then every write. In sorted path order.

    What comes back is what was *measured*: which files this call wrote, which
    were already at the recorded result, and every bound file's digest as it
    stands with the locks still held. The decision is complete only when every
    one of those digests is the recorded ``sha256_after``, so a write that did
    not land the recorded bytes refuses instead of being reported as applied.
    """

    with _bound_locks(config, prepared) as (paths, ordered):
        pending, current = _precheck_bound_files(
            paths,
            ordered,
            consequence=(
                "nothing was written, and this intent stays open."
            ),
        )
        written: list[str] = []
        for path, item in pending:
            staging.apply_prepared_update(
                path,
                staging.PreparedStagingUpdate(
                    text=item.content_after,
                    sha256_before=item.sha256_before,
                    sha256_after=item.sha256_after,
                    applied=True,
                ),
            )
            written.append(item.staging_path)
        observed: list[tuple[str, str]] = []
        for path, item in zip(paths, ordered, strict=True):
            landed = _digest(path, item.staging_path)
            observed.append((item.staging_path, landed))
            if landed != item.sha256_after:
                raise StudyCurationError(
                    f"{item.staging_path} is at {landed} after this decision "
                    f"was written, not the {item.sha256_after} it recorded. "
                    "The decision is not complete, so nothing closed it and "
                    "this intent stays open."
                )
    return tuple(written), tuple(current), tuple(observed)


def _refuse_unresolved_predecessor(
    config: ProjectConfig, plan: CurationPlan
) -> None:
    """Refuse while a decision this one would overtake is still open.

    §6.2: *a partially applied intent that was never finished or abandoned
    still blocks its successor.* Writing a fresh choice over a bound file that
    an open decision is still waiting at leaves that decision at neither digest
    it recorded — no replay can finish it, its barrier holds every file it
    named, and a decision that succeeded has left the job needing a recovery
    action. So the successor refuses here, before anything is recorded, and
    names the two routes out of the state that is actually there.
    """

    blocking = unresolved_curation_predecessors(config, plan)
    if not blocking:
        return
    barrier = blocking[0]
    wanted = set(plan.paths)
    overlap = [path for path in barrier.paths if path in wanted]
    if overlap:
        why = (
            "It binds " + ", ".join(overlap) + ", which this decision would "
            "write: writing this one now would leave that one at neither "
            "digest it recorded, so no replay could finish it and its barrier "
            "would keep holding every file it named."
        )
    else:
        why = (
            "It settles the same identity this decision settles again, and a "
            "decision that was never finished or abandoned still blocks its "
            "successor."
        )
    raise StudyCurationError(
        f"Study job {barrier.job_id} recorded curation decision "
        f"{barrier.intent_id} and it has no outcome yet ({barrier.decision}). "
        f"{why} {_barrier_routes(barrier)} Nothing was recorded and nothing "
        "was written."
    )


def apply_curation(
    config: ProjectConfig, plan: CurationPlan
) -> CurationOutcome:
    """Record the decision, then write exactly what it recorded.

    The guard is taken first, before every path lock. Under it, in order:

    1. **Every unresolved predecessor is consulted** — an open decision over a
       file this one would write, or over the identity it settles again. One
       refuses this decision outright, naming the resume and abandon routes,
       because overtaking an open decision wedges it.
    2. The whole bound set is re-read and prechecked *before* anything is
       recorded: both surfaces plan and then apply, and a file that moved in
       between would otherwise produce an intent that was unsatisfiable the
       moment it became durable — a barrier born already needing to be
       abandoned. A plan that no longer fits the files refuses here, having
       recorded nothing.
    3. A replan over a **closed** predecessor carries ``supersedes`` back to
       it, so the durable log records what this decision replaced and
       ``study_job.append_intent`` is the thing enforcing that the predecessor
       already has an outcome.

    Only then is the intent appended and fsynced, before the first effect;
    then all the path locks again, all the prechecks again, and only then any
    write. The ``applied`` outcome is appended once every bound file is at its
    recorded ``sha256_after``. The second precheck is not redundant: the locks
    are released between the two phases — they cannot be nested — and it is the
    one that guards the writes.
    """

    with curation_guard(config):
        _refuse_unresolved_predecessor(config, plan)
        with _bound_locks(config, plan.prepared) as (paths, ordered):
            _precheck_bound_files(
                paths,
                ordered,
                consequence=(
                    "nothing was recorded and nothing was written. Plan this "
                    "decision again over the files as they now stand."
                ),
            )
        job = study_job.load_study_job(config, plan.job_id)
        intent = _curation_intent(
            plan, supersedes=_replanned_predecessor(job, plan)
        )
        study_job.append_intent(
            config, plan.job_id, intent, expected_revision=job.revision
        )
        return _settle(config, plan.job_id, intent.intent_id)


def curation_intent_digest(intent: study_job.ActionIntent) -> str:
    """The digest an owner's abandonment binds: this exact recorded intent.

    Over the intent's own canonical wire, which is immutable once appended, so
    the value an owner is shown is the value their control carries back. It
    identifies *which* decision they are closing, and it is not a claim about
    any staging file.
    """

    payload = json.dumps(intent.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def abandon_intent(
    config: ProjectConfig,
    job_id: str,
    intent_id: str,
    *,
    expected_intent_sha256: str,
) -> CurationOutcome:
    """Close one recorded decision without writing it. The owner's call.

    The recovery for a decision that can never be satisfied: the process died
    between the fsynced intent and its writes, and then a bound file moved —
    an ordinary staging edit, a coverage approval, a hand-edited YAML — so it
    is now at neither recorded digest and ``resume_curation`` will refuse
    forever. Without this the barrier holds every file the decision named,
    permanently.

    It is **not** a rollback. Nothing is reverted, no file is restored and no
    evidence is deleted: it re-reads every bound path under the same locks,
    records the mixed snapshot it actually measured — which paths are at
    ``sha256_after``, which at ``sha256_before``, which at neither — and
    appends the ``abandoned`` outcome. That closes the intent, which lifts the
    barrier and unblocks the successor: an owner who still wants the change
    plans a fresh decision over these observed states.

    Nothing infers this. It is reached only from an owner control that names
    the exact intent and carries ``expected_intent_sha256`` from the same
    reading, so an intent that is not the one they were looking at refuses.
    """

    with curation_guard(config):
        job = study_job.load_study_job(config, job_id)
        intent = job.intent(intent_id)
        if intent.kind != "curation":
            raise StudyCurationError(
                f"Intent {intent_id} of study job {job_id} is a {intent.kind} "
                "action, not a curation decision."
            )
        if intent_id in job.closed_intent_ids:
            raise StudyCurationError(
                f"Curation intent {intent_id} already has an outcome; outcomes "
                "are append-only and an existing one is never rewritten."
            )
        digest = curation_intent_digest(intent)
        if not isinstance(expected_intent_sha256, str) or (
            expected_intent_sha256 != digest
        ):
            raise StudyCurationError(
                f"This abandonment names curation intent {intent_id} as "
                f"{expected_intent_sha256!r}, and study job {job_id} records it "
                f"as {digest}. Read the decision again and decide over what it "
                "actually says; nothing was closed and nothing was written."
            )
        prepared = _prepared_of(intent)
        observed: list[tuple[str, str]] = []
        at_result: list[str] = []
        at_start: list[str] = []
        at_neither: list[str] = []
        with _bound_locks(config, prepared) as (paths, ordered):
            for path, item in zip(paths, ordered, strict=True):
                try:
                    landed = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    # A path that cannot be read is part of the snapshot, not a
                    # reason to refuse: abandoning is what an owner does when
                    # the bound set has already moved out from under them.
                    landed = MISSING_DIGEST
                observed.append((item.staging_path, landed))
                if landed == item.sha256_after:
                    at_result.append(item.staging_path)
                elif landed == item.sha256_before:
                    at_start.append(item.staging_path)
                else:
                    at_neither.append(item.staging_path)
            current = study_job.load_study_job(config, job_id)
            study_job.append_outcome(
                config,
                job_id,
                study_job.IntentOutcome(
                    intent_id=intent_id,
                    state="abandoned",
                    at=datetime.now(UTC).isoformat(),
                    observed=tuple(observed),
                    consequences={
                        "at_recorded_result": at_result,
                        "at_recorded_start": at_start,
                        "at_neither": at_neither,
                        # Said plainly, because the outcome is the durable
                        # record a later reader has: closing a decision is not
                        # undoing the part of it that already landed.
                        "reverted": False,
                    },
                ),
                expected_revision=current.revision,
            )
        return CurationOutcome(
            job_id=job_id,
            intent_id=intent_id,
            state="abandoned",
            written=(),
            already_current=tuple(at_result),
            observed=tuple(observed),
            detail=(
                f"{len(at_result)} of {len(observed)} bound file(s) are at the "
                f"result it recorded, {len(at_start)} still at its start and "
                f"{len(at_neither)} at neither. Nothing was written, nothing "
                "was reverted and no evidence was deleted; the decision is "
                "closed, so it no longer holds these files back."
            ),
        )


def resume_curation(
    config: ProjectConfig, job_id: str, intent_id: str
) -> CurationOutcome:
    """Finish a decision that was recorded but not completely written.

    Re-running the recorded intent skips entries already at ``sha256_after``,
    applies entries still at ``sha256_before``, and refuses an entry at
    neither. Nothing is recomputed from today's staged values: the recorded
    text is the decision.
    """

    with curation_guard(config):
        return _settle(config, job_id, intent_id)


def _settle(
    config: ProjectConfig, job_id: str, intent_id: str
) -> CurationOutcome:
    """Write one open curation intent's files and close it. Guard held."""

    job = study_job.load_study_job(config, job_id)
    intent = job.intent(intent_id)
    if intent.kind != "curation":
        raise StudyCurationError(
            f"Intent {intent_id} of study job {job_id} is a {intent.kind} "
            "action, not a curation decision."
        )
    if intent_id in job.closed_intent_ids:
        raise StudyCurationError(
            f"Curation intent {intent_id} already has an outcome; outcomes are "
            "append-only and an existing one is never rewritten."
        )
    prepared = _prepared_of(intent)
    written, current, observed = _write_prepared(config, prepared)
    job = study_job.load_study_job(config, job_id)
    study_job.append_outcome(
        config,
        job_id,
        study_job.IntentOutcome(
            intent_id=intent_id,
            state="applied",
            at=datetime.now(UTC).isoformat(),
            # Measured with the locks held, not copied out of the intent:
            # `observed` is evidence about the files, and evidence that is
            # only a restatement of the plan cannot contradict it.
            observed=observed,
            consequences={
                "written": list(written),
                "already_current": list(current),
                "record_ids": sorted(
                    {
                        record_id
                        for item in prepared
                        for record_id in item.record_ids
                    }
                ),
            },
        ),
        expected_revision=job.revision,
    )
    return CurationOutcome(
        job_id=job_id,
        intent_id=intent_id,
        state="applied",
        written=written,
        already_current=current,
        observed=observed,
        supersedes=intent.supersedes,
        detail=(
            f"{len(written)} staged file(s) written, {len(current)} already at "
            "the recorded result."
        ),
    )


# --- the barrier promotion reads ----------------------------------------------


@dataclass(frozen=True, slots=True)
class CurationBarrier:
    """One curation intent with no terminal outcome, and what it still holds.

    The intent **is** the barrier: there is no separate marker file, so the
    barrier cannot outlive or precede its own evidence, and it is cleared only
    by a durable appended ``applied`` or ``abandoned`` outcome. Nothing clears
    it automatically and no timeout exists.
    """

    job_id: str
    intent_id: str
    decision: str
    paths: tuple[str, ...]
    remaining: tuple[str, ...] = field(default_factory=tuple)
    #: The digest an owner's abandonment of this exact intent has to carry.
    #: Disclosed here so the control that clears the barrier is reachable from
    #: the refusal that names it, without a second lookup.
    intent_sha256: str = ""
    #: Paths at neither recorded digest. While this is nonempty no resume can
    #: ever finish the decision, and an explicit abandonment is the only route
    #: that closes it.
    unsatisfiable: tuple[str, ...] = field(default_factory=tuple)


def open_curation_barriers(config: ProjectConfig) -> tuple[CurationBarrier, ...]:
    """Every open curation intent in this project, with its remaining paths.

    A job document that cannot be read **refuses** rather than being skipped: a
    skipped barrier is a lifted one, and an unreadable document is exactly
    where an open intent would be invisible. Nothing here touches a staging
    file except to hash it.
    """

    barriers: list[CurationBarrier] = []
    for job_id in study_job.list_study_jobs(config):
        try:
            job = study_job.load_study_job(config, job_id)
        except (JankiError, OSError) as exc:
            raise StudyCurationError(
                f"Study job {job_id} cannot be read ({exc}), so janki cannot "
                "prove it holds no pending curation decision. Repair or remove "
                "that document; nothing was promoted."
            ) from exc
        closed = job.closed_intent_ids
        for intent in job.intents:
            if intent.kind != "curation" or intent.intent_id in closed:
                continue
            prepared = _prepared_of(intent)
            remaining: list[str] = []
            unsatisfiable: list[str] = []
            for item in prepared:
                path = config.root / item.staging_path
                try:
                    held = path.read_bytes()
                except OSError:
                    remaining.append(item.staging_path)
                    unsatisfiable.append(item.staging_path)
                    continue
                digest = hashlib.sha256(held).hexdigest()
                if digest != item.sha256_after:
                    remaining.append(item.staging_path)
                    if digest != item.sha256_before:
                        # Neither digest: no replay of the recorded bytes can
                        # ever land, so saying "resume it" here would send the
                        # owner at a command that refuses forever.
                        unsatisfiable.append(item.staging_path)
            barriers.append(
                CurationBarrier(
                    job_id=job_id,
                    intent_id=intent.intent_id,
                    decision=str(intent.bindings.get("decision") or ""),
                    paths=tuple(item.staging_path for item in prepared),
                    remaining=tuple(remaining),
                    intent_sha256=curation_intent_digest(intent),
                    unsatisfiable=tuple(unsatisfiable),
                )
            )
    return tuple(barriers)


def _barrier_routes(barrier: CurationBarrier) -> str:
    """The two owner routes out of this open decision, in one sentence.

    One text, used by every refusal an open decision produces — the promotion
    gate and the successor's own refusal — so an owner reads the same two
    commands and the same digest wherever they meet the barrier.
    """

    if barrier.unsatisfiable:
        # Honest about which of the two routes is actually open: a bound file
        # at neither digest is a replay that can never land, and pointing at
        # `--resume` there is pointing at a refusal.
        return (
            "These bound file(s) are at neither digest this decision "
            "recorded, so replaying it can never finish: "
            + ", ".join(barrier.unsatisfiable)
            + ". Close it without writing it with `janki study curate "
            f"{barrier.job_id} --abandon {barrier.intent_id} --expect "
            f"{barrier.intent_sha256}`, then decide again over the files as "
            "they now stand."
        )
    return (
        "Finish it with `janki study curate "
        f"{barrier.job_id} --resume {barrier.intent_id}`, or close it "
        "without writing it with `janki study curate "
        f"{barrier.job_id} --abandon {barrier.intent_id} --expect "
        f"{barrier.intent_sha256}`."
    )


def unresolved_curation_predecessors(
    config: ProjectConfig, plan: CurationPlan
) -> tuple[CurationBarrier, ...]:
    """Every open decision this plan would overtake. A read; nothing is written.

    Two ways one decision stands in front of another, and both are exactly the
    §6.2 successor rule rather than a wider quiet period:

    * it binds a **file this plan would write** — whichever job recorded it,
      because the file is the shared thing and writing it strands that
      decision at neither digest it recorded;
    * it is a decision of **this job about an identity this plan settles
      again** — the replan case, where the predecessor has to reach an outcome
      before its successor can be recorded over it.

    A decision about other identities in other files blocks nothing, so an
    ordinary later curation stays ordinary.
    """

    wanted = set(plan.paths)
    try:
        job = study_job.load_study_job(config, plan.job_id)
    except (JankiError, OSError) as exc:
        raise StudyCurationError(
            f"Study job {plan.job_id} cannot be read ({exc}), so janki cannot "
            "prove this decision overtakes none of its own. Nothing was "
            "recorded and nothing was written."
        ) from exc
    subjects = _plan_subjects(plan)
    replanned = {
        intent.intent_id
        for intent in job.intents
        if intent.kind == "curation" and _recorded_subjects(intent) & subjects
    }
    return tuple(
        barrier
        for barrier in open_curation_barriers(config)
        if set(barrier.paths) & wanted
        or (barrier.job_id == plan.job_id and barrier.intent_id in replanned)
    )


def pending_curation_refusal(config: ProjectConfig, staging_path: Path) -> str:
    """Why this file cannot be promoted yet, or ``""`` when nothing blocks it.

    Bound to the exact file: an open decision about other parts does not stop
    an unrelated promotion, and an open decision about *this* file does, because
    promoting bytes a durable decision is about to rewrite would consume the
    review that decision was made over.
    """

    wanted = Path(os.path.realpath(staging_path))
    for barrier in open_curation_barriers(config):
        for path in barrier.paths:
            if Path(os.path.realpath(config.root / path)) != wanted:
                continue
            remaining = ", ".join(barrier.remaining) or "none"
            return (
                f"Study job {barrier.job_id} recorded curation decision "
                f"{barrier.intent_id} over this file and it has no outcome yet "
                f"({barrier.decision}). Files still to be written: {remaining}. "
                f"{_barrier_routes(barrier)} Nothing is promoted while it is "
                "open."
            )
    return ""
