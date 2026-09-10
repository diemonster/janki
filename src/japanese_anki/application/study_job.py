"""One committed document per study job: choices, intents, outcomes.

A study job is the owner's thread of work over one preserved source and one
destination deck. This module owns its whole durable form, and deliberately
owns nothing else: the operation journal is still the money authority, the
batch manifests and their execution receipts are still the execution
authority, staging and the ``data/staging/done/`` archives are still the
content authority, and the ledger is still the media authority. The job
document coordinates them; it reproduces none of them and adds no journal.

Three rules shape the code below.

**One file, one revision, five namespaces.** ``header`` is immutable at
creation. ``choices`` holds the owner's mutable local preferences under a
closed whitelist. ``layouts``, ``intents`` and ``outcomes`` are append-only.
``record_choice`` refuses a payload naming any namespace but ``choices``, so
an ordinary choice edit can neither rewrite a layout revision a dispatched
child bound nor remove, rewrite or supersede a pending intent. Every writer is
compare-and-swap on the file's prior revision, which is the sha256 of its own
bytes — the contract ``io.atomic_write_text_bound`` already computes.

**No progress fields.** Nothing here records what a batch did. Every count and
state in :func:`study_job_status` is derived at read time from the journal,
the batch manifests, the staging documents, the ``done`` archive, the ledger
and the publication receipts. A stored count would be a second database of
what happened, and it would be the one that is wrong.

**A backlink is found by exact id and hash.** An :class:`ActionIntent` is
appended and fsynced *before* the service it names is launched, carrying the
exact ids that service will reserve and the hash of the artifact it will
write. When the process dies between that write and the appended outcome,
:func:`discover_actions` finds the artifact at that exact path, compares that
exact hash, and — for a batch — compares the ``job_id`` the manifest itself
embeds. It never picks the newest, never orders candidates by mtime, and never
adopts a mismatched artifact.

The document grants no spending, discard or approval authority. Writing an
intent journals nothing and sends nothing; it is a record of what the owner's
existing authority is about to be spent on.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki import ledger, operations
from japanese_anki.application import extraction_batch, source_parts
from japanese_anki.application.extraction import durable_inbox_root
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    atomic_write_text_bound,
    prepare_bound_directory,
    read_bytes_bound_snapshot,
)

__all__ = [
    "CHOICE_KEYS",
    "INTENT_KINDS",
    "JOB_KINDS",
    "NAMESPACES",
    "STUDY_JOBS_DIR_NAME",
    "ActionIntent",
    "ActionReference",
    "IntentOutcome",
    "JobResumeAction",
    "StudyJob",
    "StudyJobBatchStatus",
    "StudyJobChildStatus",
    "StudyJobError",
    "StudyJobHeader",
    "StudyJobPartsStatus",
    "StudyJobPreview",
    "StudyJobStatus",
    "append_intent",
    "append_outcome",
    "discover_actions",
    "dispatch_job_batch",
    "job_destination_deck",
    "job_part_sources",
    "job_published_parts",
    "list_study_jobs",
    "load_study_job",
    "new_intent_id",
    "open_study_job",
    "plan_job_batch_retry",
    "plan_job_extraction_batch",
    "publish_job_source_parts",
    "record_choice",
    "render_job_preview",
    "resume_job_actions",
    "study_job_path",
    "study_job_status",
    "study_jobs_dir",
]


class StudyJobError(JankiError):
    """A study job could not be opened, read, or safely written."""


#: Beside the journal, exactly as ``extraction_batch.BATCH_DIR_NAME`` and
#: ``source_parts.SOURCE_PARTS_DIR_NAME`` are, so no ``[paths]`` key is added.
STUDY_JOBS_DIR_NAME = "study_jobs"

#: Wire version of the job document.
JOB_VERSION = 1

#: The five top-level namespaces, and the whole list.
NAMESPACES = ("header", "choices", "layouts", "intents", "outcomes")

#: What a job is. One kind today; a second one is a deliberate addition here.
JOB_KINDS = ("source_extraction",)

#: Every intent kind the datatype carries. Which of them may be *written* is a
#: separate question, answered by ``_WRITABLE_INTENT_KINDS``: an intent that
#: reserves a service's ids is only meaningful once that service exists.
INTENT_KINDS = ("source_parts", "extract_batch", "retry", "curation", "finish")

#: The kinds whose owning service ships today. ``curation`` belongs to the
#: cross-part curation writer and ``finish`` to the study-finish authority;
#: appending one before its writer exists would reserve ids nothing can honour.
_WRITABLE_INTENT_KINDS = ("source_parts", "extract_batch", "retry")

#: How an intent ends. Nothing else closes one, and nothing reopens it.
OUTCOME_STATES = ("applied", "refused", "abandoned")

#: The owner's local preferences this store writes today.
_WRITABLE_CHOICE_KEYS = ("directions", "part_selections", "part_layout_bindings")

#: Reserved names whose owner control does not exist yet. Refusing them by
#: name is the point: it keeps the same decision from arriving later under a
#: different, unvalidated key, and it is not a placeholder writer — the
#: editors that record these decisions ship with the finish, bound to the
#: rendering they were taken over.
_DEFERRED_CHOICE_KEYS = {
    "include_example_audio": (
        "the sentence-audio control inside the review editor"
    ),
    "review_flags": "the review editor",
    "review_patterns": "the review editor",
    "coverage_reasons": "the coverage editor",
    "dispositions": "the disposition editor",
}

#: Every choice key this schema knows, writable or reserved.
CHOICE_KEYS = tuple(sorted((*_WRITABLE_CHOICE_KEYS, *_DEFERRED_CHOICE_KEYS)))

_CARD_DIRECTIONS = ("recognition", "production", "reading")


# --- paths and identities -----------------------------------------------------


def study_jobs_dir(config: ProjectConfig) -> Path:
    """The one store directory, derived beside the journal."""

    return config.operations_file.parent / STUDY_JOBS_DIR_NAME


def _valid_job_id(job_id: Any) -> bool:
    """A job id is a uuid, so it can never name a path outside its store."""

    try:
        return str(uuid.UUID(str(job_id))) == str(job_id)
    except (AttributeError, TypeError, ValueError):
        return False


def study_job_path(config: ProjectConfig, job_id: str) -> Path:
    """The one durable document path for a job id."""

    if not _valid_job_id(job_id):
        raise StudyJobError(f"Not a study job identity: {job_id!r}")
    return study_jobs_dir(config) / f"{job_id}.json"


def new_intent_id() -> str:
    """A fresh intent identity. Opaque, and never reused."""

    return str(uuid.uuid4())


# --- the document -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyJobHeader:
    """What this job is over. Immutable from the moment it is created."""

    job_id: str
    created_at: str
    kind: str
    parent_source_name: str
    parent_sha256: str
    #: Repository-relative POSIX path of the destination deck definition.
    deck_path: str
    #: That file's bytes when the job was opened. An *initial* binding: the
    #: saved deck definition remains the authority for scope and card set.
    deck_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "kind": self.kind,
            "parent_source_name": self.parent_source_name,
            "parent_sha256": self.parent_sha256,
            "deck_path": self.deck_path,
            "deck_sha256": self.deck_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """What is about to be launched, written before it is.

    ``reserves`` holds the exact ids and hashes the owning service will use —
    a recipe id and its receipt's sha256; a batch id, its manifest sha256 and
    every reserved child operation id. ``bindings`` holds the inputs the plan
    was computed over. Both are immutable once appended: an intent is closed
    by an appended :class:`IntentOutcome`, never by editing it, and there is
    no ``superseded_by`` — a successor carries ``supersedes``.
    """

    intent_id: str
    kind: str
    decided_at: str
    reserves: Mapping[str, Any] = field(default_factory=dict)
    bindings: Mapping[str, Any] = field(default_factory=dict)
    supersedes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "kind": self.kind,
            "decided_at": self.decided_at,
            "supersedes": self.supersedes,
            "reserves": _plain(self.reserves),
            "bindings": _plain(self.bindings),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ActionIntent:
        return cls(
            intent_id=str(raw["intent_id"]),
            kind=str(raw["kind"]),
            decided_at=str(raw.get("decided_at", "")),
            reserves=_plain(raw.get("reserves") or {}),
            bindings=_plain(raw.get("bindings") or {}),
            supersedes=str(raw.get("supersedes", "")),
        )


@dataclass(frozen=True, slots=True)
class IntentOutcome:
    """How one intent ended, and what was observed when it did."""

    intent_id: str
    state: str
    at: str
    #: ``(path, sha256)`` pairs actually observed. Evidence, not a claim.
    observed: tuple[tuple[str, str], ...] = ()
    consequences: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "state": self.state,
            "at": self.at,
            "observed": [[path, digest] for path, digest in self.observed],
            "consequences": _plain(self.consequences),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> IntentOutcome:
        observed = raw.get("observed") or ()
        return cls(
            intent_id=str(raw["intent_id"]),
            state=str(raw["state"]),
            at=str(raw.get("at", "")),
            observed=tuple(
                (str(entry[0]), str(entry[1]))
                for entry in observed
                if isinstance(entry, Sequence)
                and not isinstance(entry, str)
                and len(entry) == 2
            ),
            consequences=_plain(raw.get("consequences") or {}),
        )


@dataclass(frozen=True, slots=True)
class StudyJob:
    """One job document as it stands, plus the revision it was read at."""

    path: Path
    revision: str
    header: StudyJobHeader
    choices: Mapping[str, Any]
    layouts: Mapping[str, Any]
    intents: tuple[ActionIntent, ...]
    outcomes: tuple[IntentOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": JOB_VERSION,
            "header": self.header.to_dict(),
            "choices": _plain(self.choices),
            "layouts": _plain(self.layouts),
            "intents": [intent.to_dict() for intent in self.intents],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    def intent(self, intent_id: str) -> ActionIntent:
        for candidate in self.intents:
            if candidate.intent_id == intent_id:
                return candidate
        raise StudyJobError(f"Study job {self.header.job_id} has no intent {intent_id}.")

    @property
    def closed_intent_ids(self) -> frozenset[str]:
        """Every intent an outcome has already closed."""

        return frozenset(outcome.intent_id for outcome in self.outcomes)


@dataclass(frozen=True, slots=True)
class ActionReference:
    """One open intent's artifact, resolved by exact reserved id and hash."""

    intent_id: str
    kind: str
    path: Path
    sha256: str
    #: The job id the artifact itself embeds. Empty for a source-part receipt,
    #: which is job-independent by contract and records no requesting job.
    job_id: str
    reserves: Mapping[str, Any]


def _plain(value: Any) -> Any:
    """A JSON-round-tripped copy, so nothing mutable rides inside the wire."""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _document_bytes(job: StudyJob) -> str:
    return (
        json.dumps(job.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


# --- reading ------------------------------------------------------------------


def _require_mapping(value: Any, job_id: str, namespace: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyJobError(
            f"The study job {job_id} document's {namespace} is not an object; "
            "it was hand-edited or corrupted, and janki will not guess at it."
        )
    return value


def load_study_job(config: ProjectConfig, job_id: str) -> StudyJob:
    """Read one job document back, validating every entry it records."""

    path = study_job_path(config, job_id)
    try:
        _state, revision, payload = read_bytes_bound_snapshot(path)
    except FileNotFoundError as exc:
        raise StudyJobError(f"No study job {job_id} is recorded at {path}.") from exc
    except (JankiError, OSError) as exc:
        raise StudyJobError(f"Could not read the study job {job_id}: {exc}") from exc
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StudyJobError(
            f"The study job {job_id} document is not readable JSON: {exc}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise StudyJobError(f"{path} must hold an object.")
    header_raw = _require_mapping(parsed.get("header"), job_id, "header")
    if str(header_raw.get("job_id")) != job_id:
        raise StudyJobError(f"{path} records study job {header_raw.get('job_id')!r}.")
    intents_raw = parsed.get("intents")
    outcomes_raw = parsed.get("outcomes")
    if not isinstance(intents_raw, list) or not isinstance(outcomes_raw, list):
        raise StudyJobError(
            f"The study job {job_id} document's intents and outcomes must be lists."
        )
    try:
        intents = tuple(ActionIntent.from_dict(entry) for entry in intents_raw)
        outcomes = tuple(IntentOutcome.from_dict(entry) for entry in outcomes_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyJobError(
            f"The study job {job_id} document records an unreadable entry: {exc}"
        ) from exc
    return StudyJob(
        path=path,
        revision=revision,
        header=StudyJobHeader(
            job_id=job_id,
            created_at=str(header_raw.get("created_at", "")),
            kind=str(header_raw.get("kind", "")),
            parent_source_name=str(header_raw.get("parent_source_name", "")),
            parent_sha256=str(header_raw.get("parent_sha256", "")),
            deck_path=str(header_raw.get("deck_path", "")),
            deck_sha256=str(header_raw.get("deck_sha256", "")),
        ),
        choices=_plain(_require_mapping(parsed.get("choices"), job_id, "choices")),
        layouts=_plain(_require_mapping(parsed.get("layouts"), job_id, "layouts")),
        intents=intents,
        outcomes=outcomes,
    )


def list_study_jobs(config: ProjectConfig) -> tuple[str, ...]:
    """Every job id with a document, sorted. Bad siblings are skipped."""

    try:
        with os.scandir(study_jobs_dir(config)) as scan:
            entries = sorted(entry.name for entry in scan if entry.is_file())
    except (FileNotFoundError, NotADirectoryError, OSError):
        return ()
    found: list[str] = []
    for name in entries:
        if not name.endswith(".json"):
            continue
        job_id = name[: -len(".json")]
        if _valid_job_id(job_id):
            found.append(job_id)
    return tuple(found)


# --- writing ------------------------------------------------------------------


def _save(job: StudyJob, *, expected_revision: str | None) -> StudyJob:
    """Publish this exact document under compare-and-swap, and read it back."""

    text = _document_bytes(job)
    prepare_bound_directory(job.path.parent)
    try:
        atomic_write_text_bound(
            job.path,
            text,
            expected_revision=expected_revision,
            expected_absent=expected_revision is None,
        )
    except JankiError as exc:
        raise StudyJobError(
            f"Could not save the study job {job.header.job_id}: {exc}"
        ) from exc
    return StudyJob(
        path=job.path,
        revision=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        header=job.header,
        choices=job.choices,
        layouts=job.layouts,
        intents=job.intents,
        outcomes=job.outcomes,
    )


def _at_revision(
    config: ProjectConfig, job_id: str, expected_revision: str
) -> StudyJob:
    """Read the job and prove it is the revision the caller decided against."""

    if not isinstance(expected_revision, str) or not expected_revision:
        raise StudyJobError(
            "A study job edit names the exact revision it was decided against."
        )
    job = load_study_job(config, job_id)
    if job.revision != expected_revision:
        raise StudyJobError(
            f"The study job {job_id} changed since this edit was prepared "
            f"(it is at {job.revision}, not {expected_revision}); nothing was "
            "written. Read it again and decide over the current document."
        )
    return job


def _relative_deck_path(config: ProjectConfig, deck_path: Path) -> str:
    resolved = Path(os.path.realpath(deck_path))
    root = Path(os.path.realpath(config.root))
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise StudyJobError(
            f"A study job's destination deck lives in this repository; "
            f"{deck_path} does not."
        ) from exc


def open_study_job(
    config: ProjectConfig,
    *,
    kind: str,
    parent_source: Path,
    deck_path: Path,
    deck_sha256: str = "",
) -> StudyJob:
    """Bind one source and one deck, and write the job document.

    A direct owner action — the Assistant's Create/Open, or ``janki study
    new`` — is the authority for this local write, so there is no second
    dialog and no new confirmation ladder.

    ``deck_sha256`` is the hash the deck's own writer returned when this job's
    destination was created in the same action. Empty measures the file here,
    which is what reusing an existing deck does. Supplying it matters: a
    freshly created deck's bytes are known exactly from
    ``create_study_deck``'s own plan, and re-reading the file instead would
    silently bind whatever a concurrent edit left behind.
    """

    if kind not in JOB_KINDS:
        raise StudyJobError(
            f"A study job is one of {', '.join(JOB_KINDS)}, not {kind!r}."
        )
    source = Path(parent_source)
    resolved_source = Path(os.path.realpath(source))
    corpus = Path(os.path.realpath(durable_inbox_root(config)))
    if not resolved_source.is_file() or corpus not in resolved_source.parents:
        raise StudyJobError(
            f"A study job's parent source is a preserved file under "
            f"{corpus}; {source} is not."
        )
    try:
        parent_sha256 = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    except OSError as exc:
        raise StudyJobError(f"Could not read the parent source {source}: {exc}") from exc

    deck = Path(deck_path)
    relative_deck = _relative_deck_path(config, deck)
    if deck_sha256:
        bound_deck_sha256 = deck_sha256
    else:
        try:
            bound_deck_sha256 = hashlib.sha256(
                Path(os.path.realpath(deck)).read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise StudyJobError(
                f"Could not read the destination deck definition {deck}: {exc}"
            ) from exc

    job_id = str(uuid.uuid4())
    return _save(
        StudyJob(
            path=study_job_path(config, job_id),
            revision="",
            header=StudyJobHeader(
                job_id=job_id,
                created_at=datetime.now(UTC).isoformat(),
                kind=kind,
                parent_source_name=resolved_source.name,
                parent_sha256=parent_sha256,
                deck_path=relative_deck,
                deck_sha256=bound_deck_sha256,
            ),
            choices={},
            layouts={},
            intents=(),
            outcomes=(),
        ),
        expected_revision=None,
    )


def _checked_directions(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(item not in _CARD_DIRECTIONS for item in value)
        or len(set(value)) != len(value)
    ):
        raise StudyJobError(
            "A study job's directions are one or more distinct entries of "
            f"{', '.join(_CARD_DIRECTIONS)}. Recording them changes no deck "
            "definition and no card set."
        )
    return list(value)


def _checked_part_selections(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise StudyJobError(
            "A study job's part selections are distinct nonblank part names."
        )
    return list(value)


def _checked_layout_bindings(value: Any, layouts: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyJobError(
            "A study job's part layout bindings are an object keyed by part name."
        )
    checked: dict[str, Any] = {}
    for part, reference in value.items():
        if not isinstance(part, str) or not part.strip():
            raise StudyJobError(
                "A study job's part layout binding is keyed by a nonblank part name."
            )
        if (
            not isinstance(reference, Sequence)
            or isinstance(reference, str)
            or len(reference) != 2
        ):
            raise StudyJobError(
                f"The layout binding for {part} names one (layout_id, revision) pair."
            )
        layout_id, revision = str(reference[0]), reference[1]
        if _layout_key(layout_id, revision) not in layouts:
            raise StudyJobError(
                f"The layout binding for {part} names layout {layout_id!r} "
                f"revision {revision!r}, which this job has never recorded. "
                "A binding points at an immutable revision the layout writer "
                "appended; it cannot bring one into existence."
            )
        checked[part] = [layout_id, revision]
    return checked


def _layout_key(layout_id: str, revision: Any) -> str:
    return f"{layout_id}@{revision}"


def record_choice(
    config: ProjectConfig,
    job_id: str,
    choice: Mapping[str, Any],
    *,
    expected_revision: str,
) -> StudyJob:
    """Save the owner's local preferences, and nothing else.

    Whitelisted ``choices`` keys only. A payload naming ``header``,
    ``layouts``, ``intents`` or ``outcomes`` refuses by name, because that is
    what stops a reversible local edit rewriting a layout revision a
    dispatched child bound or clearing a pending intent. Nothing here is paid
    consent, canonical content, or a review, coverage or promotion decision.
    """

    if not isinstance(choice, Mapping) or not choice:
        raise StudyJobError("A study job choice edit names at least one choice.")
    named_namespaces = [key for key in choice if key in NAMESPACES and key != "choices"]
    if named_namespaces:
        raise StudyJobError(
            "An ordinary study job choice edit writes whitelisted choices only; "
            f"it may not name the {', '.join(sorted(named_namespaces))} "
            "namespace. Nothing was written."
        )
    for key in choice:
        if key in _DEFERRED_CHOICE_KEYS:
            raise StudyJobError(
                f"The study job choice {key!r} is reserved for "
                f"{_DEFERRED_CHOICE_KEYS[key]}, which does not exist yet. "
                "janki will not record that decision through another control."
            )
        if key not in _WRITABLE_CHOICE_KEYS:
            raise StudyJobError(
                f"{key!r} is not a study job choice. This job records "
                f"{', '.join(_WRITABLE_CHOICE_KEYS)}."
            )

    job = _at_revision(config, job_id, expected_revision)
    updated = dict(job.choices)
    for key, value in choice.items():
        if key == "directions":
            updated[key] = _checked_directions(value)
        elif key == "part_selections":
            updated[key] = _checked_part_selections(value)
        else:
            updated[key] = _checked_layout_bindings(value, job.layouts)
    return _save(
        StudyJob(
            path=job.path,
            revision=job.revision,
            header=job.header,
            choices=updated,
            layouts=job.layouts,
            intents=job.intents,
            outcomes=job.outcomes,
        ),
        expected_revision=job.revision,
    )


def append_intent(
    config: ProjectConfig,
    job_id: str,
    intent: ActionIntent,
    *,
    expected_revision: str,
) -> StudyJob:
    """Append one immutable intent, before the service it names is launched.

    The write is fsynced by ``atomic_write_text_bound``, so a crash after this
    and before the outcome leaves a reserved id and hash
    :func:`discover_actions` can find the artifact by.
    """

    if not isinstance(intent, ActionIntent):
        raise StudyJobError("A study job intent is an ActionIntent.")
    if not intent.intent_id or not isinstance(intent.intent_id, str):
        raise StudyJobError("A study job intent carries a nonblank intent id.")
    if intent.kind not in INTENT_KINDS:
        raise StudyJobError(
            f"A study job intent is one of {', '.join(INTENT_KINDS)}, not "
            f"{intent.kind!r}."
        )
    if intent.kind not in _WRITABLE_INTENT_KINDS:
        raise StudyJobError(
            f"janki has no writer for a {intent.kind!r} intent yet: the service "
            "that would honour the ids it reserves does not exist. Nothing was "
            "written."
        )
    job = _at_revision(config, job_id, expected_revision)
    if any(existing.intent_id == intent.intent_id for existing in job.intents):
        raise StudyJobError(
            f"Study job {job_id} already records intent {intent.intent_id}; an "
            "intent log is append-only and an existing entry is never rewritten."
        )
    if intent.supersedes:
        if all(
            existing.intent_id != intent.supersedes for existing in job.intents
        ):
            raise StudyJobError(
                f"Intent {intent.intent_id} supersedes {intent.supersedes}, which "
                f"study job {job_id} does not record."
            )
        if intent.supersedes not in job.closed_intent_ids:
            raise StudyJobError(
                f"Intent {intent.supersedes} has no outcome yet, so it still "
                "blocks its successor. Record what became of it — applied, "
                "refused or abandoned — before replanning over it."
            )
    return _save(
        StudyJob(
            path=job.path,
            revision=job.revision,
            header=job.header,
            choices=job.choices,
            layouts=job.layouts,
            intents=(*job.intents, intent),
            outcomes=job.outcomes,
        ),
        expected_revision=job.revision,
    )


def append_outcome(
    config: ProjectConfig,
    job_id: str,
    outcome: IntentOutcome,
    *,
    expected_revision: str,
) -> StudyJob:
    """Close one intent by appending what became of it. One outcome, once."""

    if not isinstance(outcome, IntentOutcome):
        raise StudyJobError("A study job outcome is an IntentOutcome.")
    if outcome.state not in OUTCOME_STATES:
        raise StudyJobError(
            f"A study job outcome is one of {', '.join(OUTCOME_STATES)}, not "
            f"{outcome.state!r}."
        )
    job = _at_revision(config, job_id, expected_revision)
    job.intent(outcome.intent_id)
    if outcome.intent_id in job.closed_intent_ids:
        raise StudyJobError(
            f"Intent {outcome.intent_id} already has an outcome; outcomes are "
            "append-only and an existing one is never rewritten. An owner who "
            "changed their mind gets a fresh superseding intent."
        )
    return _save(
        StudyJob(
            path=job.path,
            revision=job.revision,
            header=job.header,
            choices=job.choices,
            layouts=job.layouts,
            intents=job.intents,
            outcomes=(*job.outcomes, outcome),
        ),
        expected_revision=job.revision,
    )


# --- backlink recovery --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Resolution:
    """Whether one intent's exact reserved artifact is on disk, and what it is."""

    reference: ActionReference | None
    refusal: str


def _reserved(intent: ActionIntent, key: str) -> str:
    value = intent.reserves.get(key)
    return value if isinstance(value, str) else ""


def _resolve_source_parts(
    config: ProjectConfig, intent: ActionIntent
) -> _Resolution:
    """A publication receipt, matched by recipe id and its own file hash.

    No job id is compared, and none is looked for: a receipt is
    job-independent, and reuse is bound by recipe id, receipt hash and the
    publish binding this intent itself records.
    """

    recipe_id = _reserved(intent, "recipe_id")
    expected = _reserved(intent, "receipt_sha256")
    if not recipe_id or not expected:
        return _Resolution(None, "the intent reserves no recipe id and receipt hash")
    try:
        path = source_parts.receipt_path(config, recipe_id)
        record = source_parts.load_source_part_receipt(
            config, recipe_id, verify_published=False
        )
    except (JankiError, OSError) as exc:
        return _Resolution(None, f"its receipt could not be read: {exc}")
    if record.receipt_sha256 != expected:
        return _Resolution(
            None,
            f"the receipt at {path} hashes to {record.receipt_sha256}, not the "
            f"{expected} this job bound",
        )
    return _Resolution(
        ActionReference(
            intent_id=intent.intent_id,
            kind=intent.kind,
            path=path,
            sha256=record.receipt_sha256,
            job_id="",
            reserves=intent.reserves,
        ),
        "",
    )


def _resolve_batch(
    config: ProjectConfig, job_id: str, intent: ActionIntent
) -> _Resolution:
    """A batch manifest, matched by batch id, manifest hash and embedded job id."""

    batch_id = _reserved(intent, "batch_id")
    expected = _reserved(intent, "manifest_sha256")
    if not batch_id or not expected:
        return _Resolution(None, "the intent reserves no batch id and manifest hash")
    try:
        path = extraction_batch.batch_manifest_path(config, batch_id)
    except JankiError as exc:
        return _Resolution(None, str(exc))
    if not path.is_file():
        return _Resolution(None, f"no manifest exists at {path}")
    try:
        held = path.read_bytes()
        raw = json.loads(held.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return _Resolution(None, f"its manifest could not be read: {exc}")
    found = hashlib.sha256(held).hexdigest()
    if found != expected:
        return _Resolution(
            None,
            f"the manifest at {path} hashes to {found}, not the {expected} this "
            "job bound",
        )
    if not isinstance(raw, Mapping) or str(raw.get("job_id", "")) != job_id:
        return _Resolution(
            None,
            f"the manifest at {path} does not record study job {job_id}",
        )
    reserved_children = intent.reserves.get("child_operation_ids")
    if isinstance(reserved_children, list):
        children = raw.get("children")
        actual = (
            [str(entry.get("operation_id", "")) for entry in children]
            if isinstance(children, list)
            and all(isinstance(entry, Mapping) for entry in children)
            else []
        )
        if actual != [str(entry) for entry in reserved_children]:
            return _Resolution(
                None,
                f"the manifest at {path} names different operations than this "
                "job reserved",
            )
    return _Resolution(
        ActionReference(
            intent_id=intent.intent_id,
            kind=intent.kind,
            path=path,
            sha256=found,
            job_id=job_id,
            reserves=intent.reserves,
        ),
        "",
    )


def _resolve_intent(
    config: ProjectConfig, job_id: str, intent: ActionIntent
) -> _Resolution:
    if intent.kind == "source_parts":
        return _resolve_source_parts(config, intent)
    if intent.kind in ("extract_batch", "retry"):
        return _resolve_batch(config, job_id, intent)
    return _Resolution(
        None, f"janki cannot resolve a {intent.kind!r} intent's artifact yet"
    )


def discover_actions(
    config: ProjectConfig, job_id: str
) -> tuple[ActionReference, ...]:
    """Find the artifacts of every intent no outcome has closed.

    The crash-before-the-backlink reader. Matching is by the exact reserved id
    at the exact path the owning store derives, then by the exact hash this
    job bound, then — for a batch — by the ``job_id`` the manifest embeds. It
    never picks "the newest", never orders candidates by modification time or
    path, and never adopts a mismatched artifact: an artifact that fails any
    comparison is simply not a reference, and :func:`study_job_status` reports
    it as unresolved.
    """

    job = load_study_job(config, job_id)
    closed = job.closed_intent_ids
    found: list[ActionReference] = []
    for intent in job.intents:
        if intent.intent_id in closed:
            continue
        resolution = _resolve_intent(config, job_id, intent)
        if resolution.reference is not None:
            found.append(resolution.reference)
    return tuple(found)


# --- the job-scoped services both surfaces call -------------------------------
#
# The Assistant is the primary surface and the CLI is a fully supported
# equivalent, so the intent-before-the-effect protocol lives here once rather
# than twice. Nothing below reproduces a writer: each function appends this
# job's own record and then calls the existing application service unchanged.


def _open_publication_intent(
    job: StudyJob, recipe_id: str, receipt_sha256: str
) -> ActionIntent | None:
    """An earlier, still-open intent that reserved exactly this publication.

    Recovering a publication whose parts are not all in the corpus is the
    publication service's work, not this store's — and when the owner runs it
    again, the ids it binds are the two ids an interrupted intent already
    reserved. Appending a second intent for them would reserve the same
    artifact twice and leave the first open with nothing that could ever close
    it. Anything less than an exact match on both the recipe id and the receipt
    hash is a different publication and gets its own intent.
    """

    closed = job.closed_intent_ids
    for intent in job.intents:
        if (
            intent.kind == "source_parts"
            and intent.intent_id not in closed
            and _reserved(intent, "recipe_id") == recipe_id
            and _reserved(intent, "receipt_sha256") == receipt_sha256
        ):
            return intent
    return None


def publish_job_source_parts(
    config: ProjectConfig,
    job_id: str,
    plan: Any,
    *,
    publish_token: str,
) -> Any:
    """Bind this job to one publication, then run the owner's publish.

    Two cases, and telling them apart is the whole of it.

    **Attaching an already-published receipt.** The receipt exists and is
    proved to be this plan's own by the publication service's immutability
    rule, so its exact on-disk hash is knowable now. The intent records the
    reuse binding — recipe id plus receipt hash — and the publication service
    still runs, so a partially published recipe finishes against the
    expectation it wrote. A plan that disagrees with that receipt publishes
    nothing (a receipt is immutable; different parts need a new recipe id), so
    it is refused here, before anything is recorded: an intent reserving a
    receipt this run will never write is an intent nothing can honestly close,
    and a later resume would read it as this job's own publication.

    **Initiating a new publication.** The receipt does not exist, so its hash
    does not either until it is written. ``created_at`` is frozen here and
    handed to the writer, exactly as ``coverage._approval_payload`` and
    ``ledger.record_export`` are already given their dates, so the hash the
    intent binds is the hash that lands.

    If the receipt that lands diverges — another run wrote it in between — the
    outcome records what was observed and the whole thing refuses. It is never
    adopted, and the owner replans as an attach.

    **Finishing an interrupted publication.** When this job already holds an
    open intent reserving exactly this recipe id and receipt hash, that intent
    is the one this run closes. It reserved this artifact; a second one beside
    it would say the same thing twice and orphan the first forever.
    """

    job = load_study_job(config, job_id)
    published = source_parts.published_receipt_sha256(config, plan)
    if published is not None:
        created_at = ""
        expected = published
    else:
        created_at = datetime.now(UTC).isoformat()
        expected = source_parts.planned_receipt_sha256(plan, created_at=created_at)

    already_open = _open_publication_intent(job, plan.recipe_id, expected)
    if already_open is not None:
        intent, saved = already_open, job
    else:
        intent = ActionIntent(
            intent_id=new_intent_id(),
            kind="source_parts",
            decided_at=datetime.now(UTC).isoformat(),
            reserves={"recipe_id": plan.recipe_id, "receipt_sha256": expected},
            bindings={
                "plan_fingerprint": plan.plan_fingerprint,
                "parent_name": plan.parent_name,
                "parent_sha256": plan.parent_sha256,
                "created_at": created_at,
                "target_names": [part.target_name for part in plan.parts],
                "part_sha256": [part.sha256 for part in plan.parts],
            },
        )
        saved = append_intent(config, job_id, intent, expected_revision=job.revision)

    receipt = source_parts.execute_source_parts(
        config, plan, publish_token=publish_token, created_at=created_at
    )
    if receipt.receipt_sha256 != expected:
        append_outcome(
            config,
            job_id,
            IntentOutcome(
                intent_id=intent.intent_id,
                state="refused",
                at=datetime.now(UTC).isoformat(),
                observed=((str(receipt.path), receipt.receipt_sha256),),
                consequences={"reason": "the receipt that landed is not the one bound"},
            ),
            expected_revision=saved.revision,
        )
        raise StudyJobError(
            f"The publication receipt at {receipt.path} hashes to "
            f"{receipt.receipt_sha256}, not the {expected} this job bound before "
            "publishing. Another run wrote it first; janki will not adopt it. "
            "Attach to the existing receipt instead."
        )
    append_outcome(
        config,
        job_id,
        IntentOutcome(
            intent_id=intent.intent_id,
            state="applied",
            at=datetime.now(UTC).isoformat(),
            observed=((str(receipt.path), receipt.receipt_sha256),),
            consequences={
                "published": [path.name for path in receipt.published],
                "reused": [path.name for path in receipt.reused],
            },
        ),
        expected_revision=saved.revision,
    )
    return receipt


def job_destination_deck(config: ProjectConfig, job: StudyJob) -> Path:
    """The deck this job binds, as it stands on disk right now.

    The saved definition is the authority for scope and card set; the job's
    recorded hash is only its initial binding, and a disagreement is disclosed
    by :func:`study_job_status` rather than resolved here.
    """

    path = config.root / job.header.deck_path
    if not path.is_file():
        raise StudyJobError(
            f"The destination deck {job.header.deck_path} this job binds is not "
            "in this repository any more; nothing was planned."
        )
    return path


def job_part_sources(
    config: ProjectConfig, job_id: str
) -> tuple[tuple[Path, Any], ...]:
    """Every published part this job would extract, with its ancestry.

    Read from the applied publication intents, in the order their receipts
    record, and narrowed by the owner's ``part_selections`` choice when they
    made one. Nothing here renders, publishes or re-derives geometry: a part
    that is not already in the corpus is a refusal, because a batch is planned
    over durable bytes so its children's hashes describe files that exist.
    """

    job = load_study_job(config, job_id)
    applied = frozenset(
        outcome.intent_id for outcome in job.outcomes if outcome.state == "applied"
    )
    found: list[tuple[Path, Any]] = []
    seen: set[str] = set()
    for intent in job.intents:
        if intent.kind != "source_parts" or intent.intent_id not in applied:
            continue
        recipe_id = _reserved(intent, "recipe_id")
        # One recipe's parts are one set of files however many of this job's
        # intents bound them — an interrupted publication and the run that
        # finished it name the same receipt, and a part is not sent twice for
        # having been recorded twice.
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        record = source_parts.load_source_part_receipt(
            config, recipe_id, verify_published=True
        )
        for part in record.parts:
            if not part.published:
                raise StudyJobError(
                    f"Part {part.target_name} of recipe {recipe_id} is not in the "
                    "corpus, so this job has nothing durable to send for it. "
                    "Publish the recipe's parts before planning a batch."
                )
            found.append(
                (
                    config.scan_inbox / part.target_name,
                    extraction_batch.SourcePartLineage(
                        parent_name=record.parent_name,
                        parent_sha256=record.parent_sha256,
                        descriptor=f"{recipe_id} part {part.ordinal}",
                    ),
                )
            )
    selection = job.choices.get("part_selections")
    if isinstance(selection, list) and selection:
        wanted = list(selection)
        by_name = {path.name: (path, lineage) for path, lineage in found}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise StudyJobError(
                "This job selected part(s) it has not published: "
                + ", ".join(sorted(missing))
                + "."
            )
        found = [by_name[name] for name in wanted]
    if not found:
        raise StudyJobError(
            f"Study job {job_id} has published no source parts yet, so there is "
            "nothing to send."
        )
    return tuple(found)


def job_published_parts(config: ProjectConfig, job_id: str) -> tuple[str, ...]:
    """Every part name this job has published, in receipt order.

    The unnarrowed list :func:`job_part_sources` selects from, so a surface
    offering the owner a subset shows the parts they excluded as well as the
    ones they kept. A read: nothing here publishes, renders or selects.
    """

    job = load_study_job(config, job_id)
    applied = frozenset(
        outcome.intent_id for outcome in job.outcomes if outcome.state == "applied"
    )
    names: list[str] = []
    seen: set[str] = set()
    for intent in job.intents:
        if intent.kind != "source_parts" or intent.intent_id not in applied:
            continue
        recipe_id = _reserved(intent, "recipe_id")
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        try:
            record = source_parts.load_source_part_receipt(
                config, recipe_id, verify_published=False
            )
        except (JankiError, OSError):
            continue
        names.extend(part.target_name for part in record.parts)
    return tuple(names)


def plan_job_extraction_batch(
    config: ProjectConfig,
    job_id: str,
    **plan_options: Any,
) -> Any:
    """Plan this job's batch over its own published parts. Nothing is sent."""

    job = load_study_job(config, job_id)
    parts = job_part_sources(config, job_id)
    return extraction_batch.plan_extraction_batch(
        config,
        [path for path, _lineage in parts],
        destination_deck=job_destination_deck(config, job),
        lineage=[lineage for _path, lineage in parts],
        job_id=job_id,
        **plan_options,
    )


def plan_job_batch_retry(
    config: ProjectConfig,
    job_id: str,
    batch_id: str,
    child_indices: Sequence[int],
    **plan_options: Any,
) -> Any:
    """Plan a fresh retry of exactly these children of this job's batch."""

    load_study_job(config, job_id)
    return extraction_batch.plan_extraction_batch_retry(
        config, batch_id, child_indices, job_id=job_id, **plan_options
    )


def _batch_intent(job_id: str, plan: Any, *, kind: str) -> ActionIntent:
    return ActionIntent(
        intent_id=new_intent_id(),
        kind=kind,
        decided_at=datetime.now(UTC).isoformat(),
        reserves={
            "batch_id": plan.batch_id,
            "manifest_sha256": plan.manifest_sha256,
            "child_operation_ids": [child.operation_id for child in plan.children],
        },
        bindings={
            "job_id": job_id,
            "concurrency_limit": plan.concurrency_limit,
            "provider": plan.provider,
            "model": plan.model,
            "retry_of": plan.retry_of,
            "request_fingerprints": [
                child.request_fingerprint for child in plan.children
            ],
            "source_sha256": [child.source_sha256 for child in plan.children],
        },
    )


def _batch_continues(outcome: Any) -> bool:
    """Whether this batch still holds work its own resume may legitimately do.

    A worker returning is not the statement that this job's recorded authority
    has been spent. ``dispatch_extraction_batch`` returns normally with a child
    still ``authorized`` — reserved, unsent, unspent — when one child's
    transport preflight refuses, and with a child in ``result_captured`` when a
    paid answer reached the disk and its staging write did not. Both are
    exactly what ``resume_extraction_batch`` finishes under the authority
    already recorded, so an intent closed on return would put them out of the
    reach of the job that reserved them.

    Read from the journal's own child states, which stay authoritative: this
    function decides nothing about a batch, it only asks whether the batch
    still says there is something to continue.
    """

    return bool(outcome.pending_count) or any(
        child.state == "result_captured" for child in outcome.children
    )


def _batch_detail(outcome: Any, *, closed: bool) -> str:
    """What the batch's current standing means for this job's own record."""

    if closed:
        return (
            "Nothing in this batch is still holding reserved authority or an "
            "unstaged answer, so this job's record of it is closed."
        )
    if not outcome.resume_available:
        # The batch's own durable eligibility, in its own words. An open
        # intent is not the same statement as continuable work: a child that
        # was killed mid-dispatch may already have been paid for, and what it
        # bought is a person's decision rather than something a resume — or a
        # sentence here — may take for them.
        return (
            f"{outcome.resume_refusal} Resuming this job again will not move "
            "it, and janki ended nothing on its own: settle that call with "
            "`janki operations --end` (Manage model calls, on the Assistant) "
            "first. This job's record of the batch stays open until then."
        )
    captured = sum(
        1 for child in outcome.children if child.state == "result_captured"
    )
    return (
        f"{outcome.pending_count} source(s) still hold reserved authority and "
        f"{captured} paid answer(s) are still waiting to be staged, so this "
        "job's record of them stays open. Resuming again continues exactly "
        "that authority; nothing new was authorized here."
    )


def dispatch_job_batch(
    config: ProjectConfig,
    job_id: str,
    plan: Any,
    **dispatch_options: Any,
) -> Any:
    """Record the intent, fsynced, then dispatch the owner's confirmed batch.

    The order is the point. The intent lands before any reservation, so a
    process that dies mid-dispatch leaves reserved ids this job can find the
    manifest by. It grants nothing: the owner's exact batch confirmation is
    still what authorizes the calls, and ``dispatch_extraction_batch`` still
    performs every revalidation and reservation itself.

    The intent is closed when the batch has nothing left that its own resume
    may finish — not when the call returns. See :func:`_batch_continues`.
    """

    if plan.job_id != job_id:
        planned_for = f"study job {plan.job_id}" if plan.job_id else "no study job"
        raise StudyJobError(
            f"That batch is planned for {planned_for}, not {job_id}; nothing "
            "was sent."
        )
    job = load_study_job(config, job_id)
    intent = _batch_intent(job_id, plan, kind="retry" if plan.retry_of else "extract_batch")
    saved = append_intent(config, job_id, intent, expected_revision=job.revision)

    outcome = extraction_batch.dispatch_extraction_batch(config, plan, **dispatch_options)

    if _batch_continues(outcome):
        # Deliberately no outcome: an intent is closed by what became of the
        # work it reserved, and some of that work is still reserved. Leaving it
        # open is what lets `resume_job_actions` find this exact manifest again
        # by the ids and hash already recorded, with no new approval.
        return outcome
    manifest = extraction_batch.batch_manifest_path(config, plan.batch_id)
    append_outcome(
        config,
        job_id,
        IntentOutcome(
            intent_id=intent.intent_id,
            state="applied",
            at=datetime.now(UTC).isoformat(),
            observed=((str(manifest), plan.manifest_sha256),),
            # Deliberately not a progress record: which children settled is
            # the journal's answer, re-derived by `study_job_status`.
            consequences={"children": len(plan.children)},
        ),
        expected_revision=saved.revision,
    )
    return outcome


@dataclass(frozen=True, slots=True)
class StudyJobPreview:
    """One rendered look at what this job has already saved.

    ``rendering_fingerprint`` is the previewed document's own sha256 — a pure
    function of the proposed card fields, the deck's real templates and its
    stylesheet, and nothing else. It carries no checkbox selection and no job
    revision, so saving one owner choice cannot invalidate another's.
    """

    job_id: str
    batch_id: str
    intent_id: str
    rendering_fingerprint: str
    child_indices: tuple[int, ...]
    conflicts: tuple[str, ...]
    preview: Any


def render_job_preview(config: ProjectConfig, job_id: str) -> StudyJobPreview:
    """Draw this job's saved proposals as the destination deck's real cards.

    A presentation: it accepts nothing, promotes nothing and writes nothing.
    Batches are tried in reverse *append* order — the order the job's own
    intent log records, never modification time — so a retry's proposals are
    what a person sees, and the reason each earlier batch could not be drawn
    is reported rather than swallowed.
    """

    job = load_study_job(config, job_id)
    deck = job_destination_deck(config, job)
    refusals: list[str] = []
    for intent in reversed(job.intents):
        if intent.kind not in ("extract_batch", "retry"):
            continue
        resolution = _resolve_intent(config, job_id, intent)
        if resolution.reference is None:
            refusals.append(f"{_reserved(intent, 'batch_id')}: {resolution.refusal}")
            continue
        batch_id = _reserved(intent, "batch_id")
        try:
            rendered = extraction_batch.render_extraction_batch_preview(
                config, batch_id, deck_path=deck
            )
        except (JankiError, OSError) as exc:
            refusals.append(f"{batch_id}: {exc}")
            continue
        return StudyJobPreview(
            job_id=job_id,
            batch_id=batch_id,
            intent_id=intent.intent_id,
            rendering_fingerprint=rendered.preview.sha256,
            child_indices=tuple(rendered.child_indices),
            conflicts=tuple(rendered.conflicts),
            preview=rendered.preview,
        )
    raise StudyJobError(
        f"Study job {job_id} has no batch whose proposals can be drawn"
        + (": " + "; ".join(refusals) if refusals else " yet.")
    )


@dataclass(frozen=True, slots=True)
class JobResumeAction:
    """What one open intent's resume did, and what it left behind.

    ``closed`` is the whole of it: an action that did not close its intent left
    work the same recorded authority may still finish, so the intent stays open
    and the next resume finds the same artifact by the same reserved id and
    hash. Nothing here is stored — it describes one call.
    """

    intent_id: str
    kind: str
    closed: bool
    detail: str
    #: The batch's own current standing, straight from the journal. ``None``
    #: for a publication, and for an artifact that did not resolve.
    batch: Any = None
    batch_id: str = ""
    recipe_id: str = ""

    @property
    def subject(self) -> str:
        """The artifact this action is about, named as its own store names it."""

        if self.batch_id:
            return f"Batch {self.batch_id}"
        if self.recipe_id:
            return f"Parts {self.recipe_id}"
        return f"Action {self.intent_id}"


def _bound_parts(intent: ActionIntent) -> tuple[tuple[str, str], ...] | None:
    """Exactly what this intent recorded its publication would put in the corpus.

    ``None`` when it recorded no usable pair of lists, which is not the same
    statement as an empty publication: an intent that named no parts cannot be
    proved to be a receipt's own, and that proof is what keeps a refused
    divergent recipe from being read back as an applied publication.
    """

    names = intent.bindings.get("target_names")
    hashes = intent.bindings.get("part_sha256")
    if not isinstance(names, list) or not isinstance(hashes, list):
        return None
    if len(names) != len(hashes):
        return None
    return tuple(
        (str(name), str(value)) for name, value in zip(names, hashes, strict=True)
    )


def _resume_publication(
    config: ProjectConfig,
    job_id: str,
    intent: ActionIntent,
    reference: ActionReference,
) -> JobResumeAction:
    """Close a publication whose receipt and parts landed before its outcome.

    Nothing is rendered, published or adopted here. Two comparisons have to
    agree before this job is bound to anything. The receipt was matched by the
    exact recipe id and hash this intent reserved, and the parts it names must
    be the exact parts this intent recorded it would write — a recipe id is an
    owner-written identity rather than a content hash, so a publication that
    was *refused* for naming different parts under a published id must never
    be read back as this job's own. Then the corpus itself must hold every one
    of those parts, byte for byte, which is the same measurement
    :func:`study_job_status` discloses.

    Anything else leaves the intent open with its exact evidence reported: a
    part that is absent, or present under different bytes, is the publication
    service's to write — it is the only writer that has them, and this store
    will not become a second one.
    """

    recipe_id = str(reference.reserves.get("recipe_id") or "")
    record = source_parts.load_source_part_receipt(
        config, recipe_id, verify_published=True
    )
    landed = tuple((part.target_name, part.sha256) for part in record.parts)
    bound = _bound_parts(intent)
    if bound != landed:
        planned = (
            ", ".join(name for name, _sha256 in bound)
            if bound
            else "no part names and hashes at all"
        )
        return JobResumeAction(
            intent_id=reference.intent_id,
            kind=reference.kind,
            closed=False,
            recipe_id=recipe_id,
            detail=(
                f"This action recorded that it would publish {planned}, and "
                f"the receipt at {reference.path} names "
                + ", ".join(name for name, _sha256 in landed)
                + ". A receipt is immutable, so different parts are a "
                "different recipe id: janki will not bind this job to a "
                "publication it did not make. Nothing was adopted, nothing "
                "was written, and this job's record of the action stays open."
            ),
        )
    unpublished = [part.target_name for part in record.parts if not part.published]
    if unpublished:
        return JobResumeAction(
            intent_id=reference.intent_id,
            kind=reference.kind,
            closed=False,
            recipe_id=recipe_id,
            detail=(
                f"{len(unpublished)} of {len(record.parts)} part(s) this receipt "
                "names are not in your corpus under the exact bytes it binds: "
                + ", ".join(unpublished)
                + ". Publish that recipe again — janki renders nothing and "
                "adopts nothing here — and this job's record of it stays open, "
                "with the receipt and the parts that did land untouched."
            ),
        )
    current = load_study_job(config, job_id)
    append_outcome(
        config,
        job_id,
        IntentOutcome(
            intent_id=reference.intent_id,
            state="applied",
            at=datetime.now(UTC).isoformat(),
            observed=((str(reference.path), reference.sha256),),
            consequences={
                "published": [part.target_name for part in record.parts],
                "recovered": True,
            },
        ),
        expected_revision=current.revision,
    )
    return JobResumeAction(
        intent_id=reference.intent_id,
        kind=reference.kind,
        closed=True,
        recipe_id=recipe_id,
        detail=(
            f"All {len(record.parts)} part(s) this receipt names are in your "
            "corpus: "
            + ", ".join(part.target_name for part in record.parts)
            + ". This job's binding to them is recorded, so they are what it "
            "would send. Nothing was rendered and nothing was sent."
        ),
    )


def _resume_batch(
    config: ProjectConfig,
    job_id: str,
    reference: ActionReference,
    dispatch_options: Mapping[str, Any],
) -> JobResumeAction:
    """Continue one batch under the authority this job already recorded."""

    batch_id = str(reference.reserves.get("batch_id"))
    outcome = extraction_batch.resume_extraction_batch(
        config, batch_id, **dispatch_options
    )
    if _batch_continues(outcome):
        return JobResumeAction(
            intent_id=reference.intent_id,
            kind=reference.kind,
            closed=False,
            detail=_batch_detail(outcome, closed=False),
            batch=outcome,
            batch_id=batch_id,
        )
    current = load_study_job(config, job_id)
    append_outcome(
        config,
        job_id,
        IntentOutcome(
            intent_id=reference.intent_id,
            state="applied",
            at=datetime.now(UTC).isoformat(),
            observed=((str(reference.path), reference.sha256),),
            consequences={"resumed": True},
        ),
        expected_revision=current.revision,
    )
    return JobResumeAction(
        intent_id=reference.intent_id,
        kind=reference.kind,
        closed=True,
        detail=_batch_detail(outcome, closed=True),
        batch=outcome,
        batch_id=batch_id,
    )


def resume_job_actions(
    config: ProjectConfig,
    job_id: str,
    **dispatch_options: Any,
) -> tuple[JobResumeAction, ...]:
    """Finish what this job's open intents still have authority for.

    Recovery after a crash between the fsynced intent and its outcome, and
    after a dispatch that returned with work still reserved. Each open intent's
    artifact is found by exact reserved id and hash — never by mtime and never
    by "the newest" — and then the owning service's own resume runs under the
    authority already recorded. No new approval is asked for and no new
    operation id is minted here.

    An intent whose artifact does not resolve, and one whose publication is
    only partly in the corpus, are reported and left open with their evidence.
    Neither is closed on a guess, and nothing else is adopted in their place.
    """

    job = load_study_job(config, job_id)
    closed = job.closed_intent_ids
    actions: list[JobResumeAction] = []
    for intent in job.intents:
        if intent.intent_id in closed or intent.kind not in _WRITABLE_INTENT_KINDS:
            continue
        resolution = _resolve_intent(config, job_id, intent)
        if resolution.reference is None:
            actions.append(
                JobResumeAction(
                    intent_id=intent.intent_id,
                    kind=intent.kind,
                    closed=False,
                    batch_id=_reserved(intent, "batch_id"),
                    recipe_id=_reserved(intent, "recipe_id"),
                    detail=(
                        "janki could not resolve the artifact this action "
                        f"reserved: {resolution.refusal}. Nothing was adopted "
                        "in its place and this job's record of it stays open."
                    ),
                )
            )
            continue
        if intent.kind == "source_parts":
            actions.append(
                _resume_publication(config, job_id, intent, resolution.reference)
            )
        else:
            actions.append(
                _resume_batch(config, job_id, resolution.reference, dispatch_options)
            )
    return tuple(actions)


# --- derived status -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyJobChildStatus:
    """One batch child, as the journal and its staging document stand now."""

    index: int
    operation_id: str
    state: str
    source_name: str
    staging_name: str
    records: int | None
    bookkeeping_complete: bool
    #: Whether the child's staging document is still live, and whether its
    #: exact base archive exists under ``data/staging/done/``.
    staging_present: bool
    archive_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation_id": self.operation_id,
            "state": self.state,
            "source_name": self.source_name,
            "staging_name": self.staging_name,
            "records": self.records,
            "bookkeeping_complete": self.bookkeeping_complete,
            "staging_present": self.staging_present,
            "archive_present": self.archive_present,
        }


@dataclass(frozen=True, slots=True)
class StudyJobBatchStatus:
    """One batch intent and what its artifact says, derived at read time."""

    intent_id: str
    batch_id: str
    kind: str
    manifest_sha256: str
    retry_of: str
    concurrency_limit: int
    children: tuple[StudyJobChildStatus, ...]
    resume_available: bool
    resume_refusal: str
    settled: bool
    missing: bool
    refusal: str

    @property
    def committed_count(self) -> int:
        return sum(1 for child in self.children if child.state == "committed")

    @property
    def unknown_count(self) -> int:
        return sum(1 for child in self.children if child.state == "outcome_unknown")

    @property
    def pending_count(self) -> int:
        return sum(
            1
            for child in self.children
            if child.state
            in (
                extraction_batch.UNRESERVED,
                "authorized",
                "dispatching",
                "running",
            )
        )

    @property
    def failed_count(self) -> int:
        return (
            len(self.children)
            - self.committed_count
            - self.unknown_count
            - self.pending_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "batch_id": self.batch_id,
            "kind": self.kind,
            "manifest_sha256": self.manifest_sha256,
            "retry_of": self.retry_of,
            "concurrency_limit": self.concurrency_limit,
            "settled": self.settled,
            "missing": self.missing,
            "refusal": self.refusal,
            "resume_available": self.resume_available,
            "resume_refusal": self.resume_refusal,
            "committed": self.committed_count,
            "unknown": self.unknown_count,
            "pending": self.pending_count,
            "failed": self.failed_count,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class StudyJobPartsStatus:
    """One source-part intent, with publication measured rather than recalled."""

    intent_id: str
    recipe_id: str
    receipt_sha256: str
    parent_name: str
    part_count: int
    published_count: int
    settled: bool
    missing: bool
    refusal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "recipe_id": self.recipe_id,
            "receipt_sha256": self.receipt_sha256,
            "parent_name": self.parent_name,
            "part_count": self.part_count,
            "published_count": self.published_count,
            "settled": self.settled,
            "missing": self.missing,
            "refusal": self.refusal,
        }


@dataclass(frozen=True, slots=True)
class StudyJobStatus:
    """Where one job stands, composed from the stores that actually know.

    Nothing here is stored in the job document. The child states come from the
    operation journal through ``extraction_batch.extraction_batch_status``,
    the publication counts from the corpus itself, the staging and archive
    facts from those files, and the outstanding media work from the ledger's
    own write-ahead record.
    """

    job_id: str
    kind: str
    created_at: str
    revision: str
    parent_source_name: str
    parent_sha256: str
    deck_path: str
    deck_sha256: str
    deck_present: bool
    deck_current: bool
    choices: Mapping[str, Any]
    layout_count: int
    parts: tuple[StudyJobPartsStatus, ...]
    batches: tuple[StudyJobBatchStatus, ...]
    open_intents: tuple[str, ...]
    unresolved_intents: tuple[str, ...]
    blocking_operation_ids: tuple[str, ...]
    pending_audio_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "revision": self.revision,
            "parent_source_name": self.parent_source_name,
            "parent_sha256": self.parent_sha256,
            "deck_path": self.deck_path,
            "deck_sha256": self.deck_sha256,
            "deck_present": self.deck_present,
            "deck_current": self.deck_current,
            "choices": _plain(self.choices),
            "layout_count": self.layout_count,
            "parts": [entry.to_dict() for entry in self.parts],
            "batches": [entry.to_dict() for entry in self.batches],
            "open_intents": list(self.open_intents),
            "unresolved_intents": list(self.unresolved_intents),
            "blocking_operation_ids": list(self.blocking_operation_ids),
            "pending_audio_count": self.pending_audio_count,
        }


def _deck_state(config: ProjectConfig, job: StudyJob) -> tuple[bool, bool]:
    path = config.root / job.header.deck_path
    try:
        if not path.is_file():
            return False, False
        return True, hashlib.sha256(path.read_bytes()).hexdigest() == (
            job.header.deck_sha256
        )
    except OSError:
        return False, False


def _child_statuses(
    config: ProjectConfig, outcome: Any
) -> tuple[StudyJobChildStatus, ...]:
    done = config.staging_dir / "done"
    statuses: list[StudyJobChildStatus] = []
    for child in outcome.children:
        staging_path = Path(child.staging_path)
        statuses.append(
            StudyJobChildStatus(
                index=child.index,
                operation_id=child.operation_id,
                state=child.state,
                source_name=Path(child.source).name,
                staging_name=staging_path.name,
                records=child.records,
                bookkeeping_complete=child.bookkeeping_complete,
                staging_present=staging_path.is_file(),
                archive_present=(done / staging_path.name).is_file(),
            )
        )
    return tuple(statuses)


def _batch_status(
    config: ProjectConfig,
    job: StudyJob,
    intent: ActionIntent,
    resolution: _Resolution,
) -> StudyJobBatchStatus:
    settled = intent.intent_id in job.closed_intent_ids
    batch_id = _reserved(intent, "batch_id")
    if resolution.reference is None:
        return StudyJobBatchStatus(
            intent_id=intent.intent_id,
            batch_id=batch_id,
            kind=intent.kind,
            manifest_sha256=_reserved(intent, "manifest_sha256"),
            retry_of="",
            concurrency_limit=0,
            children=(),
            resume_available=False,
            resume_refusal="",
            settled=settled,
            missing=True,
            refusal=resolution.refusal,
        )
    try:
        outcome = extraction_batch.extraction_batch_status(config, batch_id)
    except (JankiError, OSError) as exc:
        return StudyJobBatchStatus(
            intent_id=intent.intent_id,
            batch_id=batch_id,
            kind=intent.kind,
            manifest_sha256=resolution.reference.sha256,
            retry_of="",
            concurrency_limit=0,
            children=(),
            resume_available=False,
            resume_refusal="",
            settled=settled,
            missing=True,
            refusal=str(exc),
        )
    raw = json.loads(resolution.reference.path.read_text(encoding="utf-8"))
    return StudyJobBatchStatus(
        intent_id=intent.intent_id,
        batch_id=batch_id,
        kind=intent.kind,
        manifest_sha256=resolution.reference.sha256,
        retry_of=str(raw.get("retry_of", "")),
        concurrency_limit=outcome.concurrency_limit,
        children=_child_statuses(config, outcome),
        resume_available=outcome.resume_available,
        resume_refusal=outcome.resume_refusal,
        settled=settled,
        missing=False,
        refusal="",
    )


def _parts_status(
    config: ProjectConfig,
    job: StudyJob,
    intent: ActionIntent,
    resolution: _Resolution,
) -> StudyJobPartsStatus:
    settled = intent.intent_id in job.closed_intent_ids
    recipe_id = _reserved(intent, "recipe_id")
    if resolution.reference is None:
        return StudyJobPartsStatus(
            intent_id=intent.intent_id,
            recipe_id=recipe_id,
            receipt_sha256=_reserved(intent, "receipt_sha256"),
            parent_name="",
            part_count=0,
            published_count=0,
            settled=settled,
            missing=True,
            refusal=resolution.refusal,
        )
    # Measured, not remembered: publication is real state, and the reader that
    # discloses it always consults the corpus.
    record = source_parts.load_source_part_receipt(
        config, recipe_id, verify_published=True
    )
    return StudyJobPartsStatus(
        intent_id=intent.intent_id,
        recipe_id=recipe_id,
        receipt_sha256=resolution.reference.sha256,
        parent_name=record.parent_name,
        part_count=len(record.parts),
        published_count=sum(1 for part in record.parts if part.published),
        settled=settled,
        missing=False,
        refusal="",
    )


def _blocking_operation_ids(
    config: ProjectConfig, batch_ids: Sequence[str]
) -> tuple[str, ...]:
    """Every operation of this job's batches the journal says blocks spending."""

    wanted = frozenset(batch_ids)
    if not wanted:
        return ()
    try:
        journal = operations.OperationJournal.load(config.operations_file)
    except (JankiError, OSError):
        return ()
    return tuple(
        sorted(
            operation_id
            for operation_id, operation in journal.operations.items()
            if operation.batch_id in wanted and operation.blocks_spending
        )
    )


def _pending_audio_count(config: ProjectConfig) -> int:
    try:
        return len(ledger.load(config.ledger_file).pending_audio)
    except (JankiError, OSError, ValueError):
        return 0


def study_job_status(config: ProjectConfig, job_id: str) -> StudyJobStatus:
    """Compose where this job stands from the stores that actually know.

    Reads only. Every number below is derived at this moment: none of it is
    written into the job document, because a stored count is a second answer
    to a question the journal, the manifests, the corpus, the archive and the
    ledger already answer.
    """

    job = load_study_job(config, job_id)
    closed = job.closed_intent_ids
    parts: list[StudyJobPartsStatus] = []
    batches: list[StudyJobBatchStatus] = []
    unresolved: list[str] = []
    for intent in job.intents:
        resolution = _resolve_intent(config, job_id, intent)
        if resolution.reference is None:
            unresolved.append(intent.intent_id)
        if intent.kind == "source_parts":
            parts.append(_parts_status(config, job, intent, resolution))
        elif intent.kind in ("extract_batch", "retry"):
            batches.append(_batch_status(config, job, intent, resolution))
    deck_present, deck_current = _deck_state(config, job)
    return StudyJobStatus(
        job_id=job_id,
        kind=job.header.kind,
        created_at=job.header.created_at,
        revision=job.revision,
        parent_source_name=job.header.parent_source_name,
        parent_sha256=job.header.parent_sha256,
        deck_path=job.header.deck_path,
        deck_sha256=job.header.deck_sha256,
        deck_present=deck_present,
        deck_current=deck_current,
        choices=job.choices,
        layout_count=len(job.layouts),
        parts=tuple(parts),
        batches=tuple(batches),
        open_intents=tuple(
            intent.intent_id for intent in job.intents if intent.intent_id not in closed
        ),
        unresolved_intents=tuple(unresolved),
        blocking_operation_ids=_blocking_operation_ids(
            config, [entry.batch_id for entry in batches if entry.batch_id]
        ),
        pending_audio_count=_pending_audio_count(config),
    )
