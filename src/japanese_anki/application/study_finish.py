"""One owner-authorized apply-and-finish for a whole study job.

Everything this service coordinates was already prepared, extracted, curated,
assigned and previewed. **Apply and finish** is one immutable authority over
that whole set, one phase chain, and one prepared payload per owning writer::

    authorized → reviewed → promoted → enriched → audio_complete
               → packaged → complete

The rule that shapes every line below is that **this module writes no study
content**. :mod:`workbench.review` and :mod:`application.assistant_staging_review`
are still the only writers of a staging review; :mod:`application.promotion` of
canonical landings, archives and the ledger's promotion rows;
:mod:`application.character_notes` of the two reference stores;
:mod:`application.enrichment` of dictionary values; :mod:`application.audio` of
clips and their canonical references; :mod:`application.deck_package` of an
``.apkg`` and its export rows; and :mod:`card_preview` of a rendered preview.
The coordinator binds an authority, persists each writer's own prepared intent
before its first mutation, calls that writer's narrow entrypoint, and records a
receipt derived from artifacts. It holds no copy of any writer.

Three consequences worth stating out loud.

**No writer result and no phase label is evidence.** A value that was returned
but never persisted is not proof, so every recovery reads the persisted intent
and re-measures the bound components. A crash between a writer's return and the
receipt costs nothing.

**Every date-bearing value is frozen by its owning writer's prepare**, so a
resume on a later day replays the same bytes instead of recomputing a date and
finding a third digest.

**The authority is bound before the first effect.** It is written with
``expected_absent=True`` under its own content-addressed name, so re-running the
identical plan continues the receipt that already exists rather than authorizing
the same work twice.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from japanese_anki import enrich, jpdb, kanji, ledger, patterns
from japanese_anki.application import (
    assistant_staging_review,
    character_notes,
    deck_package,
    promotion,
    study_curation,
    study_job,
)
from japanese_anki.application import audio as audio_application
from japanese_anki.application import (
    coverage as coverage_application,
)
from japanese_anki.application import (
    enrichment as enrichment_application,
)
from japanese_anki.application import (
    finish as finish_application,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import project_deck_media_paths
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records_snapshot,
    prepare_bound_directory,
    read_bytes_bound,
    records_json_text,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import openai_realtime, sentence_profile_for
from japanese_anki.workbench.review import ReviewBatchRequest

__all__ = [
    "STUDY_FINISH_STATES",
    "StudyFinishAudioPlan",
    "StudyFinishError",
    "StudyFinishPart",
    "StudyFinishPlan",
    "StudyFinishResult",
    "StudyFinishScope",
    "StudyFinishSelection",
    "StudyFinishStaleAudioPlanError",
    "execute_study_finish",
    "inspect_study_finish",
    "list_study_finishes",
    "plan_study_finish",
    "receipt_path",
    "resolve_study_finish_scope",
    "resume_study_finish",
    "study_finish_directory",
]


StudyFinishState = Literal[
    "authorized",
    "reviewed",
    "promoted",
    "enriched",
    "audio_complete",
    "packaged",
    "complete",
]

#: The one phase chain, in order. Every advance is +1 under the record's lock.
STUDY_FINISH_STATES: tuple[StudyFinishState, ...] = (
    "authorized",
    "reviewed",
    "promoted",
    "enriched",
    "audio_complete",
    "packaged",
    "complete",
)
_STATE_INDEX = {state: index for index, state in enumerate(STUDY_FINISH_STATES)}

StudyFinishPhase = Literal[
    "Preparing finish",
    "Writing the reviewed staging files",
    "Promoting the reviewed cards",
    "Writing reference facts and dictionary values",
    "Creating card audio",
    "Building the Anki package",
    "Rendering the final card preview",
    "Saving finish receipt",
]
StudyFinishProgress = Callable[[StudyFinishPhase], None]

#: Receipt keys, one per phase after ``authorized``, in chain order.
_RECEIPT_KEYS: tuple[str, ...] = (
    "review_receipt",
    "promotion_receipt",
    "enrichment_receipt",
    "audio_receipt",
    "package_receipt",
    "preview_receipt",
)

#: Mutable working slots. ``*_intent`` keys are append-only by construction;
#: ``paid_clip_reservations`` is deliberately not one, because a reservation
#: that proved unsent is replaced by its successor at the same target.
_INTENT_KEYS: tuple[str, ...] = (
    "promotion_intents",
    "enrichment_intent",
    "package_intents",
)

_RECORD_KEYS = {
    "schema_version",
    "kind",
    "receipt_id",
    "state",
    "authority",
    "authorized_at",
    "updated_at",
    "paid_clip_reservations",
    *_INTENT_KEYS,
    *_RECEIPT_KEYS,
}

_AUTHORITY_KEYS = {
    "version",
    "job",
    "deck",
    "parts",
    "review",
    "promotion",
    "reference",
    "enrichment",
    "dictionary",
    "audio",
    "package",
    "preview",
    "configuration_fingerprint",
}

_KIND = "study_finish"
_SCHEMA_VERSION = 1

#: The promotion states whose prepared intent binds a canonical component. Every
#: other promotable state opens no collection, so it binds none and the fold
#: carries its checkpoint through unchanged.
_CANONICAL_STATES = frozenset({"lands", "nothing_lands"})

#: A recovery list, not an archive browser — the same cap and the same
#: unfinished-first rule ``kanji_finish.list_kanji_finishes`` uses.
_LIST_LIMIT = 20

#: Child states that still hold this job's authority or an unsettled answer.
#: A ``failed_*`` child a retry superseded is history, not a blocker.
_LIVE_CHILD_STATES = frozenset(
    {
        "unreserved",
        "authorized",
        "dispatching",
        "running",
        "result_captured",
        "outcome_unknown",
    }
)

#: Which owner choice records which decision. Named so a refusal can point at
#: the control the owner acts in rather than at a schema key.
_CHOICE_CONTROLS = {
    "review_flags": "the review editor (`janki study review`)",
    "review_patterns": (
        "the review editor's pattern control (`janki study review --patterns`)"
    ),
    "coverage_reasons": "the coverage editor (`janki study coverage`)",
    "dispositions": "the disposition editor (`janki study disposition`)",
}


class StudyFinishError(JankiError):
    """The exact study-finish authority cannot safely continue."""


class StudyFinishStaleAudioPlanError(StudyFinishError):
    """A confirmed new paid clip is already on disk from another run.

    This finish cannot claim those bytes: only a reservation it made itself,
    matched against the paid writer's own recorded attempt, proves that *this*
    call produced them. Nothing is re-sent, nothing is forced, and no approval
    is inferred — the owner plans again, and the fresh plan truthfully
    classifies the clip as reused rather than as new work.
    """


# --- small shared helpers -----------------------------------------------------


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StudyFinishError(
            f"Study finish authority is not finite JSON: {exc}"
        ) from exc


def _plain(value: object) -> Any:
    """A JSON round-trip, so nothing mutable or unserializable rides inside."""

    return json.loads(_canonical(value))


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    root = config.root.resolve()
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise StudyFinishError(
            f"Study finish {label} escapes the repository: {target}"
        ) from exc
    value = relative.as_posix()
    if not value or ".." in PurePosixPath(value).parts:
        raise StudyFinishError(f"Study finish {label} is not repository-relative.")
    return value


def _authority_path(config: ProjectConfig, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StudyFinishError(f"Study finish {label} path is malformed.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise StudyFinishError(
            f"Study finish {label} path is not repository-relative."
        )
    return (config.root.resolve() / Path(*pure.parts)).absolute()


def study_finish_directory(config: ProjectConfig) -> Path:
    """Where one job's finish receipts live, beside every other archive."""

    return (config.staging_dir / "done" / "study").absolute()


def receipt_path(config: ProjectConfig, receipt_id: str) -> Path:
    """The one durable record path for a finish receipt id."""

    if not _is_sha(receipt_id):
        raise StudyFinishError("A study finish receipt id is malformed.")
    return study_finish_directory(config) / f"study-finish-{receipt_id}.json"


def _staging_revision(path: Path) -> str:
    try:
        return _sha(read_bytes_bound(path))
    except FileNotFoundError as exc:
        raise StudyFinishError(
            f"The staged review {path} is not in this repository any more."
        ) from exc
    except (DataError, OSError) as exc:
        raise StudyFinishError(f"Could not read the staged review {path}: {exc}") from exc


def _emit(progress: StudyFinishProgress | None, phase: StudyFinishPhase) -> None:
    if progress is None:
        return
    try:
        progress(phase)
    except Exception:  # noqa: BLE001 - a reporting surface never fails a finish
        return


# --- the aggregate scope (§7.12) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyFinishSelection:
    """One part's binding to one owning promotion receipt, and what it took.

    ``record_ids`` are the ids **this job** accepted out of that receipt, in the
    authority's own order. A receipt may legitimately hold more: a retry selects
    the rows its live review still carries, and an owner who left some
    already-promoted rows out of that review did not put them in this job.
    ``archive_path`` is the part's own archive, so a receipt from somewhere else
    cannot stand in for one of this part's.
    """

    part_name: str
    receipt_id: str
    archive_path: str
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyFinishScope:
    """Every part's receipts, and the deduped identities the job's work runs over.

    ``members`` are the **unchanged** :func:`finish.resolve_finish_scope` values,
    one per distinct promotion receipt. ``selections`` retain each part's own
    receipt association and the exact ids taken from it — one entry per part
    occurrence, so a part that binds an old receipt beside its new one keeps
    both. ``projection`` is the deduped ``(record_id, owner_stem, deck_path)``
    triple over the **selected** ids two archives may legitimately both
    receipt — proved to agree on the owner deck before anything runs over it.
    """

    members: tuple[finish_application.FinishScope, ...]
    selections: tuple[StudyFinishSelection, ...]
    projection: tuple[tuple[str, str, str], ...]
    fingerprint: str

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record_id for record_id, _stem, _path in self.projection)


def resolve_study_finish_scope(
    config: ProjectConfig, selections: Sequence[StudyFinishSelection]
) -> StudyFinishScope:
    """Resolve every receipt this finish bound, and dedupe the ids it selected.

    Each selection is proved against the whole owning member: the receipt has to
    belong to that part's own archive, and every selected id has to be one the
    receipt really promoted. Nothing widens to the rest of the member — an
    unselected already-promoted row is not this job's card — and nothing narrows
    silently either, because the caller compares the projection against the ids
    the authority accepted.

    Two archives may receipt one final id; that is ordinary and is retained.
    What refuses is disagreement: the same identity owned by two different
    study decks is not something a dedupe may silently pick between.
    """

    members: dict[str, finish_application.FinishScope] = {}
    owners: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    bound: list[StudyFinishSelection] = []
    for selection in selections:
        member = members.get(selection.receipt_id)
        if member is None:
            member = finish_application.resolve_finish_scope(
                config, selection.receipt_id
            )
            members[selection.receipt_id] = member
        archive = _authority_path(config, selection.archive_path, label="archive")
        if member.archive_path.resolve() != archive.resolve():
            raise StudyFinishError(
                f"Part {selection.part_name!r} binds promotion receipt "
                f"{selection.receipt_id}, which belongs to "
                f"{member.archive_path.name} rather than this part's "
                f"{archive.name}. Nothing was run over it."
            )
        outside = [
            record_id
            for record_id in selection.record_ids
            if record_id not in member.record_ids
        ]
        if not selection.record_ids or outside:
            raise StudyFinishError(
                f"Part {selection.part_name!r} claims "
                + (", ".join(outside) or "no card at all")
                + f" from promotion receipt {selection.receipt_id}, which does not "
                "promote them. Nothing was run over it."
            )
        deck_by_stem = {group.stem: str(group.deck_path) for group in member.owner_groups}
        for record_id in selection.record_ids:
            stem = member.owner_stems[member.record_ids.index(record_id)]
            found = (stem, deck_by_stem[stem])
            seen = owners.get(record_id)
            if seen is None:
                owners[record_id] = found
                order.append(record_id)
            elif seen != found:
                raise StudyFinishError(
                    f"Receipted record {record_id} is owned by {seen[0]!r} in "
                    f"one archive and {found[0]!r} in another. Two receipts "
                    "for one identity must agree on its study deck; nothing "
                    "was run over it."
                )
        bound.append(selection)
    projection = tuple(
        (record_id, owners[record_id][0], owners[record_id][1]) for record_id in order
    )
    ordered_members = tuple(members[receipt_id] for receipt_id in members)
    fingerprint = _sha(
        _canonical(
            {
                "version": 2,
                "receipts": [member.receipt_id for member in ordered_members],
                "members": [member.fingerprint for member in ordered_members],
                "selections": [
                    [
                        selection.part_name,
                        selection.receipt_id,
                        selection.archive_path,
                        list(selection.record_ids),
                    ]
                    for selection in bound
                ],
                "projection": [list(entry) for entry in projection],
            }
        ).encode("utf-8")
    )
    return StudyFinishScope(
        members=ordered_members,
        selections=tuple(bound),
        projection=projection,
        fingerprint=fingerprint,
    )


# --- planning -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyFinishPart:
    """One part of the job, with the owner decisions taken over it."""

    part_name: str
    staging_path: Path
    staging_sha256: str
    review_record_ids: tuple[str, ...]
    review_patterns: bool
    coverage_reason: str
    coverage_approved_at: str | None
    projected_state: str
    gate: str
    #: This part's canonical checkpoints inside the fold: the digest the
    #: collection has when this part starts writing, and the digest it has when
    #: that part is done. A part is the Nth writer of one file, so these are the
    #: values its own prepared intent has to bind — a fresh decision that would
    #: write a different pair is a repository that moved under the authority.
    expected_before: str | None
    expected_after: str | None
    landed_ids: tuple[str, ...]
    held_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    archive_retry_ids: tuple[str, ...]
    #: ``exclude`` or ``defer`` with the owner's literal reason, for a part that
    #: lands nothing or holds rows. ``None`` when no disposition was needed.
    disposition: Mapping[str, Any] | None

    @property
    def lands(self) -> bool:
        return self.projected_state == "lands"


@dataclass(frozen=True, slots=True)
class StudyFinishAudioPlan:
    """Exactly what the one confirmation discloses about audio spending.

    Word and sentence numbers stay separate, and so do the three sentence
    numbers: the expected stored slots are counted from the accepted records'
    own fields, the unique clip requests from the plan, and neither is the
    other. ``omitted_by_owner`` is the explicit opt-out, which binds zero new
    example requests and is never reported as successful generation.
    """

    include_example_audio: bool
    omitted_by_owner: bool
    record_ids: tuple[str, ...]
    word_provider: Mapping[str, Any] | None
    example_provider: Mapping[str, Any] | None
    paid_word_calls: int
    paid_example_calls: int
    expected_word_slots: int
    expected_example_slots: int
    word_counts: audio_application.AudioClipCounts
    example_counts: audio_application.AudioClipCounts

    @property
    def unique_clip_requests(self) -> int:
        return self.word_counts.total + self.example_counts.total

    @property
    def paid_provider_calls(self) -> int:
        return self.paid_word_calls + self.paid_example_calls


@dataclass(frozen=True, slots=True)
class StudyFinishPlan:
    """Display-only consequences of one whole study job's apply and finish."""

    repository_root: Path
    job_id: str
    deck_path: Path
    output_path: Path
    deck_name: str
    card_types: tuple[str, ...]
    parts: tuple[StudyFinishPart, ...]
    #: sha256(text_0) … sha256(text_N): one more entry than there are parts.
    canonical_digests: tuple[str | None, ...]
    #: The deduped accepted identities every later phase runs over.
    record_ids: tuple[str, ...]
    reference: character_notes.ReferenceFactsPreparation
    enrichment_changed_ids: tuple[str, ...]
    missing_reference_facts: tuple[character_notes.MissingReferenceFact, ...]
    audio: StudyFinishAudioPlan
    #: Whole-deck package projection facts. ``note_count``/``card_count`` are
    #: the deck's configured capacity; the delivered card total is read back
    #: from the archive at ``packaged``.
    note_count: int
    card_count: int
    package_record_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    finish_directory: Path
    record_path: Path
    authority: Mapping[str, Any]
    fingerprint: str

    @property
    def paid_provider_calls(self) -> int:
        return self.audio.paid_provider_calls

    @property
    def undisposed_parts(self) -> tuple[StudyFinishPart, ...]:
        """Parts whose evidence no owner disposition covers.

        A part that accepts nothing needs one, and so does a part that holds or
        excludes a row: §9.6 keeps the job incomplete until the owner says, in
        their own words, that those cards are out of this job's scope.
        """

        undisposed: list[StudyFinishPart] = []
        for part in self.parts:
            accepted = part.landed_ids or part.archive_retry_ids
            withheld = part.held_ids or part.excluded_ids
            if accepted and not withheld:
                continue
            disposition = part.disposition
            if not isinstance(disposition, Mapping) or not str(
                disposition.get("reason") or ""
            ).strip():
                undisposed.append(part)
                continue
            selected = tuple(
                str(item) for item in disposition.get("record_ids") or ()
            )
            if withheld and selected:
                missing = [
                    record_id
                    for record_id in (*part.held_ids, *part.excluded_ids)
                    if record_id not in selected
                ]
                if missing:
                    undisposed.append(part)
        return tuple(undisposed)


def _job_parts(config: ProjectConfig, job_id: str) -> tuple[tuple[str, Path], ...]:
    """Every part of this job whose one effective child settled, by name.

    Derived at read time from the job's own durable stores — this adds no
    second progress record. Ambiguity refuses by name rather than resolving
    itself: two settled children for one part, or a part with work still in
    flight, is a state only the owner can settle.
    """

    status = study_job.study_job_status(config, job_id)
    settled: dict[str, list[str]] = {}
    unsettled: list[str] = []
    for batch in status.batches:
        if batch.missing:
            raise StudyFinishError(
                f"Batch {batch.batch_id} of study job {job_id} has no readable "
                f"manifest ({batch.refusal}), so this job's parts cannot be "
                "enumerated. Nothing was planned."
            )
        for child in batch.children:
            if child.state == "committed" and child.bookkeeping_complete:
                settled.setdefault(child.source_name, []).append(child.staging_name)
            elif child.state in _LIVE_CHILD_STATES or child.state == "committed":
                # A failed child a retry superseded is ordinary history. What
                # blocks a finish is work still holding reserved authority, an
                # unstaged paid answer, an unknown outcome, or a settled child
                # whose own bookkeeping never finished.
                unsettled.append(f"{child.source_name} ({child.state})")
    if unsettled:
        raise StudyFinishError(
            "This job still has work that is not settled: "
            + ", ".join(sorted(set(unsettled)))
            + ". Finish or settle it first; a finish never resends a call and "
            "never counts an unknown outcome as progress."
        )
    if not settled:
        raise StudyFinishError(
            f"Study job {job_id} has no settled extraction to finish yet."
        )
    ambiguous = sorted(name for name, staged in settled.items() if len(set(staged)) != 1)
    if ambiguous:
        raise StudyFinishError(
            "More than one settled extraction covers "
            + ", ".join(ambiguous)
            + ". A part has exactly one effective child; janki will not choose "
            "between them."
        )
    parts: list[tuple[str, Path]] = []
    for name in sorted(settled):
        staging_path = (config.staging_dir / settled[name][0]).absolute()
        if not staging_path.is_file():
            raise StudyFinishError(
                f"The staged review for {name} is no longer at {staging_path}. A "
                "part that was already promoted outside this job cannot be "
                "reviewed and promoted again by it."
            )
        parts.append((name, staging_path))
    return tuple(parts)


def _bound_choice(
    job: study_job.StudyJob,
    key: str,
    part_name: str,
    *,
    staging_sha256: str,
    rendering_fingerprint: str,
) -> Mapping[str, Any] | None:
    """One saved owner choice for this part, revalidated or refused as stale.

    All four bindings are compared — the job, the part, that part's current
    staging bytes and the fingerprint of the rendering the owner decided over.
    A stale saved choice is never applied: the owner decides again over the
    fresh rendering, and nothing is carried forward silently.
    """

    saved = job.choices.get(key)
    if not isinstance(saved, Mapping):
        return None
    entry = saved.get(part_name)
    if not isinstance(entry, Mapping):
        return None
    bound = (
        str(entry.get("job_id") or ""),
        str(entry.get("part") or ""),
        str(entry.get("staging_sha256") or ""),
        str(entry.get("rendering_fingerprint") or ""),
    )
    fresh = (job.header.job_id, part_name, staging_sha256, rendering_fingerprint)
    if bound != fresh:
        raise StudyFinishError(
            f"The saved {key.replace('_', ' ')} for {part_name} was taken over a "
            f"different rendering of this job ({bound} rather than {fresh}), so "
            f"it is stale. Decide again in {_CHOICE_CONTROLS[key]} over the "
            "current preview; nothing was applied."
        )
    return entry


def _rendering_fingerprint(config: ProjectConfig, job_id: str) -> str:
    """The fingerprint every owner decision on this job was bound to.

    The preview document's own sha256: a pure function of the proposed card
    fields, the deck's real templates and its stylesheet. It carries no
    checkbox selection and no job revision, so saving one owner choice cannot
    stale another's.
    """

    try:
        return study_job.render_job_preview(config, job_id).rendering_fingerprint
    except (JankiError, OSError) as exc:
        raise StudyFinishError(
            f"This job's owner decisions are bound to a rendering janki cannot "
            f"draw right now: {exc}"
        ) from exc


def _disposition_of(
    job: study_job.StudyJob,
    part_name: str,
    *,
    staging_sha256: str,
    rendering_fingerprint: str,
) -> Mapping[str, Any] | None:
    entry = _bound_choice(
        job,
        "dispositions",
        part_name,
        staging_sha256=staging_sha256,
        rendering_fingerprint=rendering_fingerprint,
    )
    if entry is None:
        return None
    return {
        "action": str(entry.get("action") or ""),
        "record_ids": [str(item) for item in entry.get("record_ids") or ()],
        "reason": str(entry.get("reason") or ""),
    }


def _coverage_approval(
    config: ProjectConfig,
    part_name: str,
    staging_path: Path,
    reason: str,
    *,
    approved_at: str,
) -> Mapping[str, Any] | None:
    """This part's frozen owner coverage payload, or ``None`` when none is due.

    Never the standalone writer: the payload is composed into the same final
    staging after-text the review write publishes, so no intermediate
    review-only document is ever on disk.
    """

    decision = coverage_application.plan_coverage(config, staging_path)
    if decision.state != "ready":
        if reason:
            raise StudyFinishError(
                f"{part_name}'s coverage is already {decision.state.replace('_', ' ')}, "
                "so the saved coverage reason has nothing to approve. Clear it "
                "and plan again; nothing was written."
            )
        return None
    if not reason.strip():
        raise StudyFinishError(
            f"{part_name}'s source coverage is unmeasured, so this part cannot "
            "promote without your own reason for accepting it. Record one in "
            f"{_CHOICE_CONTROLS['coverage_reasons']}; janki never writes one for "
            "you."
        )
    prepared = coverage_application.prepare_owner_coverage_approval(
        config, decision, reason=reason, approved_at=approved_at
    )
    return prepared.payload


def _projected_audio_references(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
) -> tuple[VocabularyRecord, ...]:
    """The exact post-audio canonical records this plan's clips imply.

    Target names are ``fp(record.id)`` and ``fp(record.id + example.japanese)``,
    which no later phase changes, so the post-audio canonical text is a real
    digest long before the WAV bytes exist. Nothing here writes: it composes the
    ledger's own public filename fingerprints with the audio module's own media
    path, and refuses any clip that does not address one of these records.
    """

    selected = set(plan.record_ids)
    by_record: dict[str, list[audio_application.AudioClipPlan]] = {}
    for clip in plan.clips:
        by_record.setdefault(clip.record_id, []).append(clip)
    used: set[tuple[str, str, str]] = set()
    projected: list[VocabularyRecord] = []
    for record in records:
        if record.id not in selected:
            projected.append(record)
            continue
        clips = by_record.get(record.id, [])
        word = [clip for clip in clips if clip.kind == "word"]
        if len(word) > 1:
            raise StudyFinishError(
                f"The audio plan repeats word audio for {record.id!r}."
            )
        audio = record.audio
        if word:
            clip = word[0]
            expected = (
                f"janki-{ledger.word_audio_filename_fingerprint(record)}"
                f"{clip.provider.suffix}"
            )
            if clip.target != expected:
                raise StudyFinishError(
                    f"The audio plan changed the word-audio identity for {record.id!r}."
                )
            audio = audio_application.audio_cmd.media_relative(
                audio_application.media_target_path(config, clip.target),
                plan.media_dir,
            )
            used.add((clip.record_id, clip.kind, clip.target))
        examples: list[ExampleSentence] = []
        for example in record.examples:
            prefix = (
                f"janki-{ledger.example_audio_filename_fingerprint(record, example)}"
            )
            matches = [
                clip
                for clip in clips
                if clip.kind == "example"
                and clip.target == f"{prefix}{clip.provider.suffix}"
            ]
            if len(matches) > 1:
                raise StudyFinishError(
                    f"The audio plan repeats one example for {record.id!r}."
                )
            if not matches:
                examples.append(example)
                continue
            clip = matches[0]
            if (
                clip.request_input != ledger.example_audio_request(example)
                or clip.content_fingerprint
                != ledger.example_audio_content_fingerprint(example)
            ):
                raise StudyFinishError(
                    f"The audio plan changed the spoken identity for {record.id!r}."
                )
            examples.append(
                replace(
                    example,
                    audio=audio_application.audio_cmd.media_relative(
                        audio_application.media_target_path(config, clip.target),
                        plan.media_dir,
                    ),
                )
            )
            used.add((clip.record_id, clip.kind, clip.target))
        projected.append(replace(record, audio=audio, examples=examples))
        selected.discard(record.id)
    expected_clips = {(clip.record_id, clip.kind, clip.target) for clip in plan.clips}
    if selected or used != expected_clips:
        raise StudyFinishError(
            "The audio plan names a card or clip outside this finish's accepted "
            "selection."
        )
    return tuple(projected)


def _projected_package_media(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
    deck_path: Path,
) -> dict[Path, str | None]:
    """Every media path the built deck will read, with its projected hash.

    ``None`` for exactly the enumerated clips whose WAV bytes cannot exist
    before the paid call. A ``current`` clip keeps the bytes on disk and a
    ``recoverable`` one its bound recovery hash, so only genuinely new work is
    left open for the realization check to close.
    """

    clip_by_path = {
        audio_application.media_target_path(config, clip.target).absolute(): clip
        for clip in plan.clips
    }
    try:
        paths = project_deck_media_paths(
            deck_path,
            config,
            config.normalized_file.resolve(),
            records,
            allowed_missing_media=frozenset(clip_by_path),
        )
    except (JankiError, OSError, ValueError) as exc:
        raise StudyFinishError(str(exc)) from exc
    projected: dict[Path, str | None] = {}
    for raw in paths:
        path = raw.absolute()
        clip = clip_by_path.get(path)
        if clip is None or clip.state == "current":
            try:
                projected[path] = _sha(read_bytes_bound(path))
            except (DataError, OSError) as exc:
                raise StudyFinishError(
                    f"Could not read the packaged media {path}: {exc}"
                ) from exc
        elif clip.state == "recoverable":
            projected[path] = clip.recovery_sha256
        else:
            projected[path] = None
    return projected


def _reference_sha256(
    config: ProjectConfig, preparation: character_notes.ReferenceFactsPreparation
) -> dict[Path, str | None]:
    """The two reference stores' prepared after-hashes, keyed by their paths."""

    mapping: dict[Path, str | None] = {}
    for label, path in zip(
        character_notes.REFERENCE_FILE_LABELS,
        (config.kanji_file.resolve(), config.jpdb_readings_file.resolve()),
        strict=True,
    ):
        proposed = preparation.file(label)
        if proposed is None:
            raise StudyFinishError(
                f"The prepared reference facts name no {label}; nothing was planned."
            )
        mapping[path] = proposed.after_sha256
    return mapping


def _assert_census_covered(
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
    slots: Sequence[audio_application.ExpectedAudioSlot],
) -> None:
    """Every expected slot has exactly one planned clip, counted independently.

    Derived from the records' own fields rather than from ``plan.clips``, so an
    empty request list cannot pass merely because every selected clip is
    already current. Several slots legitimately share one clip; each of them
    still has to reach it.
    """

    by_id = {record.id: record for record in records}
    for slot in slots:
        record = by_id[slot.record_id]
        if slot.kind == "word":
            prefix = f"janki-{ledger.word_audio_filename_fingerprint(record)}"
        else:
            example = record.examples[slot.position or 0]
            prefix = (
                f"janki-{ledger.example_audio_filename_fingerprint(record, example)}"
            )
        matches = [
            clip
            for clip in plan.clips
            if clip.record_id == slot.record_id
            and clip.kind == slot.kind
            and clip.target == f"{prefix}{clip.provider.suffix}"
        ]
        if len(matches) != 1:
            raise StudyFinishError(
                f"{record.id} slot {slot.kind}/{slot.position} is covered by "
                f"{len(matches)} planned clips; every expected slot needs exactly "
                "one, and a word clip never satisfies a sentence slot."
            )


def _paid_call_counts(plan: audio_application.AudioPlan) -> tuple[int, int]:
    words = sum(
        1
        for clip in plan.clips
        if clip.kind == "word"
        and clip.state == "provider-required"
        and clip.provider.access == "paid-network"
    )
    examples = sum(
        1
        for clip in plan.clips
        if clip.kind == "example"
        and clip.state == "provider-required"
        and clip.provider.access == "paid-network"
    )
    return words, examples


def _default_client_factory() -> enrich.DictionaryLookup:
    return jpdb.JpdbClient(jpdb.api_key_from_env())


def plan_study_finish(
    config: ProjectConfig,
    job_id: str,
    *,
    client_factory: Callable[[], enrich.DictionaryLookup] | None = None,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
    now: date | None = None,
) -> StudyFinishPlan:
    """Project this whole job's finish, and write nothing.

    Every dictionary fact this job will ever use is fetched here, once, through
    a recording client, and frozen into the authority. The projection, the
    preview and every later apply replay that book; **no phase refetches after
    review**, so no dictionary refresh can change what the owner approved.
    """

    job = study_job.load_study_job(config, job_id)
    deck_path = study_job.job_destination_deck(config, job)
    parts = _job_parts(config, job_id)
    day = now or date.today()
    approved_at = day.isoformat()

    rendering = _rendering_fingerprint(config, job_id)
    recording = enrich.RecordingDictionaryClient(
        (client_factory or _default_client_factory)()
    )

    requests: list[ReviewBatchRequest] = []
    chosen: dict[str, dict[str, Any]] = {}
    for part_name, staging_path in parts:
        staging_sha256 = _staging_revision(staging_path)
        flags = _bound_choice(
            job,
            "review_flags",
            part_name,
            staging_sha256=staging_sha256,
            rendering_fingerprint=rendering,
        )
        record_ids = tuple(str(item) for item in (flags or {}).get("record_ids") or ())
        # §9.1's standalone `review_patterns` choice is the one fact this mark
        # has, read and revalidated on its own: a stale pattern decision refuses
        # here even when the owner saved no review selection for the part, and no
        # saved review selection carries a second copy to disagree with. An
        # absent choice is no mark — §7.2 writes to the pattern store only where
        # the owner selected one — and is never inferred from anything else.
        pattern_choice = _bound_choice(
            job,
            "review_patterns",
            part_name,
            staging_sha256=staging_sha256,
            rendering_fingerprint=rendering,
        )
        review_patterns = bool((pattern_choice or {}).get("value"))
        reason_entry = _bound_choice(
            job,
            "coverage_reasons",
            part_name,
            staging_sha256=staging_sha256,
            rendering_fingerprint=rendering,
        )
        reason = str((reason_entry or {}).get("reason") or "")
        approval = _coverage_approval(
            config, part_name, staging_path, reason, approved_at=approved_at
        )
        chosen[part_name] = {
            "staging_path": staging_path,
            "staging_sha256": staging_sha256,
            "record_ids": record_ids,
            "review_patterns": review_patterns,
            "coverage_reason": reason,
            "coverage_approval": approval,
            "coverage_approved_at": approved_at if approval is not None else None,
            "disposition": _disposition_of(
                job,
                part_name,
                staging_sha256=staging_sha256,
                rendering_fingerprint=rendering,
            ),
        }
        requests.append(
            ReviewBatchRequest(
                part_name=part_name,
                staging_path=staging_path,
                record_ids=record_ids,
                review_patterns=review_patterns,
                coverage_approval=approval,
                expected_revision=staging_sha256,
            )
        )

    review_batch = assistant_staging_review.prepare_staging_review_batch(
        config, requests
    )
    store_text = review_batch.batch.patterns.after_text
    if store_text is None:
        store_text = (
            "{}"
            if review_batch.batch.patterns.expected_after is None
            else _read_text_or_empty(Path(review_batch.batch.patterns.path))
        )
    pattern_store = patterns.load_store_text(
        store_text, source="<prepared pattern store>"
    )

    collection, collection_revision = load_records_snapshot(
        config.normalized_file.resolve()
    )
    fold = promotion.fold_source_extraction_promotions(
        config,
        [
            promotion.SourcePromotionPart(
                part_name=name,
                staging_path=values["staging_path"],
                expected_revision=values["staging_sha256"],
                review_record_ids=values["record_ids"],
                review_patterns=values["review_patterns"],
                coverage_approval=values["coverage_approval"],
            )
            for name, values in chosen.items()
        ],
        collection=collection,
        collection_revision=collection_revision,
        pattern_store=pattern_store,
        witness=recording,
    )

    finish_parts = tuple(
        StudyFinishPart(
            part_name=projection.part_name,
            staging_path=projection.staging_path,
            staging_sha256=projection.staging_sha256,
            review_record_ids=chosen[projection.part_name]["record_ids"],
            review_patterns=chosen[projection.part_name]["review_patterns"],
            coverage_reason=chosen[projection.part_name]["coverage_reason"],
            coverage_approved_at=chosen[projection.part_name]["coverage_approved_at"],
            projected_state=projection.state,
            gate=projection.gate,
            expected_before=projection.expected_before,
            expected_after=projection.expected_after,
            landed_ids=projection.landed_ids,
            held_ids=projection.held_ids,
            excluded_ids=projection.excluded_ids,
            archive_retry_ids=projection.archive_retry_ids,
            disposition=chosen[projection.part_name]["disposition"],
        )
        for projection in fold.parts
    )

    # A blocked part's disclosed ids are what makes its refusal readable, not
    # cards this finish may claim: §7.6 writes no canonical byte for it and §7.7
    # keeps it a blocker rather than a disposition. Planning audio and a package
    # over them would be planning over cards that do not exist.
    accepted: list[str] = []
    for part in finish_parts:
        if part.projected_state == "blocked":
            continue
        for record_id in (*part.landed_ids, *part.archive_retry_ids):
            if record_id not in accepted:
                accepted.append(record_id)
    if not accepted:
        blocked = [
            f"{part.part_name} (blocked at the {part.gate} gate)"
            for part in finish_parts
            if part.projected_state == "blocked"
        ]
        raise StudyFinishError(
            "This finish would land no card at all."
            + (" " + ", ".join(blocked) + "." if blocked else "")
            + " A job completes over the cards it accepted; clear what is "
            "blocking these parts, or record a disposition for every part that "
            "lands nothing, and plan again."
        )

    canonical_path = config.normalized_file.resolve()
    after_text = fold.canonical_text_after
    if after_text is None:
        raise StudyFinishError(
            "This finish projects no canonical collection to enrich; nothing "
            "was planned."
        )
    after_revision = RecordsRevision(canonical_path, after_text)

    characters: list[str] = []
    for record in fold.records_after:
        for character in kanji.kanji_in(record.expression):
            if character not in characters:
                characters.append(character)
    reference = character_notes.prepare_reference_facts(config, characters)
    kanji_store = kanji.parse_store(
        _proposed_text(reference, character_notes.REFERENCE_FILE_LABELS[0]),
        source="<prepared kanji reference store>",
    )

    enrichment_decision = enrichment_application.plan_dictionary_enrichment_revision(
        config,
        recording,
        fold.records_after,
        after_revision,
        accepted,
        kanji_store=kanji_store,
    )
    book = recording.freeze()
    enriched_records = tuple(enrichment_decision.result.records)
    enriched_text = records_json_text(enriched_records)
    enriched_revision = RecordsRevision(canonical_path, enriched_text)

    audio_choice = job.choices.get("include_example_audio")
    include_examples = True
    if isinstance(audio_choice, Mapping) and "value" in audio_choice:
        include_examples = bool(audio_choice.get("value"))
    audio_plan = audio_application.plan_targeted_audio_revision(
        config,
        enriched_records,
        enriched_revision,
        accepted,
        words=True,
        examples=include_examples,
        force=False,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    slots = audio_application.expected_audio_slots(
        enriched_records, tuple(accepted), words=True, examples=include_examples
    )
    _assert_census_covered(audio_plan, enriched_records, slots)
    audio_application.preflight_paid_deck_audio_plan(
        audio_plan,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    confirmed = audio_application.confirm_audio_plan(config, audio_plan)
    paid_words, paid_examples = _paid_call_counts(audio_plan)

    post_audio = _projected_audio_references(config, audio_plan, enriched_records)
    post_audio_text = records_json_text(post_audio)
    media_sha256 = _projected_package_media(config, audio_plan, post_audio, deck_path)
    reference_sha256 = _reference_sha256(config, reference)
    projection = deck_package.plan_vocabulary_deck_package_revision(
        config,
        deck_path,
        post_audio,
        RecordsRevision(canonical_path, post_audio_text),
        media_sha256=media_sha256,
        reference_sha256=reference_sha256,
    )

    audio_facts = StudyFinishAudioPlan(
        include_example_audio=include_examples,
        omitted_by_owner=not include_examples,
        record_ids=tuple(accepted),
        word_provider=audio_application.provider_plan_wire(audio_plan.word_provider),
        example_provider=audio_application.provider_plan_wire(
            audio_plan.example_provider
        ),
        paid_word_calls=paid_words,
        paid_example_calls=paid_examples,
        expected_word_slots=sum(1 for slot in slots if slot.kind == "word"),
        expected_example_slots=sum(1 for slot in slots if slot.kind == "example"),
        word_counts=audio_plan.word_counts,
        example_counts=audio_plan.example_counts,
    )

    authority = {
        "version": 1,
        "job": {
            "job_id": job_id,
            "parent_source_name": job.header.parent_source_name,
            "parent_sha256": job.header.parent_sha256,
            "rendering_fingerprint": rendering,
        },
        "deck": {
            "path": _relative(config, deck_path, label="deck"),
            "sha256": _sha(read_bytes_bound(deck_path)),
            "name": projection.deck_name,
            "card_types": list(projection.card_types),
        },
        "parts": [
            {
                "part_name": part.part_name,
                "staging_path": _relative(
                    config, part.staging_path, label="staged review"
                ),
                "staging_sha256": part.staging_sha256,
                "review_record_ids": list(part.review_record_ids),
                "review_patterns": part.review_patterns,
                "coverage_reason": part.coverage_reason,
                "coverage_approved_at": part.coverage_approved_at,
                "projected_state": part.projected_state,
                "gate": part.gate,
                "expected_before": part.expected_before,
                "expected_after": part.expected_after,
                "landed_ids": list(part.landed_ids),
                "held_ids": list(part.held_ids),
                "excluded_ids": list(part.excluded_ids),
                "archive_retry_ids": list(part.archive_retry_ids),
                "disposition": None
                if part.disposition is None
                else _plain(part.disposition),
            }
            for part in finish_parts
        ],
        "review": review_batch.to_dict(),
        "promotion": {
            "canonical_digests": list(fold.canonical_digests),
            "record_ids": list(accepted),
            "ledger_dates": {"day": approved_at},
        },
        "reference": reference.to_dict(),
        "enrichment": {
            "record_ids": list(enrichment_decision.record_ids),
            "force_fields": list(enrichment_decision.force_fields),
            "enriched_at": approved_at,
            "decision_fingerprint": enrichment_decision.fingerprint,
            "bound_values": _bound_enrichment_values(enrichment_decision),
            "cleared_fields": {
                record_id: sorted(fields)
                for record_id, fields in enrichment_decision.result.cleared.items()
            },
            "projected_input_sha256": _sha(after_text.encode("utf-8")),
            "projected_output_sha256": _sha(enriched_text.encode("utf-8")),
        },
        "dictionary": {
            "fingerprint": book.fingerprint,
            "book": book.to_dict(),
        },
        "audio": {
            "include_example_audio": include_examples,
            "record_ids": list(accepted),
            "expected_slots": [
                {
                    "record_id": slot.record_id,
                    "kind": slot.kind,
                    "position": slot.position,
                }
                for slot in slots
            ],
            "confirmed_plan": confirmed.to_wire(),
            "counts": {
                "word": _counts_wire(audio_plan.word_counts),
                "example": _counts_wire(audio_plan.example_counts),
                "paid_word_calls": paid_words,
                "paid_example_calls": paid_examples,
            },
        },
        # The whole bound projection, as `deck_package` itself serializes it.
        # Not a fingerprint and a list of inputs this module would plan from
        # again later: re-deriving a projection is planning a second time, and
        # the second plan is a second authority. `plan_from_wire` restores this
        # exact value for B's realization comparator.
        "package": {"projection": deck_package.plan_wire(config, projection)},
        "preview": {
            "rendering_fingerprint": rendering,
            "content_fingerprint": _sha(
                _canonical(
                    {
                        "records": [
                            _approved_content(record) for record in post_audio
                        ],
                        "card_types": list(projection.card_types),
                    }
                ).encode("utf-8")
            ),
        },
        "configuration_fingerprint": _configuration_fingerprint(config),
    }
    fingerprint = _sha(_canonical(authority).encode("utf-8"))
    directory = study_finish_directory(config)
    return StudyFinishPlan(
        repository_root=config.root.resolve(),
        job_id=job_id,
        deck_path=deck_path,
        output_path=projection.output_path,
        deck_name=projection.deck_name,
        card_types=projection.card_types,
        parts=finish_parts,
        canonical_digests=fold.canonical_digests,
        record_ids=tuple(accepted),
        reference=reference,
        enrichment_changed_ids=tuple(sorted(enrichment_decision.result.changes)),
        missing_reference_facts=reference.missing,
        audio=audio_facts,
        note_count=projection.note_count,
        card_count=projection.card_count,
        package_record_ids=projection.record_ids,
        warnings=tuple(enrichment_decision.result.warnings),
        finish_directory=directory,
        record_path=directory / f"study-finish-{fingerprint}.json",
        authority=authority,
        fingerprint=fingerprint,
    )


def _read_text_or_empty(path: Path) -> str:
    try:
        return read_bytes_bound(path).decode("utf-8")
    except (FileNotFoundError, DataError, OSError, UnicodeDecodeError):
        return "{}"


def _proposed_text(
    preparation: character_notes.ReferenceFactsPreparation, label: str
) -> str | None:
    proposed = preparation.file(label)
    if proposed is None:
        raise StudyFinishError(
            f"The prepared reference facts name no {label}; nothing was planned."
        )
    if proposed.after_text is not None:
        return proposed.after_text
    if proposed.before_sha256 is None:
        return None
    return _read_text_or_empty(proposed.path)


def _counts_wire(counts: audio_application.AudioClipCounts) -> dict[str, int]:
    return {
        "total": counts.total,
        "current": counts.current,
        "recoverable": counts.recoverable,
        "provider_required": counts.provider_required,
    }


def _bound_enrichment_values(
    decision: enrichment_application.DictionaryEnrichmentDecision,
) -> dict[str, dict[str, Any]]:
    """Record id → field → the exact new value this pass is authorized to write."""

    return {
        record_id: {field: change[1] for field, change in fields.items()}
        for record_id, fields in decision.result.changes.items()
    }


def _approved_content(record: VocabularyRecord) -> dict[str, Any]:
    """Every visible field except audio, which the finish is allowed to change."""

    payload = record.to_dict()
    payload.pop("audio", None)
    for example in payload.get("examples") or ():
        if isinstance(example, dict):
            example.pop("audio", None)
    return payload


def _configuration_fingerprint(config: ProjectConfig) -> str:
    root = config.root.resolve()
    return _sha(
        _canonical(
            {
                "root": str(root),
                "normalized_file": _relative(
                    config, config.normalized_file, label="collection"
                ),
                "ledger_file": _relative(config, config.ledger_file, label="ledger"),
                "deck_dir": _relative(config, config.deck_dir, label="deck directory"),
                "dist_dir": _relative(config, config.dist_dir, label="package directory"),
                "media_dir": _relative(config, config.media_dir, label="media directory"),
                "staging_dir": _relative(
                    config, config.staging_dir, label="staging directory"
                ),
            }
        ).encode("utf-8")
    )


# --- the durable record -------------------------------------------------------


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise StudyFinishError(f"Study finish JSON repeats key {key!r}.")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise StudyFinishError(f"Study finish JSON contains non-finite number {value}.")


def _record_text(record: Mapping[str, Any]) -> str:
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def _strict_record(wire: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            wire.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except StudyFinishError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StudyFinishError(f"Could not parse study finish {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise StudyFinishError("Study finish has invalid top-level fields.")
    receipt_id = value.get("receipt_id")
    authority = value.get("authority")
    if (
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("kind") != _KIND
        or not _is_sha(receipt_id)
        or path.name != f"study-finish-{receipt_id}.json"
        or not isinstance(authority, Mapping)
        or set(authority) != _AUTHORITY_KEYS
        or authority.get("version") != 1
        or _sha(_canonical(authority).encode("utf-8")) != receipt_id
    ):
        raise StudyFinishError("Study finish identity or authority is corrupt.")
    state = value.get("state")
    if state not in _STATE_INDEX:
        raise StudyFinishError("Study finish state is invalid.")
    for key in ("authorized_at", "updated_at"):
        timestamp = value.get(key)
        if not isinstance(timestamp, str):
            raise StudyFinishError(f"Study finish {key} is malformed.")
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise StudyFinishError(f"Study finish {key} is malformed.") from exc
    required = _STATE_INDEX[state]
    for index, key in enumerate(_RECEIPT_KEYS):
        present = isinstance(value.get(key), Mapping)
        if (index < required) != present:
            raise StudyFinishError(
                "Study finish receipts do not match its durable state."
            )
    if not isinstance(value.get("promotion_intents"), Mapping):
        raise StudyFinishError("Study finish promotion intents are malformed.")
    for key in ("package_intents", "paid_clip_reservations"):
        if not isinstance(value.get(key), list):
            raise StudyFinishError(f"Study finish {key} are malformed.")
    if not isinstance(value.get("enrichment_intent"), Mapping | None):
        raise StudyFinishError("Study finish enrichment intent is malformed.")
    return value


def _read_record(path: Path) -> tuple[dict[str, Any], str]:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError as exc:
        raise StudyFinishError(f"Study finish no longer exists: {path}") from exc
    except (DataError, OSError) as exc:
        raise StudyFinishError(
            f"Could not safely read study finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _read_record_optional(path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise StudyFinishError(
            f"Could not safely read study finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _new_record(plan: StudyFinishPlan) -> dict[str, Any]:
    stamp = datetime.now(UTC).isoformat()
    record: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": _KIND,
        "receipt_id": plan.fingerprint,
        "state": "authorized",
        "authority": _plain(plan.authority),
        "authorized_at": stamp,
        "updated_at": stamp,
        "promotion_intents": {},
        "enrichment_intent": None,
        "package_intents": [],
        "paid_clip_reservations": [],
    }
    for key in _RECEIPT_KEYS:
        record[key] = None
    return record


def _write_new(path: Path, record: Mapping[str, Any]) -> str:
    text = _record_text(record)
    _strict_record(text.encode("utf-8"), path)
    try:
        atomic_write_text_bound(path, text, expected_absent=True)
    except (DataError, OSError) as exc:
        raise StudyFinishError(
            f"Could not record the finish authority before writing anything: {exc}"
        ) from exc
    return _sha(text.encode("utf-8"))


def _assert_intents_append_only(
    record: Mapping[str, Any], updates: Mapping[str, Any]
) -> None:
    """An intent is never rewritten once it is present.

    A prepared intent is what a recovery replays, so changing one is changing
    the evidence. A per-part promotion intent may be added and a superseding
    package preparation appended; neither may replace what is already there.
    """

    if "promotion_intents" in updates:
        held = record.get("promotion_intents")
        fresh = updates["promotion_intents"]
        assert isinstance(held, Mapping)
        if not isinstance(fresh, Mapping) or any(
            key not in fresh or fresh[key] != value for key, value in held.items()
        ):
            raise StudyFinishError(
                "A recorded promotion intent is never rewritten; a finish that "
                "needs a different one is a different authority."
            )
    if (
        "enrichment_intent" in updates
        and record.get("enrichment_intent") is not None
        and updates["enrichment_intent"] != record.get("enrichment_intent")
    ):
        raise StudyFinishError(
            "This finish already recorded its dictionary enrichment intent; an "
            "intent is never rewritten."
        )
    if "package_intents" in updates:
        held = record.get("package_intents")
        fresh = updates["package_intents"]
        assert isinstance(held, list)
        if (
            not isinstance(fresh, list)
            or len(fresh) < len(held)
            or fresh[: len(held)] != held
        ):
            raise StudyFinishError(
                "A recorded package preparation is never rewritten; a rebuild "
                "appends a new superseding attempt instead."
            )


def _replace_record(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    **updates: Any,
) -> tuple[dict[str, Any], str]:
    """One CAS write of a working slot, without advancing the phase."""

    if any(key not in _RECORD_KEYS for key in updates):
        raise StudyFinishError("A study finish records no such field.")
    if "state" in updates or "authority" in updates:
        raise StudyFinishError(
            "The finish authority and its phase are never edited in place."
        )
    _assert_intents_append_only(record, updates)
    updated = dict(record)
    updated.update({key: _plain(value) for key, value in updates.items()})
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current != dict(record) or current_revision != revision:
            raise StudyFinishError(
                "The study finish record changed while an intent was recorded."
            )
        try:
            atomic_write_text_bound(path, text, expected_revision=revision)
        except (DataError, OSError) as exc:
            raise StudyFinishError(
                f"Could not record a study finish intent: {exc}"
            ) from exc
    return updated, _sha(text.encode("utf-8"))


def _advance(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    *,
    to_state: StudyFinishState,
    receipt_key: str,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if _STATE_INDEX[to_state] != _STATE_INDEX[str(record.get("state"))] + 1:
        raise StudyFinishError(
            f"A study finish cannot advance from {record.get('state')!r} to "
            f"{to_state!r}."
        )
    updated = dict(record)
    updated["state"] = to_state
    updated[receipt_key] = _plain(receipt)
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current != dict(record) or current_revision != revision:
            raise StudyFinishError(
                "The study finish record changed while a completed phase was "
                "recorded."
            )
        try:
            atomic_write_text_bound(path, text, expected_revision=revision)
        except (DataError, OSError) as exc:
            raise StudyFinishError(
                f"Could not record a completed study finish phase: {exc}"
            ) from exc
    return updated, _sha(text.encode("utf-8"))


def _authority_section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    authority = record.get("authority")
    section = authority.get(key) if isinstance(authority, Mapping) else None
    if not isinstance(section, Mapping):
        raise StudyFinishError(f"Study finish {key} authority is malformed.")
    return section


def _authority_output_path(config: ProjectConfig, record: Mapping[str, Any]) -> Path:
    """Where the bound projection publishes, read off the projection itself.

    Not a second copy of the path in the authority: the wire is the projection,
    so asking it is asking the one bound value.
    """

    wire = _authority_section(record, "package").get("projection")
    if not isinstance(wire, Mapping):
        raise StudyFinishError("This finish records no bound package projection.")
    return _authority_path(config, wire.get("output_path"), label="package")


def _authority_parts(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    authority = record.get("authority")
    parts = authority.get("parts") if isinstance(authority, Mapping) else None
    if not isinstance(parts, list) or not parts or any(
        not isinstance(item, Mapping) for item in parts
    ):
        raise StudyFinishError("Study finish part authority is malformed.")
    return tuple(parts)


# --- the result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyFinishResult:
    """Truthful durable state after an initial or resumed finish execution."""

    receipt_id: str
    state: StudyFinishState
    record_path: Path
    job_id: str
    deck_path: Path
    output_path: Path
    deck_name: str = ""
    record_ids: tuple[str, ...] = ()
    landed_ids: tuple[str, ...] = ()
    held_ids: tuple[str, ...] = ()
    note_count: int | None = None
    card_count: int | None = None
    media_count: int | None = None
    package_sha256: str | None = None
    preview_sha256: str | None = None
    outstanding: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.state == "complete" and not self.outstanding


def _result(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
    *,
    outstanding: Sequence[str] = (),
) -> StudyFinishResult:
    job = _authority_section(record, "job")
    deck = _authority_section(record, "deck")
    promoted = _authority_section(record, "promotion")
    parts = _authority_parts(record)
    package_receipt = record.get("package_receipt")
    counts: Mapping[str, Any] = (
        package_receipt if isinstance(package_receipt, Mapping) else {}
    )
    preview_receipt = record.get("preview_receipt")
    promotion_receipt = record.get("promotion_receipt")
    landed: tuple[str, ...] = ()
    if isinstance(promotion_receipt, Mapping):
        landed = tuple(str(item) for item in promotion_receipt.get("landed_ids") or ())
    held: list[str] = []
    for part in parts:
        held.extend(str(item) for item in part.get("held_ids") or ())
    return StudyFinishResult(
        receipt_id=str(record["receipt_id"]),
        state=str(record["state"]),  # type: ignore[arg-type]
        record_path=path,
        job_id=str(job.get("job_id") or ""),
        deck_path=_authority_path(config, deck.get("path"), label="deck"),
        output_path=_authority_output_path(config, record),
        deck_name=str(deck.get("name") or ""),
        record_ids=tuple(str(item) for item in promoted.get("record_ids") or ()),
        landed_ids=landed,
        held_ids=tuple(held),
        note_count=(
            int(counts["note_count"]) if isinstance(counts.get("note_count"), int) else None
        ),
        card_count=(
            int(counts["card_count"]) if isinstance(counts.get("card_count"), int) else None
        ),
        media_count=(
            int(counts["media_count"])
            if isinstance(counts.get("media_count"), int)
            else None
        ),
        package_sha256=(
            str(counts["package_sha256"])
            if isinstance(counts.get("package_sha256"), str)
            else None
        ),
        preview_sha256=(
            str(preview_receipt["preview_sha256"])
            if isinstance(preview_receipt, Mapping)
            and isinstance(preview_receipt.get("preview_sha256"), str)
            else None
        ),
        outstanding=tuple(outstanding),
        warnings=tuple(str(item) for item in counts.get("warnings") or ()),
    )


# --- phase 1: reviewed --------------------------------------------------------


def _review_batch(
    record: Mapping[str, Any],
) -> assistant_staging_review.PreparedStagingReviewBatch:
    return assistant_staging_review.PreparedStagingReviewBatch.from_dict(
        _authority_section(record, "review")
    )


def _apply_reviewed(
    config: ProjectConfig, record: Mapping[str, Any], *, resumed: bool
) -> dict[str, Any]:
    """Write the one aggregate review payload the authority already binds.

    The intent is already durable: it is part of the immutable authority, which
    `_write_new` published before the first effect. A resume finishes only the
    writes this vector still owes, so a crash between the staging replace and
    the pattern-store replace costs nothing.
    """

    prepared = _review_batch(record)
    if resumed:
        outcome = assistant_staging_review.recover_prepared_staging_review_batch_under_guard(
            config, prepared
        )
    else:
        outcome = assistant_staging_review.apply_prepared_staging_review_batch_under_guard(
            config, prepared
        )
    return {
        "fingerprint": prepared.fingerprint,
        "patterns_path": str(prepared.batch.patterns.path),
        "patterns_sha256": prepared.batch.patterns.expected_after,
        "parts": {
            name: {
                "record_ids": list(part.accepted_record_ids),
                "pattern_reviewed": part.pattern_reviewed,
            }
            for name, part in outcome.parts.items()
        },
    }


# --- phase 2: promoted --------------------------------------------------------


def _assert_part_matches(
    part: Mapping[str, Any], prepared: promotion.PreparedSourcePromotion
) -> None:
    """The freshly decided intent must be the projection the owner approved.

    Every disclosed id set, the state and the gate's canonical checkpoints are
    compared; a difference names both sides rather than being resolved. The
    equality is an identity by construction — the fold and the intent use one
    ``_disclosed_ids`` — so a mismatch really is a repository that moved.
    """

    expected = (
        str(part.get("projected_state") or ""),
        tuple(str(item) for item in part.get("landed_ids") or ()),
        tuple(str(item) for item in part.get("held_ids") or ()),
        tuple(str(item) for item in part.get("excluded_ids") or ()),
        tuple(str(item) for item in part.get("archive_retry_ids") or ()),
    )
    found = (
        prepared.projected_state,
        prepared.landed_ids,
        prepared.held_ids,
        prepared.excluded_ids,
        prepared.archive_retry_ids,
    )
    if expected != found:
        raise StudyFinishError(
            f"Part {part.get('part_name')!r} now promotes {found} where this "
            f"finish was authorized for {expected}. Nothing was written; plan "
            "again over the current files."
        )
    canonical = [
        component
        for component in prepared.components
        if component.role == "canonical"
    ]
    projected = (part.get("expected_before"), part.get("expected_after"))
    if prepared.projected_state in _CANONICAL_STATES:
        # `lands` writes the collection and `nothing_lands` binds it unwritten;
        # either way the gate's own checkpoints are this part's pair.
        if len(canonical) != 1:
            raise StudyFinishError(
                f"Part {part.get('part_name')!r} binds {len(canonical)} canonical "
                "components where its state binds exactly one."
            )
        bound = (canonical[0].expected_before, canonical[0].expected_after)
        if bound != projected:
            raise StudyFinishError(
                f"Part {part.get('part_name')!r} would write canonical {bound} "
                f"where this finish projected {projected}. Nothing was written."
            )
        return
    # `nothing`, `pattern_only` and `archive_retry` open no collection at all
    # (§7.6), so an intent that bound one would write a file its own state never
    # looks at — and the fold carries the checkpoint past them unchanged.
    if canonical:
        raise StudyFinishError(
            f"Part {part.get('part_name')!r} would write the collection for a "
            f"{prepared.projected_state!r} promotion, which writes none. Nothing "
            "was written."
        )
    if projected[0] != projected[1]:
        raise StudyFinishError(
            f"Part {part.get('part_name')!r} projected canonical {projected} for a "
            f"{prepared.projected_state!r} promotion, which changes nothing. "
            "Nothing was written."
        )


def _accepted_ids(part: Mapping[str, Any]) -> tuple[str, ...]:
    """What one part's authority accepted: newly landed, then already archived.

    An archive retry accepts cards it never landed itself, so the two sets are
    read together and deduped in the authority's own order.
    """

    accepted: list[str] = []
    for record_id in (
        *(str(item) for item in part.get("landed_ids") or ()),
        *(str(item) for item in part.get("archive_retry_ids") or ()),
    ):
        if record_id not in accepted:
            accepted.append(record_id)
    return tuple(accepted)


def _bound_source_file(meta: Mapping[str, Any] | None) -> str | None:
    """The source identity every receipt in this part's archive carries, if bound.

    Read only from metadata this part's own promotion already bound — never from
    the archive, which stays the owning reader's to validate. Two of that
    writer's invariants make a *declared* identity exact rather than a guess:
    ``promotion.promotion_batches`` refuses an archive whose declared
    ``source_file`` disagrees with any of its batches, and
    ``promotion.validate_record_archive`` refuses a same-run archive whose
    metadata differs from the live review's. So when the live review names a
    source, every batch in that archive — this pass's and every earlier one a
    retry needs — reports that same name, and the owning discovery re-proves it
    on every read.

    ``None`` is absence, not a filter: a review that declared no ``source_file``
    constrains no prior batch, so nothing may be skipped on its behalf.
    """

    if not isinstance(meta, Mapping):
        return None
    named = meta.get("source_file")
    if not isinstance(named, str) or not named.strip():
        return None
    return named.strip()


def _receipt_candidates(
    config: ProjectConfig, primary: str | None, source_file: str | None
) -> Iterator[str]:
    """This part's own receipt first, then every archive receipt of its source.

    The listing is one validated scan of the whole done namespace; *resolving*
    one of its receipts is a second scan plus a canonical read plus a
    deck-ownership re-proof under the deck lock (`finish.resolve_finish_scope`).
    ``source_file`` keeps that second cost off receipts this part's own archive
    cannot hold, and decides nothing: the archive check in
    :func:`_part_selections` remains the authority for every candidate kept.
    """

    if primary:
        yield primary
    for receipt in finish_application.list_finish_receipts(config):
        if source_file is not None and receipt.source_file != source_file:
            continue
        yield receipt.receipt_id


def _part_selections(
    config: ProjectConfig,
    *,
    part_name: str,
    accepted: Sequence[str],
    primary: str | None,
    archive_path: Path | None,
    source_file: str | None,
) -> tuple[StudyFinishSelection, ...]:
    """Which owning receipts account for this part's accepted ids, and for which.

    Resolved through the archive reader that owns them — never minted here, and
    never inferred from a writer's return value alone. The writer's own receipt
    is asked first, because it is the batch this pass wrote; a retry's ids
    legitimately belong to earlier batches of the **same** archive, which is the
    only place this part's receipts may come from.

    A receipt that holds more than this part accepted is fine and is recorded as
    such: the ids outside the selection are somebody's promoted rows, not this
    job's cards.

    ``source_file`` is the identity :func:`_bound_source_file` took from this
    part's own promotion. It narrows which receipts are worth a full resolve and
    proves nothing by itself — a kept candidate still has to resolve to this
    part's own archive and really promote every id taken from it.
    """

    if not accepted:
        return ()
    if archive_path is None:
        raise StudyFinishError(
            f"Part {part_name!r} accepted {len(accepted)} card(s) but retains no "
            "archive for them, so nothing receipts them. The job stays outstanding."
        )
    relative = _relative(config, archive_path, label="archive")
    remaining = list(accepted)
    bound: list[StudyFinishSelection] = []
    unreadable: list[str] = []
    seen: set[str] = set()
    candidates = _receipt_candidates(config, primary, source_file)
    while remaining:
        # Lazily, and that buys exactly one thing: a part its own receipt already
        # covers pays for no second full archive scan and no ownership re-proof of
        # somebody else's batch. It does **not** make an unrelated invalid archive
        # harmless — the discovery listing validates the whole done namespace, so
        # such an archive refuses here, and the refusal belongs to this part's
        # source rather than escaping the fold as the reader's own error.
        try:
            receipt_id = next(candidates)
        except StopIteration:
            break
        except JankiError as exc:
            unreadable.append(f"the done archive listing: {exc}")
            break
        if receipt_id in seen:
            continue
        seen.add(receipt_id)
        try:
            member = finish_application.resolve_finish_scope(config, receipt_id)
        except JankiError as exc:
            # One receipt's own problem — a record missing from the collection, an
            # owner deck that changed — is not this part's refusal unless this
            # part actually needed that receipt, which the coverage check below is
            # what decides. An invalid archive is not in that class: the owning
            # reader refuses it for every receipt, this part's included.
            unreadable.append(f"{receipt_id}: {exc}")
            continue
        if member.archive_path.resolve() != archive_path.resolve():
            continue
        selected = tuple(
            record_id for record_id in remaining if record_id in member.record_ids
        )
        if not selected:
            continue
        bound.append(
            StudyFinishSelection(
                part_name=part_name,
                receipt_id=receipt_id,
                archive_path=relative,
                record_ids=selected,
            )
        )
        remaining = [record_id for record_id in remaining if record_id not in selected]
    if remaining:
        raise StudyFinishError(
            f"Part {part_name!r} accepted "
            + ", ".join(remaining)
            + f", and no promotion receipt in {archive_path.name} accounts for "
            "them"
            + ("; " + "; ".join(unreadable) if unreadable else "")
            + ". The job stays outstanding."
        )
    return tuple(bound)


def _apply_promoted(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    replay: enrich.DictionaryLookup,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Promote every part in fold order, each through its own durable intent.

    A part whose intent is already recorded is **finished from that intent**,
    never re-decided: a fresh decision reads this finish's own writes as
    somebody else's. A genuinely unstarted part is decided again from disk with
    the replay witness and the frozen day, and the fresh outcome must equal the
    projection the owner approved.
    """

    day = date.fromisoformat(str(_authority_section(record, "promotion").get(
        "ledger_dates", {}
    ).get("day")))
    parts = _authority_parts(record)
    receipts: list[dict[str, Any]] = []
    selections: list[StudyFinishSelection] = []
    landed: list[str] = []
    outstanding: list[str] = []
    for part in parts:
        name = str(part.get("part_name") or "")
        disposition = part.get("disposition")
        if str(part.get("projected_state") or "") == "blocked":
            if not isinstance(disposition, Mapping):
                raise StudyFinishError(
                    f"Part {name!r} is blocked at the {part.get('gate')!r} gate "
                    "and no owner disposition covers it."
                )
            receipts.append(
                {
                    "part_name": name,
                    "state": "disposed",
                    "receipt_id": None,
                    "archive_path": None,
                    "landed_ids": [],
                    "receipts": [],
                    "disposition": _plain(disposition),
                }
            )
            continue
        staging_path = _authority_path(
            config, part.get("staging_path"), label="staged review"
        )
        held = record.get("promotion_intents")
        assert isinstance(held, Mapping)
        recorded = held.get(name)
        if isinstance(recorded, Mapping):
            prepared = promotion.PreparedSourcePromotion.from_dict(recorded)
            recovery = promotion.recover_promotion_intent_under_guard(
                config, prepared, witness=replay
            )
            state = recovery.state
            receipt_id = recovery.receipt_id
            archive_path = recovery.archive_path
            part_landed = recovery.landed_ids
            # The intent's own archive payload, which a landing binds and the
            # states that write no archive do not. Absence filters nothing.
            source_file = _bound_source_file(prepared.archive_meta)
        else:
            decision = promotion.decide_promotion(
                config, staging_path, source=name, client=replay
            )
            prepared = promotion.prepare_source_promotion(
                config, decision, part_name=name, now=day
            )
            _assert_part_matches(part, prepared)
            record, revision = _replace_record(
                path,
                record,
                revision,
                promotion_intents={**dict(held), name: prepared.to_dict()},
            )
            executed = promotion.apply_prepared_source_promotion(
                config, prepared, decision=decision, witness=replay
            )
            state = executed.state
            receipt_id = executed.receipt_id
            archive_path = executed.archive_path
            part_landed = executed.promoted_ids
            # The live review the decision was taken under, read at the same seam
            # the promotion writer reads its own receipt's source from.
            source_file = _bound_source_file(decision.meta)
        if state in {"landed_ledger_incomplete", "landed_ai_ledger_incomplete"}:
            outstanding.append(
                f"{name}: the promotion writer finished the cards but not its "
                f"bookkeeping ({state}). Run `janki promote` for that part to "
                "finish it; this job stays incomplete until it does."
            )
        landed.extend(part_landed)
        bound = _part_selections(
            config,
            part_name=name,
            accepted=_accepted_ids(part),
            primary=receipt_id,
            archive_path=archive_path,
            source_file=source_file,
        )
        selections.extend(bound)
        receipts.append(
            {
                "part_name": name,
                "state": state,
                "receipt_id": receipt_id,
                "archive_path": None
                if archive_path is None
                else _relative(config, archive_path, label="archive"),
                "landed_ids": list(part_landed),
                # Plural, and each with the ids this part took from it: §9.6.2's
                # completion re-derives exactly this, and a receipt's unselected
                # ids never enter the selection the later phases run over.
                "receipts": [
                    {
                        "receipt_id": selection.receipt_id,
                        "selected_ids": list(selection.record_ids),
                    }
                    for selection in bound
                ],
                "disposition": None
                if not isinstance(disposition, Mapping)
                else _plain(disposition),
            }
        )
    receipt_ids = list(dict.fromkeys(item.receipt_id for item in selections))
    scope = resolve_study_finish_scope(config, selections)
    expected = tuple(
        str(item) for item in _authority_section(record, "promotion").get("record_ids") or ()
    )
    if tuple(sorted(scope.record_ids)) != tuple(sorted(expected)):
        raise StudyFinishError(
            f"This finish promoted {sorted(scope.record_ids)} where it was "
            f"authorized for {sorted(expected)}. The job stays outstanding."
        )
    receipt = {
        "parts": receipts,
        "landed_ids": sorted(set(landed)),
        "receipt_ids": receipt_ids,
        # The deduped owner bindings — identity, owner stem, study deck — are
        # what §9.6 re-resolves later. `scope_fingerprint` is disclosed beside
        # them as this moment's whole-scope identity, and is deliberately *not*
        # what the later check compares: a `FinishScope` binds the canonical
        # revision it was resolved against, and the phases after this one
        # rewrite canonical on purpose.
        "projection": [list(entry) for entry in scope.projection],
        "scope_fingerprint": scope.fingerprint,
        "outstanding": outstanding,
    }
    return record, revision, receipt


# --- phase 3: enriched --------------------------------------------------------


def _assert_enrichment_matches(
    bound: Mapping[str, Any], prepared: enrichment_application.PreparedDictionaryEnrichment
) -> None:
    """The re-planned pass must write exactly the values the owner approved."""

    expected_values = _plain(bound.get("bound_values") or {})
    found_values = _plain(dict(prepared.bound_values))
    if expected_values != found_values:
        raise StudyFinishError(
            "Dictionary enrichment now proposes different values than this "
            f"finish bound: authorized {expected_values}, recomputed "
            f"{found_values}. Nothing was written."
        )
    expected_cleared = {
        str(key): sorted(str(item) for item in value)
        for key, value in (bound.get("cleared_fields") or {}).items()
    }
    found_cleared = {
        str(key): sorted(str(item) for item in value)
        for key, value in dict(prepared.cleared_fields).items()
    }
    if expected_cleared != found_cleared:
        raise StudyFinishError(
            "Dictionary enrichment now clears different provisional marks than "
            f"this finish bound: authorized {expected_cleared}, recomputed "
            f"{found_cleared}. Nothing was written."
        )
    if prepared.enriched_at != str(bound.get("enriched_at") or ""):
        raise StudyFinishError(
            "Dictionary enrichment would attribute another day than the one "
            "this finish froze. Nothing was written."
        )


def _apply_enriched(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    replay: enrich.DictionaryLookup,
    book: enrich.DictionaryFactBook,
    resumed: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Write the two reference stores first, then the vocabulary values.

    The order is the whole of §7.9: the enrichment apply re-plans against the
    live reference store, so it has to observe what this phase's first half
    wrote. Running it the other way round computes values the preview never
    showed and refuses naming both.
    """

    bound = _authority_section(record, "enrichment")
    preparation = character_notes.ReferenceFactsPreparation.from_dict(
        _authority_section(record, "reference")
    )
    if resumed:
        reference = character_notes.recover_prepared_reference_facts(
            config, preparation
        )
    else:
        reference = character_notes.apply_prepared_reference_facts(config, preparation)

    held = record.get("enrichment_intent")
    outstanding: list[str] = []
    if isinstance(held, Mapping):
        prepared = enrichment_application.PreparedDictionaryEnrichment.from_dict(held)
        recovery = enrichment_application.recover_prepared_dictionary_enrichment_under_guard(
            config, prepared, client=replay
        )
        state = recovery.state
        changed = recovery.changed_record_ids
        cleared = recovery.cleared_record_ids
        ledger_error = recovery.ledger_error
    else:
        decision = enrichment_application.plan_dictionary_enrichment(
            config,
            replay,
            tuple(str(item) for item in bound.get("record_ids") or ()),
            force_fields=tuple(str(item) for item in bound.get("force_fields") or ()),
        )
        prepared = enrichment_application.prepare_dictionary_enrichment(
            config,
            decision,
            now=date.fromisoformat(str(bound.get("enriched_at"))),
            book=book,
        )
        _assert_enrichment_matches(bound, prepared)
        record, revision = _replace_record(
            path, record, revision, enrichment_intent=prepared.to_dict()
        )
        commit = enrichment_application.apply_prepared_dictionary_enrichment_under_guard(
            config, prepared, client=replay, decision=decision
        )
        state = commit.state
        changed = commit.changed_record_ids
        cleared = commit.cleared_record_ids
        ledger_error = commit.ledger_error
    if state == "committed_ledger_incomplete":
        outstanding.append(
            "The dictionary values landed but their ledger attribution did not"
            + (f" ({ledger_error})" if ledger_error is not None else "")
            + ". Run `janki enrich` for this scope to finish it; this job stays "
            "incomplete until it does."
        )
    return (
        record,
        revision,
        {
            "reference": {
                "already_complete": list(reference.already_complete),
                "changed": list(reference.changed),
                "missing": [item.to_dict() for item in reference.missing],
                "fingerprint": preparation.fingerprint,
            },
            "dictionary": {
                "state": state,
                "changed_record_ids": list(changed),
                "cleared_record_ids": list(cleared),
                "book_fingerprint": book.fingerprint,
                "projected_output_sha256": bound.get("projected_output_sha256"),
            },
            "outstanding": outstanding,
        },
    )


# --- phase 4: audio_complete --------------------------------------------------


def _paid_profile(
    clip: audio_application.AudioClipPlan,
    *,
    word_provider: Any,
    sentence_provider: Any,
) -> openai_realtime.OpenAiRealtimeProvider:
    selected = word_provider if clip.kind == "word" else sentence_provider
    if selected is None:
        raise StudyFinishError(
            f"The exact {clip.provider.name} provider for {clip.kind} audio is "
            "required before a paid clip can be reserved."
        )
    try:
        provider = sentence_profile_for(selected, clip.record_id)
    except JankiError as exc:
        raise StudyFinishError(str(exc)) from exc
    if not isinstance(provider, openai_realtime.OpenAiRealtimeProvider):
        raise StudyFinishError(
            "Paid study audio is supported only through OpenAI Realtime."
        )
    return provider


def _expected_request_fp(
    clip: audio_application.AudioClipPlan,
    *,
    word_provider: Any,
    sentence_provider: Any,
) -> str:
    """The fingerprint the dispatcher independently expects of this call.

    Derived from the approved request and the resolved provider profile before
    the call leaves, and saved durably at ``before_paid_dispatch``. Proving
    compares the paid writer's own recorded attempt against *this* value; a
    reservation id alone, or a later journal absence, proves nothing.
    """

    provider = _paid_profile(
        clip, word_provider=word_provider, sentence_provider=sentence_provider
    )
    return provider.request_fingerprint(
        clip.request_input, forced_accent=clip.forced_accent
    )


def _assert_reservation_request(
    operation: Any,
    operation_id: str,
    clip: audio_application.AudioClipPlan,
    expected_request_fp: str,
) -> None:
    """This journal entry has to be the exact call this reservation names.

    Both halves of the paid witness meet here: the request fingerprint this
    coordinator derived and saved durably, and the entry the paid provider
    authorized. An entry describing another request under the same id proves
    nothing about this clip, so it refuses rather than being adopted.
    """

    expected_source = audio_application.audio_cmd.audio_journal_source(
        clip.record_id, of=clip.kind, target=clip.target
    )
    if (
        operation.kind != "audio-realtime"
        or operation.source_file != expected_source
        or operation.source_sha256 != clip.content_fingerprint
        or operation.request_fp != expected_request_fp
        or operation.model != openai_realtime.MODEL
    ):
        raise StudyFinishError(
            f"Paid audio reservation {operation_id} no longer matches its exact "
            "request."
        )


def _unsent_reserved_targets(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: audio_application.AudioPlan,
) -> frozenset[str]:
    """Which recorded reservations the journal itself proves never reached a provider.

    Read-only, deliberately. The operation journal belongs to
    :mod:`japanese_anki.operations` and the paid provider: advancing a stranded
    entry to ``failed_before_send`` or forgetting it here would be this
    coordinator keeping another writer's books, and it would do it at exactly the
    moment when guessing costs money. So an entry that may have been billed is
    disclosed with the route that settles it, and nothing is re-sent.

    A reservation whose clip now has bytes is left exactly as it is: it carries
    the EXPECTED request fingerprint the completion proof matches the paid
    writer's own recorded attempt against, and the writer legitimately forgets
    its journal entry after a successful commit.
    """

    from japanese_anki import operations

    reservations = record.get("paid_clip_reservations")
    assert isinstance(reservations, list)
    clips = {clip.target: clip for clip in fresh.clips}
    journal = operations.OperationJournal.load(config.operations_file)
    unsent: set[str] = set()
    for item in reservations:
        target = str(item.get("target") or "")
        clip = clips.get(target)
        if clip is None:
            raise StudyFinishError(
                f"A reserved paid clip {target!r} disappeared from this finish's "
                "audio plan."
            )
        if clip.state != "provider-required":
            continue
        operation_id = str(item.get("operation_id") or "")
        operation = journal.operations.get(operation_id)
        if operation is None:
            raise StudyFinishError(
                f"This finish reserved paid clip {target!r} as operation "
                f"{operation_id}, that operation is no longer in the journal, and "
                "the clip still has no bytes. janki will not bill it again on "
                "that basis: settle the operation's own record first, then plan "
                "this finish again."
            )
        _assert_reservation_request(
            operation,
            operation_id,
            clip,
            str(item.get("expected_request_fp") or ""),
        )
        if operation.cleanup is not None:
            raise StudyFinishError(
                f"Paid audio reservation {operation_id} has unfinished cleanup, so "
                f"{target!r} is not ready to be dispatched again."
            )
        if (
            operation.state in operations.TERMINAL_STATES
            and not operation.money_may_have_been_spent
        ):
            unsent.add(target)
            continue
        raise StudyFinishError(
            f"This finish's paid clip {target!r} is reserved as operation "
            f"{operation_id}, which stands at {operation.state}. A call that may "
            "already have been billed is never re-sent automatically and no fresh "
            "retry authority is inferred from it. Settle that operation with "
            "`janki operations`, then plan this finish again."
        )
    return frozenset(unsent)


def _reserve_paid_clip(
    config: ProjectConfig,
    path: Path,
    durable: dict[str, Any],
    dispatch: audio_application.PaidAudioDispatch,
    *,
    confirmed: audio_application.ConfirmedAudioPlan,
    unsent: frozenset[str],
    word_provider: Any,
    sentence_provider: Any,
) -> None:
    """Save this clip's reservation durably, before the call leaves.

    Two values that have to meet later are written by two different writers:
    the ``expected_request_fp`` here, derived from the approved request and the
    resolved profile, and the paid writer's own ``paid_attempt`` on the clip's
    ledger entry. Neither is accepted on its own, so this has to be durable
    before the transport sees anything.
    """

    from japanese_anki import operations

    clip = dispatch.clip
    approved = confirmed.clip_for(clip.target)
    if (
        approved is None
        or approved.initial_state != "provider-required"
        or approved.provider.access != "paid-network"
        or (approved.record_id, approved.kind, approved.request_input) != (
            clip.record_id,
            clip.kind,
            clip.request_input,
        )
        or approved.forced_accent != clip.forced_accent
        or approved.content_fingerprint != clip.content_fingerprint
        or approved.provider != clip.provider
    ):
        raise StudyFinishError(
            "Paid audio dispatch exceeds this finish's confirmed authority."
        )
    expected_request_fp = _expected_request_fp(
        clip, word_provider=word_provider, sentence_provider=sentence_provider
    )
    journal = operations.OperationJournal.load(config.operations_file)
    operation = journal.operations.get(dispatch.operation_id)
    if operation is None or operation.state != "authorized":
        raise StudyFinishError(
            "Paid audio dispatch has no exact newly authorized operation."
        )
    _assert_reservation_request(
        operation, dispatch.operation_id, clip, expected_request_fp
    )
    record = durable["record"]
    revision = durable["revision"]
    reservations = record.get("paid_clip_reservations")
    assert isinstance(reservations, list)
    same_target = [
        item
        for item in reservations
        if isinstance(item, Mapping) and item.get("target") == clip.target
    ]
    if same_target and clip.target not in unsent:
        raise StudyFinishError(
            f"Paid audio clip {clip.target!r} already consumed its one dispatch "
            "reservation, and the journal does not prove that attempt went unsent."
        )
    # The current reservation replaces its own proven-unsent predecessor at the
    # same target: the evidence rule takes one current attempt per target, not a
    # history list. Earlier attempts stay in the operation journal, which is the
    # store that owns them.
    remaining = [
        item
        for item in reservations
        if not isinstance(item, Mapping) or item.get("target") != clip.target
    ]
    remaining.append(
        {
            "target": clip.target,
            "operation_id": dispatch.operation_id,
            "expected_request_fp": expected_request_fp,
        }
    )
    updated, updated_revision = _replace_record(
        path, record, revision, paid_clip_reservations=remaining
    )
    durable["record"] = updated
    durable["revision"] = updated_revision


def _assert_paid_plan_is_current(
    confirmed: audio_application.ConfirmedAudioPlan,
    fresh: audio_application.AudioPlan,
    reservations: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse a plan whose new paid work another run already supplied.

    A clip this finish confirmed as new work, now sitting on disk with no
    reservation of this finish's own, cannot be claimed: its bytes are some
    other authorized run's, and no evidence here says otherwise. The route out
    is a fresh plan whose classification is truthful — never a forced
    regeneration, never an inferred approval, and never another run's media
    attributed to this finish.
    """

    reserved = {str(item.get("target") or "") for item in reservations}
    stranded = [
        clip.target
        for clip in confirmed.clips
        if clip.initial_state == "provider-required"
        and clip.provider.access == "paid-network"
        and clip.target not in reserved
        and any(
            fresh_clip.target == clip.target and fresh_clip.state != "provider-required"
            for fresh_clip in fresh.clips
        )
    ]
    if stranded:
        raise StudyFinishStaleAudioPlanError(
            "This finish was confirmed to pay for "
            + ", ".join(sorted(stranded))
            + ", and those clips are already on disk from work it never "
            "dispatched. Janki will not claim another run's media, force a "
            "second call, or infer that you approved one. Plan this finish "
            "again — the fresh plan reports those clips as reused, and its "
            "confirmation costs nothing for them."
        )


def _apply_audio_complete(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    chosen_provider: str | None,
    word_provider: Any,
    sentence_provider: Any,
    progress: StudyFinishProgress | None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """Run this finish's audio through the audio writer, and prove the result.

    The caller already holds ``.janki-audio-operation``. Every paid clip is
    reserved durably at the writer's own ``before_paid_dispatch`` seam, and the
    completion proof is the audio module's — this coordinator writes no second
    witness and keeps no capture after the writer's successful cleanup.
    """

    bound = _authority_section(record, "audio")
    confirmed = audio_application.ConfirmedAudioPlan.from_wire(
        config, bound.get("confirmed_plan") or {}
    )
    record_ids = tuple(str(item) for item in bound.get("record_ids") or ())
    include_examples = bool(bound.get("include_example_audio"))
    slots = tuple(
        audio_application.ExpectedAudioSlot(
            record_id=str(item.get("record_id")),
            kind="word" if item.get("kind") == "word" else "example",
            position=None if item.get("position") is None else int(item["position"]),
        )
        for item in bound.get("expected_slots") or ()
    )

    def fresh_plan() -> audio_application.AudioPlan:
        return audio_application.plan_targeted_audio(
            config,
            record_ids,
            words=True,
            examples=include_examples,
            force=False,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )

    try:
        planned = fresh_plan()
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise StudyFinishError(str(exc)) from exc
    unsent = _unsent_reserved_targets(config, record, planned)
    reservations = record.get("paid_clip_reservations")
    assert isinstance(reservations, list)
    _assert_paid_plan_is_current(confirmed, planned, reservations)
    try:
        audio_application.preflight_paid_deck_audio_plan(
            planned, word_provider=word_provider, sentence_provider=sentence_provider
        )
    except JankiError as exc:
        raise StudyFinishError(str(exc)) from exc

    _emit(progress, "Creating card audio")
    durable = {"record": record, "revision": revision}

    def reserve(dispatch: audio_application.PaidAudioDispatch) -> None:
        _reserve_paid_clip(
            config,
            path,
            durable,
            dispatch,
            confirmed=confirmed,
            unsent=unsent,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )

    outcome = audio_application.execute_targeted_audio_locked(
        config,
        record_ids,
        words=True,
        examples=include_examples,
        expected_fingerprint=planned.fingerprint,
        force=False,
        prune=False,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        before_paid_dispatch=reserve,
    )
    record = durable["record"]
    revision = durable["revision"]
    if not outcome.succeeded:
        return record, revision, None

    final = fresh_plan()
    reservations = record.get("paid_clip_reservations")
    assert isinstance(reservations, list)
    # One current attempt per target, which is what B's evidence rule takes: a
    # superseded predecessor was removed when its successor was reserved, and
    # the history of earlier attempts stays in the operation journal.
    attempts = tuple(
        audio_application.ReservedPaidAttempt(
            target=str(item["target"]),
            operation_id=str(item["operation_id"]),
            expected_request_fp=str(item["expected_request_fp"]),
        )
        for item in reservations
    )
    try:
        proof = audio_application.prove_audio_completion(
            config,
            final,
            authority=confirmed,
            expected_slots=slots,
            reservations=attempts,
        )
    except audio_application.AudioProofError as exc:
        raise StudyFinishError(
            f"This finish's audio could not be proven complete: {exc}"
        ) from exc
    return (
        record,
        revision,
        {
            "include_example_audio": include_examples,
            "omitted_by_owner": not include_examples,
            "proof": proof.to_wire(),
            "proof_fingerprint": proof.fingerprint,
            "canonical_sha256": proof.canonical_sha256,
            "stored_slot_count": proof.stored_slot_count,
            "clip_count": proof.clip_count,
            "paid_clip_count": sum(
                1 for slot in proof.slots if slot.paid_operation is not None
            ),
        },
    )


# --- phase 5: packaged --------------------------------------------------------


def _bound_projection(
    config: ProjectConfig, record: Mapping[str, Any]
) -> deck_package.DeckPackagePlan:
    """Restore the exact whole-deck projection this authority bound.

    The authority carries :func:`deck_package.plan_wire`'s complete projection,
    written before the first effect, so this restores that value rather than
    planning a second one. ``plan_from_wire`` is pure and re-earns the plan's own
    fingerprint from the wire, and :func:`deck_package.prepare_deck_package`
    compares it against the freshly realized plan — which is where drift is
    caught, by the writer that owns the comparison.
    """

    bound = _authority_section(record, "package")
    wire = bound.get("projection")
    if not isinstance(wire, Mapping):
        raise StudyFinishError("This finish records no bound package projection.")
    try:
        return deck_package.plan_from_wire(config, wire)
    except deck_package.DeckPackageError as exc:
        raise StudyFinishError(
            f"The package projection this finish bound is unreadable: {exc}"
        ) from exc


def _audio_proof(
    config: ProjectConfig, record: Mapping[str, Any]
) -> audio_application.AudioCompletionProof:
    receipt = record.get("audio_receipt")
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("proof"), Mapping):
        raise StudyFinishError("This finish has no durable audio completion proof.")
    try:
        return audio_application.AudioCompletionProof.from_wire(
            config, receipt["proof"]
        )
    except audio_application.AudioProofError as exc:
        raise StudyFinishError(
            f"This finish's audio completion proof is unreadable: {exc}"
        ) from exc


def _apply_packaged(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Build privately, persist the intent, then publish the exact bytes."""

    projection = _bound_projection(config, record)
    proof = _audio_proof(config, record)
    held = record.get("package_intents")
    assert isinstance(held, list)
    preparation: deck_package.DeckPackagePreparation | None = None
    result: deck_package.DeckPackageResult | None = None
    if held:
        preparation = deck_package.DeckPackagePreparation.from_wire(config, held[-1])
        result = deck_package.recover_prepared_deck_package(config, preparation)
        if result is None:
            # `dist/` is disposable, so a vanished private stage is a free
            # rebuild — recorded as a new immutable attempt that links as
            # superseding, never as an edit of the intent it replaces.
            superseded = preparation
            preparation = deck_package.prepare_deck_package(
                config, projection, audio_completion=proof, supersedes=superseded
            )
            record, revision = _replace_record(
                path,
                record,
                revision,
                package_intents=[*held, preparation.to_wire(config)],
            )
            result = deck_package.publish_prepared_deck_package(config, preparation)
    else:
        preparation = deck_package.prepare_deck_package(
            config, projection, audio_completion=proof
        )
        record, revision = _replace_record(
            path, record, revision, package_intents=[preparation.to_wire(config)]
        )
        result = deck_package.publish_prepared_deck_package(config, preparation)
    if result.package_sha256 != preparation.package_sha256:
        raise StudyFinishError(
            "The published package is not the artifact this finish prepared."
        )
    # B's own receipt, verbatim: its `card_count` is the archive's validated
    # native expansion rather than the plan's notes-times-directions capacity,
    # and its `audio_selection` numbers are explicitly the proof's record ids.
    # Restating any of them here would be a second, unvalidated scope.
    return record, revision, dict(preparation.packaged_receipt(config))


# --- phase 6: complete --------------------------------------------------------


def _apply_preview(
    config: ProjectConfig, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Draw the EXACT proven package, and build, publish and fetch nothing.

    A preview-only resume happens here and nowhere else: the package is already
    proven, so this reads those exact bytes back and renders them with the
    owning card renderer. Re-building would draw an artifact nobody delivered.
    """

    from japanese_anki import card_preview

    receipt = record.get("package_receipt")
    if not isinstance(receipt, Mapping):
        raise StudyFinishError("This finish has no durable package receipt.")
    package = _authority_path(config, receipt.get("output_path"), label="package")
    expected = str(receipt.get("package_sha256") or "")
    try:
        payload = read_bytes_bound(package)
    except (FileNotFoundError, DataError, OSError) as exc:
        raise StudyFinishError(
            f"The packaged deck at {package} could not be read: {exc}"
        ) from exc
    if _sha(payload) != expected:
        raise StudyFinishError(
            f"The packaged deck at {package} is no longer the artifact this "
            f"finish proved ({expected}); nothing was previewed."
        )
    deck = _authority_section(record, "deck")
    # The two scopes §7.12 keeps apart: the ids this job accepted, which are the
    # cards it draws, and the whole delivered deck, which the renderer counts
    # separately from them. Drawing the package without a selection would show
    # every card the deck ships and report the deck's totals as this job's.
    accepted = tuple(
        str(item)
        for item in _authority_section(record, "promotion").get("record_ids") or ()
    )
    promotion_receipt = record.get("promotion_receipt")
    new_ids = (
        tuple(str(item) for item in promotion_receipt.get("landed_ids") or ())
        if isinstance(promotion_receipt, Mapping)
        else ()
    )
    audio_receipt = record.get("audio_receipt")
    omitted = isinstance(audio_receipt, Mapping) and bool(
        audio_receipt.get("omitted_by_owner")
    )
    notices: list[str] = []
    if omitted:
        notices.append(
            "Sentence audio was omitted by owner for this job; existing clips "
            "and references were kept."
        )
    # `pending` is never this preview's answer: the package it draws is already
    # proven, so a clip that is missing from these exact bytes is either the
    # owner's opt-out or a fault, and it is labelled as whichever one it is.
    bound = _authority_section(record, "preview")
    try:
        preview = card_preview.render_packaged_card_preview(
            config,
            package,
            package_sha256=expected,
            deck_name=str(deck.get("name") or ""),
            deck_kind="vocabulary",
            directions=tuple(str(item) for item in deck.get("card_types") or ()),
            new_record_ids=new_ids,
            scope_record_ids=accepted,
            subtitle="The cards this study job delivered",
            notices=notices,
            audio_state="omitted" if omitted else "packaged",
        )
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise StudyFinishError(
            f"The finished package could not be drawn: {exc}"
        ) from exc
    output = (config.dist_dir / f"{package.stem}-preview.html").absolute()
    card_preview.write_card_preview(preview, output)
    return {
        "preview_path": _relative(config, output, label="preview"),
        "preview_sha256": preview.sha256,
        "package_sha256": expected,
        "inventory_fingerprint": receipt.get("inventory_fingerprint"),
        "note_count": preview.note_count,
        "card_count": preview.card_count,
        "deck_note_count": preview.deck_note_count,
        "deck_card_count": preview.deck_card_count,
        "render_assets": dict(preview.render_assets),
        "content_security_policy": preview.content_security_policy,
        "audio_state": "omitted" if omitted else "packaged",
        "notices": notices,
        # What the owner approved, carried into the receipt beside what was
        # drawn: the rendering their decisions were bound to, and the content
        # fingerprint of the projected cards. Stated as provenance — the proof
        # that these are the delivered cards is the package hash above.
        "authorized_rendering_fingerprint": bound.get("rendering_fingerprint"),
        "authorized_content_fingerprint": bound.get("content_fingerprint"),
    }


# --- §9.6: the fresh conjunction a `complete` receipt requires ----------------


def _settled_children(config: ProjectConfig, job_id: str) -> dict[str, list[str]]:
    """Part name → the settled staging documents its children produced."""

    status = study_job.study_job_status(config, job_id)
    settled: dict[str, list[str]] = {}
    for batch in status.batches:
        if batch.missing:
            continue
        for child in batch.children:
            if child.state == "committed" and child.bookkeeping_complete:
                settled.setdefault(child.source_name, []).append(child.staging_name)
    return settled


def _assert_complete(config: ProjectConfig, record: Mapping[str, Any]) -> None:
    """Every §9.6 condition, re-derived now — never from a helper's say-so.

    A package alone does not make a job complete, and neither does a phase
    label. This re-asks the durable stores: the frontier, each part's
    disposition and receipts, every held row's exclusion, the aggregate scope,
    the ledger's write-ahead record, the audio proof, and the archive's own
    whole-deck inventory with the selection contained in it.
    """

    job_id = str(_authority_section(record, "job").get("job_id") or "")
    status = study_job.study_job_status(config, job_id)
    if status.blocking_operation_ids:
        raise StudyFinishError(
            f"{len(status.blocking_operation_ids)} model call(s) of this job "
            "still block spending, so it is not complete: "
            + ", ".join(status.blocking_operation_ids)
        )
    settled = _settled_children(config, job_id)
    parts = _authority_parts(record)
    for part in parts:
        name = str(part.get("part_name") or "")
        staged = settled.get(name) or []
        if len(set(staged)) != 1:
            raise StudyFinishError(
                f"Part {name!r} no longer has exactly one settled extraction, so "
                "this job is not complete."
            )

    promotion_receipt = record.get("promotion_receipt")
    if not isinstance(promotion_receipt, Mapping):
        raise StudyFinishError("This finish has no durable promotion receipt.")
    if promotion_receipt.get("outstanding"):
        raise StudyFinishError(
            "A promotion writer's bookkeeping is unfinished: "
            + "; ".join(str(item) for item in promotion_receipt["outstanding"])
        )
    enrichment_receipt = record.get("enrichment_receipt")
    if isinstance(enrichment_receipt, Mapping) and enrichment_receipt.get("outstanding"):
        raise StudyFinishError(
            "A dictionary writer's bookkeeping is unfinished: "
            + "; ".join(str(item) for item in enrichment_receipt["outstanding"])
        )

    by_part = {
        str(entry.get("part_name") or ""): entry
        for entry in promotion_receipt.get("parts") or ()
        if isinstance(entry, Mapping)
    }
    selections: list[StudyFinishSelection] = []
    for part in parts:
        name = str(part.get("part_name") or "")
        accepted = _accepted_ids(part)
        disposition = part.get("disposition")
        outcome = by_part.get(name)
        if outcome is None:
            raise StudyFinishError(
                f"Part {name!r} has no recorded promotion outcome; this job is "
                "not complete."
            )
        bound = outcome.get("receipts")
        if not isinstance(bound, list) or any(
            not isinstance(item, Mapping) for item in bound
        ):
            raise StudyFinishError(
                f"Part {name!r} records no readable promotion receipt binding; "
                "this job is not complete."
            )
        if accepted:
            # §9.6.2: the exact source-bound receipts, plural, with the ids this
            # part took from each. `landed_ids` alone is vacuously accounted for
            # by an archive retry, which lands nothing and accepts cards anyway.
            archive = outcome.get("archive_path")
            if not bound or not isinstance(archive, str) or not archive:
                raise StudyFinishError(
                    f"Part {name!r} accepted {len(accepted)} card(s) but retains "
                    "no promotion receipt; this job is not complete."
                )
            taken: list[str] = []
            for item in bound:
                selections.append(
                    StudyFinishSelection(
                        part_name=name,
                        receipt_id=str(item.get("receipt_id") or ""),
                        archive_path=archive,
                        record_ids=tuple(
                            str(one) for one in item.get("selected_ids") or ()
                        ),
                    )
                )
                taken.extend(selections[-1].record_ids)
            if sorted(taken) != sorted(accepted) or len(set(taken)) != len(taken):
                raise StudyFinishError(
                    f"Part {name!r}'s promotion receipts select {sorted(taken)} "
                    f"where this job accepted {sorted(accepted)}; this job is not "
                    "complete."
                )
        elif bound:
            raise StudyFinishError(
                f"Part {name!r} accepted no card and may retain no promotion "
                "receipt for one; this job is not complete."
            )
        elif not isinstance(disposition, Mapping) or not str(
            disposition.get("reason") or ""
        ).strip():
            raise StudyFinishError(
                f"Part {name!r} accepted no card and carries no owner "
                "disposition with a reason; this job is not complete."
            )
        held = tuple(str(item) for item in part.get("held_ids") or ())
        excluded = tuple(str(item) for item in part.get("excluded_ids") or ())
        if held or excluded:
            covered = (
                frozenset(str(item) for item in disposition.get("record_ids") or ())
                if isinstance(disposition, Mapping)
                else frozenset()
            )
            whole_part = isinstance(disposition, Mapping) and not disposition.get(
                "record_ids"
            )
            uncovered = [
                record_id
                for record_id in (*held, *excluded)
                if not whole_part and record_id not in covered
            ]
            if uncovered:
                raise StudyFinishError(
                    f"Part {name!r} holds back "
                    + ", ".join(sorted(uncovered))
                    + " and no owner exclusion or deferral covers them; this job "
                    "is not complete."
                )

    scope = resolve_study_finish_scope(config, selections)
    if [list(entry) for entry in scope.projection] != promotion_receipt.get(
        "projection"
    ):
        raise StudyFinishError(
            "The aggregate scope no longer resolves to the selected canonical "
            "projection this finish bound, or its owner bindings disagree; the "
            "job is not complete."
        )

    book = ledger.load(config.ledger_file)
    if book.pending_audio:
        raise StudyFinishError(
            f"{len(book.pending_audio)} audio clip(s) are still in the ledger's "
            "write-ahead record, so this job is not complete."
        )
    proof = _audio_proof(config, record)
    try:
        audio_application.revalidate_audio_completion(config, proof)
    except audio_application.AudioProofError as exc:
        raise StudyFinishError(
            f"This job's audio is no longer complete: {exc}"
        ) from exc
    audio_receipt = record.get("audio_receipt")
    if (
        isinstance(audio_receipt, Mapping)
        and not audio_receipt.get("include_example_audio")
        and not audio_receipt.get("omitted_by_owner")
    ):
        raise StudyFinishError(
            "Sentence audio is neither included nor recorded as the owner's "
            "explicit opt-out; this job is not complete."
        )

    package_receipt = record.get("package_receipt")
    if not isinstance(package_receipt, Mapping):
        raise StudyFinishError("This finish has no durable package receipt.")
    intents = record.get("package_intents")
    assert isinstance(intents, list)
    if not intents:
        raise StudyFinishError("This finish recorded no package preparation.")
    preparation = deck_package.DeckPackagePreparation.from_wire(config, intents[-1])
    package = _authority_path(
        config, package_receipt.get("output_path"), label="package"
    )
    try:
        payload = read_bytes_bound(package)
    except (FileNotFoundError, DataError, OSError) as exc:
        raise StudyFinishError(
            f"The finished package at {package} could not be read: {exc}"
        ) from exc
    if _sha(payload) != str(package_receipt.get("package_sha256") or ""):
        raise StudyFinishError(
            f"The finished package at {package} is not the artifact this job's "
            "receipt proves."
        )
    if preparation.inventory.fingerprint != str(
        package_receipt.get("inventory_fingerprint") or ""
    ):
        raise StudyFinishError(
            "The package receipt's whole-deck inventory is not the one the "
            "preparation validated."
        )
    packaged_ids = {note.record_id for note in preparation.inventory.notes}
    outside = [
        record_id for record_id in scope.record_ids if record_id not in packaged_ids
    ]
    if outside:
        raise StudyFinishError(
            "The package does not contain this job's selection: "
            + ", ".join(sorted(outside))
        )
    # §9.6.6 asks for this job's export entries, not for a ledger file nobody
    # else may write to: the publication transaction is over, and an ordinary
    # later build legitimately moves those bytes. Each frozen row is offered back
    # to the ledger's own writer on the copy loaded above, which returns False
    # only when the current entry already **is** that exact value. Nothing here
    # is saved; this in-memory ledger is dropped with the check.
    moved: list[str] = []
    for entry in preparation.export_delta:
        try:
            changed = book.record_export(
                entry.record_id, entry.deck_stem, gaps=entry.gaps, at=entry.at
            )
        except JankiError as exc:
            raise StudyFinishError(
                f"This job's export entry for {entry.record_id} could not be read "
                f"back from the ledger: {exc}"
            ) from exc
        if changed:
            moved.append(f"{entry.record_id} in {entry.deck_stem}")
    if moved:
        raise StudyFinishError(
            "The export ledger no longer records this job's publication for "
            + ", ".join(sorted(moved))
            + "; this job is not complete."
        )


# --- running the chain --------------------------------------------------------


def _replay_client(
    record: Mapping[str, Any],
) -> tuple[enrich.ReplayDictionaryClient, enrich.DictionaryFactBook]:
    section = _authority_section(record, "dictionary")
    raw = section.get("book")
    if not isinstance(raw, Mapping):
        raise StudyFinishError("This finish records no dictionary fact book.")
    book = enrich.DictionaryFactBook.from_dict(raw)
    if book.fingerprint != str(section.get("fingerprint") or ""):
        raise StudyFinishError(
            "This finish's dictionary fact book does not match its bound "
            "fingerprint."
        )
    return enrich.ReplayDictionaryClient(book), book


def _run(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    resumed: bool,
    progress: StudyFinishProgress | None,
    chosen_provider: str | None,
    word_provider: Any,
    sentence_provider: Any,
) -> StudyFinishResult:
    """Advance the one phase chain as far as this run's evidence allows."""

    replay, book = _replay_client(record)
    outstanding: list[str] = []

    if record["state"] == "authorized":
        _emit(progress, "Writing the reviewed staging files")
        receipt = _apply_reviewed(config, record, resumed=resumed)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="reviewed",
            receipt_key="review_receipt",
            receipt=receipt,
        )

    if record["state"] == "reviewed":
        _emit(progress, "Promoting the reviewed cards")
        record, revision, receipt = _apply_promoted(
            config, path, record, revision, replay=replay
        )
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="promoted",
            receipt_key="promotion_receipt",
            receipt=receipt,
        )

    if record["state"] == "promoted":
        _emit(progress, "Writing reference facts and dictionary values")
        record, revision, receipt = _apply_enriched(
            config,
            path,
            record,
            revision,
            replay=replay,
            book=book,
            resumed=resumed,
        )
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="enriched",
            receipt_key="enrichment_receipt",
            receipt=receipt,
        )

    if record["state"] == "enriched":
        record, revision, receipt = _apply_audio_complete(
            config,
            path,
            record,
            revision,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            progress=progress,
        )
        if receipt is None:
            return _result(
                config,
                path,
                record,
                outstanding=(
                    "The audio writer did not finish this job's clips. Nothing "
                    "was re-sent; resume this finish to continue under the "
                    "authority it already recorded.",
                ),
            )
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="audio_complete",
            receipt_key="audio_receipt",
            receipt=receipt,
        )

    if record["state"] == "audio_complete":
        _emit(progress, "Building the Anki package")
        record, revision, receipt = _apply_packaged(config, path, record, revision)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="packaged",
            receipt_key="package_receipt",
            receipt=receipt,
        )

    if record["state"] == "packaged":
        _emit(progress, "Rendering the final card preview")
        receipt = _apply_preview(config, record)
        _assert_complete(config, record)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="complete",
            receipt_key="preview_receipt",
            receipt=receipt,
        )

    for key in ("promotion_receipt", "enrichment_receipt"):
        held = record.get(key)
        if isinstance(held, Mapping):
            outstanding.extend(str(item) for item in held.get("outstanding") or ())
    return _result(config, path, record, outstanding=outstanding)


@contextlib.contextmanager
def _finish_locks(config: ProjectConfig):
    """§6.5's shared curation guard first, then the audio operation lock.

    One global lock always taken ahead of the per-path locks is what stops a
    curation holding file A and waiting for B from deadlocking a promotion
    holding B and waiting for A. The audio-operation lock is held from here
    through publication because §7.10 requires the build's ledger before-digest
    to be taken under it. Neither is re-entrant, so every writer below is
    called through its ``*_under_guard`` entry.
    """

    with (
        study_curation.curation_guard(config),
        exclusive_path_lock(config.root / ".janki-audio-operation"),
    ):
        yield


def _assert_dispositions_cover(plan: StudyFinishPlan) -> None:
    undisposed = plan.undisposed_parts
    if undisposed:
        raise StudyFinishError(
            "These parts accept no card, or hold rows back, and no owner "
            "disposition covers them: "
            + ", ".join(part.part_name for part in undisposed)
            + ". Record an exclusion or deferral with your reason in "
            f"{_CHOICE_CONTROLS['dispositions']}, then plan again. Nothing was "
            "written."
        )


def execute_study_finish(
    config: ProjectConfig,
    plan: StudyFinishPlan,
    *,
    progress: StudyFinishProgress | None = None,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> StudyFinishResult:
    """Record the owner's authority, then apply the whole chain under it once."""

    if plan.repository_root != config.root.resolve():
        raise StudyFinishError("This study finish plan belongs to another repository.")
    _assert_dispositions_cover(plan)
    _emit(progress, "Preparing finish")
    path = plan.record_path
    try:
        prepare_bound_directory(plan.finish_directory)
    except (DataError, OSError) as exc:
        raise StudyFinishError(
            f"Could not prepare the study finish directory: {exc}"
        ) from exc
    existing = _read_record_optional(path)
    if existing is None:
        record = _new_record(plan)
        revision = _write_new(path, record)
        resumed = False
    else:
        # The same exact authority was recorded and interrupted. Continue that
        # receipt rather than authorizing identical work a second time.
        record, revision = existing
        resumed = True
    with _finish_locks(config):
        return _run(
            config,
            path,
            record,
            revision,
            resumed=resumed,
            progress=progress,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )


def resume_study_finish(
    config: ProjectConfig,
    receipt_id: str,
    *,
    progress: StudyFinishProgress | None = None,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> StudyFinishResult:
    """Continue one durable receipt with no new owner decision and no refetch."""

    path = receipt_path(config, receipt_id)
    record, revision = _read_record(path)
    _emit(progress, "Preparing finish")
    with _finish_locks(config):
        return _run(
            config,
            path,
            record,
            revision,
            resumed=True,
            progress=progress,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )


def inspect_study_finish(config: ProjectConfig, receipt_id: str) -> StudyFinishResult:
    """Report one durable receipt's truthful state without changing it."""

    path = receipt_path(config, receipt_id)
    record, _revision = _read_record(path)
    outstanding: list[str] = []
    for key in ("promotion_receipt", "enrichment_receipt"):
        held = record.get(key)
        if isinstance(held, Mapping):
            outstanding.extend(str(item) for item in held.get("outstanding") or ())
    return _result(config, path, record, outstanding=outstanding)


def list_study_finishes(
    config: ProjectConfig,
    *,
    job_id: str = "",
    limit: int = _LIST_LIMIT,
) -> tuple[StudyFinishResult, ...]:
    """The durable receipts a fresh process can still act on, unfinished first."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise StudyFinishError("A study finish list limit must be an integer.")
    if limit <= 0:
        return ()
    directory = study_finish_directory(config)
    try:
        names = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.name.startswith("study-finish-") and entry.name.endswith(".json")
        )
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise StudyFinishError(
            f"Could not list study finish receipts: {exc}"
        ) from exc
    found: list[tuple[str, StudyFinishResult]] = []
    for name in names:
        path = directory / name
        try:
            record, _revision = _read_record(path)
            result = _result(config, path, record)
        except StudyFinishError:
            continue
        if job_id and result.job_id != job_id:
            continue
        found.append((str(record["updated_at"]), result))
    found.sort(key=lambda item: (item[0], item[1].receipt_id), reverse=True)
    found.sort(key=lambda item: item[1].succeeded)
    return tuple(result for _updated_at, result in found[:limit])
