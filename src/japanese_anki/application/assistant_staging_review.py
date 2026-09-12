"""Plan and execute exact Assistant staging-review decisions.

The ordinary Assistant may point at one opaque proposal and describe a closed
selection, but it cannot mark its own answer reviewed.  This broker turns that
selection into a canonical confirmation projection containing every Japanese
sentence and grammar pattern the click would approve.  Execution resolves the
opaque proposal again, rebuilds the projection, and delegates the only writes
to :class:`japanese_anki.workbench.review.ReviewPanel`.

No path supplied by a provider is accepted, no review choice is inferred, and
this module owns no second review writer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.application import study_curation
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.patterns import PatternSet
from japanese_anki.workbench.review import (
    PartialReviewBatchError,
    PartialReviewError,
    PreparedReview,
    PreparedReviewBatch,
    ReviewBatchOutcome,
    ReviewBatchRequest,
    ReviewOutcome,
    ReviewPanel,
    apply_prepared_review,
    apply_prepared_review_batch,
    prepare_review_batch,
    recover_prepared_review,
    recover_prepared_review_batch,
    review_target_paths,
)

__all__ = [
    "AssistantStagingReviewBatchPartialError",
    "AssistantStagingReviewError",
    "AssistantStagingReviewExecution",
    "AssistantStagingReviewPartialError",
    "AssistantStagingReviewPlan",
    "PreparedStagingReview",
    "PreparedStagingReviewBatch",
    "ReviewBatchRequest",
    "apply_prepared_staging_review",
    "apply_prepared_staging_review_batch",
    "apply_prepared_staging_review_batch_under_guard",
    "apply_prepared_staging_review_under_guard",
    "execute_staging_review",
    "plan_staging_review",
    "prepare_staging_review",
    "prepare_staging_review_batch",
    "recover_prepared_staging_review",
    "recover_prepared_staging_review_batch",
    "recover_prepared_staging_review_batch_under_guard",
    "recover_prepared_staging_review_under_guard",
]


class AssistantStagingReviewError(JankiError):
    """An Assistant staging-review action is not exactly plan-bound."""


class AssistantStagingReviewPartialError(AssistantStagingReviewError):
    """One monotonic review mark landed before the other write failed."""

    def __init__(self, message: str, outcome: ReviewOutcome) -> None:
        self.outcome = outcome
        super().__init__(message)


class AssistantStagingReviewBatchPartialError(AssistantStagingReviewError):
    """Part of one aggregate review landed before a later write failed."""

    def __init__(self, message: str, outcome: ReviewBatchOutcome) -> None:
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AssistantStagingReviewPlan:
    """One exact, browser-safe review projection and its private target."""

    repository_root: Path
    proposal_resource_id: str
    proposal_kind: str
    proposal_path: Path
    source_name: str
    record_ids: tuple[str, ...]
    review_patterns: bool
    staging_snapshot: str
    patterns_snapshot: str
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute() or not self.proposal_path.is_absolute():
            raise ValueError("Assistant staging-review paths must be absolute")
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Assistant staging-review repository root must be canonical")
        if self.proposal_path != self.proposal_path.absolute():
            raise ValueError("Assistant staging-review proposal path must be absolute")
        try:
            self.proposal_path.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError(
                "Assistant staging-review proposal must stay in its repository"
            ) from exc
        if not self.proposal_resource_id.strip():
            raise ValueError("Assistant staging-review proposal resource must be nonblank")
        if self.proposal_kind != "source_extraction":
            raise ValueError("Assistant staging review only accepts source extractions")
        if not self.source_name.strip():
            raise ValueError("Assistant staging-review source name must be nonblank")
        if not isinstance(self.review_patterns, bool):
            raise ValueError("Assistant staging-review pattern choice must be boolean")
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("Assistant staging-review record ids must be unique")
        if any(not isinstance(item, str) or not item.strip() for item in self.record_ids):
            raise ValueError("Assistant staging-review record ids must be nonblank text")
        if not self.record_ids and not self.review_patterns:
            raise ValueError("Assistant staging review must select at least one decision")
        if not _is_sha256(self.staging_snapshot) or not _is_sha256(
            self.patterns_snapshot
        ):
            raise ValueError("Assistant staging-review snapshots must be SHA-256")
        try:
            projection = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Assistant staging-review projection must be JSON") from exc
        if not isinstance(projection, Mapping):
            raise ValueError("Assistant staging-review projection must be a JSON object")
        if _canonical_json(projection) != self.projection_wire:
            raise ValueError("Assistant staging-review projection must use canonical JSON")
        if projection.get("proposal") != {
            "kind": self.proposal_kind,
            "resource_id": self.proposal_resource_id,
            "source_name": self.source_name,
        }:
            raise ValueError("Assistant staging-review projection target is not bound")
        selection = projection.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("Assistant staging-review projection has no selection")
        if selection.get("record_ids") != list(self.record_ids):
            raise ValueError("Assistant staging-review projection record ids are not bound")
        if selection.get("review_patterns") is not self.review_patterns:
            raise ValueError("Assistant staging-review projection pattern choice is not bound")
        if projection.get("snapshots") != {
            "patterns_sha256": self.patterns_snapshot,
            "staging_sha256": self.staging_snapshot,
        }:
            raise ValueError("Assistant staging-review projection snapshots are not bound")
        if hashlib.sha256(self.projection_wire.encode("utf-8")).hexdigest() != self.fingerprint:
            raise ValueError("Assistant staging-review fingerprint does not bind its projection")

    @property
    def projection(self) -> Mapping[str, Any]:
        """The exact parsed value a confirmation card may render."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class AssistantStagingReviewExecution:
    """The fresh matching plan and the existing review transaction's result."""

    plan: AssistantStagingReviewPlan
    outcome: ReviewOutcome


@dataclass(frozen=True, slots=True)
class _PreparedReview:
    plan: AssistantStagingReviewPlan
    panel: ReviewPanel


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AssistantStagingReviewError(
            f"Assistant staging review cannot be fingerprinted: {exc}"
        ) from exc


def _selected_ids(record_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(record_ids, str | bytes) or not isinstance(record_ids, Sequence):
        raise AssistantStagingReviewError(
            "Assistant staging review record ids must be an explicit list of text."
        )
    selected = tuple(record_ids)
    if any(not isinstance(item, str) or not item.strip() for item in selected):
        raise AssistantStagingReviewError(
            "Every Assistant staging review record id must be nonblank text."
        )
    repeated = [item for item, count in Counter(selected).items() if count > 1]
    if repeated:
        raise AssistantStagingReviewError(
            f"Assistant staging review record id {repeated[0]!r} was supplied twice."
        )
    return selected


def _example_value(example: ExampleSentence) -> dict[str, str]:
    """The complete human-readable context for one approved Japanese string."""

    return {
        "english": example.english,
        "furigana": example.furigana,
        "japanese": example.japanese,
        "register": example.register,
        "romaji": example.romaji,
        "spoken_japanese": example.spoken_japanese,
    }


def _record_value(record: VocabularyRecord) -> dict[str, Any]:
    return {
        "examples": [
            _example_value(example) for example in record.examples if example.japanese
        ],
        "expression": record.expression,
        "reading": record.reading,
        "record_id": record.id,
    }


def _pattern_value(pattern_set: PatternSet | None) -> dict[str, Any] | None:
    if pattern_set is None:
        return None
    return {
        "kind": pattern_set.kind,
        "patterns": [pattern.to_dict() for pattern in pattern_set.patterns],
        "source_name": pattern_set.source,
        "title": pattern_set.title,
    }


def _prepare(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    record_ids: Sequence[str],
    review_patterns: bool,
) -> _PreparedReview:
    if not isinstance(proposal_resource_id, str) or not proposal_resource_id.strip():
        raise AssistantStagingReviewError(
            "Assistant staging review needs one nonblank proposal resource id."
        )
    selected = _selected_ids(record_ids)
    if not isinstance(review_patterns, bool):
        raise AssistantStagingReviewError(
            "Assistant staging review needs an explicit boolean pattern decision."
        )
    if not selected and not review_patterns:
        raise AssistantStagingReviewError(
            "Assistant staging review selected no cards or patterns; nothing would change."
        )

    try:
        resolved = AssistantContextBroker(config).proposal_context(
            proposal_resource_id
        )
    except AssistantContextError as exc:
        raise AssistantStagingReviewError(
            f"Could not resolve Assistant proposal {proposal_resource_id!r}: {exc}"
        ) from exc
    if resolved.proposal_kind != "source_extraction":
        raise AssistantStagingReviewError(
            "Assistant review approvals require a source-extraction proposal; "
            f"{resolved.proposal_kind!r} has a different review workflow."
        )

    try:
        panel = ReviewPanel.open(
            resolved.path,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
            collection_name=config.normalized_file.name,
        )
        if not hmac.compare_digest(
            panel.staging_fingerprint,
            resolved.proposal_sha256,
        ):
            raise AssistantStagingReviewError(
                "The selected proposal changed while Janki was resolving it; "
                "nothing was reviewed. Refresh the proposal list and try again."
            )
        validated = panel.validate_actions(selected, review_patterns)
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"Could not plan the exact staging review: {exc}"
        ) from exc

    records_by_id = {record.id: record for record in panel.records}
    chosen_records = [records_by_id[record_id] for record_id in validated]
    pattern = _pattern_value(panel.pattern_set) if review_patterns else None
    projection = {
        "action": "review_staging",
        "proposal": {
            "kind": resolved.proposal_kind,
            "resource_id": proposal_resource_id,
            "source_name": panel.source,
        },
        "selection": {
            "record_ids": list(validated),
            "records": [_record_value(record) for record in chosen_records],
            "review_patterns": review_patterns,
            "patterns": pattern,
        },
        "snapshots": {
            "patterns_sha256": panel.patterns_fingerprint,
            "staging_sha256": panel.staging_fingerprint,
        },
        "writes": {
            "pattern_store_review_mark": review_patterns,
            "staging_example_authority": list(validated),
        },
    }
    wire = _canonical_json(projection)
    plan = AssistantStagingReviewPlan(
        repository_root=config.root.resolve(),
        proposal_resource_id=proposal_resource_id,
        proposal_kind=resolved.proposal_kind,
        # Preserve the lexical no-follow name which supplied the bound bytes.
        # Calling resolve() after that read could follow a swapped symlink.
        proposal_path=resolved.path.absolute(),
        source_name=panel.source,
        record_ids=validated,
        review_patterns=review_patterns,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        projection_wire=wire,
        fingerprint=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
    )
    return _PreparedReview(plan=plan, panel=panel)


def plan_staging_review(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    record_ids: Sequence[str],
    review_patterns: bool,
) -> AssistantStagingReviewPlan:
    """Build one side-effect-free exact review confirmation."""

    return _prepare(
        config,
        proposal_resource_id=proposal_resource_id,
        record_ids=record_ids,
        review_patterns=review_patterns,
    ).plan


def execute_staging_review(
    config: ProjectConfig,
    expected: AssistantStagingReviewPlan,
) -> AssistantStagingReviewExecution:
    """Re-plan one confirmed review and delegate its exact monotonic writes."""

    if expected.repository_root != config.root.resolve():
        raise AssistantStagingReviewError(
            "The staging-review plan belongs to another repository."
        )
    fresh = _prepare(
        config,
        proposal_resource_id=expected.proposal_resource_id,
        record_ids=expected.record_ids,
        review_patterns=expected.review_patterns,
    )
    if not hmac.compare_digest(
        fresh.plan.fingerprint, expected.fingerprint
    ) or not hmac.compare_digest(
        fresh.plan.projection_wire.encode("utf-8"),
        expected.projection_wire.encode("utf-8"),
    ):
        raise AssistantStagingReviewError(
            "The staging review changed after confirmation; nothing was reviewed."
        )
    try:
        outcome = fresh.panel.submit(
            record_ids=fresh.plan.record_ids,
            review_patterns=fresh.plan.review_patterns,
        )
    except PartialReviewError as exc:
        raise AssistantStagingReviewPartialError(str(exc), exc.outcome) from exc
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The confirmed staging review could not be saved: {exc}"
        ) from exc
    return AssistantStagingReviewExecution(plan=fresh.plan, outcome=outcome)


# --- the prepared review phase ------------------------------------------------
#
# The confirmation above and the study finish reach the same writer from two
# different places. The Assistant action re-plans from an opaque proposal at
# click time; a study-job part is a staging path whose review decision was
# durably persisted before anything was written, and it resumes from that
# intent rather than from a returned value nobody saved.


@dataclass(frozen=True, slots=True)
class PreparedStagingReview:
    """One part's complete review payload, its coverage decision included."""

    repository_root: Path
    staging_path: Path
    part_name: str
    source_name: str
    review: PreparedReview

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute() or not self.staging_path.is_absolute():
            raise ValueError("Prepared staging-review paths must be absolute")
        if Path(self.review.staging.path).absolute() != self.staging_path:
            raise ValueError("A prepared staging review must bind its own staging path")

    @property
    def fingerprint(self) -> str:
        """One digest binding every byte of this prepared review."""
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": str(self.repository_root),
            "staging_path": str(self.staging_path),
            "part_name": self.part_name,
            "source_name": self.source_name,
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedStagingReview:
        try:
            return cls(
                repository_root=Path(str(raw["repository_root"])),
                staging_path=Path(str(raw["staging_path"])),
                part_name=str(raw["part_name"]),
                source_name=str(raw["source_name"]),
                review=PreparedReview.from_dict(raw["review"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AssistantStagingReviewError(
                f"A recorded staging-review intent is unreadable: {exc}"
            ) from exc


def prepare_staging_review(
    config: ProjectConfig,
    staging_path: Path,
    *,
    part_name: str = "",
    record_ids: Sequence[str],
    review_patterns: bool,
    coverage_approval: Mapping[str, Any] | None = None,
    expected_revision: str | None = None,
) -> PreparedStagingReview:
    """Compute one part's exact review payload without writing anything.

    ``coverage_approval`` is the owner's already-authorized payload with its
    frozen date; this composes it into the same staging after-text as the
    review marks. Preparation publishes nothing, so a plan that is never
    confirmed leaves no trace.
    """

    path = Path(staging_path).absolute()
    try:
        panel = ReviewPanel.open(
            path,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
            collection_name=config.normalized_file.name,
        )
        if expected_revision is not None and not hmac.compare_digest(
            panel.staging_fingerprint, expected_revision
        ):
            raise AssistantStagingReviewError(
                f"[staging-review-stale] {path.name} changed before its review "
                "could be prepared; nothing was reviewed."
            )
        review = panel.prepare(
            record_ids=_selected_ids(record_ids),
            review_patterns=review_patterns,
            coverage_approval=coverage_approval,
        )
    except AssistantStagingReviewError:
        raise
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"Could not prepare the exact staging review: {exc}"
        ) from exc
    return PreparedStagingReview(
        repository_root=config.root.resolve(),
        staging_path=panel.staging_path,
        part_name=part_name or panel.source,
        source_name=panel.source,
        review=review,
    )


def _bound_to(config: ProjectConfig, prepared: PreparedStagingReview) -> None:
    """Re-resolve every bound path through the **live** configuration.

    A repository root is not the binding. Two configurations in one root can
    name different pattern stores and different active staging directories, and
    a durable intent's paths are exactly the part of it an editor or a later
    configuration can move — so each component's role is resolved from this
    configuration and compared to the name the intent carries, before any
    effect. A substituted target holding byte-identical content passes every
    digest check; only this separates it from the file the owner reviewed.

    The comparison is lexical, and deliberately so: resolving both sides first
    would accept a symlink standing in for the reviewed file.
    ``review_target_paths`` proves the shapes as well as the names.
    """

    if prepared.repository_root != config.root.resolve():
        raise AssistantStagingReviewError(
            "The prepared staging review belongs to another repository."
        )
    active = _bound_part_to(
        config,
        staging_path=Path(prepared.review.staging.path),
        patterns_path=Path(prepared.review.patterns.path),
        source_name=prepared.source_name,
        record_ids=prepared.review.record_ids,
        expected_authority=prepared.review.expected_authority,
    )
    if prepared.staging_path != active:
        raise AssistantStagingReviewError(
            f"The prepared staging review names {prepared.staging_path}, which "
            f"this configuration resolves to {active}. Nothing was reviewed."
        )


def _bound_part_to(
    config: ProjectConfig,
    *,
    staging_path: Path,
    patterns_path: Path,
    source_name: str,
    record_ids: Sequence[str],
    expected_authority: Mapping[str, str],
) -> Path:
    """Re-resolve one part's two bound paths, and return the active staging one.

    Shared by the single review and every part of an aggregate one, so a batch
    is held to exactly the binding rules a single confirmed review is.
    """
    try:
        active, pattern_target = review_target_paths(
            staging_path,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
        )
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The prepared staging review does not name a reviewable staging "
            f"target in this configuration: {exc}"
        ) from exc
    for role, bound, expected in (
        ("staging file", staging_path, active),
        ("pattern store", patterns_path, pattern_target),
    ):
        if bound.absolute() != expected:
            raise AssistantStagingReviewError(
                f"The prepared staging review binds the {role} {bound}, but this "
                f"configuration resolves that role to {expected}. Nothing was "
                "reviewed."
            )
    if set(expected_authority) != set(record_ids):
        raise AssistantStagingReviewError(
            "The prepared staging review's expected authority does not name "
            "exactly the rows it reviews. Nothing was reviewed."
        )
    if not source_name.strip() or source_name != Path(source_name).name:
        raise AssistantStagingReviewError(
            "The prepared staging review's source is not a pattern-store key. "
            "Nothing was reviewed."
        )
    return active


def apply_prepared_staging_review_under_guard(
    config: ProjectConfig, prepared: PreparedStagingReview
) -> ReviewOutcome:
    """Write one prepared review while the caller holds the curation guard."""

    _bound_to(config, prepared)
    try:
        return apply_prepared_review(prepared.review)
    except PartialReviewError as exc:
        raise AssistantStagingReviewPartialError(str(exc), exc.outcome) from exc
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The prepared staging review could not be saved: {exc}"
        ) from exc


def apply_prepared_staging_review(
    config: ProjectConfig, prepared: PreparedStagingReview
) -> ReviewOutcome:
    """Take the shared curation guard, then write one prepared review.

    The guard is outermost, ahead of the two sorted review path locks, for the
    same reason promotion takes it: one global lock always ahead of the
    per-path locks is what stops a curation holding file A and waiting for B
    from deadlocking a writer holding B and waiting for A.
    """

    with study_curation.curation_guard(config):
        return apply_prepared_staging_review_under_guard(config, prepared)


def recover_prepared_staging_review_under_guard(
    config: ProjectConfig, prepared: PreparedStagingReview
) -> ReviewOutcome:
    """Finish one partially applied review, the curation guard already held."""

    _bound_to(config, prepared)
    try:
        return recover_prepared_review(prepared.review)
    except PartialReviewError as exc:
        raise AssistantStagingReviewPartialError(str(exc), exc.outcome) from exc
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The prepared staging review could not be recovered: {exc}"
        ) from exc


def recover_prepared_staging_review(
    config: ProjectConfig, prepared: PreparedStagingReview
) -> ReviewOutcome:
    """Take the shared curation guard, then finish one interrupted review."""

    with study_curation.curation_guard(config):
        return recover_prepared_staging_review_under_guard(config, prepared)


# --- the aggregate reviewed phase ---------------------------------------------
#
# One study job reviews several parts, and every part that marks a pattern set
# writes the *same* store. §7.2 and §7.6 give that store one final after-payload
# the whole batch shares; these are the configuration-bound entries around the
# writer that owns it.


@dataclass(frozen=True, slots=True)
class PreparedStagingReviewBatch:
    """Every part's review payload, over one aggregate pattern-store payload."""

    repository_root: Path
    batch: PreparedReviewBatch

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("Prepared staging-review paths must be absolute")
        for part in self.batch.parts:
            if not Path(part.staging.path).is_absolute():
                raise ValueError("Prepared staging-review paths must be absolute")

    @property
    def fingerprint(self) -> str:
        """One digest binding every byte of this prepared aggregate review."""
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": str(self.repository_root),
            "batch": self.batch.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedStagingReviewBatch:
        try:
            return cls(
                repository_root=Path(str(raw["repository_root"])),
                batch=PreparedReviewBatch.from_dict(raw["batch"]),
            )
        except (KeyError, TypeError, ValueError, JankiError) as exc:
            raise AssistantStagingReviewError(
                f"A recorded staging-review intent is unreadable: {exc}"
            ) from exc


def prepare_staging_review_batch(
    config: ProjectConfig, requests: Sequence[ReviewBatchRequest]
) -> PreparedStagingReviewBatch:
    """Compute every part's exact review payload without writing anything.

    The pattern store is captured once and every selected mark is composed over
    that one capture, so the batch carries a single final store payload rather
    than one per part. Preparation publishes nothing.
    """

    try:
        batch = prepare_review_batch(
            requests,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
            collection_name=config.normalized_file.name,
        )
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"Could not prepare the exact staging review: {exc}"
        ) from exc
    return PreparedStagingReviewBatch(
        repository_root=config.root.resolve(), batch=batch
    )


def _bound_batch_to(
    config: ProjectConfig, prepared: PreparedStagingReviewBatch
) -> None:
    """Re-resolve every part's bound paths through the live configuration."""

    if prepared.repository_root != config.root.resolve():
        raise AssistantStagingReviewError(
            "The prepared staging review belongs to another repository."
        )
    for part in prepared.batch.parts:
        _bound_part_to(
            config,
            staging_path=Path(part.staging.path),
            patterns_path=Path(prepared.batch.patterns.path),
            source_name=part.source,
            record_ids=part.record_ids,
            expected_authority=part.expected_authority,
        )


def apply_prepared_staging_review_batch_under_guard(
    config: ProjectConfig, prepared: PreparedStagingReviewBatch
) -> ReviewBatchOutcome:
    """Write one prepared aggregate review, the curation guard already held."""

    _bound_batch_to(config, prepared)
    try:
        return apply_prepared_review_batch(prepared.batch)
    except PartialReviewBatchError as exc:
        raise AssistantStagingReviewBatchPartialError(str(exc), exc.outcome) from exc
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The prepared staging review could not be saved: {exc}"
        ) from exc


def apply_prepared_staging_review_batch(
    config: ProjectConfig, prepared: PreparedStagingReviewBatch
) -> ReviewBatchOutcome:
    """Take the shared curation guard, then write one aggregate review."""

    with study_curation.curation_guard(config):
        return apply_prepared_staging_review_batch_under_guard(config, prepared)


def recover_prepared_staging_review_batch_under_guard(
    config: ProjectConfig, prepared: PreparedStagingReviewBatch
) -> ReviewBatchOutcome:
    """Finish one partially applied aggregate review, the guard already held."""

    _bound_batch_to(config, prepared)
    try:
        return recover_prepared_review_batch(prepared.batch)
    except PartialReviewBatchError as exc:
        raise AssistantStagingReviewBatchPartialError(str(exc), exc.outcome) from exc
    except JankiError as exc:
        raise AssistantStagingReviewError(
            f"The prepared staging review could not be recovered: {exc}"
        ) from exc


def recover_prepared_staging_review_batch(
    config: ProjectConfig, prepared: PreparedStagingReviewBatch
) -> ReviewBatchOutcome:
    """Take the shared curation guard, then finish one interrupted batch."""

    with study_curation.curation_guard(config):
        return recover_prepared_staging_review_batch_under_guard(config, prepared)
