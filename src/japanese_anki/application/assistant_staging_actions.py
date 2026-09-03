"""Plan-bound owner actions over one opaque live staging proposal.

These adapters expose existing staging operations to the conversational
surface without giving the model a path or a writer.  Planning captures the
exact no-follow proposal bytes, renders every consequence, and fingerprints
that projection.  Execution resolves and plans the action again before it
delegates to the same compare-and-swap writers used by the ordinary workbench.

Identity values, deletion selections, and coverage reasons are owner inputs.
Nothing in this module reads Japanese or manufactures one of those decisions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import ledger, staging
from japanese_anki.application import coverage as coverage_application
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    ProposalContext,
    assistant_record_value,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    exclusive_path_lock,
    load_records_snapshot,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.workbench import reidentify, review

__all__ = [
    "AssistantCoverageApprovalExecution",
    "AssistantCoverageApprovalPlan",
    "AssistantReidentificationExecution",
    "AssistantReidentificationPlan",
    "AssistantStagedDeletionExecution",
    "AssistantStagedDeletionPlan",
    "AssistantStagingActionError",
    "execute_coverage_approval",
    "execute_reidentification",
    "execute_staged_deletion",
    "plan_coverage_approval",
    "plan_reidentification",
    "plan_staged_deletion",
]


class AssistantStagingActionError(JankiError):
    """A staged owner action is unsafe, stale, or incompletely specified."""


@dataclass(frozen=True, slots=True)
class _OpenedProposal:
    target: ProposalContext
    panel: review.ReviewPanel


@dataclass(frozen=True, slots=True)
class AssistantStagedDeletionPlan:
    """Exact rows and replacement bytes behind one destructive confirmation."""

    repository_root: Path
    proposal_resource_id: str
    instruction: str
    proposal_kind: str
    proposal_path: Path
    record_ids: tuple[str, ...]
    staging_bytes: bytes
    replacement_text: str
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_common_plan(
            self.repository_root,
            self.proposal_resource_id,
            self.instruction,
            self.proposal_path,
            self.projection_wire,
            self.fingerprint,
        )
        if not self.record_ids or len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("Staged deletion needs unique selected record ids")
        if any(not item for item in self.record_ids):
            raise ValueError("Staged deletion record ids must be nonblank")
        if not isinstance(self.staging_bytes, bytes) or not self.staging_bytes:
            raise ValueError("Staged deletion needs captured proposal bytes")
        if not isinstance(self.replacement_text, str) or not self.replacement_text:
            raise ValueError("Staged deletion needs exact replacement text")

    @property
    def projection(self) -> Mapping[str, Any]:
        return _parsed_projection(self.projection_wire)


@dataclass(frozen=True, slots=True)
class AssistantStagedDeletionExecution:
    plan: AssistantStagedDeletionPlan
    removed_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssistantReidentificationPlan:
    """One explicit old/new staged identity and its exact consequences."""

    repository_root: Path
    proposal_resource_id: str
    instruction: str
    proposal_kind: str
    proposal_path: Path
    record_id: str
    new_expression: str
    new_reading: str
    staging_bytes: bytes
    replacement_text: str
    service_plan: reidentify.Reidentification
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_common_plan(
            self.repository_root,
            self.proposal_resource_id,
            self.instruction,
            self.proposal_path,
            self.projection_wire,
            self.fingerprint,
        )
        for label, value in (
            ("record id", self.record_id),
            ("new expression", self.new_expression),
            ("new reading", self.new_reading),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Staged reidentification {label} must be nonblank")
        if self.service_plan.old_id != self.record_id:
            raise ValueError("Staged reidentification service plan targets another row")
        if (
            self.service_plan.new_expression != self.new_expression
            or self.service_plan.new_reading != self.new_reading
        ):
            raise ValueError("Staged reidentification values are not service-bound")
        if not isinstance(self.staging_bytes, bytes) or not self.staging_bytes:
            raise ValueError("Staged reidentification needs captured proposal bytes")
        if not isinstance(self.replacement_text, str) or not self.replacement_text:
            raise ValueError("Staged reidentification needs exact replacement text")

    @property
    def projection(self) -> Mapping[str, Any]:
        return _parsed_projection(self.projection_wire)


@dataclass(frozen=True, slots=True)
class AssistantReidentificationExecution:
    plan: AssistantReidentificationPlan
    old_record_id: str
    new_record_id: str


@dataclass(frozen=True, slots=True)
class AssistantCoverageApprovalPlan:
    """One explicit owner coverage reason over an exact rendered account."""

    repository_root: Path
    proposal_resource_id: str
    instruction: str
    proposal_path: Path
    reason: str
    decision: coverage_application.CoverageDecision
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_common_plan(
            self.repository_root,
            self.proposal_resource_id,
            self.instruction,
            self.proposal_path,
            self.projection_wire,
            self.fingerprint,
        )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Coverage approval needs a nonblank owner reason")
        if self.decision.staging_path.absolute() != self.proposal_path:
            raise ValueError("Coverage approval decision targets another proposal")
        if self.decision.state != "ready" or self.decision.replace_existing:
            raise ValueError("Coverage approval plan is not a fresh owner decision")

    @property
    def projection(self) -> Mapping[str, Any]:
        return _parsed_projection(self.projection_wire)


@dataclass(frozen=True, slots=True)
class AssistantCoverageApprovalExecution:
    plan: AssistantCoverageApprovalPlan
    staging_path: Path


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
        raise AssistantStagingActionError(
            f"Assistant staging action cannot be fingerprinted: {exc}"
        ) from exc


def _parsed_projection(wire: str) -> Mapping[str, Any]:
    value = json.loads(wire)
    assert isinstance(value, Mapping)
    return value


def _validate_common_plan(
    repository_root: Path,
    proposal_resource_id: str,
    instruction: str,
    proposal_path: Path,
    projection_wire: str,
    fingerprint: str,
) -> None:
    if repository_root != repository_root.resolve():
        raise ValueError("Assistant staging repository root must be canonical")
    if not proposal_path.is_absolute() or proposal_path != proposal_path.absolute():
        raise ValueError("Assistant staging proposal path must be lexical absolute")
    try:
        proposal_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("Assistant staging proposal must stay in its repository") from exc
    for label, value in (
        ("proposal resource id", proposal_resource_id),
        ("instruction", instruction),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Assistant staging {label} must be nonblank")
    try:
        projection = json.loads(projection_wire)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Assistant staging projection must be JSON") from exc
    if not isinstance(projection, Mapping) or _canonical_json(projection) != projection_wire:
        raise ValueError("Assistant staging projection must be one canonical JSON object")
    if hashlib.sha256(projection_wire.encode("utf-8")).hexdigest() != fingerprint:
        raise ValueError("Assistant staging fingerprint does not bind its projection")


def _selected_ids(record_ids: Sequence[str], *, action: str) -> tuple[str, ...]:
    if isinstance(record_ids, str | bytes) or not isinstance(record_ids, Sequence):
        raise AssistantStagingActionError(
            f"Assistant {action} record ids must be an explicit list of text."
        )
    selected = tuple(record_ids)
    if not selected:
        raise AssistantStagingActionError(
            f"Assistant {action} selected no cards; nothing would change."
        )
    if any(not isinstance(item, str) or not item.strip() for item in selected):
        raise AssistantStagingActionError(
            f"Every Assistant {action} record id must be nonblank text."
        )
    repeated = [item for item, count in Counter(selected).items() if count > 1]
    if repeated:
        raise AssistantStagingActionError(
            f"Assistant {action} record id {repeated[0]!r} was supplied twice."
        )
    return selected


def _open_proposal(
    config: ProjectConfig,
    proposal_resource_id: str,
    *,
    allowed_kinds: frozenset[str],
    locked_proposal_path: Path | None = None,
) -> _OpenedProposal:
    if not isinstance(proposal_resource_id, str) or not proposal_resource_id.strip():
        raise AssistantStagingActionError(
            "Assistant staging action needs one opaque proposal resource id."
        )
    try:
        target = AssistantContextBroker(config).proposal_context(proposal_resource_id)
    except AssistantContextError as exc:
        raise AssistantStagingActionError(
            f"Could not resolve Assistant proposal {proposal_resource_id!r}: {exc}"
        ) from exc
    if target.proposal_kind not in allowed_kinds:
        raise AssistantStagingActionError(
            f"This action does not support {target.proposal_kind!r} proposals."
        )
    if (
        locked_proposal_path is not None
        and target.path.absolute() != locked_proposal_path.absolute()
    ):
        raise AssistantStagingActionError(
            "The selected proposal resolved to another path after confirmation."
        )
    try:
        opener = (
            review.ReviewPanel.open_under_lock
            if locked_proposal_path is not None
            else review.ReviewPanel.open
        )
        panel = opener(
            target.path,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
            collection_name=config.normalized_file.name,
        )
    except JankiError as exc:
        raise AssistantStagingActionError(
            f"Could not open the selected staging proposal: {exc}"
        ) from exc
    if not _is_sha256(target.proposal_sha256) or not hmac.compare_digest(
        target.proposal_sha256,
        panel.staging_fingerprint,
    ):
        raise AssistantStagingActionError(
            "The selected proposal changed while Janki was resolving it; refresh "
            "the proposal list before continuing."
        )
    return _OpenedProposal(target=target, panel=panel)


def _relative(config: ProjectConfig, path: Path) -> str:
    try:
        return path.absolute().relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantStagingActionError(
            "The selected staging proposal escapes the configured repository."
        ) from exc


def _fingerprinted(value: Mapping[str, Any]) -> tuple[str, str]:
    wire = _canonical_json(value)
    return wire, hashlib.sha256(wire.encode("utf-8")).hexdigest()


def plan_staged_deletion(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    record_ids: Sequence[str],
    instruction: str,
) -> AssistantStagedDeletionPlan:
    """Render an exact destructive staging-row batch without writing it."""

    selected = _selected_ids(record_ids, action="staged deletion")
    opened = _open_proposal(
        config,
        proposal_resource_id,
        allowed_kinds=frozenset(
            {"source_extraction", "ai_enrichment", "card_revision"}
        ),
    )
    panel = opened.panel
    counts = Counter(record.id for record in panel.records)
    unknown = [record_id for record_id in selected if counts[record_id] == 0]
    ambiguous = [record_id for record_id in selected if counts[record_id] > 1]
    if unknown:
        raise AssistantStagingActionError(
            f"Record {unknown[0]!r} is not in the selected proposal."
        )
    if ambiguous:
        raise AssistantStagingActionError(
            f"Record {ambiguous[0]!r} occurs more than once; select that row in the "
            "ordinary staging editor so its occurrence is unambiguous."
        )
    selected_set = set(selected)
    keep = tuple(record.id not in selected_set for record in panel.records)
    replacement = staging.render_staging_prune(
        panel.staging_bytes,
        keep,
        source=str(panel.staging_path),
    )
    if replacement is None:
        raise AssistantStagingActionError("The staged deletion would remove nothing.")
    records_by_id = {record.id: record for record in panel.records}
    projection = {
        "schema_version": 1,
        "kind": "delete_staged_cards",
        "instruction": instruction,
        "target": {
            "proposal_kind": opened.target.proposal_kind,
            "resource_id": proposal_resource_id,
            "staging_proposal": _relative(config, panel.staging_path),
        },
        "selection": {
            "record_ids": list(selected),
            "records": [
                assistant_record_value(records_by_id[record_id])
                for record_id in selected
            ],
        },
        "snapshots": {
            "before_sha256": panel.staging_fingerprint,
            "after_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
        },
        "effects": {
            "rows_before": len(panel.records),
            "rows_removed": len(selected),
            "rows_after": len(panel.records) - len(selected),
            "canonical_cards_changed": False,
            "paid_provider_call": False,
        },
    }
    wire, fingerprint = _fingerprinted(projection)
    return AssistantStagedDeletionPlan(
        repository_root=config.root.resolve(),
        proposal_resource_id=proposal_resource_id,
        instruction=instruction,
        proposal_kind=opened.target.proposal_kind,
        proposal_path=panel.staging_path.absolute(),
        record_ids=selected,
        staging_bytes=panel.staging_bytes,
        replacement_text=replacement,
        projection_wire=wire,
        fingerprint=fingerprint,
    )


def execute_staged_deletion(
    config: ProjectConfig,
    expected: AssistantStagedDeletionPlan,
) -> AssistantStagedDeletionExecution:
    """Re-plan and compare before using the workbench staging CAS writer."""

    if expected.repository_root != config.root.resolve():
        raise AssistantStagingActionError(
            "The staged-deletion plan belongs to another repository."
        )
    fresh = plan_staged_deletion(
        config,
        proposal_resource_id=expected.proposal_resource_id,
        record_ids=expected.record_ids,
        instruction=expected.instruction,
    )
    if not hmac.compare_digest(
        fresh.fingerprint,
        expected.fingerprint,
    ) or not hmac.compare_digest(
        fresh.projection_wire.encode("utf-8"),
        expected.projection_wire.encode("utf-8"),
    ):
        raise AssistantStagingActionError(
            "The staged deletion changed after confirmation; nothing was removed."
        )
    try:
        review.bound_replace(
            fresh.proposal_path,
            fresh.replacement_text,
            fresh.staging_bytes,
            label="staging file",
        )
    except review.IndeterminateWriteError as exc:
        raise _indeterminate_staging_write("staged deletion", exc) from exc
    except JankiError as exc:
        raise AssistantStagingActionError(
            f"The confirmed staged deletion could not be saved: {exc}"
        ) from exc
    return AssistantStagedDeletionExecution(
        plan=fresh,
        removed_record_ids=fresh.record_ids,
    )


def _existing_records(config: ProjectConfig) -> list[VocabularyRecord]:
    records, _revision = load_records_snapshot(config.normalized_file)
    return records


def _exported_ids(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
) -> frozenset[str]:
    try:
        wire = read_bytes_bound(config.ledger_file)
    except FileNotFoundError:
        return frozenset()
    book = ledger.load_snapshot(config.ledger_file, wire)
    return book.ever_exported(record.id for record in records)


def _indeterminate_staging_write(
    action: str,
    exc: review.IndeterminateWriteError,
) -> AssistantStagingActionError:
    if exc.intended_bytes_are_live:
        status = (
            f"The confirmed {action} did reach the staging file, but Janki could "
            "not finish verifying the write"
        )
    else:
        status = (
            f"Janki could not determine whether the confirmed {action} reached "
            "the staging file"
        )
    return AssistantStagingActionError(
        f"{status}. Reload and inspect the current proposal before trying again: "
        f"{exc}"
    )


def _plan_reidentification(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    record_id: str,
    new_expression: str,
    new_reading: str,
    instruction: str,
    locked_proposal_path: Path | None = None,
) -> AssistantReidentificationPlan:
    """Render the existing identity consequences for one explicit owner choice."""

    selected = _selected_ids((record_id,), action="reidentification")
    opened = _open_proposal(
        config,
        proposal_resource_id,
        allowed_kinds=frozenset({"source_extraction"}),
        locked_proposal_path=locked_proposal_path,
    )
    panel = opened.panel
    if not panel.reidentifiable:
        raise AssistantStagingActionError(
            "This model-authored proposal is bound to its existing card identity."
        )
    positions = [
        index for index, record in enumerate(panel.records) if record.id == selected[0]
    ]
    if len(positions) != 1:
        raise AssistantStagingActionError(
            "The selected staged card is missing or its identity is ambiguous."
        )
    try:
        service = reidentify.plan_reidentification(
            panel.records,
            positions[0],
            new_expression,
            new_reading,
            existing=_existing_records(config),
            exported_ids=_exported_ids(config, panel.records),
        )
        updated = reidentify.apply_reidentification(panel.records, service)
        replacement = staging.render_staging_update(
            panel.staging_bytes,
            updated,
            source=str(panel.staging_path),
        )
    except JankiError as exc:
        raise AssistantStagingActionError(
            f"Could not plan the staged identity change: {exc}"
        ) from exc
    if not service.is_change:
        raise AssistantStagingActionError("The staged identity would not change.")
    projection = {
        "schema_version": 1,
        "kind": "reidentify_staged_card",
        "instruction": instruction,
        "target": {
            "proposal_kind": opened.target.proposal_kind,
            "resource_id": proposal_resource_id,
            "staging_proposal": _relative(config, panel.staging_path),
        },
        "identity": {
            "old": {
                "record_id": service.old_id,
                "expression": service.old_expression,
                "reading": service.old_reading,
            },
            "new": {
                "record_id": service.new_id,
                "expression": service.new_expression,
                "reading": service.new_reading,
            },
            "neighbours": [
                {
                    "record_id": item.record_id,
                    "expression": item.expression,
                    "reading": item.reading,
                    "relation": item.relation,
                    "where": item.where,
                }
                for item in service.neighbours
            ],
            "consequences": service.consequences(),
            "was_exported": service.was_exported,
        },
        "snapshots": {
            "before_sha256": panel.staging_fingerprint,
            "after_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
        },
        "effects": {
            "japanese_examples_changed": False,
            "canonical_cards_changed": False,
            "paid_provider_call": False,
        },
    }
    wire, fingerprint = _fingerprinted(projection)
    return AssistantReidentificationPlan(
        repository_root=config.root.resolve(),
        proposal_resource_id=proposal_resource_id,
        instruction=instruction,
        proposal_kind=opened.target.proposal_kind,
        proposal_path=panel.staging_path.absolute(),
        record_id=service.old_id,
        new_expression=service.new_expression,
        new_reading=service.new_reading,
        staging_bytes=panel.staging_bytes,
        replacement_text=replacement,
        service_plan=service,
        projection_wire=wire,
        fingerprint=fingerprint,
    )


def plan_reidentification(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    record_id: str,
    new_expression: str,
    new_reading: str,
    instruction: str,
) -> AssistantReidentificationPlan:
    """Render the existing identity consequences for one explicit owner choice."""

    return _plan_reidentification(
        config,
        proposal_resource_id=proposal_resource_id,
        record_id=record_id,
        new_expression=new_expression,
        new_reading=new_reading,
        instruction=instruction,
    )


def execute_reidentification(
    config: ProjectConfig,
    expected: AssistantReidentificationPlan,
) -> AssistantReidentificationExecution:
    """Re-plan one explicit identity decision and save it through the staging CAS."""

    if expected.repository_root != config.root.resolve():
        raise AssistantStagingActionError(
            "The reidentification plan belongs to another repository."
        )
    try:
        dependencies = sorted(
            {
                Path(os.path.realpath(expected.proposal_path)),
                Path(os.path.realpath(config.patterns_file)),
                Path(os.path.realpath(config.normalized_file)),
                Path(os.path.realpath(config.ledger_file)),
            },
            key=os.fspath,
        )
        with ExitStack() as locks:
            for path in dependencies:
                locks.enter_context(exclusive_path_lock(path))
            fresh = _plan_reidentification(
                config,
                proposal_resource_id=expected.proposal_resource_id,
                record_id=expected.record_id,
                new_expression=expected.new_expression,
                new_reading=expected.new_reading,
                instruction=expected.instruction,
                locked_proposal_path=expected.proposal_path,
            )
            if not hmac.compare_digest(
                fresh.fingerprint,
                expected.fingerprint,
            ) or not hmac.compare_digest(
                fresh.projection_wire.encode("utf-8"),
                expected.projection_wire.encode("utf-8"),
            ):
                raise AssistantStagingActionError(
                    "The staged identity plan changed after confirmation; nothing "
                    "was changed."
                )
            review.bound_replace_under_lock(
                fresh.proposal_path,
                fresh.replacement_text,
                fresh.staging_bytes,
                label="staging file",
            )
    except review.IndeterminateWriteError as exc:
        raise _indeterminate_staging_write("staged identity change", exc) from exc
    except AssistantStagingActionError:
        raise
    except JankiError as exc:
        raise AssistantStagingActionError(
            f"The confirmed staged identity could not be saved: {exc}"
        ) from exc
    return AssistantReidentificationExecution(
        plan=fresh,
        old_record_id=fresh.service_plan.old_id,
        new_record_id=fresh.service_plan.new_id,
    )


def plan_coverage_approval(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    reason: str,
    instruction: str,
) -> AssistantCoverageApprovalPlan:
    """Render one current coverage account and explicit repository-owner reason."""

    if not isinstance(reason, str) or not reason.strip():
        raise AssistantStagingActionError(
            "Owner coverage approval needs the owner's nonblank reason."
        )
    opened = _open_proposal(
        config,
        proposal_resource_id,
        allowed_kinds=frozenset({"source_extraction"}),
    )
    try:
        decision = coverage_application.plan_coverage(
            config,
            opened.panel.staging_path,
        )
        preview = coverage_application.project_coverage(decision)
    except JankiError as exc:
        raise AssistantStagingActionError(
            f"Could not plan owner coverage approval: {exc}"
        ) from exc
    if decision.state != "ready" or decision.replace_existing:
        detail = decision.detail or "coverage is not awaiting a fresh owner decision"
        raise AssistantStagingActionError(
            f"This proposal does not offer owner coverage approval: {detail}."
        )
    if not hmac.compare_digest(
        decision.staging_revision,
        opened.panel.staging_fingerprint,
    ):
        raise AssistantStagingActionError(
            "The coverage account changed while it was being planned; refresh it."
        )
    projection = {
        "schema_version": 1,
        "kind": "approve_coverage",
        "instruction": instruction,
        "target": {
            "proposal_kind": opened.target.proposal_kind,
            "resource_id": proposal_resource_id,
            "source_name": preview.source_file,
            "staging_proposal": _relative(config, opened.panel.staging_path),
        },
        "coverage": {
            "account": preview.account,
            "candidate_units": preview.candidate_units,
            "source_units": preview.source_units,
            "staging_sha256": preview.staging_fingerprint,
        },
        "owner_decision": {
            "authority": "repository-owner",
            "reason": reason.strip(),
        },
        "effects": {
            "writes_coverage_approval_to_staging": True,
            "promotes_cards": False,
            "paid_provider_call": False,
        },
    }
    wire, fingerprint = _fingerprinted(projection)
    return AssistantCoverageApprovalPlan(
        repository_root=config.root.resolve(),
        proposal_resource_id=proposal_resource_id,
        instruction=instruction,
        proposal_path=opened.panel.staging_path.absolute(),
        reason=reason.strip(),
        decision=decision,
        projection_wire=wire,
        fingerprint=fingerprint,
    )


def execute_coverage_approval(
    config: ProjectConfig,
    expected: AssistantCoverageApprovalPlan,
) -> AssistantCoverageApprovalExecution:
    """Re-plan and record one exact owner coverage decision."""

    if expected.repository_root != config.root.resolve():
        raise AssistantStagingActionError(
            "The coverage-approval plan belongs to another repository."
        )
    fresh = plan_coverage_approval(
        config,
        proposal_resource_id=expected.proposal_resource_id,
        reason=expected.reason,
        instruction=expected.instruction,
    )
    if not hmac.compare_digest(
        fresh.fingerprint,
        expected.fingerprint,
    ) or not hmac.compare_digest(
        fresh.projection_wire.encode("utf-8"),
        expected.projection_wire.encode("utf-8"),
    ):
        raise AssistantStagingActionError(
            "The coverage approval changed after confirmation; nothing was approved."
        )
    try:
        path = coverage_application.approve_coverage_as_owner(
            config,
            fresh.decision,
            reason=fresh.reason,
        )
    except JankiError as exc:
        raise AssistantStagingActionError(
            "Janki could not prove whether the confirmed owner coverage approval "
            "was saved. Reload and inspect the current coverage decision before "
            f"trying again: {exc}"
        ) from exc
    return AssistantCoverageApprovalExecution(plan=fresh, staging_path=path)
