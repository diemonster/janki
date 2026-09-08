"""Plan-bound Assistant assignment of staged cards to one study deck.

The provider names one opaque source-extraction proposal, explicit stable card
ids, and one opaque configured-deck destination.  This module resolves those
ids locally, delegates every ownership decision to :mod:`assignment`, and
renders the exact tag changes for one confirmation.

Execution resolves and plans the same request again.  Only a byte-for-byte
matching plan reaches the staging compare-and-swap writer, so neither stale
proposal content nor changed deck selectors can ride an older confirmation.
No Japanese is interpreted or changed here; assignment changes only the tags
already authorized by the existing deck-assignment service.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import yaml

from japanese_anki import staging, status
from japanese_anki.application import assignment
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    ProposalContext,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import YAML_LOADER, exclusive_path_lock, read_bytes_bound
from japanese_anki.models import VocabularyRecord
from japanese_anki.workbench import review

__all__ = [
    "AssistantAssignmentError",
    "AssistantAssignmentExecution",
    "AssistantAssignmentPlan",
    "execute_assignment",
    "plan_assignment",
]


class AssistantAssignmentError(JankiError):
    """An Assistant assignment is not an exact, currently valid deck choice."""


@dataclass(frozen=True, slots=True)
class AssistantAssignmentPlan:
    """One exact staging edit plus its browser-safe confirmation projection."""

    repository_root: Path
    proposal_resource_id: str
    destination_resource_id: str
    instruction: str
    proposal_path: Path
    destination_path: Path
    source_name: str
    destination_name: str
    selected_record_ids: tuple[str, ...]
    staging_bytes: bytes
    staging_snapshot: str
    records: tuple[VocabularyRecord, ...]
    assigned_records: tuple[VocabularyRecord, ...]
    service_plans: tuple[assignment.DeckAssignmentPlan, ...]
    service_fingerprint: str
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            not self.repository_root.is_absolute()
            or self.repository_root != self.repository_root.resolve()
        ):
            raise ValueError("Assistant assignment root must be canonical and absolute")
        for path in (self.proposal_path, self.destination_path):
            if not path.is_absolute() or path != Path(os.path.normpath(path)):
                raise ValueError(
                    "Assistant assignment targets must be lexical absolute paths"
                )
        for path in (self.proposal_path, self.destination_path):
            try:
                path.relative_to(self.repository_root)
            except ValueError as exc:
                raise ValueError(
                    "Assistant assignment targets must stay in their repository"
                ) from exc
        for label, value in (
            ("proposal resource id", self.proposal_resource_id),
            ("destination resource id", self.destination_resource_id),
            ("instruction", self.instruction),
            ("source name", self.source_name),
            ("destination name", self.destination_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Assistant assignment {label} must be nonblank")
        if not self.selected_record_ids or len(self.selected_record_ids) != len(
            set(self.selected_record_ids)
        ):
            raise ValueError(
                "Assistant assignment must bind unique selected record ids"
            )
        if any(not isinstance(item, str) or not item for item in self.selected_record_ids):
            raise ValueError("Assistant assignment record ids must be nonblank text")
        if hashlib.sha256(self.staging_bytes).hexdigest() != self.staging_snapshot:
            raise ValueError("Assistant assignment staging bytes do not match their snapshot")
        if len(self.records) != len(self.assigned_records):
            raise ValueError("Assistant assignment must preserve the proposal row count")
        if tuple(plan.record_id for plan in self.service_plans) != self.selected_record_ids:
            raise ValueError("Assistant assignment service plans do not match the selection")
        if any(
            plan.destination.path.absolute() != self.destination_path
            for plan in self.service_plans
        ):
            raise ValueError("Assistant assignment service plans target another deck")
        if not _is_sha256(self.service_fingerprint):
            raise ValueError("Assistant assignment service fingerprint must be SHA-256")
        try:
            projection = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Assistant assignment projection must be JSON") from exc
        if not isinstance(projection, Mapping):
            raise ValueError("Assistant assignment projection must be a JSON object")
        if _canonical_json(projection) != self.projection_wire:
            raise ValueError("Assistant assignment projection must use canonical JSON")
        if projection.get("service_fingerprint") != self.service_fingerprint:
            raise ValueError("Assistant assignment projection does not bind its service plan")
        if hashlib.sha256(self.projection_wire.encode("utf-8")).hexdigest() != self.fingerprint:
            raise ValueError("Assistant assignment fingerprint does not bind its projection")

    @property
    def projection(self) -> Mapping[str, Any]:
        """Return the exact parsed value a confirmation card may render."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class AssistantAssignmentExecution:
    """The fresh matching plan after its one atomic staging-file write."""

    plan: AssistantAssignmentPlan
    assigned_record_ids: tuple[str, ...]


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
        raise AssistantAssignmentError(
            f"Assistant assignment plan cannot be fingerprinted: {exc}"
        ) from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    target = Path(os.path.normpath(path.absolute()))
    try:
        return target.relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantAssignmentError(
            f"Assistant assignment {label} escapes the configured repository."
        ) from exc


def _safe_name(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    name = PurePath(value.replace("\\", "/")).name
    return fallback if name in {"", ".", ".."} else name


def _selected_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise AssistantAssignmentError(
            "Assistant assignment record ids must be an explicit list of text."
        )
    selected = tuple(value)
    if not selected:
        raise AssistantAssignmentError(
            "Assistant assignment must select at least one staged card."
        )
    if any(not isinstance(item, str) or not item.strip() for item in selected):
        raise AssistantAssignmentError(
            "Every Assistant assignment record id must be nonblank text."
        )
    repeated = [item for item, count in Counter(selected).items() if count > 1]
    if repeated:
        raise AssistantAssignmentError(
            f"Assistant assignment record id {repeated[0]!r} was supplied twice."
        )
    return selected


def _resolve_proposal(config: ProjectConfig, resource_id: str) -> ProposalContext:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise AssistantAssignmentError(
            "Assistant assignment needs one nonblank proposal resource id."
        )
    try:
        target = AssistantContextBroker(config).proposal_context(resource_id)
    except AssistantContextError as exc:
        raise AssistantAssignmentError(
            f"Could not resolve Assistant proposal {resource_id!r}: {exc}"
        ) from exc
    if target.proposal_kind != "source_extraction":
        raise AssistantAssignmentError(
            "Assistant assignment accepts source-extraction proposals only."
        )
    return target


def _resolve_destination(config: ProjectConfig, resource_id: str) -> Path:
    """Resolve one opaque id to exactly one currently configured deck path."""

    if not isinstance(resource_id, str) or not resource_id.strip():
        raise AssistantAssignmentError(
            "Assistant assignment needs one nonblank destination deck resource id."
        )
    try:
        broker = AssistantContextBroker(config)
        configured = tuple(path.absolute() for path in status.deck_files(config))
        matches = tuple(
            path
            for path in configured
            if broker.resource_id_for_deck(path) == resource_id
        )
        if len(matches) == 1:
            broker.deck_context(resource_id)
    except (AssistantContextError, JankiError, OSError, ValueError) as exc:
        raise AssistantAssignmentError(
            f"Could not resolve destination deck resource {resource_id!r}: {exc}"
        ) from exc
    if len(matches) != 1:
        raise AssistantAssignmentError(
            f"Destination deck resource {resource_id!r} is not one exact current "
            "configured deck; choose it from a fresh deck list."
        )
    destination = matches[0]
    if sum(path.stem == destination.stem for path in configured) != 1:
        raise AssistantAssignmentError(
            f"Configured deck stem {destination.stem!r} is ambiguous; rename the "
            "decks before assigning cards."
        )
    return destination


def _captured_proposal(
    target: ProposalContext,
) -> tuple[bytes, tuple[VocabularyRecord, ...], Mapping[str, Any]]:
    try:
        snapshot = read_bytes_bound(target.path)
    except (JankiError, OSError) as exc:
        raise AssistantAssignmentError(
            f"Could not read the selected staging proposal: {exc}"
        ) from exc
    snapshot_sha = hashlib.sha256(snapshot).hexdigest()
    if snapshot_sha != target.proposal_sha256:
        raise AssistantAssignmentError(
            "The selected staging proposal changed while it was being resolved; "
            "request a fresh proposal list."
        )
    try:
        text = snapshot.decode("utf-8", errors="strict")
        records, metadata = staging.read_staging_text(text, source=str(target.path))
    except (JankiError, UnicodeError, ValueError) as exc:
        raise AssistantAssignmentError(
            f"Could not parse the selected staging proposal: {exc}"
        ) from exc
    return snapshot, tuple(records), metadata


def _assignment_value(plan: assignment.DeckAssignmentPlan) -> dict[str, object]:
    assigned_wire = _canonical_json(plan.assigned_record.to_dict())
    prospective_wire = _canonical_json(plan.prospective_record.to_dict())
    return {
        "record_id": plan.record_id,
        # What the selected row becomes. Equal to ``record_id`` for a shared
        # destination; a standalone deck's independent copy carries its own
        # identity, and the owner sees both before confirming anything.
        "target_record_id": plan.assigned_record.id,
        "expression": plan.assigned_record.expression,
        "reading": plan.assigned_record.reading,
        "existing_owner": plan.existing_owner,
        "assigned_record_sha256": hashlib.sha256(
            assigned_wire.encode("utf-8")
        ).hexdigest(),
        "prospective_record_sha256": hashlib.sha256(
            prospective_wire.encode("utf-8")
        ).hexdigest(),
        "tags": {
            "before": list(plan.tag_diff.before),
            "after": list(plan.tag_diff.after),
            "removed": list(plan.tag_diff.removed),
            "added": list(plan.tag_diff.added),
        },
        "proposal_occurrences": [
            {
                "source_name": _safe_name(item.imported_from, "unknown source"),
                "row": item.row,
                "source_type": item.source_type,
            }
            for item in plan.proposals
        ],
        "resulting_memberships": [
            {
                "deck": membership.name,
                "stem": membership.stem,
                "takes": membership.takes,
                "refusal": membership.refusal,
            }
            for membership in plan.memberships
        ],
    }


def _assignment_dependency_paths(
    config: ProjectConfig,
    proposal_path: Path,
) -> tuple[Path, ...]:
    """Return every mutable file whose value can change an assignment plan."""

    try:
        deck_paths = tuple(path.absolute() for path in status.deck_files(config))
        staging_paths = tuple(
            Path(os.path.abspath(path))
            for path in config.staging_dir.iterdir()
            if path.suffix.lower() in staging.STAGING_SUFFIXES
        )
    except (JankiError, OSError) as exc:
        raise AssistantAssignmentError(
            f"Could not enumerate the assignment's repository dependencies: {exc}"
        ) from exc

    dependencies = {
        config.normalized_file.absolute(),
        Path(os.path.abspath(proposal_path)),
        *deck_paths,
        *staging_paths,
    }
    for deck_path in deck_paths:
        try:
            raw = yaml.load(
                read_bytes_bound(deck_path).decode("utf-8", errors="strict"),
                Loader=YAML_LOADER,
            )
        except (JankiError, OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            raise AssistantAssignmentError(
                f"Could not inspect configured deck dependency {deck_path}: {exc}"
            ) from exc
        section = raw.get("deck") if isinstance(raw, Mapping) else None
        if isinstance(section, Mapping) and section.get("source"):
            dependencies.add(
                (deck_path.parent / str(section["source"])).resolve()
            )
    return tuple(sorted(dependencies, key=lambda path: os.fspath(path)))


@contextmanager
def _locked_assignment_dependencies(
    config: ProjectConfig,
    proposal_path: Path,
) -> Iterator[None]:
    """Hold a stable complete dependency census across re-plan and commit."""

    expected = _assignment_dependency_paths(config, proposal_path)
    real_paths = sorted(
        {Path(os.path.realpath(path)) for path in expected},
        key=os.fspath,
    )
    with ExitStack() as stack:
        for path in real_paths:
            stack.enter_context(exclusive_path_lock(path))
        if _assignment_dependency_paths(config, proposal_path) != expected:
            raise AssistantAssignmentError(
                "The assignment's deck or staging dependency set changed while "
                "Janki was locking it; refresh and review the current assignment."
            )
        yield


def _prepare(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    destination_resource_id: str,
    record_ids: Sequence[str],
    instruction: str,
) -> AssistantAssignmentPlan:
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantAssignmentError(
            "Assistant assignment needs the owner's nonblank instruction."
        )
    selected = _selected_ids(record_ids)
    target = _resolve_proposal(config, proposal_resource_id)
    destination = _resolve_destination(config, destination_resource_id)
    snapshot, records, metadata = _captured_proposal(target)

    by_id: dict[str, list[tuple[int, VocabularyRecord]]] = {}
    for index, record in enumerate(records):
        by_id.setdefault(record.id, []).append((index, record))
    selected_rows: list[tuple[int, VocabularyRecord]] = []
    for record_id in selected:
        matches = by_id.get(record_id, [])
        if len(matches) != 1:
            detail = "is absent" if not matches else "appears more than once"
            raise AssistantAssignmentError(
                f"Selected card {record_id!r} {detail} in this proposal; "
                "refresh and choose one unambiguous staged card."
            )
        selected_rows.append(matches[0])

    try:
        siblings = assignment.staged_sibling_proposals(
            config,
            target.path,
            frozenset(selected),
        )
        sibling_rows = tuple(siblings[record_id] for record_id in selected)
        matrix = assignment.plan_deck_assignments(
            config,
            [record for _index, record in selected_rows],
            destination_stems=(destination.stem,),
            sibling_proposals=sibling_rows,
        )
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantAssignmentError(
            f"Could not plan the exact staged-card assignment: {exc}"
        ) from exc

    plans: list[assignment.DeckAssignmentPlan] = []
    for record_id, attempts in zip(selected, matrix, strict=True):
        if len(attempts) != 1:
            raise AssistantAssignmentError(
                "The assignment planner returned an ambiguous destination matrix."
            )
        attempt = attempts[0]
        if attempt.plan is None:
            raise AssistantAssignmentError(
                f"Card {record_id!r} cannot be assigned to "
                f"{attempt.destination.name}: {attempt.refusal or 'the deck refused it'}."
            )
        if attempt.plan.destination.path.absolute() != destination:
            raise AssistantAssignmentError(
                "The assignment planner resolved the opaque destination to another deck."
            )
        plans.append(attempt.plan)

    updated = list(records)
    for (index, _record), plan in zip(selected_rows, plans, strict=True):
        updated[index] = plan.assigned_record

    assignment_values = [_assignment_value(plan) for plan in plans]
    try:
        destination_bytes = read_bytes_bound(destination)
    except (JankiError, OSError) as exc:
        raise AssistantAssignmentError(
            f"Could not read the destination deck definition: {exc}"
        ) from exc
    service_value = {
        "staging_sha256": hashlib.sha256(snapshot).hexdigest(),
        "destination_file": _relative(
            config, destination, label="destination deck"
        ),
        # The deck file's exact bytes, because its scope decides what identity
        # every assigned row becomes. A scope edited between the plan and the
        # confirmation must invalidate the plan, not silently re-target it.
        "destination_sha256": hashlib.sha256(destination_bytes).hexdigest(),
        "destination_scope_id": plans[0].destination.scope_id,
        "assignments": assignment_values,
    }
    service_wire = _canonical_json(service_value)
    service_fingerprint = hashlib.sha256(service_wire.encode("utf-8")).hexdigest()
    source_name = _safe_name(metadata.get("source_file"), target.path.name)
    destination_name = plans[0].destination.name
    projection = {
        "schema_version": 1,
        "kind": "assign_cards",
        "instruction": instruction,
        "proposal": {
            "resource_id": proposal_resource_id,
            "kind": target.proposal_kind,
            "source_name": source_name,
            "configured_file": _relative(
                config, target.path, label="staging proposal"
            ),
            "sha256": service_value["staging_sha256"],
        },
        "destination": {
            "resource_id": destination_resource_id,
            "name": destination_name,
            "stem": plans[0].destination.stem,
            "intake_tag": plans[0].destination.intake_tag,
            # Empty for an ordinary shared deck. A nonempty scope is what makes
            # every assigned row below an independent copy rather than the same
            # card gaining a tag, so the confirmation says so plainly.
            "scope_id": service_value["destination_scope_id"],
            "configured_file": service_value["destination_file"],
            "deck_sha256": service_value["destination_sha256"],
        },
        "selection": {
            "record_ids": list(selected),
            "target_record_ids": [plan.assigned_record.id for plan in plans],
            "assignments": assignment_values,
        },
        "writes": {
            "staging_proposal": _relative(
                config, target.path, label="staging proposal"
            ),
            "canonical_cards": False,
            "deck_definition": False,
        },
        "service_fingerprint": service_fingerprint,
    }
    wire = _canonical_json(projection)
    return AssistantAssignmentPlan(
        repository_root=config.root.resolve(),
        proposal_resource_id=proposal_resource_id,
        destination_resource_id=destination_resource_id,
        instruction=instruction,
        proposal_path=target.path.absolute(),
        destination_path=destination,
        source_name=source_name,
        destination_name=destination_name,
        selected_record_ids=selected,
        staging_bytes=snapshot,
        staging_snapshot=str(service_value["staging_sha256"]),
        records=records,
        assigned_records=tuple(updated),
        service_plans=tuple(plans),
        service_fingerprint=service_fingerprint,
        projection_wire=wire,
        fingerprint=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
    )


def plan_assignment(
    config: ProjectConfig,
    *,
    proposal_resource_id: str,
    destination_resource_id: str,
    record_ids: Sequence[str],
    instruction: str,
) -> AssistantAssignmentPlan:
    """Plan one exact multi-card staging assignment without writing."""

    return _prepare(
        config,
        proposal_resource_id=proposal_resource_id,
        destination_resource_id=destination_resource_id,
        record_ids=record_ids,
        instruction=instruction,
    )


def execute_assignment(
    config: ProjectConfig,
    expected: AssistantAssignmentPlan,
) -> AssistantAssignmentExecution:
    """Re-plan a confirmed assignment and atomically update its staging file."""

    if expected.repository_root != config.root.resolve():
        raise AssistantAssignmentError(
            "This Assistant assignment belongs to another repository."
        )
    try:
        with _locked_assignment_dependencies(config, expected.proposal_path):
            fresh = _prepare(
                config,
                proposal_resource_id=expected.proposal_resource_id,
                destination_resource_id=expected.destination_resource_id,
                record_ids=expected.selected_record_ids,
                instruction=expected.instruction,
            )
            if (
                fresh.fingerprint != expected.fingerprint
                or fresh.projection_wire != expected.projection_wire
            ):
                raise AssistantAssignmentError(
                    "The staged-card assignment changed after it was displayed; "
                    "refresh and review the current tag changes before confirming it."
                )
            text = staging.render_staging_update(
                fresh.staging_bytes,
                fresh.assigned_records,
                source=str(fresh.proposal_path),
            )
            review.bound_replace_under_lock(
                fresh.proposal_path,
                text,
                fresh.staging_bytes,
                label="staging file",
            )
    except AssistantAssignmentError:
        raise
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantAssignmentError(
            f"Could not save the exact staged-card assignment: {exc}"
        ) from exc
    return AssistantAssignmentExecution(
        plan=fresh,
        # What is now in the staging file. For a standalone destination these
        # are the copies' new identities, so the review, promotion and preview
        # that follow focus the rows this assignment actually wrote.
        assigned_record_ids=tuple(
            plan.assigned_record.id for plan in fresh.service_plans
        ),
    )
