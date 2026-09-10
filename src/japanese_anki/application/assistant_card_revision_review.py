"""Exact owner review for staged generic card revisions."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import staging
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records


class AssistantCardRevisionReviewError(JankiError):
    """A card-revision review is invalid or stale."""


@dataclass(frozen=True, slots=True)
class AssistantCardRevisionReviewPlan:
    repository_root: Path
    resource_id: str
    proposal_path: Path
    record_ids: tuple[str, ...]
    proposal_sha256: str
    projection_wire: str
    fingerprint: str

    @property
    def projection(self) -> Mapping[str, Any]:
        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


def _wire(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _prepare(
    config: ProjectConfig, resource_id: str, record_ids: Sequence[str]
) -> AssistantCardRevisionReviewPlan:
    selected = tuple(record_ids)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(not isinstance(item, str) or not item.strip() for item in selected)
    ):
        raise AssistantCardRevisionReviewError(
            "Card revision review needs unique selected card ids."
        )
    try:
        target = AssistantContextBroker(config).proposal_context(resource_id)
    except AssistantContextError as exc:
        raise AssistantCardRevisionReviewError(str(exc)) from exc
    if target.proposal_kind != "card_revision":
        raise AssistantCardRevisionReviewError(
            "This review action requires a card_revision proposal."
        )
    proposed, meta = staging.read_staging(target.path)
    if staging.CARD_REVISION_REVIEW_KEY in meta:
        raise AssistantCardRevisionReviewError(
            "This card revision already has durable owner review."
        )
    if len({record.id for record in proposed}) != len(proposed):
        raise AssistantCardRevisionReviewError(
            "The card revision proposal contains duplicate card ids."
        )
    canonical_records = load_records(config.normalized_file)
    if len({record.id for record in canonical_records}) != len(canonical_records):
        raise AssistantCardRevisionReviewError(
            "The canonical collection contains duplicate card ids."
        )
    current = {record.id: record for record in canonical_records}
    staged = {record.id: record for record in proposed}
    if any(item not in staged or item not in current for item in selected):
        raise AssistantCardRevisionReviewError(
            "Every selected card must belong to this proposal and the canonical collection."
        )
    revision = meta.get(staging.CARD_REVISION_KEY)
    proofs = meta.get(staging.FIELD_REPLACEMENTS_KEY)
    fields = revision.get("fields") if isinstance(revision, Mapping) else None
    proof_records = proofs.get("records") if isinstance(proofs, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(proof_records, Mapping):
        raise AssistantCardRevisionReviewError("The card revision has incomplete field provenance.")
    selected_set = set(selected)
    ordered_selected = tuple(record.id for record in proposed if record.id in selected_set)
    changes = []
    for record_id in ordered_selected:
        names = fields.get(record_id)
        bound = proof_records.get(record_id)
        if (
            not isinstance(names, list)
            or not isinstance(bound, Mapping)
            or set(names) != set(bound)
        ):
            raise AssistantCardRevisionReviewError(
                f"The card revision has incomplete fields for {record_id}."
            )
        old_wire = current[record_id].to_dict()
        new_wire = staged[record_id].to_dict()
        for name in names:
            if staging.replacement_fingerprint(current[record_id], name) != bound[name]:
                raise AssistantCardRevisionReviewError(
                    f"Canonical card {record_id}.{name} changed after the revision was staged."
                )
            changes.append(
                {
                    "record_id": record_id,
                    "expression": current[record_id].expression,
                    "field": name,
                    # `.get`, because the canonical serialization is sparse for
                    # the optional fields: a record with no `source_forms` key
                    # emits none, and an add-a-table proposal binds `null` as
                    # its old value. Indexing here raised `KeyError` on a
                    # staged proposal this planner is meant to review.
                    "old_value": old_wire.get(name),
                    "proposed_value": new_wire.get(name),
                }
            )
    projection = {
        "action": "review_staging",
        "proposal": {
            "kind": "card_revision",
            "resource_id": resource_id,
            "sha256": target.proposal_sha256,
        },
        "selection": {"record_ids": list(ordered_selected), "changes": changes},
        "writes": {
            "durable_owner_review": True,
            "unselected_rows_removed": [
                record.id for record in proposed if record.id not in selected_set
            ],
        },
    }
    projection_wire = _wire(projection)
    return AssistantCardRevisionReviewPlan(
        config.root.resolve(),
        resource_id,
        target.path.absolute(),
        ordered_selected,
        target.proposal_sha256,
        projection_wire,
        hashlib.sha256(projection_wire.encode()).hexdigest(),
    )


def plan_card_revision_review(
    config: ProjectConfig, *, resource_id: str, record_ids: Sequence[str]
) -> AssistantCardRevisionReviewPlan:
    return _prepare(config, resource_id, record_ids)


def execute_card_revision_review(
    config: ProjectConfig, expected: AssistantCardRevisionReviewPlan
) -> AssistantCardRevisionReviewPlan:
    if expected.repository_root != config.root.resolve():
        raise AssistantCardRevisionReviewError(
            "The card revision review belongs to another repository."
        )
    fresh = _prepare(config, expected.resource_id, expected.record_ids)
    if (
        not hmac.compare_digest(fresh.fingerprint, expected.fingerprint)
        or fresh.projection_wire != expected.projection_wire
    ):
        raise AssistantCardRevisionReviewError(
            "The card revision changed after confirmation; nothing was reviewed."
        )
    try:
        staging.record_card_revision_review(
            fresh.proposal_path, fresh.record_ids, expected_revision=fresh.proposal_sha256
        )
    except JankiError as exc:
        raise AssistantCardRevisionReviewError(
            f"The confirmed owner review could not be saved: {exc}"
        ) from exc
    return fresh
