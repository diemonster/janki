"""Exact repository-owner review for staged AI-enrichment proposals."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import staging
from japanese_anki.application import promotion
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records, read_bytes_bound
from japanese_anki.models import ExampleSentence


class AssistantAiEnrichmentReviewError(JankiError):
    """An AI-enrichment review is incomplete, invalid, or stale."""


@dataclass(frozen=True, slots=True)
class AssistantAiEnrichmentReviewPlan:
    """The exact old/proposed content selected for owner review."""

    repository_root: Path
    resource_id: str
    proposal_path: Path
    record_ids: tuple[str, ...]
    operation_id: str
    focus_resource_id: str | None
    proposal_sha256: str
    projection_wire: str
    fingerprint: str

    @property
    def projection(self) -> Mapping[str, Any]:
        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


def _wire(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AssistantAiEnrichmentReviewError(
            f"AI-enrichment review cannot be fingerprinted: {exc}"
        ) from exc


def _example(example: ExampleSentence) -> dict[str, str]:
    return {
        "register": example.register,
        "japanese": example.japanese,
        "furigana": example.furigana,
        "romaji": example.romaji,
        "english": example.english,
        "spoken_japanese": example.spoken_japanese,
        "audio": example.audio,
    }


def _prepare(
    config: ProjectConfig,
    resource_id: str,
    record_ids: Sequence[str],
) -> AssistantAiEnrichmentReviewPlan:
    selected = tuple(record_ids)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(not isinstance(item, str) or not item.strip() for item in selected)
    ):
        raise AssistantAiEnrichmentReviewError(
            "AI-enrichment review needs unique selected card ids."
        )
    try:
        target = AssistantContextBroker(config).proposal_context(resource_id)
    except AssistantContextError as exc:
        raise AssistantAiEnrichmentReviewError(str(exc)) from exc
    if target.proposal_kind != "ai_enrichment":
        raise AssistantAiEnrichmentReviewError(
            "This review action requires an ai_enrichment proposal."
        )
    try:
        captured = read_bytes_bound(target.path)
        text = captured.decode("utf-8", errors="strict")
        proposed, meta = staging.read_staging_text(text, source=str(target.path))
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantAiEnrichmentReviewError(
            f"Could not read the AI-enrichment proposal: {exc}"
        ) from exc
    proposal_sha256 = hashlib.sha256(captured).hexdigest()
    if not hmac.compare_digest(proposal_sha256, target.proposal_sha256):
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal changed while Janki resolved it."
        )
    if staging.AI_ENRICHMENT_REVIEW_KEY in meta:
        raise AssistantAiEnrichmentReviewError(
            "This AI-enrichment proposal already has durable owner review."
        )
    if len({record.id for record in proposed}) != len(proposed):
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal contains duplicate card ids."
        )
    try:
        promotion.staged_ai_enrichment(meta, proposed)
    except JankiError as exc:
        raise AssistantAiEnrichmentReviewError(str(exc)) from exc
    enrichment = meta.get(staging.AI_ENRICHMENT_KEY)
    fields = enrichment.get("fields") if isinstance(enrichment, Mapping) else None
    proofs = meta.get(staging.FIELD_REPLACEMENTS_KEY)
    proof_records = proofs.get("records") if isinstance(proofs, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(proof_records, Mapping):
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal has incomplete field provenance."
        )
    operation_id = meta.get("review_run_id")
    focus_resource_id = (
        enrichment.get("focus_resource_id")
        if isinstance(enrichment, Mapping)
        else None
    )
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal has no durable operation identity."
        )
    if focus_resource_id is not None and (
        not isinstance(focus_resource_id, str) or not focus_resource_id.strip()
    ):
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal has a malformed focused deck identity."
        )
    canonical = load_records(config.normalized_file)
    if len({record.id for record in canonical}) != len(canonical):
        raise AssistantAiEnrichmentReviewError(
            "The canonical collection contains duplicate card ids."
        )
    current = {record.id: record for record in canonical}
    staged = {record.id: record for record in proposed}
    if any(item not in staged or item not in current for item in selected):
        raise AssistantAiEnrichmentReviewError(
            "Every selected card must belong to this proposal and the canonical collection."
        )
    selected_set = set(selected)
    ordered = tuple(record.id for record in proposed if record.id in selected_set)
    changes: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for record_id in ordered:
        names = fields.get(record_id)
        bound = proof_records.get(record_id)
        if (
            not isinstance(names, list)
            or not names
            or not isinstance(bound, Mapping)
            or set(names) != set(bound)
        ):
            raise AssistantAiEnrichmentReviewError(
                f"The AI-enrichment proposal has incomplete fields for {record_id}."
            )
        old_wire = current[record_id].to_dict()
        proposed_wire = staged[record_id].to_dict()
        for name in names:
            if staging.replacement_fingerprint(current[record_id], name) != bound[name]:
                raise AssistantAiEnrichmentReviewError(
                    f"Canonical card {record_id}.{name} changed after enrichment was staged."
                )
            changes.append(
                {
                    "record_id": record_id,
                    "expression": current[record_id].expression,
                    "field": name,
                    "current": old_wire[name],
                    "proposed": proposed_wire[name],
                }
            )
        examples.append(
            {
                "record_id": record_id,
                "expression": current[record_id].expression,
                "current": [_example(item) for item in current[record_id].examples],
                "proposed": [_example(item) for item in staged[record_id].examples],
            }
        )
    projection = {
        "action": "review_ai_enrichment",
        "proposal": {
            "kind": "ai_enrichment",
            "resource_id": resource_id,
            "sha256": proposal_sha256,
            "operation_id": operation_id,
            "focus_resource_id": focus_resource_id,
        },
        "selection": {
            "record_ids": list(ordered),
            "changes": changes,
            "examples": examples,
        },
        "writes": {
            "durable_owner_review": True,
            "exact_example_authority": True,
            "unselected_rows_removed": [
                record.id for record in proposed if record.id not in selected_set
            ],
        },
    }
    projection_wire = _wire(projection)
    return AssistantAiEnrichmentReviewPlan(
        repository_root=config.root.resolve(),
        resource_id=resource_id,
        proposal_path=target.path.absolute(),
        record_ids=ordered,
        operation_id=operation_id,
        focus_resource_id=focus_resource_id,
        proposal_sha256=proposal_sha256,
        projection_wire=projection_wire,
        fingerprint=hashlib.sha256(projection_wire.encode("utf-8")).hexdigest(),
    )


def plan_ai_enrichment_review(
    config: ProjectConfig,
    *,
    resource_id: str,
    record_ids: Sequence[str],
) -> AssistantAiEnrichmentReviewPlan:
    """Plan an exact visible AI-enrichment review without writes."""

    return _prepare(config, resource_id, record_ids)


def execute_ai_enrichment_review(
    config: ProjectConfig,
    expected: AssistantAiEnrichmentReviewPlan,
) -> AssistantAiEnrichmentReviewPlan:
    """Re-plan and persist one confirmed exact AI-enrichment review."""

    if expected.repository_root != config.root.resolve():
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment review belongs to another repository."
        )
    fresh = _prepare(config, expected.resource_id, expected.record_ids)
    if (
        not hmac.compare_digest(fresh.fingerprint, expected.fingerprint)
        or fresh.projection_wire != expected.projection_wire
        or fresh.proposal_sha256 != expected.proposal_sha256
    ):
        raise AssistantAiEnrichmentReviewError(
            "The AI-enrichment proposal changed after confirmation; nothing was reviewed."
        )
    try:
        staging.record_ai_enrichment_review(
            fresh.proposal_path,
            fresh.record_ids,
            expected_revision=fresh.proposal_sha256,
        )
    except JankiError as exc:
        raise AssistantAiEnrichmentReviewError(
            f"The confirmed AI-enrichment review could not be saved: {exc}"
        ) from exc
    return fresh
