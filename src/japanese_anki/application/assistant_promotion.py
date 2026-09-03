"""Plan-bound promotion for one opaque Assistant proposal resource.

This module does not review cards, accept coverage, choose an identity, or
assign a deck.  It only exposes the existing promotion transaction after the
repository already says those owner decisions are durable.  The Assistant
names an opaque proposal id; the local context broker resolves the current
staging file, and :mod:`promotion` remains the sole writer.

Confirmation and execution are separate on purpose.  Execution resolves the
opaque id again, makes the same offline plan again, compares its exact
canonical projection, then performs the ordinary jpdb reading check and calls
the shared promotion writer.  A stale or newly incomplete review cannot ride
an older confirmation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal

from japanese_anki import jpdb
from japanese_anki.application import promotion as promotion_application
from japanese_anki.application import promotion_action
from japanese_anki.application.assignment import DeckOwnershipEvaluation
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    ProposalContext,
    assistant_record_value,
)
from japanese_anki.application.authority import example_authority_state
from japanese_anki.application.journey import (
    GRAMMAR_ONLY,
    READY_TO_ADD,
    source_journeys,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

__all__ = [
    "AssistantPromotionError",
    "AssistantPromotionExecution",
    "AssistantPromotionPlan",
    "execute_promotion_action",
    "plan_promotion_action",
]


ProposalKind = Literal["source_extraction", "ai_enrichment", "card_revision"]


class AssistantPromotionError(JankiError):
    """An Assistant proposal is not exactly ready for ordinary promotion."""


@dataclass(frozen=True, slots=True)
class AssistantPromotionPlan:
    """One current promotion decision plus its exact confirmation projection."""

    repository_root: Path
    resource_id: str
    instruction: str
    proposal_kind: ProposalKind
    proposal_path: Path
    source: str
    service_fingerprint: str
    projection_wire: str
    fingerprint: str
    decision: promotion_application.PromotionDecision

    def __post_init__(self) -> None:
        if (
            not self.repository_root.is_absolute()
            or not self.proposal_path.is_absolute()
        ):
            raise ValueError("Assistant promotion paths must be absolute")
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Assistant promotion repository root must be canonical")
        if self.proposal_path != self.proposal_path.absolute():
            raise ValueError("Assistant promotion proposal path must be absolute")
        try:
            self.proposal_path.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError(
                "Assistant promotion proposal must stay in its repository"
            ) from exc
        if self.proposal_kind not in {
            "source_extraction",
            "ai_enrichment",
            "card_revision",
        }:
            raise ValueError("Assistant promotion proposal kind is unsupported")
        for label, value in (
            ("resource id", self.resource_id),
            ("instruction", self.instruction),
            ("source", self.source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Assistant promotion {label} must be nonblank")
        if not _is_sha256(self.service_fingerprint):
            raise ValueError("Assistant promotion service fingerprint must be SHA-256")
        if (
            promotion_action.promotion_preview_fingerprint(self.decision)
            != self.service_fingerprint
        ):
            raise ValueError(
                "Assistant promotion decision does not match its service fingerprint"
            )
        if self.decision.staging_path != self.proposal_path:
            raise ValueError("Assistant promotion decision targets another proposal")
        try:
            projection = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Assistant promotion projection must be JSON") from exc
        if not isinstance(projection, Mapping):
            raise ValueError("Assistant promotion projection must be a JSON object")
        if _canonical_json(projection) != self.projection_wire:
            raise ValueError("Assistant promotion projection must use canonical JSON")
        if _sha256(self.projection_wire.encode("utf-8")) != self.fingerprint:
            raise ValueError(
                "Assistant promotion fingerprint does not bind its projection"
            )

    @property
    def projection(self) -> Mapping[str, Any]:
        """The exact parsed value a confirmation card may render."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class AssistantPromotionExecution:
    """The fresh confirmed plan and the sole promotion writer's result."""

    plan: AssistantPromotionPlan
    result: promotion_application.PromotionExecutionResult


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise AssistantPromotionError(
            f"Assistant promotion plan cannot be fingerprinted: {exc}"
        ) from exc


def _safe_name(value: str) -> str:
    name = PurePath(value.replace("\\", "/")).name if value else ""
    return "" if name in {".", ".."} else name


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    try:
        return path.absolute().relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantPromotionError(
            f"Assistant promotion {label} escapes the configured repository."
        ) from exc


def _resolve_proposal(config: ProjectConfig, resource_id: str) -> ProposalContext:
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise AssistantPromotionError(
            "Assistant promotion needs one nonblank opaque proposal resource id."
        )
    try:
        return AssistantContextBroker(config).proposal_context(resource_id)
    except AssistantContextError as exc:
        raise AssistantPromotionError(
            f"Could not resolve Assistant proposal {resource_id!r}: {exc}"
        ) from exc


def _ready_journey(config: ProjectConfig, target: ProposalContext) -> str:
    """Return the durable source name only when no prior owner choice remains."""

    try:
        journeys, _warnings = source_journeys(config)
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantPromotionError(
            f"Could not verify the proposal's review state: {exc}"
        ) from exc
    matches = [
        journey
        for journey in journeys
        if journey.staging_path is not None
        and journey.staging_path.absolute() == target.path.absolute()
    ]
    if len(matches) != 1:
        raise AssistantPromotionError(
            "The selected proposal no longer has one exact current review journey; "
            "refresh the proposal list."
        )
    journey = matches[0]
    ready = journey.state == READY_TO_ADD or (
        journey.state == GRAMMAR_ONLY and not journey.needs_a_person
    )
    if not ready:
        raise AssistantPromotionError(
            f"Proposal {_safe_name(target.path.name)!r} is not ready to add: "
            f"{journey.state}. {journey.next_action}. The Assistant will not "
            "supply that owner decision."
        )
    if (
        not journey.source
        or _safe_name(journey.source) != journey.source
        or "/" in journey.source
        or "\\" in journey.source
    ):
        raise AssistantPromotionError(
            "The selected proposal's source is not a basename-shaped repository "
            "identity; repair its durable review before promoting it."
        )
    return journey.source


def _require_durable_example_authority(
    records: tuple[Any, ...],
) -> None:
    """Refuse every model-authored example whose exact review is not durable."""

    unresolved = [
        (record.id, state)
        for record in records
        if (state := example_authority_state(record)) not in {"existing", "ineligible"}
    ]
    if unresolved:
        rendered = ", ".join(
            f"{record_id} ({state})" for record_id, state in unresolved
        )
        raise AssistantPromotionError(
            "The selected proposal does not carry exact durable example review "
            f"authority for: {rendered}. Repair or review those cards outside the "
            "promotion action; the Assistant will not infer approval."
        )


def _require_ai_enrichment_owner_review(
    decision: promotion_application.PromotionDecision,
) -> None:
    """Keep direct Assistant promotion behind the exact enrichment review."""

    try:
        provenance = promotion_application.staged_ai_enrichment(
            decision.meta,
            decision.work,
            archived_ids=[record.id for record in decision.archived],
            archived_records=decision.archived,
            require_owner_review=True,
        )
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantPromotionError(
            "Use Apply and finish for this AI-enrichment proposal so its exact "
            f"owner review is recorded before promotion: {exc}"
        ) from exc
    if provenance is None:
        raise AssistantPromotionError(
            "Use Apply and finish for this AI-enrichment proposal so its exact "
            "owner review is recorded before promotion."
        )


def _ownership(value: DeckOwnershipEvaluation) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "state": value.state,
        "decks": [
            {
                "name": membership.name,
                "selected": membership.takes,
                "refusal": membership.refusal,
            }
            for membership in value.memberships
        ],
    }


def _projection(
    config: ProjectConfig,
    target: ProposalContext,
    instruction: str,
    source: str,
    decision: promotion_application.PromotionDecision,
    service_fingerprint: str,
) -> dict[str, object]:
    preview = promotion_application.project_promotion(decision)
    landing = [
        {
            "proposed_record_id": card.staged.id,
            "landing_record_id": card.landing.id,
            "result": "add" if card.is_new else "merge",
            "reminted_from": card.reminted_from or None,
            "keeps_existing_meanings": card.keeps_existing_meanings,
            "proposed": assistant_record_value(card.staged),
            "current": (
                assistant_record_value(card.existing)
                if card.existing is not None
                else None
            ),
            "landing": assistant_record_value(card.landing),
        }
        for card in preview.landing
    ]
    held = [
        {
            "record_id": card.record.id,
            "reason": card.reason,
            "record": assistant_record_value(card.record),
        }
        for card in preview.held
    ]
    writes: dict[str, object] = {
        "live_review": _relative(config, target.path, label="live review"),
        "archive": (
            _relative(config, decision.done, label="review archive")
            if decision.done is not None
            else None
        ),
    }
    if decision.output_path is not None:
        writes["collection"] = _relative(
            config, decision.output_path, label="canonical collection"
        )
    if decision.state == "lands":
        writes["ledger"] = _relative(
            config, config.ledger_file, label="promotion ledger"
        )
    return {
        "schema_version": 1,
        "kind": "promote_staging",
        "instruction": instruction,
        "target": {
            "resource_id": target.resource_id,
            "proposal_kind": target.proposal_kind,
            "proposal": _relative(config, target.path, label="proposal"),
            "proposal_sha256": _sha256(decision.wire),
            "source_name": _safe_name(source),
        },
        "authority": {
            "owner_decisions_remaining_before_reading_check": [],
            "coverage_acceptance_requested": False,
            "review_and_deck_choices_are_already_durable": True,
            "reading_conflicts_are_never_resolved_by_the_assistant": True,
        },
        "decision": {
            "state": decision.state,
            "proposal_records": len(decision.records),
            "landing": landing,
            "held": held,
            "already_archived_record_ids": list(preview.already_archived),
            "deck_ownership": [_ownership(item) for item in preview.deck_ownership],
            "warnings": list(preview.warnings),
            "readings_checked_at_execution": decision.state
            in {"lands", "nothing_lands"},
            "reading_conflicts_stay_in_the_live_review": True,
        },
        "writes": writes,
        "service_fingerprint": service_fingerprint,
    }


def plan_promotion_action(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
) -> AssistantPromotionPlan:
    """Resolve one reviewed proposal into an exact display-only promotion plan."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantPromotionError(
            "Assistant promotion needs the owner's nonblank instruction."
        )
    target = _resolve_proposal(config, resource_id)
    if target.proposal_kind not in {
        "source_extraction",
        "ai_enrichment",
        "card_revision",
    }:
        raise AssistantPromotionError(
            f"Assistant promotion does not apply {target.proposal_kind!r} proposals. "
            "Deck and card revisions keep their separate reviewed apply/finish path."
        )
    if target.proposal_kind == "card_revision":
        try:
            _records, revision_meta = promotion_application.read_staging(target.path)
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantPromotionError(
                f"Could not read the reviewed card revision: {exc}"
            ) from exc
        raw_source = revision_meta.get("source_file")
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise AssistantPromotionError(
                "The reviewed card revision has no durable source identity."
            )
        source = raw_source
    else:
        source = _ready_journey(config, target)
    try:
        decision = promotion_application.decide_promotion(
            config,
            target.path,
            source=source,
            skip_reading_check=None,
        )
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantPromotionError(
            f"Could not plan promotion for {_safe_name(target.path.name)!r}: {exc}"
        ) from exc
    if decision.staging_path != target.path.absolute():
        raise AssistantPromotionError(
            "The selected proposal path changed after its safe resolution; refresh "
            "the proposal list before confirming it."
        )
    if (
        not _is_sha256(target.proposal_sha256)
        or _sha256(decision.wire) != target.proposal_sha256
    ):
        raise AssistantPromotionError(
            "The selected proposal bytes changed after their safe resolution; "
            "refresh the proposal list before confirming it."
        )
    recorded_source = decision.meta.get("source_file")
    if recorded_source != source:
        raise AssistantPromotionError(
            "The selected proposal's source changed while its promotion was being "
            "planned; refresh the proposal before confirming it."
        )
    if decision.is_blocked:
        raise AssistantPromotionError(
            f"Proposal {_safe_name(target.path.name)!r} is not ready for ordinary "
            f"promotion: {decision.error}. Resolve that refusal outside the "
            "Assistant action; no coverage, review, identity, or deck decision "
            "was inferred."
        )
    if target.proposal_kind == "ai_enrichment":
        _require_ai_enrichment_owner_review(decision)
    if target.proposal_kind != "card_revision":
        _require_durable_example_authority(decision.records)
    service_fingerprint = promotion_action.promotion_preview_fingerprint(decision)
    projection = _projection(
        config,
        target,
        instruction,
        source,
        decision,
        service_fingerprint,
    )
    wire = _canonical_json(projection)
    return AssistantPromotionPlan(
        repository_root=config.root.resolve(),
        resource_id=resource_id,
        instruction=instruction,
        proposal_kind=target.proposal_kind,
        # Keep the lexical no-follow name validated by the context broker.
        # Resolving it here would reopen a symlink-swap window after the bound
        # read that produced ``proposal_sha256``.
        proposal_path=target.path.absolute(),
        source=source,
        service_fingerprint=service_fingerprint,
        projection_wire=wire,
        fingerprint=_sha256(wire.encode("utf-8")),
        decision=decision,
    )


def _assert_fresh(
    expected: AssistantPromotionPlan, fresh: AssistantPromotionPlan
) -> None:
    if (
        fresh.fingerprint != expected.fingerprint
        or fresh.projection_wire != expected.projection_wire
        or fresh.service_fingerprint != expected.service_fingerprint
    ):
        raise AssistantPromotionError(
            "The Assistant promotion plan changed after it was displayed; reload "
            "and review the fresh plan before confirming it."
        )


def _emit(progress: Callable[[str], None] | None, state: str) -> None:
    if progress is not None:
        progress(state)


def execute_promotion_action(
    config: ProjectConfig,
    expected: AssistantPromotionPlan,
    *,
    client_factory: Callable[[], jpdb.JpdbClient] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AssistantPromotionExecution:
    """Re-plan one confirmed proposal, compare it, then use the sole writer."""

    if expected.repository_root != config.root.resolve():
        raise AssistantPromotionError(
            "The Assistant promotion plan belongs to another repository."
        )
    _emit(progress, "Checking the reviewed proposal")
    fresh = plan_promotion_action(
        config,
        expected.resource_id,
        expected.instruction,
    )
    _assert_fresh(expected, fresh)

    factory = client_factory or (lambda: jpdb.JpdbClient(jpdb.api_key_from_env()))
    if fresh.decision.state in {"lands", "nothing_lands"}:
        _emit(progress, "Checking readings")
    try:
        checked = promotion_action.resolve_promotion_for_execution(
            config,
            fresh.decision,
            client_factory=factory,
            expected_preview_fingerprint=fresh.service_fingerprint,
        )
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantPromotionError(
            f"The confirmed proposal could not pass its fresh reading check: {exc}"
        ) from exc
    if checked.is_blocked:
        raise AssistantPromotionError(
            f"The confirmed proposal is no longer promotable after checking "
            f"readings: {checked.error}. Nothing was promoted."
        )

    _emit(progress, "Saving reviewed cards")
    try:
        result = promotion_application.execute_promotion(config, checked)
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantPromotionError(
            f"The confirmed promotion did not complete: {exc}"
        ) from exc
    return AssistantPromotionExecution(plan=fresh, result=result)
