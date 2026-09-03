"""Bind Janki's bounded repository agent and application plans to ChatKit.

This module is imported only when ``[assistant] enabled = true``. A selected
deck is optional conversational focus. It is neither a read grant nor a change
capability: opaque resources are resolved by the repository context broker and
all writes still go through Janki's existing typed application services.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from japanese_anki import status
from japanese_anki.application import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    OUTCOME_UNKNOWN,
    ExtractionCompletionError,
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    ai_enrichment,
    assistant_actions,
    assistant_agent,
    assistant_ai_enrichment_review,
    assistant_assignment,
    assistant_card_revision_review,
    assistant_context,
    assistant_deck_creation,
    assistant_deletion,
    assistant_operations,
    assistant_promotion,
    assistant_staging_actions,
    assistant_staging_review,
    card_revision,
    card_revision_finish,
    describe_extraction,
    dispatch_extraction,
    revision,
    revision_finish,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.exporters.pattern_cards import read_drill_deck_content
from japanese_anki.models import ExampleSentence
from japanese_anki.workbench.assistant import (
    AssistantDeckChoice,
    ChatReply,
    OperationActionChoice,
    OperationChoice,
    RevisionConfirmation,
    RevisionExampleReview,
    RevisionExecution,
    RevisionFinishConfirmation,
    RevisionFinishExecution,
    RevisionFinishReview,
    RevisionRecordReview,
    RevisionRefusal,
    SourceExtractionConfirmation,
    SourceExtractionExecution,
    SourceExtractionPlan,
    StagedContentFinishReview,
)
from japanese_anki.workbench.assistant import RevisionPlan as AssistantRevisionPlan

__all__ = [
    "RevisionAssistantAdapter",
    "discover_revision_adapter",
]

_RICH_DRILL_ONLY = "Deck changes currently support rich conjugation practice decks only."
_DRILL_EXAMPLES_REQUIRED = (
    "This conjugation deck needs complete rich drill examples before Janki can "
    "select it for changes."
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decoded_disclosure(
    disclosure: assistant_context.ContextDisclosure,
) -> dict[str, object]:
    value = json.loads(disclosure.wire)
    if not isinstance(value, dict):
        raise RevisionRefusal("The Assistant context broker returned an invalid value.")
    return value


def _markdown_disclosure(
    disclosure: assistant_context.ContextDisclosure,
) -> str:
    """Render one exact local projection without treating its values as prose."""

    value = _decoded_disclosure(disclosure)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    longest = 0
    run = 0
    for character in rendered:
        if character == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    fence = "`" * max(3, longest + 1)
    return (
        f"#### {disclosure.kind.title()} resource\n\n"
        f"{disclosure.item_count} item(s) · {disclosure.utf8_bytes} UTF-8 bytes · "
        f"SHA-256 `{disclosure.sha256}`\n\n"
        f"{fence}json\n{rendered}\n{fence}"
    )


def _agent_context(
    config: ProjectConfig,
    *,
    deck_scope: str,
) -> assistant_agent.AgentContext:
    """Build one exact bounded provider disclosure with optional deck focus."""

    broker = assistant_context.AssistantContextBroker(config)
    catalog = broker.catalog()
    status_disclosure = broker.project_status()
    catalog_value = _decoded_disclosure(catalog)
    data = catalog_value.get("data")
    resources = data.get("resources") if isinstance(data, dict) else None
    if not isinstance(resources, list):
        raise RevisionRefusal("The Assistant resource catalog is invalid.")
    resource_ids: list[str] = []
    for item in resources:
        resource_id = item.get("resource_id") if isinstance(item, dict) else None
        if not isinstance(resource_id, str) or not resource_id:
            raise RevisionRefusal("The Assistant resource catalog is incomplete.")
        resource_ids.append(resource_id)
    if len(resource_ids) != len(set(resource_ids)):
        raise RevisionRefusal("The Assistant resource catalog repeats an identifier.")

    disclosures = [catalog, status_disclosure]
    editable_records = ()
    focus_resource_id: str | None = None
    if deck_scope:
        focus_resource_id = broker.resource_id_for_deck(deck_scope)
        deck = broker.deck_context(focus_resource_id)
        disclosures.append(deck.disclosure)
        editable_records = deck.records

    context_value = {
        "schema_version": 1,
        "disclosures": [
            {
                "resource_id": disclosure.resource_id,
                "kind": disclosure.kind,
                "sha256": disclosure.sha256,
                "item_count": disclosure.item_count,
                "utf8_bytes": disclosure.utf8_bytes,
                "value": _decoded_disclosure(disclosure),
            }
            for disclosure in disclosures
        ],
    }
    wire = _canonical_json(context_value)
    return assistant_agent.AgentContext(
        wire=wire,
        fingerprint=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
        resource_ids=tuple(resource_ids),
        editable_records=tuple(editable_records),
        focus_resource_id=focus_resource_id,
    )


@dataclass(frozen=True, slots=True)
class _RevisionDeckTarget:
    """One startup-allowlisted deck behind an opaque browser-facing id."""

    choice: AssistantDeckChoice
    path: Path
    record_ids: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _ResolvedCardRevisionTarget:
    """One fresh opaque deck/card selection resolved entirely server-side."""

    target: _RevisionDeckTarget
    resource_id: str
    record_ids: tuple[str, ...]
    focus_resource_id: str | None


@dataclass(frozen=True, slots=True)
class _PreparedAgentAction:
    """One local application plan behind an owner-visible fingerprint."""

    kind: str
    focus_scope: str
    instruction: str
    target: str
    plan: object


def _deck_label(deck_config: object, deck_path: Path) -> str:
    if isinstance(deck_config, dict):
        label = str(deck_config.get("name") or "").strip()
        if label:
            return label
    return deck_path.stem


def _inspect_deck(
    deck_path: Path,
) -> tuple[str, tuple[str, ...] | None, str | None, str | None]:
    """Return display/support facts using only local deck readers."""

    try:
        deck_config, _records = resolve_deck_records(deck_path)
    except (JankiError, OSError) as exc:
        detail = f"This configured deck could not be read safely: {exc}"
        return deck_path.stem, None, detail, detail

    label = _deck_label(deck_config, deck_path)
    kind = str(deck_config.get("kind") or "").strip().lower()
    if kind != "conjugation":
        return label, None, _RICH_DRILL_ONLY, None
    if not deck_config.get("drill_examples"):
        return label, None, _DRILL_EXAMPLES_REQUIRED, None

    try:
        content = read_drill_deck_content(deck_path)
    except (JankiError, OSError) as exc:
        detail = f"Deck changes cannot safely read this rich conjugation deck: {exc}"
        return label, None, detail, None
    if not content.record_ids or not content.drill_examples:
        return label, None, _DRILL_EXAMPLES_REQUIRED, None
    return label, tuple(content.record_ids), None, None


def _extraction_refusal(error: ExtractionDispatchError) -> RevisionRefusal:
    """Preserve the journal's recovery truth in the conversational surface."""

    detail = str(error)
    if error.phase in {"binding", "preparation"}:
        return RevisionRefusal(
            f"{detail} Nothing was sent to the provider and no paid request was made."
        )
    if error.phase == "authorization":
        return RevisionRefusal(
            f"{detail} Nothing was sent to the provider. The operation journal may "
            "contain unused authority if its final pre-dispatch write failed. Run "
            "'janki operations' and settle anything it shows before trying again."
        )

    operation_id = error.operation_id
    operation = f" Operation {operation_id}." if operation_id is not None else ""
    if error.phase == "dispatch":
        if error.journal_error is not None:
            return RevisionRefusal(
                f"{detail}{operation} The provider request was dispatched and may "
                "have been billed, but janki could not settle the operation journal "
                f"or prove whether recovery bytes were saved: {error.journal_error} "
                "Do not retry. Repair the journal, then run 'janki operations'."
            )
        failure = error.failure
        if failure is None:
            return RevisionRefusal(
                f"{detail}{operation} The provider request was dispatched and may "
                "have been billed. Do not retry; inspect 'janki operations' first."
            )
        operation_id = failure.operation_id
        if failure.outcome == ANSWER_SAVED:
            recovery = (
                "The exact provider answer is saved in paid-operation recovery. "
                f"Read it with 'janki operations --show-reply {operation_id}', then "
                "decide whether to keep or forget it."
            )
        elif failure.outcome == ANSWER_EMPTY:
            recovery = (
                "The captured provider reply contains no answer. "
                f"Inspect its exact bytes with 'janki operations --show-reply "
                f"{operation_id}'."
            )
        elif failure.outcome == ANSWER_UNAVAILABLE:
            recovery = (
                "The reply is recorded as captured, but its exact recovery bytes "
                f"are unavailable. Inspect operation {operation_id} with "
                "'janki operations' before accepting that loss or making a new call."
            )
        elif failure.outcome == FORGOTTEN and failure.cleanup_pending:
            recovery = (
                "Its forget decision is recorded, but exact recovery-data cleanup "
                f"remains. Finish it with 'janki operations --forget {operation_id}'."
            )
        elif failure.outcome == FORGOTTEN:
            recovery = (
                "The operation was already forgotten, so no recovery answer remains. "
                "Prepare a fresh plan only after checking the current source state."
            )
        elif failure.outcome == OUTCOME_UNKNOWN:
            recovery = (
                "The journal records an unsettled provider outcome; a retry may pay "
                f"twice. Do not retry. Inspect operation {operation_id} with "
                "'janki operations' and settle it first."
            )
        else:
            recovery = (
                f"The journal reports unrecognized outcome {failure.outcome!r}. Do "
                f"not retry. Inspect operation {operation_id} with 'janki operations'."
            )
        return RevisionRefusal(
            f"{detail} Operation {operation_id}. The provider request was dispatched "
            f"and may have been billed. {recovery}"
        )

    cause = error.cause
    if isinstance(cause, ExtractionCompletionError):
        return RevisionRefusal(
            f"{detail}{operation} The proposals are saved at {cause.staging_path}, "
            "including the embedded pattern set, but the separate pattern store was "
            "not updated. Review that saved proposal; do not repeat extraction to "
            "repair the pattern-store copy."
        )
    return RevisionRefusal(
        f"{detail}{operation} The provider answer arrived, but janki could not prove "
        "where every proposal and journal transition landed. Do not retry. Inspect "
        "'janki operations' and the staging directory first."
    )


@dataclass(slots=True)
class RevisionAssistantAdapter:
    """Project intake plus exact allowlisted revision targets behind ChatKit."""

    config: ProjectConfig
    deck_choices: tuple[AssistantDeckChoice, ...]
    _targets: tuple[_RevisionDeckTarget, ...] = field(repr=False)
    _agent_plans: dict[str, _PreparedAgentAction] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _agent_plan_duplicates: dict[str, list[_PreparedAgentAction]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _extraction_expectations: dict[str, ExtractionDispatchExpectation] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _finish_plans: dict[
        str,
        revision_finish.RevisionFinishPlan
        | card_revision_finish.CardRevisionFinishPlan,
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _plan_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def _remember_agent_plan(
        self,
        fingerprint: str,
        prepared: _PreparedAgentAction,
    ) -> None:
        """Retain each independently rendered capability, even for equal plans."""

        with self._plan_lock:
            while self._agent_plan_count_locked() >= 256:
                self._evict_oldest_agent_plan_locked()
            current = self._agent_plans.get(fingerprint)
            if current is None:
                self._agent_plans[fingerprint] = prepared
                return
            self._agent_plan_duplicates.setdefault(fingerprint, []).append(prepared)

    def _agent_plan_count_locked(self) -> int:
        return len(self._agent_plans) + sum(
            len(plans) for plans in self._agent_plan_duplicates.values()
        )

    def _evict_oldest_agent_plan_locked(self) -> None:
        fingerprint = next(iter(self._agent_plans))
        duplicates = self._agent_plan_duplicates.get(fingerprint)
        if duplicates:
            self._agent_plans[fingerprint] = duplicates.pop(0)
            if not duplicates:
                self._agent_plan_duplicates.pop(fingerprint, None)
            return
        self._agent_plans.pop(fingerprint)

    def _take_agent_plan(
        self,
        confirmation: RevisionConfirmation,
    ) -> _PreparedAgentAction | None:
        """Consume the one cached plan bound to this exact browser confirmation."""

        with self._plan_lock:
            fingerprint = confirmation.expected_fingerprint
            primary = self._agent_plans.get(fingerprint)
            if primary is None:
                return None
            candidates = [primary, *self._agent_plan_duplicates.get(fingerprint, ())]
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.focus_scope == confirmation.deck_scope
                    and candidate.instruction == confirmation.instruction
                    and candidate.target == confirmation.target
                ),
                None,
            )
            if selected_index is None:
                raise RevisionRefusal(
                    "This Assistant action is stale, already used, or belongs to a "
                    "different focus or target. Nothing was sent."
                )
            selected = candidates[selected_index]
            if selected_index == 0:
                duplicates = self._agent_plan_duplicates.get(fingerprint)
                if duplicates:
                    self._agent_plans[fingerprint] = duplicates.pop(0)
                    if not duplicates:
                        self._agent_plan_duplicates.pop(fingerprint, None)
                else:
                    self._agent_plans.pop(fingerprint)
            else:
                duplicates = self._agent_plan_duplicates[fingerprint]
                duplicates.pop(selected_index - 1)
                if not duplicates:
                    self._agent_plan_duplicates.pop(fingerprint, None)
            return selected

    def _chat_target_for_scope(self, deck_scope: str) -> _RevisionDeckTarget:
        if not isinstance(deck_scope, str) or not deck_scope.strip():
            raise RevisionRefusal("Choose one available deck before asking about it.")
        for target in self._targets:
            if target.choice.scope != deck_scope:
                continue
            if not target.choice.chat_supported:
                raise RevisionRefusal(
                    target.choice.unavailable_reason
                    or "This configured deck is not available for Assistant chat."
                )
            return target
        raise RevisionRefusal("The requested deck is outside this assistant's allowlist.")

    def resolve_deck_selection(self, deck_id: str) -> AssistantDeckChoice:
        """Freshly validate one opaque startup choice without probing a provider."""

        if not isinstance(deck_id, str) or not deck_id:
            raise RevisionRefusal("The selected deck id is unknown.")
        selected: _RevisionDeckTarget | None = None
        candidate = deck_id.encode("utf-8")
        for target in self._targets:
            if secrets.compare_digest(target.choice.deck_id.encode("utf-8"), candidate):
                selected = target
                break
        if selected is None:
            raise RevisionRefusal("The selected deck id is unknown.")
        if not selected.choice.chat_supported:
            raise RevisionRefusal(
                selected.choice.unavailable_reason
                or "This configured deck is not available for Assistant chat."
            )

        label, record_ids, reason, warning = _inspect_deck(selected.path)
        chat_supported = warning is None
        revision_supported = record_ids is not None
        if not chat_supported:
            raise RevisionRefusal(
                reason or "This configured deck is no longer available for Assistant chat."
            )
        if (
            chat_supported != selected.choice.chat_supported
            or revision_supported != selected.choice.revision_supported
        ):
            raise RevisionRefusal(
                "This deck's Assistant capabilities changed after the selector was "
                "prepared. Restart Janki before selecting it."
            )
        if revision_supported and record_ids != selected.record_ids:
            raise RevisionRefusal(
                "This deck's revision scope changed after the selector was prepared. "
                "Restart Janki before selecting it."
            )
        return replace(selected.choice, label=label, unavailable_reason=reason)

    def list_operation_choices(self) -> tuple[OperationChoice, ...]:
        """Expose current recovery choices without dispatching the chat provider."""

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            choices = assistant_operations.list_operation_choices(fresh_config)
        except (JankiError, OSError, TypeError, ValueError) as exc:
            raise RevisionRefusal(str(exc)) from exc
        return tuple(
            OperationChoice(
                operation_id=choice.operation_id,
                kind=choice.kind,
                state=choice.state,
                source_name=choice.source_name,
                model=choice.model,
                authorized_at=choice.authorized_at,
                blocks_spending=choice.blocks_spending,
                money_may_have_been_spent=choice.money_may_have_been_spent,
                has_captured_reply=choice.has_captured_reply,
                has_response_spool=choice.has_response_spool,
                cleanup_pending=choice.cleanup_pending,
                actions=tuple(
                    OperationActionChoice(
                        action=option.action,
                        label=option.label,
                        accept_paid_output_loss=option.accept_paid_output_loss,
                    )
                    for option in choice.actions
                ),
            )
            for choice in choices
        )

    def prepare_operation_action(
        self,
        *,
        operation_id: str,
        action: str,
        accept_paid_output_loss: bool,
        deck_scope: str,
    ) -> ChatReply:
        """Prepare one local selector action even while paid chat is blocked."""

        if action not in {"recover", "show_reply", "end", "forget"}:
            raise RevisionRefusal("Choose recover, show reply, end, or forget.")
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = assistant_operations.plan_operation_action(
                fresh_config,
                operation_id=operation_id,
                action=action,
                accept_paid_output_loss=accept_paid_output_loss,
            )
        except (JankiError, OSError, TypeError, ValueError) as exc:
            raise RevisionRefusal(str(exc)) from exc
        effects, disclosures, confirm_label = self._operation_action_description(plan)
        instruction = f"{action} paid operation {operation_id}"
        target = f"Paid operation {operation_id}"
        prepared = _PreparedAgentAction(
            kind="manage_operation",
            focus_scope=deck_scope,
            instruction=instruction,
            target=target,
            plan=plan,
        )
        self._remember_agent_plan(plan.fingerprint, prepared)
        return ChatReply(
            text=(
                f"Prepared a local {action.replace('_', ' ')} action for paid "
                f"operation {operation_id}. No model call was made."
            ),
            action=AssistantRevisionPlan(
                request_fingerprint=plan.fingerprint,
                target=target,
                effects=effects,
                disclosures=disclosures,
                confirm_label=confirm_label,
                progress_label="Checking paid operation",
            ),
            action_instruction=instruction,
        )

    @staticmethod
    def _catalog_by_id(
        broker: assistant_context.AssistantContextBroker,
    ) -> dict[str, dict[str, object]]:
        catalog = _decoded_disclosure(broker.catalog())
        data = catalog.get("data")
        resources = data.get("resources") if isinstance(data, dict) else None
        if not isinstance(resources, list):
            raise RevisionRefusal("The Assistant resource catalog is invalid.")
        indexed: dict[str, dict[str, object]] = {}
        for item in resources:
            if not isinstance(item, dict):
                raise RevisionRefusal("The Assistant resource catalog is invalid.")
            resource_id = item.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise RevisionRefusal("The Assistant resource catalog is incomplete.")
            if resource_id in indexed:
                raise RevisionRefusal("The Assistant resource catalog repeats an id.")
            indexed[resource_id] = item
        return indexed

    def _target_for_resource_id(
        self,
        broker: assistant_context.AssistantContextBroker,
        resource_id: str,
    ) -> _RevisionDeckTarget:
        matches: list[_RevisionDeckTarget] = []
        for target in self._targets:
            try:
                candidate = broker.resource_id_for_deck(target.choice.scope)
            except JankiError:
                continue
            if secrets.compare_digest(candidate, resource_id):
                matches.append(target)
        if len(matches) != 1:
            raise RevisionRefusal(
                "The requested deck is missing, ambiguous, or no longer configured."
            )
        return matches[0]

    def _resolve_card_revision_target(
        self,
        config: ProjectConfig,
        *,
        intent: assistant_agent.AgentActionIntent,
        deck_scope: str,
    ) -> _ResolvedCardRevisionTarget:
        broker = assistant_context.AssistantContextBroker(config)
        catalog = self._catalog_by_id(broker)
        requested_entries: list[tuple[str, dict[str, object]]] = []
        for resource_id in intent.resource_ids:
            entry = catalog.get(resource_id)
            if entry is None:
                raise RevisionRefusal(
                    "The requested resource changed after the Assistant turn."
                )
            requested_entries.append((resource_id, entry))
        invalid_kinds = sorted(
            {
                str(entry.get("kind") or "")
                for _resource_id, entry in requested_entries
                if entry.get("kind") not in {"deck", "card"}
            }
        )
        if invalid_kinds:
            raise RevisionRefusal(
                "A card revision can target only deck and card resources, not "
                + ", ".join(invalid_kinds)
                + "."
            )
        deck_resource_ids = [
            resource_id
            for resource_id, entry in requested_entries
            if entry.get("kind") == "deck"
        ]
        focus_resource_id: str | None = None
        if deck_scope:
            focus_resource_id = broker.resource_id_for_deck(deck_scope)
        if not deck_resource_ids and focus_resource_id is not None:
            deck_resource_ids.append(focus_resource_id)
        if len(deck_resource_ids) != 1:
            raise RevisionRefusal(
                "Choose exactly one deck resource for this card revision. Deck "
                "focus is optional, but an unfocused change must name its deck."
            )
        deck_resource_id = deck_resource_ids[0]
        deck_context = broker.deck_context(deck_resource_id)
        if not deck_context.records:
            raise RevisionRefusal(
                "This deck does not expose canonical vocabulary records to the "
                "card-revision service."
            )
        target = self._target_for_resource_id(broker, deck_resource_id)
        selected_ids = set(intent.record_ids)
        for resource_id, entry in requested_entries:
            if entry.get("kind") != "card":
                continue
            snapshot = _decoded_disclosure(broker.snapshot(resource_id))
            data = snapshot.get("data")
            card = data.get("card") if isinstance(data, dict) else None
            record_id = card.get("id") if isinstance(card, dict) else None
            if not isinstance(record_id, str) or not record_id:
                raise RevisionRefusal("A requested card resource is invalid.")
            selected_ids.add(record_id)
        available_ids = {record.id for record in deck_context.records}
        unknown = sorted(selected_ids - available_ids)
        if unknown:
            raise RevisionRefusal(
                "The requested card ids are not members of the selected deck: "
                + ", ".join(unknown)
            )
        ordered_ids = tuple(
            record.id
            for record in deck_context.records
            if not selected_ids or record.id in selected_ids
        )
        if not ordered_ids:
            raise RevisionRefusal("The requested deck currently resolves to no cards.")
        return _ResolvedCardRevisionTarget(
            target=target,
            resource_id=deck_resource_id,
            record_ids=ordered_ids,
            focus_resource_id=focus_resource_id,
        )

    def _resolve_enrichment_record_ids(
        self,
        config: ProjectConfig,
        *,
        intent: assistant_agent.AgentActionIntent,
        deck_scope: str,
    ) -> tuple[tuple[str, ...], str | None]:
        """Resolve opaque card/deck resources to an exact canonical-card order."""

        broker = assistant_context.AssistantContextBroker(config)
        catalog = self._catalog_by_id(broker)
        selected: list[str] = []
        raw_record_ids = tuple(intent.record_ids)
        deck_records: tuple[str, ...] | None = None
        deck_resource: str | None = None
        for resource_id in intent.resource_ids:
            entry = catalog.get(resource_id)
            if entry is None:
                raise RevisionRefusal(
                    "The requested enrichment resource changed after the "
                    "Assistant turn."
                )
            kind = entry.get("kind")
            if kind == "card":
                snapshot = _decoded_disclosure(broker.snapshot(resource_id))
                data = snapshot.get("data")
                card = data.get("card") if isinstance(data, dict) else None
                record_id = card.get("id") if isinstance(card, dict) else None
                if not isinstance(record_id, str) or not record_id:
                    raise RevisionRefusal(
                        "A requested AI-enrichment card resource is invalid."
                    )
                selected.append(record_id)
                continue
            if kind == "deck":
                if deck_resource is not None:
                    raise RevisionRefusal(
                        "AI enrichment may expand at most one exact deck resource "
                        "per confirmed batch."
                    )
                context = broker.deck_context(resource_id)
                if context.deck_kind != "vocabulary":
                    raise RevisionRefusal(
                        "AI enrichment can finish only into an exact vocabulary "
                        "deck, not this deck type."
                    )
                if not context.records:
                    raise RevisionRefusal(
                        "The selected deck exposes no canonical vocabulary cards "
                        "for AI enrichment."
                    )
                deck_resource = resource_id
                deck_records = tuple(record.id for record in context.records)
                continue
            raise RevisionRefusal(
                "AI enrichment can target only canonical card or deck resources, "
                f"not {str(kind or 'unknown')}."
            )

        focus_resource_id = (
            broker.resource_id_for_deck(deck_scope) if deck_scope else None
        )
        if deck_resource is not None:
            focus_resource_id = deck_resource
        constrained_records = deck_records
        if constrained_records is None and focus_resource_id is not None:
            focus_context = broker.deck_context(focus_resource_id)
            if focus_context.deck_kind != "vocabulary":
                raise RevisionRefusal(
                    "AI enrichment can finish only into an exact vocabulary "
                    "deck, not this active deck type."
                )
            constrained_records = tuple(record.id for record in focus_context.records)
        if (
            constrained_records is None
            and raw_record_ids
            and focus_resource_id is None
        ):
            raise RevisionRefusal(
                "An unfocused AI-enrichment action must name each exact opaque "
                "card or deck resource; a bare record id is not authority."
            )
        if constrained_records is not None:
            requested = [*selected, *raw_record_ids]
            if requested:
                outside = sorted(set(requested) - set(constrained_records))
                if outside:
                    raise RevisionRefusal(
                        "The requested enrichment cards are not members of the "
                        "selected deck: " + ", ".join(outside)
                    )
                wanted = set(requested)
                selected = [
                    record_id
                    for record_id in constrained_records
                    if record_id in wanted
                ]
            else:
                selected = list(constrained_records)
        ordered = tuple(dict.fromkeys(selected))
        if not ordered:
            raise RevisionRefusal(
                "Choose at least one exact canonical card or one canonical-backed "
                "deck to enrich."
            )
        if focus_resource_id is None:
            raise RevisionRefusal(
                "AI enrichment needs an active deck or one exact deck resource "
                "before any paid call, so Apply and finish has a bound destination."
            )
        return ordered, focus_resource_id

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
    ) -> ChatReply:
        """Answer one bounded repository turn and prepare at most one typed plan."""

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            context = _agent_context(fresh_config, deck_scope=deck_scope)
            plan = assistant_agent.plan_agent(
                fresh_config,
                context=context,
                message=message,
                history=history,
            )
            result = assistant_agent.run_agent(
                fresh_config,
                plan,
                context_loader=lambda: _agent_context(
                    ProjectConfig.load(self.config.root),
                    deck_scope=deck_scope,
                ),
                progress=progress,
            )
        except (JankiError, OSError, TypeError, ValueError) as exc:
            raise RevisionRefusal(str(exc)) from exc
        if not result.action_intents:
            return ChatReply(text=result.answer)
        try:
            return self._prepare_agent_intent(
                fresh_config,
                result=result,
                deck_scope=deck_scope,
                owner_message=message,
            )
        except (RevisionRefusal, JankiError, OSError, TypeError, ValueError) as exc:
            return ChatReply(
                text=(
                    f"{result.answer}\n\nJanki could not prepare that exact action: "
                    f"{exc} Nothing was changed."
                )
            )

    def _prepare_agent_intent(
        self,
        config: ProjectConfig,
        *,
        result: assistant_agent.AgentRunResult,
        deck_scope: str,
        owner_message: str | None = None,
    ) -> ChatReply:
        """Resolve one untrusted model intent through a local Janki planner."""

        [intent] = result.action_intents
        if intent.kind in {"inspect_resources", "search_cards"}:
            broker = assistant_context.AssistantContextBroker(config)
            if intent.record_ids:
                raise RevisionRefusal(
                    "A deterministic repository read cannot carry write-target card ids."
                )
            if intent.kind == "inspect_resources":
                if not 1 <= len(intent.resource_ids) <= 3 or intent.options:
                    raise RevisionRefusal(
                        "Inspect one to three exact repository resources with no "
                        "write options."
                    )
                disclosures = [
                    broker.snapshot(resource_id)
                    for resource_id in intent.resource_ids
                ]
            else:
                if intent.resource_ids:
                    raise RevisionRefusal(
                        "Card search uses its exact literal, not a resource target."
                    )
                options = dict(intent.options)
                if not set(options) <= {"search_literal", "search_limit"}:
                    raise RevisionRefusal("The card-search options are invalid.")
                literal = options.get("search_literal")
                limit = options.get("search_limit", 10)
                if (
                    not isinstance(literal, str)
                    or not literal
                    or isinstance(limit, bool)
                    or not isinstance(limit, int)
                    or not 1 <= limit <= 20
                ):
                    raise RevisionRefusal(
                        "Card search needs one exact literal and a limit from 1 to 20."
                    )
                disclosures = [broker.search_cards(literal, limit=limit)]
            return ChatReply(
                text=(
                    f"{result.answer}\n\n### Local repository result\n\n"
                    + "\n\n".join(
                        _markdown_disclosure(disclosure)
                        for disclosure in disclosures
                    )
                )
            )
        if intent.kind == "enrich_cards":
            if intent.options:
                raise RevisionRefusal(
                    "AI enrichment fills missing meanings, examples, and usage "
                    "notes only; use card revision for an explicit replacement."
                )
            record_ids, focus_resource_id = self._resolve_enrichment_record_ids(
                config,
                intent=intent,
                deck_scope=deck_scope,
            )
            try:
                plan = ai_enrichment.plan_ai_enrichment(
                    config,
                    record_ids,
                    focus_resource_id=focus_resource_id,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            target = f"{len(plan.calls)} canonical card"
            if len(plan.calls) != 1:
                target += "s"
            first_provider = plan.calls[0].provider_plan
            auth = ", ".join(
                f"{key.replace('_', ' ')}="
                f"{value if value is not None else 'none'}"
                for key, value in first_provider.auth_metadata.items()
            )
            effects = [
                (
                    f"Make {len(plan.calls)} paid enrichment call"
                    f"{'s' if len(plan.calls) != 1 else ''} for "
                    f"{len(plan.calls)} selected canonical card"
                    f"{'s' if len(plan.calls) != 1 else ''}"
                ),
                f"Billing: {plan.billing_display}",
                f"Authentication: {auth}",
                f"Model: {plan.model}",
                (
                    f"Canonical input: {plan.canonical_relative_path}; SHA-256 "
                    f"{plan.canonical_sha256}"
                ),
                f"Reviewed-pattern input SHA-256: {plan.patterns_sha256}",
            ]
            for index, call in enumerate(plan.calls, start=1):
                exact_card = _canonical_json(
                    assistant_context.assistant_record_value(call.record)
                )
                request_sha256 = hashlib.sha256(
                    call.provider_plan.request_bytes
                ).hexdigest()
                manifest = self._display_path(config, call.request_manifest_path)
                staged = self._display_path(config, call.staging_path)
                effects.extend(
                    (
                        (
                            f"Call {index}: {call.record_id} "
                            f"({call.record.expression}); operation {call.operation_id}"
                        ),
                        f"Call {index} — Exact current card fields: {exact_card}",
                        (
                            f"Call {index} input fingerprint: "
                            f"{call.input_fingerprint}"
                        ),
                        (
                            f"Call {index} exact provider request: "
                            f"{len(call.provider_plan.request_bytes)} bytes; SHA-256 "
                            f"{request_sha256}"
                        ),
                        (
                            f"Call {index} provider request identity: "
                            f"{call.request_fingerprint}"
                        ),
                        f"Call {index} writes request/result manifest: {manifest}",
                        f"Call {index} stages any proposed fields at: {staged}",
                    )
                )
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.plan_fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.plan_fingerprint,
                    target=target,
                    effects=tuple(effects),
                    disclosures=(
                        (
                            f"Confirming makes {len(plan.calls)} external paid "
                            f"model call{'s' if len(plan.calls) != 1 else ''} using "
                            f"{plan.billing_display}."
                        ),
                        (
                            "Each call discloses only its displayed current card, "
                            "the exact enrichment prompt, and the bound reviewed "
                            "lesson-pattern context."
                        ),
                        (
                            "Every unseen answer stops as an AI-enrichment staging "
                            "proposal. This confirmation does not review, promote, "
                            "voice, build, or directly change canonical cards."
                        ),
                    ),
                    confirm_label="Enrich and stage these exact cards",
                    progress_label="Reading the source",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "revise_deck":
            if intent.record_ids or intent.options:
                raise RevisionRefusal(
                    "A whole-deck revision targets exactly one deck and cannot "
                    "carry card ids or extra choices."
                )
            broker = assistant_context.AssistantContextBroker(config)
            resource_ids = list(intent.resource_ids)
            if not resource_ids and deck_scope:
                resource_ids.append(broker.resource_id_for_deck(deck_scope))
            if len(resource_ids) != 1:
                raise RevisionRefusal(
                    "Choose exactly one rich conjugation deck for this whole-deck "
                    "teaching-content revision."
                )
            catalog = self._catalog_by_id(broker)
            entry = catalog.get(resource_ids[0])
            if entry is None or entry.get("kind") != "deck":
                raise RevisionRefusal(
                    "A whole-deck revision requires one current deck resource."
                )
            target = self._target_for_resource_id(broker, resource_ids[0])
            if not target.choice.revision_supported or target.record_ids is None:
                raise RevisionRefusal(
                    target.choice.unavailable_reason
                    or "Whole-deck teaching revisions currently require a rich "
                    "conjugation-practice deck."
                )
            application_plan, rendered = self._plan_deck_revision(
                config,
                target,
                instruction=intent.instruction,
            )
            self._remember_agent_plan(
                rendered.request_fingerprint,
                _PreparedAgentAction(
                    kind=intent.kind,
                    focus_scope=deck_scope,
                    instruction=intent.instruction,
                    target=rendered.target,
                    plan=application_plan,
                ),
            )
            return ChatReply(
                text=result.answer,
                action=rendered,
                action_instruction=intent.instruction,
            )
        if intent.kind == "revise_cards":
            resolved = self._resolve_card_revision_target(
                config,
                intent=intent,
                deck_scope=deck_scope,
            )
            try:
                plan = card_revision.plan_card_revision(
                    config,
                    resolved.target.path,
                    resolved.record_ids,
                    intent.instruction,
                    focus_resource_id=resolved.focus_resource_id,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=plan.deck_relative_path,
                plan=plan,
            )
            self._remember_agent_plan(plan.plan_fingerprint, prepared)
            request_sha256 = hashlib.sha256(
                plan.provider_plan.request_bytes
            ).hexdigest()
            selection = ", ".join(plan.selected_record_ids)
            auth = ", ".join(
                f"{key.replace('_', ' ')}="
                f"{value if value is not None else 'none'}"
                for key, value in plan.provider_plan.auth_metadata.items()
            )
            effects = [
                (
                    f"Send {len(plan.selected_record_ids)} selected canonical "
                    f"card(s) through {plan.provider}: {selection}"
                ),
                f"Billing: {plan.billing_display}",
                f"Authentication: {auth}",
                f"Model: {plan.model}",
            ]
            if tuple(record.id for record in plan.current_records) != tuple(
                plan.selected_record_ids
            ):
                raise RevisionRefusal(
                    "The selected current cards no longer match the revision plan."
                )
            effects.extend(
                f"Selected current card {index}: "
                + _canonical_json(assistant_context.assistant_record_value(record))
                for index, record in enumerate(plan.current_records, start=1)
            )
            cli_version = plan.provider_plan.transport.get("cli_version")
            if cli_version is not None:
                effects.append(f"Claude Code version: {cli_version}")
            effects.extend(
                (
                    (
                        f"Exact provider request: "
                        f"{len(plan.provider_plan.request_bytes)} bytes; SHA-256 "
                        f"{request_sha256}"
                    ),
                    f"Provider request identity: {plan.request_fingerprint}",
                    (
                        "Save the answer as one unapproved card-revision staging "
                        "proposal; leave canonical cards unchanged"
                    ),
                )
            )
            action = AssistantRevisionPlan(
                request_fingerprint=plan.plan_fingerprint,
                target=plan.deck_relative_path,
                effects=tuple(effects),
                disclosures=(
                    (
                        "Confirming makes one external revision-model call using "
                        f"{plan.billing_display}."
                    ),
                    (
                        "Only the selected current card records and this exact "
                        "instruction are disclosed by the revision call."
                    ),
                    (
                        "The result stops in staging for Japanese-content review; "
                        "this confirmation does not approve, promote, voice, or "
                        "build it."
                    ),
                ),
                confirm_label="Send revision and stage proposal",
            )
            return ChatReply(
                text=result.answer,
                action=action,
                action_instruction=intent.instruction,
            )
        if intent.kind in {"generate_audio", "build_deck"}:
            if intent.kind == "generate_audio":
                options = dict(intent.options)
                self._require_owner_audio_authority(
                    owner_message,
                    force=options.get("audio_force") is True,
                    prune=options.get("audio_prune") is True,
                )
            try:
                plan = assistant_actions.plan_action(config, intent)
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            projection = plan.projection
            target = projection.get("target")
            if not isinstance(target, dict):
                raise RevisionRefusal("The Assistant action target is invalid.")
            configured_file = target.get("configured_file")
            if not isinstance(configured_file, str) or not configured_file:
                raise RevisionRefusal("The Assistant action target is incomplete.")
            if intent.kind == "generate_audio":
                effects, disclosures = self._audio_action_description(plan)
                confirm_label = "Generate this exact audio"
                progress_label = "Preparing audio"
            else:
                effects, disclosures = self._build_action_description(plan)
                confirm_label = "Build this exact deck"
                progress_label = "Preparing build"
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=configured_file,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=configured_file,
                    effects=effects,
                    disclosures=disclosures,
                    confirm_label=confirm_label,
                    progress_label=progress_label,
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "create_deck":
            options = dict(intent.options)
            if set(options) != {"deck_name", "card_directions"}:
                raise RevisionRefusal(
                    "Deck creation needs exactly one deck name and explicit card "
                    "directions."
                )
            if intent.resource_ids or intent.record_ids:
                raise RevisionRefusal(
                    "Deck creation cannot reuse an existing resource or card identity."
                )
            deck_name = options.get("deck_name")
            directions = options.get("card_directions")
            if (
                not isinstance(deck_name, str)
                or not isinstance(directions, list)
                or any(not isinstance(item, str) for item in directions)
                or len(directions) != len(set(directions))
            ):
                raise RevisionRefusal(
                    "Deck creation needs one exact name and unique card directions."
                )
            selected = set(directions)
            if not selected or not selected <= {"recognition", "production", "reading"}:
                raise RevisionRefusal(
                    "Choose at least one of recognition, production, or reading."
                )
            self._require_owner_literal(owner_message, deck_name, label="deck name")
            normalized_owner_message = (owner_message or "").casefold()
            if any(direction not in normalized_owner_message for direction in selected):
                raise RevisionRefusal(
                    "The owner must explicitly name every requested card direction in "
                    "the current message."
                )
            request = assistant_deck_creation.AssistantDeckCreationRequest(
                name=deck_name,
                recognition="recognition" in selected,
                production="production" in selected,
                reading="reading" in selected,
                instruction=intent.instruction,
            )
            try:
                plan = assistant_deck_creation.plan_deck_creation(config, request)
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            projection = plan.projection
            target = projection.get("target")
            definition = projection.get("definition")
            inputs = projection.get("inputs")
            if (
                not isinstance(target, dict)
                or not isinstance(definition, dict)
                or not isinstance(inputs, dict)
            ):
                raise RevisionRefusal("The deck-creation plan is incomplete.")
            configured_file = target.get("configured_file")
            package = target.get("future_package")
            intake_tag = target.get("intake_tag")
            deck_id = target.get("deck_id")
            definition_sha = definition.get("sha256")
            yaml_text = definition.get("yaml")
            deck_set_sha = inputs.get("deck_set_sha256")
            if not all(
                isinstance(value, str) and value
                for value in (
                    configured_file,
                    package,
                    intake_tag,
                    definition_sha,
                    yaml_text,
                    deck_set_sha,
                )
            ) or not isinstance(deck_id, int):
                raise RevisionRefusal("The deck-creation plan is incomplete.")
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=configured_file,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=configured_file,
                    effects=(
                        f"Create the configured study deck {deck_name!r}",
                        "Enable card directions: " + ", ".join(directions),
                        f"Assign Anki deck id {deck_id} and intake tag {intake_tag}",
                        f"Reserve its future package path at {package}",
                        f"Bind the current configured-deck set: SHA-256 {deck_set_sha}",
                        f"Write this exact UTF-8 YAML (SHA-256 {definition_sha}):\n{yaml_text}",
                    ),
                    disclosures=(
                        "This local action makes no model or audio-provider call.",
                        (
                            "The new deck definition is written once; existing "
                            "decks and cards are unchanged."
                        ),
                    ),
                    confirm_label="Create this exact deck",
                    progress_label="Preparing deck",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "extract_source":
            if len(intent.resource_ids) != 1 or intent.record_ids or intent.options:
                raise RevisionRefusal(
                    "Source extraction needs exactly one preserved source resource "
                    "and no card targets or inferred options."
                )
            try:
                source_path = assistant_context.AssistantContextBroker(
                    config
                ).source_path(intent.resource_ids[0])
                extraction = self.prepare_source_extraction(source_path=source_path)
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=extraction.target,
                plan=extraction,
            )
            self._remember_agent_plan(extraction.request_fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=extraction.request_fingerprint,
                    target=extraction.target,
                    effects=extraction.effects,
                    disclosures=extraction.disclosures,
                    confirm_label=extraction.confirm_label,
                    progress_label="Preparing pages",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "review_staging":
            options = dict(intent.options)
            if len(intent.resource_ids) != 1:
                raise RevisionRefusal(
                    "Staging review needs exactly one proposal."
                )
            if "review_patterns" in options and not isinstance(
                options["review_patterns"], bool
            ):
                raise RevisionRefusal(
                    "Pattern review needs an explicit true-or-false decision."
                )
            try:
                proposal = assistant_context.AssistantContextBroker(
                    config
                ).proposal_context(intent.resource_ids[0])
                if proposal.proposal_kind == "card_revision":
                    if (
                        not set(options) <= {"review_patterns"}
                        or options.get("review_patterns", False) is not False
                    ):
                        raise RevisionRefusal(
                            "Card revisions do not carry grammar-pattern review."
                        )
                    plan = card_revision_finish.plan_card_revision_finish(
                        config,
                        resource_id=intent.resource_ids[0],
                        instruction=intent.instruction,
                        record_ids=intent.record_ids,
                    )
                    target = self._display_path(config, plan.deck_path)
                elif proposal.proposal_kind == "ai_enrichment":
                    if (
                        not set(options) <= {"review_patterns"}
                        or options.get("review_patterns", False) is not False
                    ):
                        raise RevisionRefusal(
                            "AI enrichment does not carry grammar-pattern review."
                        )
                    plan = card_revision_finish.plan_ai_enrichment_finish(
                        config,
                        intent.resource_ids[0],
                        intent.instruction,
                        record_ids=intent.record_ids,
                    )
                    target = self._display_path(config, plan.deck_path)
                else:
                    if (
                        set(options) != {"review_patterns"}
                        or not isinstance(options.get("review_patterns"), bool)
                    ):
                        raise RevisionRefusal(
                            "Source staging review needs an explicit true-or-false "
                            "grammar-pattern decision."
                        )
                    plan = assistant_staging_review.plan_staging_review(
                        config,
                        proposal_resource_id=intent.resource_ids[0],
                        record_ids=intent.record_ids,
                        review_patterns=options["review_patterns"],
                    )
                    target = plan.source_name
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            if isinstance(plan, card_revision_finish.CardRevisionFinishPlan):
                effects, disclosures = self._card_revision_finish_description(
                    config, plan
                )
                confirm_label = "Apply and finish"
                progress_label = "Preparing finish"
            else:
                effects = self._staging_review_description(plan)
                disclosures = (
                    (
                        "This click is the owner's review decision for exactly "
                        "the Japanese and patterns displayed above."
                    ),
                    (
                        "It makes no provider call and does not promote cards, "
                        "generate audio, or build a package."
                    ),
                )
                confirm_label = "Approve this exact review"
                progress_label = "Saving review"
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=target,
                    effects=effects,
                    disclosures=disclosures,
                    confirm_label=confirm_label,
                    progress_label=progress_label,
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "assign_cards":
            options = dict(intent.options)
            destination = options.get("destination_resource_id")
            if (
                len(intent.resource_ids) != 1
                or not intent.record_ids
                or set(options) != {"destination_resource_id"}
                or not isinstance(destination, str)
                or not destination
            ):
                raise RevisionRefusal(
                    "Card assignment needs exactly one source-extraction proposal, "
                    "one or more explicit staged card ids, and one exact destination "
                    "deck resource."
                )
            try:
                plan = assistant_assignment.plan_assignment(
                    config,
                    proposal_resource_id=intent.resource_ids[0],
                    destination_resource_id=destination,
                    record_ids=intent.record_ids,
                    instruction=intent.instruction,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            effects = self._assignment_description(plan)
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=plan.source_name,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=plan.source_name,
                    effects=effects,
                    disclosures=(
                        (
                            "This local action changes only the displayed ownership "
                            "tags in the selected staging proposal."
                        ),
                        (
                            "It does not approve Japanese, promote canonical cards, "
                            "change a deck definition, generate audio, or build a package."
                        ),
                    ),
                    confirm_label="Assign these exact staged cards",
                    progress_label="Saving deck assignments",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "delete_content":
            options = dict(intent.options)
            deletion_kind = options.get("deletion_kind")
            if (
                set(options) != {"deletion_kind"}
                or not isinstance(deletion_kind, str)
                or not deletion_kind
            ):
                raise RevisionRefusal(
                    "Deletion needs exactly one explicit deletion_kind: "
                    "staged_cards, canonical_cards, or deck."
                )
            if deletion_kind == "canonical_cards":
                if (
                    not intent.resource_ids
                    or len(intent.resource_ids) != len(intent.record_ids)
                ):
                    raise RevisionRefusal(
                        "Canonical-card deletion needs one or more exact card "
                        "resources and the same number of explicit canonical card ids."
                    )
                try:
                    plan = assistant_deletion.plan_canonical_deletion(
                        config,
                        card_resource_ids=intent.resource_ids,
                        record_ids=intent.record_ids,
                        instruction=intent.instruction,
                    )
                    effects = self._canonical_deletion_description(plan)
                    collection = plan.projection.get("canonical_collection")
                    target = (
                        collection.get("configured_file")
                        if isinstance(collection, dict)
                        else None
                    )
                    if not isinstance(target, str) or not target:
                        raise RevisionRefusal(
                            "The canonical-card deletion target is incomplete."
                        )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                disclosures = (
                    (
                        "This permanently removes the exact displayed cards from "
                        "the canonical collection."
                    ),
                    (
                        "Configured deck definitions, ledger history, media, and "
                        "staging proposals are retained. Existing generated packages "
                        "are retained and remain stale until you rebuild them. No "
                        "provider call is made."
                    ),
                )
                confirm_label = "Delete these exact canonical cards"
                progress_label = "Deleting canonical cards"
            elif deletion_kind == "deck":
                if len(intent.resource_ids) != 1 or intent.record_ids:
                    raise RevisionRefusal(
                        "Deck deletion needs exactly one deck resource and no card ids."
                    )
                try:
                    plan = assistant_deletion.plan_deck_deletion(
                        config,
                        deck_resource_id=intent.resource_ids[0],
                        instruction=intent.instruction,
                    )
                    effects = self._deck_deletion_description(plan)
                    plan_target = plan.projection.get("target")
                    target = (
                        plan_target.get("configured_file")
                        if isinstance(plan_target, dict)
                        else None
                    )
                    if not isinstance(target, str) or not target:
                        raise RevisionRefusal(
                            "The configured-deck deletion target is incomplete."
                        )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                disclosures = (
                    (
                        "This permanently removes only the exact displayed configured "
                        "deck definition."
                    ),
                    (
                        "Canonical cards, ledger history, media, and staging proposals "
                        "are retained. The generated package is retained and may be "
                        "stale; deleting a deck definition does not uninstall a deck "
                        "from Anki. No provider call is made."
                    ),
                )
                confirm_label = "Delete this exact deck definition"
                progress_label = "Deleting deck definition"
            elif deletion_kind != "staged_cards":
                raise RevisionRefusal(
                    "The deletion_kind must be staged_cards, canonical_cards, or deck."
                )
            else:
                if len(intent.resource_ids) != 1 or not intent.record_ids:
                    raise RevisionRefusal(
                        "Staged-card deletion needs exactly one staging proposal and "
                        "one or more explicit staged card ids."
                    )
                try:
                    plan = assistant_staging_actions.plan_staged_deletion(
                        config,
                        proposal_resource_id=intent.resource_ids[0],
                        record_ids=intent.record_ids,
                        instruction=intent.instruction,
                    )
                    effects = self._staged_deletion_description(plan)
                    target = self._staging_action_target(
                        plan.projection,
                        action="staged-card deletion",
                    )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                disclosures = (
                    (
                        "This permanently removes only the exact displayed rows "
                        "from the live staging proposal."
                    ),
                    (
                        "Canonical cards, deck definitions, review archives, and "
                        "provider records are unchanged. No provider call is made."
                    ),
                )
                confirm_label = "Delete these exact staged cards"
                progress_label = "Removing staged cards"
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=target,
                    effects=effects,
                    disclosures=disclosures,
                    confirm_label=confirm_label,
                    progress_label=progress_label,
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "reidentify_staged_card":
            options = dict(intent.options)
            new_expression = options.get("new_expression")
            new_reading = options.get("new_reading")
            if (
                len(intent.resource_ids) != 1
                or len(intent.record_ids) != 1
                or set(options) != {"new_expression", "new_reading"}
                or not isinstance(new_expression, str)
                or not new_expression.strip()
                or not isinstance(new_reading, str)
                or not new_reading.strip()
            ):
                raise RevisionRefusal(
                    "Staged-card reidentification needs exactly one source-extraction "
                    "proposal, one staged card id, and the owner's explicit nonblank "
                    "new_expression and new_reading."
                )
            self._require_owner_literal(
                owner_message,
                new_expression,
                label="new expression",
            )
            self._require_owner_literal(
                owner_message,
                new_reading,
                label="new reading",
            )
            try:
                plan = assistant_staging_actions.plan_reidentification(
                    config,
                    proposal_resource_id=intent.resource_ids[0],
                    record_id=intent.record_ids[0],
                    new_expression=new_expression,
                    new_reading=new_reading,
                    instruction=intent.instruction,
                )
                effects = self._reidentification_description(plan)
                target = self._staging_action_target(
                    plan.projection,
                    action="staged-card reidentification",
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=target,
                    effects=effects,
                    disclosures=(
                        (
                            "This click records the owner's exact staged identity "
                            "decision and its displayed consequences."
                        ),
                        (
                            "It does not rewrite Japanese examples, change canonical "
                            "cards, promote anything, or make a provider call."
                        ),
                    ),
                    confirm_label="Change this exact staged identity",
                    progress_label="Saving staged identity",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "approve_coverage":
            options = dict(intent.options)
            reason = options.get("coverage_reason")
            if (
                len(intent.resource_ids) != 1
                or intent.record_ids
                or set(options) != {"coverage_reason"}
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise RevisionRefusal(
                    "Coverage approval needs exactly one source-extraction proposal, "
                    "no card ids, and the owner's explicit nonblank coverage_reason."
                )
            self._require_owner_literal(
                owner_message,
                reason,
                label="coverage reason",
            )
            try:
                plan = assistant_staging_actions.plan_coverage_approval(
                    config,
                    proposal_resource_id=intent.resource_ids[0],
                    reason=reason,
                    instruction=intent.instruction,
                )
                effects = self._coverage_approval_description(plan)
                target = self._staging_action_target(
                    plan.projection,
                    action="coverage approval",
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=target,
                    effects=effects,
                    disclosures=(
                        (
                            "This records the owner's explicit decision about the "
                            "displayed source-unit coverage account."
                        ),
                        (
                            "It is not a Japanese-content review, promotes no cards, "
                            "and makes no provider call."
                        ),
                    ),
                    confirm_label="Approve this exact coverage account",
                    progress_label="Saving coverage approval",
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "promote_staging":
            if len(intent.resource_ids) != 1 or intent.record_ids or intent.options:
                raise RevisionRefusal(
                    "Promotion needs exactly one reviewed proposal resource and "
                    "cannot carry inferred review, identity, deck, or coverage choices."
            )
            try:
                promotion = assistant_promotion.plan_promotion_action(
                    config,
                    intent.resource_ids[0],
                    intent.instruction,
                )
                plan = (
                    card_revision_finish.plan_card_revision_finish(
                        config,
                        intent.resource_ids[0],
                        intent.instruction,
                    )
                    if getattr(promotion, "proposal_kind", None) == "card_revision"
                    else card_revision_finish.plan_ai_enrichment_finish(
                        config,
                        intent.resource_ids[0],
                        intent.instruction,
                    )
                    if getattr(promotion, "proposal_kind", None) == "ai_enrichment"
                    else promotion
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            if isinstance(plan, card_revision_finish.CardRevisionFinishPlan):
                effects, disclosures = self._card_revision_finish_description(
                    config, plan
                )
                target = self._display_path(config, plan.deck_path)
                confirm_label = "Apply and finish"
                progress_label = "Preparing finish"
            else:
                effects = self._promotion_description(plan)
                disclosures = (
                    (
                        "All required review, identity, deck, and coverage "
                        "decisions must already be durable; Janki supplies none."
                    ),
                    (
                        "Readings are checked at the click. Conflicts stay in "
                        "the live review instead of being guessed or discarded."
                    ),
                    (
                        "This exact action promotes only. It does not generate "
                        "audio or build a package."
                    ),
                )
                target = plan.source
                confirm_label = "Promote this exact proposal"
                progress_label = "Checking the reviewed proposal"
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=target,
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=target,
                    effects=effects,
                    disclosures=disclosures,
                    confirm_label=confirm_label,
                    progress_label=progress_label,
                ),
                action_instruction=intent.instruction,
            )
        if intent.kind == "manage_operation":
            if intent.record_ids or len(intent.resource_ids) != 1:
                raise RevisionRefusal(
                    "A paid-operation action needs exactly one operation-status "
                    "resource and no card ids."
                )
            catalog = self._catalog_by_id(
                assistant_context.AssistantContextBroker(config)
            )
            resource = catalog.get(intent.resource_ids[0])
            if resource is None or resource.get("kind") != "operations":
                raise RevisionRefusal(
                    "A paid-operation action must target the current operation-status "
                    "resource."
                )
            options = dict(intent.options)
            allowed = {
                "operation_id",
                "operation_action",
                "accept_paid_output_loss",
            }
            if any(value is not None and key not in allowed for key, value in options.items()):
                raise RevisionRefusal("The paid-operation action options are invalid.")
            operation_id = options.get("operation_id")
            operation_action = options.get("operation_action")
            accept_loss = options.get("accept_paid_output_loss")
            if not isinstance(operation_id, str) or not operation_id.strip():
                raise RevisionRefusal("Choose one exact paid operation id.")
            if operation_action not in {"recover", "show_reply", "end", "forget"}:
                raise RevisionRefusal("Choose recover, show reply, end, or forget.")
            self._require_owner_literal(
                owner_message,
                operation_id,
                label="paid operation id",
            )
            try:
                plan = assistant_operations.plan_operation_action(
                    config,
                    operation_id=operation_id,
                    action=operation_action,
                    accept_paid_output_loss=accept_loss is True,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            effects, disclosures, confirm_label = self._operation_action_description(
                plan
            )
            prepared = _PreparedAgentAction(
                kind=intent.kind,
                focus_scope=deck_scope,
                instruction=intent.instruction,
                target=f"Paid operation {operation_id}",
                plan=plan,
            )
            self._remember_agent_plan(plan.fingerprint, prepared)
            return ChatReply(
                text=result.answer,
                action=AssistantRevisionPlan(
                    request_fingerprint=plan.fingerprint,
                    target=prepared.target,
                    effects=effects,
                    disclosures=disclosures,
                    confirm_label=confirm_label,
                    progress_label="Checking paid operation",
                ),
                action_instruction=intent.instruction,
            )
        return ChatReply(
            text=(
                f"{result.answer}\n\nJanki recognized the `{intent.kind}` request, "
                "but that local action service is not connected yet. Nothing was "
                "changed."
            )
        )

    @staticmethod
    def _require_owner_literal(
        owner_message: str | None,
        value: str,
        *,
        label: str,
    ) -> None:
        """Refuse owner-only free text that exists only in model output."""

        if not owner_message or value not in owner_message:
            raise RevisionRefusal(
                f"The {label} must appear verbatim in the owner's current message; "
                "the Assistant may not invent or paraphrase it."
            )

    @staticmethod
    def _require_owner_audio_authority(
        owner_message: str | None,
        *,
        force: bool,
        prune: bool,
    ) -> None:
        """Bind destructive model options to exact words in this owner turn."""

        words = set(re.findall(r"[a-z]+", (owner_message or "").casefold()))
        if force and words.isdisjoint({"force", "regenerate", "regeneration"}):
            raise RevisionRefusal(
                "Force regeneration requires the owner to say force or regenerate "
                "in the current message."
            )
        if prune and "prune" not in words:
            raise RevisionRefusal(
                "Audio pruning requires the owner to say prune in the current message."
            )

    @staticmethod
    def _operation_action_description(
        plan: assistant_operations.OperationActionPlan,
    ) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        operation = plan.projection.get("operation")
        if not isinstance(operation, dict):
            raise RevisionRefusal("The paid-operation plan is incomplete.")
        operation_id = operation.get("operation_id")
        state = operation.get("state")
        kind = operation.get("kind")
        source_name = operation.get("source_name")
        revision = operation.get("operation_revision")
        if any(
            not isinstance(value, str) or not value
            for value in (operation_id, state, kind, source_name, revision)
        ):
            raise RevisionRefusal("The paid-operation plan is incomplete.")
        facts = (
            f"Operation: {operation_id}",
            f"Current state: {state} · kind: {kind} · source: {source_name}",
            f"Exact operation revision: {revision}",
        )
        if plan.action == "recover":
            recovery = plan.projection.get("recovery")
            if not isinstance(recovery, dict):
                raise RevisionRefusal("The paid-operation recovery plan is incomplete.")
            recovery_kind = recovery.get("kind")
            manifest_name = recovery.get("manifest_name")
            manifest_sha256 = recovery.get("manifest_sha256")
            reply_sha256 = recovery.get("captured_reply_sha256")
            consequence = recovery.get("consequence")
            result_names = recovery.get("result_names")
            if (
                any(
                    not isinstance(value, str) or not value
                    for value in (
                        recovery_kind,
                        manifest_name,
                        manifest_sha256,
                        reply_sha256,
                        consequence,
                    )
                )
                or not isinstance(result_names, list)
                or not result_names
                or any(not isinstance(value, str) or not value for value in result_names)
            ):
                raise RevisionRefusal("The paid-operation recovery plan is incomplete.")
            return (
                (
                    *facts,
                    f"Captured reply SHA-256: {reply_sha256}",
                    f"Exact request manifest: {manifest_name} · SHA-256 {manifest_sha256}",
                    f"Recovery consequence: {consequence}",
                    f"Durable result: {', '.join(result_names)}",
                    "Reuse the captured reply locally; make no provider call",
                ),
                (
                    "Recovery uses the existing operation-specific parser and durable "
                    "staging/turn writer. Raw private reply bytes are not added to "
                    "later model context.",
                ),
                "Recover this exact captured result",
            )
        if plan.action == "show_reply":
            size = operation.get("captured_reply_bytes")
            digest = operation.get("captured_reply_sha256")
            frames = operation.get("committed_response_frames")
            effects = (
                *facts,
                (
                    "Display the exact private recovery reply"
                    + (f" ({size} bytes · SHA-256 {digest})" if size is not None else "")
                    + (f"; committed response frames: {frames}" if frames else "")
                ),
                "Do not change or settle the paid operation",
            )
            return (
                effects,
                (
                    "The reply is private repository recovery data. It will be "
                    "shown in this thread but excluded from later model context.",
                ),
                "Show this exact private reply",
            )
        if plan.action == "end":
            next_state = (
                "canceled_before_send" if state == "authorized" else "outcome_unknown"
            )
            return (
                (*facts, f"End the operation and record state: {next_state}"),
                (
                    "This records that the original process is gone. It does not "
                    "claim an uncertain paid request succeeded or failed, and it "
                    "does not delete recovery evidence.",
                ),
                "End this exact operation",
            )
        if plan.force:
            return (
                (
                    *facts,
                    "Delete this journal entry and retire its exact recovery evidence",
                    "Permanently discard paid output that never reached a durable destination",
                ),
                (
                    "This is destructive and cannot be undone. The confirmation "
                    "does not authorize another paid call.",
                ),
                "Discard paid output and forget",
            )
        return (
            (*facts, "Forget this finished journal entry and its bound cleanup evidence"),
            (
                "This operation no longer carries uncommitted paid output according "
                "to the exact rendered state.",
            ),
            "Forget this exact operation",
        )

    @staticmethod
    def _audio_action_description(
        plan: assistant_actions.AssistantActionPlan,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        projection = plan.projection
        options = projection.get("options")
        counts = projection.get("counts")
        providers = projection.get("providers")
        clips = projection.get("clips")
        writes = projection.get("writes")
        force_replacements = projection.get("force_replacements")
        cleanup = projection.get("cleanup")
        if (
            not isinstance(options, dict)
            or not isinstance(counts, dict)
            or not isinstance(providers, list)
            or not isinstance(clips, list)
            or not isinstance(writes, dict)
            or not isinstance(force_replacements, list)
            or not isinstance(cleanup, dict)
        ):
            raise RevisionRefusal("The Assistant audio plan is incomplete.")
        effects: list[str] = []
        paid_calls = 0
        words = options.get("words")
        examples = options.get("examples")
        force = options.get("force")
        prune = options.get("prune")
        class_source = options.get("clip_classes_source")
        if (
            not isinstance(words, bool)
            or not isinstance(examples, bool)
            or not isinstance(force, bool)
            or not isinstance(prune, bool)
            or class_source not in {"deck-default", "owner-explicit"}
            or (not words and not examples)
        ):
            raise RevisionRefusal("The Assistant audio choices are invalid.")
        class_names = " and ".join(
            name
            for name, selected in (("words", words), ("examples", examples))
            if selected
        )
        source_label = "deck default" if class_source == "deck-default" else "owner explicit"
        effects.append(f"Clip classes: {class_names} ({source_label})")
        for kind in ("words", "examples"):
            values = counts.get(kind)
            if not isinstance(values, dict):
                raise RevisionRefusal("The Assistant audio counts are invalid.")
            try:
                total = int(values["total"])
                current = int(values["current"])
                recoverable = int(values["recoverable"])
                provider_required = int(values["provider_required"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RevisionRefusal("The Assistant audio counts are invalid.") from exc
            effects.append(
                f"{kind.title()}: {total} exact clip(s) — {current} current, "
                f"{recoverable} recoverable without dispatch, {provider_required} "
                "provider call(s) required"
            )
        for provider in providers:
            if not isinstance(provider, dict):
                raise RevisionRefusal("The Assistant audio provider plan is invalid.")
            name = provider.get("name")
            access = provider.get("access")
            clip_kind = provider.get("clip_kind")
            voice = provider.get("voice")
            speed = provider.get("speed")
            required = provider.get("provider_required")
            if (
                not isinstance(name, str)
                or not isinstance(access, str)
                or not isinstance(clip_kind, str)
                or not isinstance(required, int)
            ):
                raise RevisionRefusal("The Assistant audio provider plan is incomplete.")
            if access == "paid-network":
                paid_calls += required
            voice_text = (
                f" · voice {voice}"
                if isinstance(voice, str | int)
                and not isinstance(voice, bool)
                and str(voice)
                else ""
            )
            speed_text = f" · speed {speed}" if isinstance(speed, int | float) else ""
            effects.append(
                f"{clip_kind.title()} provider: {name} · {access}{voice_text}{speed_text}"
            )
        for clip in clips:
            if not isinstance(clip, dict):
                raise RevisionRefusal("The Assistant audio clip plan is invalid.")
            record_id = clip.get("record_id")
            kind = clip.get("kind")
            target = clip.get("target")
            request_input = clip.get("request_input")
            content_fingerprint = clip.get("content_fingerprint")
            state = clip.get("state")
            provider = clip.get("provider")
            billing_class = clip.get("billing_class")
            recovery_sha256 = clip.get("recovery_sha256")
            if (
                not isinstance(record_id, str)
                or not isinstance(kind, str)
                or not isinstance(target, str)
                or not isinstance(content_fingerprint, str)
                or not isinstance(state, str)
                or not isinstance(provider, str)
                or not isinstance(billing_class, str)
                or (
                    recovery_sha256 is not None
                    and not isinstance(recovery_sha256, str)
                )
            ):
                raise RevisionRefusal("The Assistant audio clip plan is incomplete.")
            try:
                request_wire = json.dumps(
                    request_input,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise RevisionRefusal(
                    "The Assistant audio clip request is invalid."
                ) from exc
            recovery = (
                f" · recovery SHA-256 {recovery_sha256}"
                if recovery_sha256 is not None
                else " · no recovery artifact"
            )
            effects.append(
                f"{kind.title()} clip for {record_id}: {target} · {state} · "
                f"{provider} · {billing_class} · content SHA-256 "
                f"{content_fingerprint} · request {request_wire}{recovery}"
            )
        media_directory = writes.get("media_directory")
        ledger = writes.get("ledger")
        if not isinstance(media_directory, str) or not isinstance(ledger, str):
            raise RevisionRefusal("The Assistant audio write plan is incomplete.")

        def bound_entry(value: object) -> tuple[str, str | None]:
            if not isinstance(value, dict):
                raise RevisionRefusal("The Assistant audio file binding is invalid.")
            file = value.get("file")
            entry_type = value.get("entry_type")
            identity = value.get("identity")
            byte_count = value.get("bytes")
            digest = value.get("sha256")
            if (
                not isinstance(file, str)
                or not isinstance(entry_type, str)
                or not isinstance(identity, list)
                or len(identity) != 4
                or any(not isinstance(item, int) for item in identity)
                or not isinstance(byte_count, int)
                or (digest is not None and not isinstance(digest, str))
            ):
                raise RevisionRefusal("The Assistant audio file binding is incomplete.")
            binding = (
                f"{file} · {entry_type} · {byte_count} bytes · identity "
                + json.dumps(identity)
            )
            return binding, digest

        if force:
            effects.append(
                "Force regeneration is on: current and recoverable selected clips "
                "will be regenerated using the provider-required counts above."
            )
            for replacement in force_replacements:
                if not isinstance(replacement, dict):
                    raise RevisionRefusal(
                        "The Assistant forced-audio replacement is invalid."
                    )
                record_id = replacement.get("record_id")
                kind = replacement.get("kind")
                target = replacement.get("target")
                if not all(isinstance(value, str) for value in (record_id, kind, target)):
                    raise RevisionRefusal(
                        "The Assistant forced-audio replacement is incomplete."
                    )
                binding, digest = bound_entry(replacement.get("existing"))
                hash_text = (
                    f" · content SHA-256 {digest}"
                    if digest is not None
                    else " · content not followed"
                )
                effects.append(
                    f"Replace existing {kind} clip for {record_id}: {binding}{hash_text}"
                )
        else:
            if force_replacements:
                raise RevisionRefusal(
                    "The Assistant audio plan names replacements without force authority."
                )
            effects.append(
                "Force regeneration is off; current and recoverable clips are reused"
            )

        cleanup_enabled = cleanup.get("enabled")
        cleanup_scope = cleanup.get("scope")
        removals = cleanup.get("media_files_removed")
        forgotten = cleanup.get("ledger_audio_files_forgotten")
        missing_pending = cleanup.get("missing_record_pending_audio_removed")
        if (
            not isinstance(cleanup_enabled, bool)
            or cleanup_enabled != prune
            or not isinstance(cleanup_scope, str)
            or not isinstance(removals, list)
            or not isinstance(forgotten, list)
            or any(not isinstance(item, str) for item in forgotten)
            or not isinstance(missing_pending, list)
            or any(not isinstance(item, dict) for item in missing_pending)
        ):
            raise RevisionRefusal("The Assistant audio cleanup plan is invalid.")
        if prune:
            if cleanup_scope != "repository-wide-unreferenced-janki-audio":
                raise RevisionRefusal("The Assistant audio prune scope is invalid.")
            effects.append(
                "Pruning is on for repository-wide unreferenced janki audio; this "
                "destructive cleanup is not limited to the selected deck."
            )
            for removal in removals:
                binding, digest = bound_entry(removal)
                hash_text = (
                    f" · content SHA-256 {digest}"
                    if digest is not None
                    else " · content not followed"
                )
                effects.append(f"Permanently delete: {binding}{hash_text}")
            effects.append(
                "Remove ledger audio references: "
                + json.dumps(forgotten, ensure_ascii=False, sort_keys=True)
            )
            effects.append(
                "Retire missing-record pending audio rows and their bound stages: "
                + json.dumps(missing_pending, ensure_ascii=False, sort_keys=True)
            )
        else:
            if cleanup_scope != "none" or removals or forgotten or missing_pending:
                raise RevisionRefusal(
                    "The Assistant audio plan names cleanup without prune authority."
                )
            effects.append("Pruning is off; no unreferenced media is removed")
        effects.extend(
            (
                f"Publish generated clips under {media_directory} and update {ledger}",
                (
                    "The action fingerprint binds every clip's exact text, content "
                    "identity, provider, request profile, and recovery state"
                ),
            )
        )
        if paid_calls:
            disclosures = (
                f"Confirming may make {paid_calls} paid audio provider call(s).",
                (
                    "Each paid call is separately journaled before dispatch. An "
                    "interruption preserves exact recovery state and is not silently retried."
                ),
            )
        else:
            disclosures = (
                "This exact plan needs no paid audio-provider calls.",
                "Current and recoverable clips remain bound to their existing ledger evidence.",
            )
        return tuple(effects), disclosures

    @staticmethod
    def _build_action_description(
        plan: assistant_actions.AssistantActionPlan,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        projection = plan.projection
        target = projection.get("target")
        inputs = projection.get("inputs")
        writes = projection.get("writes")
        if (
            not isinstance(target, dict)
            or not isinstance(inputs, dict)
            or not isinstance(writes, dict)
        ):
            raise RevisionRefusal("The Assistant build plan is incomplete.")
        card_count = target.get("card_count")
        package = writes.get("package")
        package_precondition = writes.get("package_precondition")
        export_history = writes.get("export_history")
        deck_input = inputs.get("deck")
        sources = inputs.get("sources")
        templates = inputs.get("templates")
        media = inputs.get("media")
        if (
            not isinstance(card_count, int)
            or not isinstance(package, str)
            or not isinstance(package_precondition, dict)
            or not isinstance(deck_input, dict)
            or not isinstance(sources, list)
            or not isinstance(templates, list)
            or not isinstance(media, list)
        ):
            raise RevisionRefusal("The Assistant build plan is incomplete.")

        def bound_input(item: object, *, fallback_label: str) -> tuple[str, str, str]:
            if not isinstance(item, dict):
                raise RevisionRefusal("The Assistant build input is invalid.")
            label = item.get("label", fallback_label)
            file = item.get("file")
            sha256 = item.get("sha256")
            if (
                not isinstance(label, str)
                or not label.strip()
                or not isinstance(file, str)
                or not isinstance(sha256, str)
            ):
                raise RevisionRefusal("The Assistant build input is incomplete.")
            return label.strip().capitalize(), file, sha256

        deck_label, deck_file, deck_sha = bound_input(
            deck_input,
            fallback_label="configured deck",
        )
        bound_sources = [bound_input(item, fallback_label="source") for item in sources]
        bound_templates = [
            bound_input(item, fallback_label="template") for item in templates
        ]
        bound_media = [bound_input(item, fallback_label="media") for item in media]
        effects = [
            f"Build all {card_count} card(s)",
            f"{deck_label}: {deck_file} · SHA-256 {deck_sha}",
        ]
        effects.extend(
            f"{label}: {file} · SHA-256 {sha256}"
            for label, file, sha256 in bound_sources
            if (file, sha256) != (deck_file, deck_sha)
        )
        effects.extend(
            f"{label}: {file} · SHA-256 {sha256}"
            for label, file, sha256 in bound_templates
        )
        effects.extend(
            f"{label}: {file} · SHA-256 {sha256}"
            for label, file, sha256 in bound_media
        )
        output_state = package_precondition.get("state")
        output_sha = package_precondition.get("sha256")
        output_identity = package_precondition.get("identity")
        if output_state == "absent":
            if output_sha is not None or output_identity is not None:
                raise RevisionRefusal(
                    "The Assistant build output-absence plan is invalid."
                )
            effects.append(
                f"Require {package} to remain absent until this build publishes it"
            )
        elif output_state == "replace_exact":
            if (
                not isinstance(output_sha, str)
                or not isinstance(output_identity, list)
                or len(output_identity) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in output_identity
                )
            ):
                raise RevisionRefusal(
                    "The Assistant build output-replacement plan is invalid."
                )
            effects.append(
                f"Replace only the existing package at {package}: SHA-256 "
                f"{output_sha} · device/inode {output_identity[0]}/{output_identity[1]}"
            )
        else:
            raise RevisionRefusal("The Assistant build output precondition is invalid.")
        effects.append(f"Write the Anki package to {package}")
        if export_history is not None:
            if not isinstance(export_history, str):
                raise RevisionRefusal("The Assistant build export-history write is invalid.")
            effects.append(f"Write export history to {export_history}")
        return (
            tuple(effects),
            (
                "This is a local build and makes no model or audio-provider call.",
                "Janki re-reads and hashes every bound input before accepting the package.",
            ),
        )

    @staticmethod
    def _staging_review_description(
        plan: (
            assistant_staging_review.AssistantStagingReviewPlan
            | assistant_card_revision_review.AssistantCardRevisionReviewPlan
            | assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan
        ),
    ) -> tuple[str, ...]:
        projection = plan.projection
        selection = projection.get("selection")
        writes = projection.get("writes")
        if not isinstance(selection, dict) or not isinstance(writes, dict):
            raise RevisionRefusal("The staging-review plan is incomplete.")
        if isinstance(
            plan,
            assistant_card_revision_review.AssistantCardRevisionReviewPlan,
        ):
            changes = selection.get("changes")
            if not isinstance(changes, list) or not changes:
                raise RevisionRefusal("The card-revision review has no exact changes.")
            effects = []
            for change in changes:
                if not isinstance(change, dict):
                    raise RevisionRefusal("A card-revision change is invalid.")
                effects.append(
                    f"Accept {change.get('expression')} [{change.get('record_id')}] "
                    f"field {change.get('field')}: old "
                    f"{json.dumps(change.get('old_value'), ensure_ascii=False)} → new "
                    f"{json.dumps(change.get('proposed_value'), ensure_ascii=False)}"
                )
            effects.append(
                f"Bind proposal SHA-256 {plan.proposal_sha256} and save "
                "durable repository-owner review authority"
            )
            removed = writes.get("unselected_rows_removed")
            if removed:
                effects.append(
                    "Leave unselected cards unapproved and remove them from this "
                    "proposal: " + json.dumps(removed, ensure_ascii=False)
                )
            return tuple(effects)
        if isinstance(
            plan,
            assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan,
        ):
            changes = selection.get("changes")
            examples = selection.get("examples")
            if (
                not isinstance(changes, list)
                or not changes
                or any(not isinstance(change, dict) for change in changes)
                or not isinstance(examples, list)
                or any(not isinstance(item, dict) for item in examples)
            ):
                raise RevisionRefusal(
                    "The AI-enrichment review has no exact selected content."
                )
            effects = [
                (
                    f"Review {change.get('expression')} [{change.get('record_id')}] "
                    f"field {change.get('field')}: current "
                    f"{json.dumps(change.get('current'), ensure_ascii=False)} → "
                    f"proposed {json.dumps(change.get('proposed'), ensure_ascii=False)}"
                )
                for change in changes
            ]
            for item in examples:
                current = json.dumps(
                    item.get("current"), ensure_ascii=False, sort_keys=True
                )
                proposed = json.dumps(
                    item.get("proposed"), ensure_ascii=False, sort_keys=True
                )
                effects.append(
                    f"Explicitly review every current and proposed example for "
                    f"{item.get('expression')} [{item.get('record_id')}]: "
                    f"current {current} → proposed {proposed}"
                )
            effects.append(
                f"Bind proposal SHA-256 {plan.proposal_sha256} and save exact "
                "repository-owner enrichment review authority"
            )
            removed = writes.get("unselected_rows_removed")
            if removed:
                effects.append(
                    "Leave unselected cards unapproved and remove them from this "
                    "proposal: " + json.dumps(removed, ensure_ascii=False)
                )
            return tuple(effects)
        snapshots = projection.get("snapshots")
        if not isinstance(snapshots, dict):
            raise RevisionRefusal("The staging-review snapshots are invalid.")
        records = selection.get("records")
        patterns = selection.get("patterns")
        if not isinstance(records, list):
            raise RevisionRefusal("The staging-review records are invalid.")
        effects: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                raise RevisionRefusal("The staging-review record is invalid.")
            record_id = record.get("record_id")
            expression = record.get("expression")
            reading = record.get("reading")
            examples = record.get("examples")
            if not all(
                isinstance(value, str) for value in (record_id, expression, reading)
            ) or not isinstance(examples, list):
                raise RevisionRefusal("The staging-review record is incomplete.")
            effects.append(
                f"Accept the displayed examples for {expression}（{reading}） "
                f"[{record_id}]"
            )
            for example in examples:
                if not isinstance(example, dict):
                    raise RevisionRefusal("A staging-review example is invalid.")
                effects.append(
                    "  "
                    + " · ".join(
                        f"{key}: {example.get(key, '')}"
                        for key in (
                            "register",
                            "japanese",
                            "furigana",
                            "english",
                            "romaji",
                            "spoken_japanese",
                        )
                    )
                )
        if selection.get("review_patterns") is True:
            effects.append(
                "Mark this exact grammar pattern set reviewed: "
                + json.dumps(patterns, ensure_ascii=False, sort_keys=True)
            )
        effects.extend(
            (
                (
                    "Bind staging SHA-256 "
                    f"{snapshots.get('staging_sha256')} and pattern-store SHA-256 "
                    f"{snapshots.get('patterns_sha256')}"
                ),
                (
                    "Write review authority: "
                    + json.dumps(writes, ensure_ascii=False, sort_keys=True)
                ),
            )
        )
        return tuple(effects)

    @staticmethod
    def _assignment_description(
        plan: assistant_assignment.AssistantAssignmentPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        proposal = projection.get("proposal")
        destination = projection.get("destination")
        selection = projection.get("selection")
        writes = projection.get("writes")
        if not all(
            isinstance(value, dict)
            for value in (proposal, destination, selection, writes)
        ):
            raise RevisionRefusal("The staged-card assignment plan is incomplete.")
        assert isinstance(proposal, dict)
        assert isinstance(destination, dict)
        assert isinstance(selection, dict)
        assert isinstance(writes, dict)
        assignments = selection.get("assignments")
        if not isinstance(assignments, list):
            raise RevisionRefusal("The staged-card assignment changes are invalid.")
        effects: list[str] = []
        for change in assignments:
            if not isinstance(change, dict):
                raise RevisionRefusal("A staged-card assignment change is invalid.")
            tags = change.get("tags")
            memberships = change.get("resulting_memberships")
            occurrences = change.get("proposal_occurrences")
            if (
                not isinstance(tags, dict)
                or not isinstance(memberships, list)
                or not isinstance(occurrences, list)
            ):
                raise RevisionRefusal("A staged-card assignment change is incomplete.")
            effects.extend(
                (
                    (
                        f"Assign {change.get('expression')}（{change.get('reading')}） "
                        f"[{change.get('record_id')}] to {destination.get('name')}"
                    ),
                    (
                        "Exact ownership-tag change: "
                        + json.dumps(tags, ensure_ascii=False, sort_keys=True)
                    ),
                    (
                        "Resulting configured-deck membership: "
                        + json.dumps(memberships, ensure_ascii=False, sort_keys=True)
                    ),
                    (
                        "Known staged source occurrences: "
                        + json.dumps(occurrences, ensure_ascii=False, sort_keys=True)
                    ),
                    (
                        "Bind assigned/prospective record SHA-256: "
                        f"{change.get('assigned_record_sha256')} / "
                        f"{change.get('prospective_record_sha256')}"
                    ),
                )
            )
        effects.extend(
            (
                (
                    f"Bind proposal SHA-256 {proposal.get('sha256')} and assignment-"
                    f"service fingerprint {plan.service_fingerprint}"
                ),
                (
                    "Write only this staging target: "
                    + json.dumps(writes, ensure_ascii=False, sort_keys=True)
                ),
            )
        )
        return tuple(effects)

    @staticmethod
    def _staging_action_target(
        projection: object,
        *,
        action: str,
    ) -> str:
        if not isinstance(projection, dict):
            raise RevisionRefusal(f"The {action} plan is invalid.")
        target = projection.get("target")
        if not isinstance(target, dict):
            raise RevisionRefusal(f"The {action} target is invalid.")
        staging_proposal = target.get("staging_proposal")
        if not isinstance(staging_proposal, str) or not staging_proposal:
            raise RevisionRefusal(f"The {action} target is incomplete.")
        return staging_proposal

    @staticmethod
    def _canonical_deletion_description(
        plan: assistant_deletion.AssistantCanonicalDeletionPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        selection = projection.get("selection")
        collection = projection.get("canonical_collection")
        affected = projection.get("affected_decks")
        retained = projection.get("retained")
        writes = projection.get("writes")
        inputs = projection.get("inputs")
        if (
            not isinstance(selection, dict)
            or not isinstance(collection, dict)
            or not isinstance(affected, list)
            or not isinstance(retained, dict)
            or not isinstance(writes, dict)
            or not isinstance(inputs, dict)
        ):
            raise RevisionRefusal("The canonical-card deletion plan is incomplete.")
        repository_files = inputs.get("repository_files")
        if not isinstance(repository_files, list) or not repository_files:
            raise RevisionRefusal(
                "The canonical-card deletion input binding is incomplete."
            )
        record_ids = selection.get("record_ids")
        cards = selection.get("cards")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or not isinstance(cards, list)
            or len(cards) != len(record_ids)
            or any(not isinstance(card, dict) for card in cards)
        ):
            raise RevisionRefusal("The canonical-card deletion selection is invalid.")
        return (
            "Permanently delete these exact canonical card ids: "
            + json.dumps(record_ids, ensure_ascii=False, sort_keys=True),
            *(
                "Permanently remove this complete canonical card: "
                + json.dumps(card, ensure_ascii=False, sort_keys=True)
                for card in cards
            ),
            (
                f"Change canonical row count from {collection.get('rows_before')} to "
                f"{collection.get('rows_after')} by removing "
                f"{collection.get('rows_removed')} row(s)"
            ),
            (
                f"Bind canonical SHA-256 {collection.get('before_sha256')} and exact "
                f"replacement SHA-256 {collection.get('after_sha256')}"
            ),
            "Affected configured decks after their next rebuild: "
            + json.dumps(affected, ensure_ascii=False, sort_keys=True),
            "Retained repository state: "
            + json.dumps(retained, ensure_ascii=False, sort_keys=True),
            "Bound repository inputs: "
            + json.dumps(inputs, ensure_ascii=False, sort_keys=True),
            "Write only this canonical replacement: "
            + json.dumps(writes, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _deck_deletion_description(
        plan: assistant_deletion.AssistantDeckDeletionPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        target = projection.get("target")
        consequences = projection.get("consequences")
        retained = projection.get("retained")
        removals = projection.get("removals")
        inputs = projection.get("inputs")
        if (
            not isinstance(target, dict)
            or not isinstance(consequences, dict)
            or not isinstance(retained, dict)
            or not isinstance(removals, list)
            or len(removals) != 1
            or not isinstance(removals[0], str)
            or not isinstance(inputs, dict)
        ):
            raise RevisionRefusal("The configured-deck deletion plan is incomplete.")
        repository_files = inputs.get("repository_files")
        if not isinstance(repository_files, list) or not repository_files:
            raise RevisionRefusal(
                "The configured-deck deletion input binding is incomplete."
            )
        return (
            "Permanently delete this exact configured deck definition: "
            + json.dumps(target, ensure_ascii=False, sort_keys=True),
            (
                f"Change configured deck count from "
                f"{consequences.get('configured_deck_count_before')} to "
                f"{consequences.get('configured_deck_count_after')}"
            ),
            "Cards currently selected by this definition: "
            + json.dumps(
                consequences.get("currently_selected_record_ids"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Inline-only cards that leave the configured deck library: "
            + json.dumps(
                consequences.get("inline_only_record_ids_removed_from_library"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Canonical cards removed: "
            + json.dumps(
                consequences.get("canonical_cards_removed"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Retained repository state, including the generated package: "
            + json.dumps(retained, ensure_ascii=False, sort_keys=True),
            "Bound repository inputs: "
            + json.dumps(inputs, ensure_ascii=False, sort_keys=True),
            "Remove only: "
            + json.dumps(removals, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _staged_deletion_description(
        plan: assistant_staging_actions.AssistantStagedDeletionPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        target = projection.get("target")
        selection = projection.get("selection")
        snapshots = projection.get("snapshots")
        effects = projection.get("effects")
        if not all(
            isinstance(value, dict)
            for value in (target, selection, snapshots, effects)
        ):
            raise RevisionRefusal("The staged-card deletion plan is incomplete.")
        assert isinstance(target, dict)
        assert isinstance(selection, dict)
        assert isinstance(snapshots, dict)
        assert isinstance(effects, dict)
        record_ids = selection.get("record_ids")
        records = selection.get("records")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or not isinstance(records, list)
            or len(records) != len(record_ids)
            or any(not isinstance(record, dict) for record in records)
        ):
            raise RevisionRefusal("The staged-card deletion selection is invalid.")
        return (
            "Delete these exact staged card ids: "
            + json.dumps(record_ids, ensure_ascii=False, sort_keys=True),
            *(
                "Permanently remove this complete staged row: "
                + json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in records
            ),
            (
                f"Change staged row count from {effects.get('rows_before')} to "
                f"{effects.get('rows_after')} by removing "
                f"{effects.get('rows_removed')} row(s)"
            ),
            (
                f"Bind proposal SHA-256 {snapshots.get('before_sha256')} and exact "
                f"replacement SHA-256 {snapshots.get('after_sha256')}"
            ),
            "Write only this staging target: "
            + json.dumps(target, ensure_ascii=False, sort_keys=True),
            "Action boundaries: "
            + json.dumps(effects, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _reidentification_description(
        plan: assistant_staging_actions.AssistantReidentificationPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        target = projection.get("target")
        identity = projection.get("identity")
        snapshots = projection.get("snapshots")
        effects = projection.get("effects")
        if not all(
            isinstance(value, dict)
            for value in (target, identity, snapshots, effects)
        ):
            raise RevisionRefusal("The staged-card reidentification plan is incomplete.")
        assert isinstance(target, dict)
        assert isinstance(identity, dict)
        assert isinstance(snapshots, dict)
        assert isinstance(effects, dict)
        old = identity.get("old")
        new = identity.get("new")
        neighbours = identity.get("neighbours")
        consequences = identity.get("consequences")
        if (
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or not isinstance(neighbours, list)
            or any(not isinstance(item, dict) for item in neighbours)
            or not isinstance(consequences, list)
            or any(not isinstance(item, str) for item in consequences)
        ):
            raise RevisionRefusal("The staged-card identity consequences are invalid.")
        return (
            "Change this exact staged identity from "
            + json.dumps(old, ensure_ascii=False, sort_keys=True)
            + " to "
            + json.dumps(new, ensure_ascii=False, sort_keys=True),
            "Identity neighbours: "
            + json.dumps(neighbours, ensure_ascii=False, sort_keys=True),
            "Identity consequences: "
            + json.dumps(consequences, ensure_ascii=False, sort_keys=True),
            f"Previously exported under the old identity: {identity.get('was_exported')}",
            (
                f"Bind proposal SHA-256 {snapshots.get('before_sha256')} and exact "
                f"replacement SHA-256 {snapshots.get('after_sha256')}"
            ),
            "Write only this staging target: "
            + json.dumps(target, ensure_ascii=False, sort_keys=True),
            "Action boundaries: "
            + json.dumps(effects, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _coverage_approval_description(
        plan: assistant_staging_actions.AssistantCoverageApprovalPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        target = projection.get("target")
        coverage = projection.get("coverage")
        owner_decision = projection.get("owner_decision")
        effects = projection.get("effects")
        if not all(
            isinstance(value, dict)
            for value in (target, coverage, owner_decision, effects)
        ):
            raise RevisionRefusal("The owner coverage-approval plan is incomplete.")
        assert isinstance(target, dict)
        assert isinstance(coverage, dict)
        assert isinstance(owner_decision, dict)
        assert isinstance(effects, dict)
        account = coverage.get("account")
        reason = owner_decision.get("reason")
        if (
            not isinstance(account, str)
            or not account
            or owner_decision.get("authority") != "repository-owner"
            or not isinstance(reason, str)
            or not reason
        ):
            raise RevisionRefusal("The owner coverage decision is invalid.")
        return (
            f"Approve this exact source-unit coverage account:\n{account}",
            (
                f"Accounted candidate/source units: "
                f"{coverage.get('candidate_units')} / {coverage.get('source_units')}"
            ),
            f"Owner reason: {reason}",
            f"Bind staging SHA-256 {coverage.get('staging_sha256')}",
            "Write approval only to this staging target: "
            + json.dumps(target, ensure_ascii=False, sort_keys=True),
            "Action boundaries: "
            + json.dumps(effects, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _promotion_description(
        plan: assistant_promotion.AssistantPromotionPlan,
    ) -> tuple[str, ...]:
        projection = plan.projection
        target = projection.get("target")
        decision = projection.get("decision")
        writes = projection.get("writes")
        if (
            not isinstance(target, dict)
            or not isinstance(decision, dict)
            or not isinstance(writes, dict)
        ):
            raise RevisionRefusal("The promotion plan is incomplete.")
        landing = decision.get("landing")
        held = decision.get("held")
        if not isinstance(landing, list) or not isinstance(held, list):
            raise RevisionRefusal("The promotion landing plan is invalid.")
        effects = [
            (
                f"Promote proposal state {decision.get('state')} with "
                f"{len(landing)} landing card(s) and {len(held)} held card(s)"
            ),
            (
                "Exact card landing: "
                + json.dumps(landing, ensure_ascii=False, sort_keys=True)
            ),
            (
                "Cards that remain held: "
                + json.dumps(held, ensure_ascii=False, sort_keys=True)
            ),
            (
                "Write/archive targets: "
                + json.dumps(writes, ensure_ascii=False, sort_keys=True)
            ),
            f"Bind proposal SHA-256 {target.get('proposal_sha256')}",
            f"Bind promotion-service fingerprint {plan.service_fingerprint}",
        ]
        return tuple(effects)

    def _card_revision_finish_description(
        self,
        config: ProjectConfig,
        plan: card_revision_finish.CardRevisionFinishPlan,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Render every bound review, promotion, audio, and build consequence."""

        if plan.review is not None:
            effects = list(self._staging_review_description(plan.review))
        elif plan.promotion is not None:
            effects = list(self._promotion_description(plan.promotion))
        else:
            raise RevisionRefusal(
                "The card Apply-and-finish plan has no exact review projection."
            )
        record_ids = ", ".join(plan.record_ids)
        effects.append(
            "Apply only the reviewed card ids to the canonical collection: "
            + record_ids
        )
        effects.append(
            "Bind projected canonical SHA-256 "
            + hashlib.sha256(
                plan.projected_canonical_text.encode("utf-8")
            ).hexdigest()
        )
        audio = plan.audio
        effects.append(
            "Create selected-card word audio: "
            f"{audio.word_counts.total} total · {audio.word_counts.current} current · "
            f"{audio.word_counts.recoverable} recoverable · "
            f"{audio.word_counts.provider_required} provider-required"
        )
        effects.append(
            "Create selected-card example audio: "
            f"{audio.example_counts.total} total · "
            f"{audio.example_counts.current} current · "
            f"{audio.example_counts.recoverable} recoverable · "
            f"{audio.example_counts.provider_required} provider-required"
        )
        for clip in audio.clips:
            provider = clip.provider
            voice = (
                f" · voice {provider.voice}"
                if provider.voice is not None
                else ""
            )
            speed = (
                f" · speed {provider.speed}"
                if provider.speed is not None
                else ""
            )
            effects.append(
                f"Audio {clip.kind} for [{clip.record_id}] → {clip.target}: "
                f"{clip.state} · {provider.name} ({provider.access}){voice}{speed} · "
                f"spoken input {json.dumps(clip.request_input, ensure_ascii=False)}"
            )
        build = plan.build
        effects.append(
            f"Build {build.card_count} card(s) from {build.note_count} note(s) "
            f"for configured deck {build.deck_name!r} at "
            f"{self._display_path(config, build.output_path)}"
        )
        effects.append(
            f"Package variant {build.variant}; card types "
            f"{', '.join(build.card_types)}; configuration fingerprint "
            f"{build.configuration_fingerprint}"
        )
        if build.output_revision is None:
            effects.append("Require the package output to remain absent before build")
        elif build.output_identity is not None:
            effects.append(
                "Replace only the exact existing package: SHA-256 "
                f"{build.output_revision} · device/inode "
                f"{build.output_identity[0]}/{build.output_identity[1]}"
            )
        else:
            raise RevisionRefusal(
                "The card Apply-and-finish package precondition is incomplete."
            )
        inputs = (
            (build.deck_input,)
            + build.source_inputs
            + build.template_inputs
            + build.media_inputs
        )
        effects.append(
            "Bind every package input: "
            + json.dumps(
                [
                    {
                        "label": item.label,
                        "path": self._display_path(config, item.path),
                        "sha256": item.sha256,
                    }
                    for item in inputs
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        effects.append(
            "Persist resumable finish receipt "
            f"{plan.fingerprint} at {self._display_path(config, plan.record_path)}"
        )
        paid = sum(
            clip.state == "provider-required"
            and clip.provider.access == "paid-network"
            for clip in audio.clips
        )
        return tuple(effects), (
            (
                "This one click records owner review of exactly the displayed old and "
                "proposed fields, applies that selection, creates its missing audio, "
                "and builds only the displayed configured deck."
            ),
            (
                f"At most {paid} paid audio-provider call(s) may be dispatched. "
                "Current or recoverable clips are reused and force is false."
            ),
            (
                "The authority receipt is saved before review or promotion. If a "
                "later phase stops, resume that receipt instead of authorizing or "
                "paying for the work again."
            ),
        )

    def _plan_deck_revision(
        self,
        config: ProjectConfig,
        target: _RevisionDeckTarget,
        *,
        instruction: str,
    ) -> tuple[revision.RevisionPlan, AssistantRevisionPlan]:
        """Plan one rich-deck revision without retaining or granting authority."""

        try:
            plan = revision.plan_revision(
                config,
                target.path,
                target.record_ids,
                instruction,
            )
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        record_list = ", ".join(plan.selected_record_ids)
        auth = ", ".join(
            f"{key.replace('_', ' ')}={value if value is not None else 'none'}"
            for key, value in plan.auth_metadata.items()
        )
        effects = [
            (
                f"Send all {len(plan.selected_record_ids)} drill cards through "
                f"{plan.provider} in this deck order: {record_list}"
            ),
            f"Billing: {plan.billing_display}",
            f"Authentication: {auth}",
            f"Model: {plan.model}",
        ]
        cli_version = plan.transport.get("cli_version")
        if cli_version is not None:
            effects.append(f"Claude Code version: {cli_version}")
        request_bytes_sha256 = hashlib.sha256(plan.provider_plan.request_bytes).hexdigest()
        effects.extend(
            (
                (
                    f"Exact provider request bytes: "
                    f"{len(plan.provider_plan.request_bytes)} bytes; request bytes "
                    f"SHA-256 {request_bytes_sha256}"
                ),
                f"Provider request identity: {plan.request_fingerprint}",
                (
                    "Ask for a revised form note plus one polite and one casual "
                    "example for every selected card"
                ),
                "Save the answer as an unapproved staging proposal; leave the deck unchanged",
            )
        )
        rendered = AssistantRevisionPlan(
            # The application plan binds request content *and* local staging
            # state. It is therefore the exact owner-visible confirmation
            # identity, rather than the narrower provider request hash.
            request_fingerprint=plan.plan_fingerprint,
            target=plan.deck_relative_path,
            effects=tuple(effects),
            disclosures=(
                f"Confirming makes one external model call against {plan.billing_display}.",
                (
                    "The answer is staged for exact owner review. This confirmation "
                    "does not accept Japanese, apply it, generate audio, or build a package."
                ),
            ),
        )
        return plan, rendered

    def prepare_source_extraction(self, *, source_path: Path) -> SourceExtractionPlan:
        """Describe one saved source and retain only its scalar dispatch binding."""

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            consent = describe_extraction(fresh_config, source_path, mode=None)
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        if not consent.sendable or consent.target is None:
            reason = consent.refusal or consent.busy or "This source cannot be extracted."
            raise RevisionRefusal(reason)

        target = consent.target
        request_fingerprint = str(target.provenance["request_fingerprint"])
        expectation = ExtractionDispatchExpectation(
            source=source_path,
            model=consent.model,
            mode=consent.mode,
            source_sha256=target.source_sha256,
            request_fingerprint=request_fingerprint,
            replacement_revision=consent.replacement_revision,
            # A rendered replacement is not authority. The explicit ChatKit
            # confirmation sets this only inside consume_replan_and_extract.
            replacement_confirmed=False,
            staging_path=target.staging_path,
            patterns_path=target.patterns_path,
            operations_path=fresh_config.operations_file,
        )
        preparation_id = secrets.token_urlsafe(32)
        with self._plan_lock:
            while len(self._extraction_expectations) >= 256:
                self._extraction_expectations.pop(next(iter(self._extraction_expectations)))
            self._extraction_expectations[preparation_id] = expectation

        try:
            target_display = target.staging_path.relative_to(fresh_config.root.resolve()).as_posix()
        except ValueError:
            target_display = str(target.staging_path)
        mode_display = consent.mode or "automatic source-shape selection"
        effects = [
            (
                f"Send the whole {consent.name} source to {consent.model}; "
                "page ranges or other extraction scope typed in chat are not applied"
            ),
            f"Use {mode_display} and write proposals to {target_display}",
            "Propose vocabulary cards and grammar for owner review",
        ]
        if consent.sends_known_words:
            effects.append("Also send the existing expression list so prose extraction can skip it")
        replaces = consent.replaces is not None
        if replaces:
            card_review = (
                f"{consent.replaces_cards} cards, {consent.replaces_state}"
                if consent.replaces_state
                else "the existing card review"
            )
            grammar = f"; {consent.replaces_grammar}" if consent.replaces_grammar else ""
            effects.append(
                "Permanently replace the named review: "
                f"{card_review}{grammar}. That work is not recoverable"
            )
        disclosures = (
            "This is one paid Anthropic API call; Claude Pro or Max does not pay for it.",
            (
                "Only the named source (and the existing expression list in prose "
                "mode) leaves this computer; the saved local copy stays in the inbox."
            ),
            (
                "Extraction creates unapproved staging proposals. It does not approve, "
                "assign, voice, build, or install a deck."
            ),
        )
        confirm_label = (
            f"Replace the named review and send {consent.name} — paid API call"
            if replaces
            else f"Send {consent.name} using {consent.model} — paid API call"
        )
        return SourceExtractionPlan(
            preparation_id=preparation_id,
            source_name=consent.name,
            request_fingerprint=request_fingerprint,
            target=target_display,
            effects=tuple(effects),
            disclosures=disclosures,
            confirm_label=confirm_label,
            replaces=replaces,
        )

    def consume_replan_and_extract(
        self,
        confirmation: SourceExtractionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> SourceExtractionExecution:
        """Consume one binding and run the shared W3 extraction transaction."""

        with self._plan_lock:
            expected = self._extraction_expectations.pop(
                confirmation.preparation_id,
                None,
            )
        if (
            expected is None
            or expected.source.name != confirmation.source_name
            or expected.request_fingerprint != confirmation.expected_fingerprint
        ):
            raise RevisionRefusal(
                "This extraction plan is missing, stale, already used, or belongs "
                "elsewhere. Prepare it again; nothing was sent."
            )

        # Reaching this callback proves the owner clicked the one button whose
        # label explicitly names any replacement. Only that event can set force.
        expected = replace(
            expected,
            replacement_confirmed=expected.replacement_revision is not None,
        )
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            outcome = dispatch_extraction(
                fresh_config,
                expected,
                progress=progress,
            )
        except ExtractionDispatchError as exc:
            raise _extraction_refusal(exc) from exc

        try:
            target_display = outcome.target.relative_to(fresh_config.root.resolve()).as_posix()
        except ValueError:
            target_display = str(outcome.target)
        return SourceExtractionExecution(
            message=(
                f"Extraction saved {outcome.records} card proposal(s) at "
                f"{target_display} for owner review. Return to the Workbench to "
                "review and assign them; nothing was approved, voiced, built, or "
                "installed automatically."
            )
        )

    def consume_replan_and_execute(
        self,
        confirmation: RevisionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> RevisionExecution:
        """Stage unseen content, then prepare its exact aggregate owner review."""

        agent_expected = self._take_agent_plan(confirmation)
        if agent_expected is not None:
            return self._consume_agent_action(
                confirmation,
                agent_expected,
                progress=progress,
            )
        raise RevisionRefusal(
            "This Assistant action is missing, stale, already consumed, or belongs "
            "elsewhere. Prepare it again; nothing was sent."
        )

    def _stage_deck_revision(
        self,
        expected: revision.RevisionPlan,
        *,
        progress: Callable[[str], None],
    ) -> RevisionExecution:
        """Run one bound rich-deck revision and prepare its aggregate finish."""

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            result = revision.run_revision(fresh_config, expected, progress=progress)
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        try:
            staged = result.staging_path.relative_to(fresh_config.root.resolve()).as_posix()
        except ValueError:
            staged = str(result.staging_path)
        preparation_id = secrets.token_urlsafe(32)
        try:
            finish = revision_finish.plan_revision_finish(
                fresh_config,
                result.staging_path,
            )
            review = self._finish_review(
                fresh_config,
                finish,
                preparation_id=preparation_id,
            )
        except Exception as exc:  # noqa: BLE001 - the paid proposal is already durable
            detail = str(exc) or type(exc).__name__
            return RevisionExecution(
                message=(
                    f"The revision proposal is staged at {staged}. Nothing in it has "
                    "been accepted or applied. Do not repeat the paid revision call."
                ),
                finish=None,
                finish_unavailable=(
                    f"Janki could not prepare its Apply and finish review: {detail} "
                    "The staged proposal remains the deliverable; repair the local "
                    "finish prerequisite, then review this exact proposal."
                ),
            )
        with self._plan_lock:
            while len(self._finish_plans) >= 256:
                self._finish_plans.pop(next(iter(self._finish_plans)))
            self._finish_plans[preparation_id] = finish
        return RevisionExecution(
            message=(
                f"The revision proposal is staged at {staged}. Nothing in it has "
                "been accepted or applied. Review the exact current and proposed "
                "content below."
            ),
            finish=review,
        )

    @staticmethod
    def _proposal_resource_for_path(
        config: ProjectConfig,
        path: Path,
        *,
        proposal_kind: str,
    ) -> str:
        """Resolve one service-produced path back through the opaque live catalog."""

        broker = assistant_context.AssistantContextBroker(config)
        catalog = _decoded_disclosure(broker.catalog())
        data = catalog.get("data")
        resources = data.get("resources") if isinstance(data, dict) else None
        if not isinstance(resources, list):
            raise RevisionRefusal("The Assistant proposal catalog is incomplete.")
        expected = path.absolute()
        matches: list[str] = []
        for item in resources:
            if (
                not isinstance(item, dict)
                or item.get("kind") != "proposal"
                or item.get("proposal_kind") != proposal_kind
            ):
                continue
            resource_id = item.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise RevisionRefusal(
                    "The Assistant proposal catalog contains an invalid resource."
                )
            proposal = broker.proposal_context(resource_id)
            if proposal.path.absolute() == expected:
                matches.append(resource_id)
        if len(matches) != 1:
            raise RevisionRefusal(
                "The staged AI-enrichment result does not resolve to one unique "
                "live proposal resource."
            )
        return matches[0]

    def _prepare_ai_enrichment_finish(
        self,
        config: ProjectConfig,
        *,
        staging_path: Path,
        record_id: str,
        instruction: str,
    ) -> StagedContentFinishReview:
        resource_id = self._proposal_resource_for_path(
            config,
            staging_path,
            proposal_kind="ai_enrichment",
        )
        plan = card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            instruction,
            record_ids=(record_id,),
        )
        effects, disclosures = self._card_revision_finish_description(config, plan)
        preparation_id = secrets.token_urlsafe(32)
        review = StagedContentFinishReview(
            preparation_id=preparation_id,
            request_fingerprint=plan.fingerprint,
            target=self._display_path(config, plan.deck_path),
            effects=effects,
            disclosures=disclosures,
        )
        with self._plan_lock:
            while len(self._finish_plans) >= 256:
                self._finish_plans.pop(next(iter(self._finish_plans)))
            self._finish_plans[preparation_id] = plan
        return review

    def _consume_agent_action(
        self,
        confirmation: RevisionConfirmation,
        expected: _PreparedAgentAction,
        *,
        progress: Callable[[str], None],
    ) -> RevisionExecution:
        """Consume one typed Assistant plan through its existing application service."""

        if (
            confirmation.deck_scope != expected.focus_scope
            or confirmation.instruction != expected.instruction
            or confirmation.target != expected.target
        ):
            raise RevisionRefusal(
                "This Assistant action is stale, already used, or belongs to a "
                "different focus or target. Nothing was sent."
            )
        if isinstance(
            expected.plan,
            card_revision_finish.CardRevisionFinishPlan,
        ):
            plan = expected.plan
            if (
                expected.kind not in {"review_staging", "promote_staging"}
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This card Apply-and-finish action no longer matches the "
                    "rendered review and consequences. Nothing was executed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                result = card_revision_finish.execute_card_revision_finish(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                try:
                    durable = card_revision_finish.inspect_card_revision_finish(
                        self.config,
                        plan.fingerprint,
                    )
                except (JankiError, OSError, TypeError, ValueError):
                    recovery = ""
                else:
                    recovery = (
                        f" Durable receipt {durable.receipt_id} is in state "
                        f"{durable.state}; resume that exact receipt instead of "
                        "confirming or paying again."
                    )
                raise RevisionRefusal(f"{exc}{recovery}") from exc
            if not result.succeeded:
                detail = (
                    f" Audio stopped in state {result.audio.state}: "
                    f"{result.audio.stopped_by or 'the transaction did not complete'}."
                    if result.audio is not None
                    else ""
                )
                raise RevisionRefusal(
                    f"Apply and finish paused in durable state {result.state}."
                    f"{detail} Resume exact receipt {result.receipt_id}; do not "
                    "confirm or pay for this work again."
                )
            output = self._display_path(fresh_config, result.output_path)
            return RevisionExecution(
                message=(
                    f"Applied reviewed cards {', '.join(plan.record_ids)}, created "
                    f"their selected audio, and built {result.card_count} card(s) "
                    f"at {output}. Finish receipt: {result.receipt_id}. Package "
                    f"SHA-256: {result.package_sha256}."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "enrich_cards":
            plan = expected.plan
            if (
                not isinstance(plan, ai_enrichment.AiEnrichmentPlan)
                or confirmation.expected_fingerprint != plan.plan_fingerprint
            ):
                raise RevisionRefusal(
                    "This AI-enrichment batch no longer matches the rendered plan. "
                    "Nothing was sent."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = ai_enrichment.run_ai_enrichment(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            staged = [item for item in execution.results if item.state == "staged"]
            unchanged = [
                item for item in execution.results if item.state == "no_changes"
            ]
            paths = ", ".join(
                self._display_path(fresh_config, item.staging_path)
                for item in staged
                if item.staging_path is not None
            )
            message = (
                f"Saved {len(staged)} AI-enrichment proposal(s)"
                + (f" at {paths}" if paths else "")
                + f"; {len(unchanged)} answer(s) proposed no writable change. "
                "Canonical cards are unchanged."
            )
            if len(staged) == 1 and staged[0].staging_path is not None:
                try:
                    review = self._prepare_ai_enrichment_finish(
                        fresh_config,
                        staging_path=staged[0].staging_path,
                        record_id=staged[0].record_id,
                        instruction=expected.instruction,
                    )
                except Exception as exc:  # noqa: BLE001 - paid output is durable
                    detail = str(exc) or type(exc).__name__
                    return RevisionExecution(
                        message=message,
                        finish=None,
                        finish_unavailable=(
                            "The paid AI-enrichment proposal is safely staged, but "
                            f"Janki could not prepare its one Apply and finish review: "
                            f"{detail} Do not repeat enrichment. Repair the named local "
                            "prerequisite, then reopen this exact proposal."
                        ),
                    )
                return RevisionExecution(
                    message=(
                        message
                        + " Review the exact current and proposed fields and examples "
                        "below; one Apply and finish confirmation records that review, "
                        "promotes it, creates selected-card audio, and builds its "
                        "persisted focused deck."
                    ),
                    finish=review,
                )
            return RevisionExecution(
                message=message,
                finish=None,
                complete=not staged,
                review_required=(
                    "This batch produced more than one independent enrichment proposal. "
                    "Open one exact proposal at a time for its single Apply and finish "
                    "review; do not repeat the paid enrichment calls."
                    if staged
                    else None
                ),
            )
        if expected.kind == "revise_deck":
            plan = expected.plan
            if confirmation.expected_fingerprint != getattr(
                plan, "plan_fingerprint", None
            ):
                raise RevisionRefusal(
                    "This whole-deck revision no longer matches the rendered plan. "
                    "Nothing was sent."
                )
            return self._stage_deck_revision(plan, progress=progress)
        if expected.kind == "revise_cards":
            plan = expected.plan
            if (
                not isinstance(plan, card_revision.CardRevisionPlan)
                or confirmation.expected_fingerprint != plan.plan_fingerprint
            ):
                raise RevisionRefusal(
                    "This card-revision plan no longer matches the rendered action. "
                    "Nothing was sent."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                result = card_revision.run_card_revision(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            staged = self._display_path(fresh_config, result.staging_path)
            return RevisionExecution(
                message=(
                    f"The card revision is staged at {staged}. Canonical cards "
                    "are unchanged."
                ),
                finish=None,
                review_required=(
                    "Review the exact proposed Japanese and field changes in the "
                    "Workbench. Generic card review, promotion, audio, and build are "
                    "currently separate exact actions; Janki will not pretend that "
                    "one confirmation covers consequences it has not bound."
                ),
            )
        if expected.kind in {"generate_audio", "build_deck"}:
            plan = expected.plan
            if (
                not isinstance(plan, assistant_actions.AssistantActionPlan)
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This Assistant action no longer matches the rendered plan. "
                    "Nothing was executed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_actions.execute_action(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            if expected.kind == "build_deck":
                built = execution.result
                if not hasattr(built, "package_sha256"):
                    raise RevisionRefusal("The deck build returned an invalid result.")
                output = self._display_path(fresh_config, built.output_path)
                return RevisionExecution(
                    message=(
                        f"Built {built.card_count} card(s) at {output}. Package "
                        f"SHA-256: {built.package_sha256}."
                    ),
                    finish=None,
                    complete=True,
                )

            audio = execution.result
            if not hasattr(audio, "state"):
                raise RevisionRefusal("The audio service returned an invalid result.")
            detail = (
                f"Audio state {audio.state}: generated {audio.file_count} clip(s); "
                f"{audio.up_to_date} were already current."
            )
            prune_requested = bool(getattr(audio, "prune_requested", False))
            pruned_paths = tuple(getattr(audio, "pruned_paths", ()))
            if prune_requested:
                detail += f" Pruned {len(pruned_paths)} unreferenced clip(s)."
            if audio.state not in {"complete", "no-records"}:
                recovery = (
                    " Paid bytes remain recoverable; do not repeat the request blindly."
                    if audio.pending_recovery
                    else ""
                )
                failures = " ".join(
                    str(value)
                    for value in (
                        getattr(audio, "stopped_by", None),
                        getattr(audio, "prune_error", None),
                        getattr(audio, "ledger_error", None),
                    )
                    if value
                )
                raise RevisionRefusal(
                    detail + recovery + (f" {failures}" if failures else "")
                )
            warnings = " " + " ".join(audio.warnings) if audio.warnings else ""
            return RevisionExecution(
                message=detail + warnings,
                finish=None,
                complete=True,
            )
        if expected.kind == "create_deck":
            plan = expected.plan
            if (
                not isinstance(
                    plan,
                    assistant_deck_creation.AssistantDeckCreationPlan,
                )
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This deck-creation action no longer matches the rendered plan. "
                    "Nothing was created."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_deck_creation.execute_deck_creation(
                    fresh_config,
                    plan,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            path = self._display_path(fresh_config, execution.result.path)
            return RevisionExecution(
                message=(
                    f"Created {plan.request.name!r} at {path}. It is ready to "
                    "receive explicitly assigned cards."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "extract_source":
            plan = expected.plan
            if (
                not isinstance(plan, SourceExtractionPlan)
                or confirmation.expected_fingerprint != plan.request_fingerprint
            ):
                raise RevisionRefusal(
                    "This extraction action no longer matches the rendered plan. "
                    "Nothing was sent."
                )
            result = self.consume_replan_and_extract(
                SourceExtractionConfirmation(
                    preparation_id=plan.preparation_id,
                    source_name=plan.source_name,
                    expected_fingerprint=plan.request_fingerprint,
                ),
                progress=progress,
            )
            return RevisionExecution(
                message=result.message,
                finish=None,
                complete=True,
            )
        if expected.kind == "review_staging":
            plan = expected.plan
            if isinstance(
                plan,
                assistant_card_revision_review.AssistantCardRevisionReviewPlan,
            ):
                if confirmation.expected_fingerprint != plan.fingerprint:
                    raise RevisionRefusal(
                        "This card-revision review no longer matches the rendered "
                        "content. Nothing was reviewed."
                    )
                try:
                    fresh_config = ProjectConfig.load(self.config.root)
                    saved = assistant_card_revision_review.execute_card_revision_review(
                        fresh_config,
                        plan,
                    )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                accepted = ", ".join(saved.record_ids)
                return RevisionExecution(
                    message=(
                        f"Saved exact owner review for cards: {accepted}. The "
                        "reviewed proposal is now eligible for a separately "
                        "confirmed canonical promotion."
                    ),
                    finish=None,
                    complete=True,
                )
            if (
                not isinstance(
                    plan,
                    assistant_staging_review.AssistantStagingReviewPlan,
                )
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This staging-review action no longer matches the rendered "
                    "content. Nothing was reviewed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_staging_review.execute_staging_review(
                    fresh_config,
                    plan,
                )
            except assistant_staging_review.AssistantStagingReviewPartialError as exc:
                accepted = ", ".join(exc.outcome.accepted_record_ids) or "none"
                patterns = "saved" if exc.outcome.pattern_reviewed else "not saved"
                raise RevisionRefusal(
                    f"The review only partly completed: card approvals {accepted}; "
                    f"pattern review {patterns}. {exc} Refresh before any retry."
                ) from exc
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            accepted = ", ".join(execution.outcome.accepted_record_ids) or "none"
            pattern_note = (
                " The grammar pattern set was also marked reviewed."
                if execution.outcome.pattern_reviewed
                else ""
            )
            return RevisionExecution(
                message=f"Saved review authority for cards: {accepted}.{pattern_note}",
                finish=None,
                complete=True,
            )
        if expected.kind == "assign_cards":
            plan = expected.plan
            if (
                not isinstance(plan, assistant_assignment.AssistantAssignmentPlan)
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This staged-card assignment no longer matches the rendered "
                    "plan. Nothing was assigned."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_assignment.execute_assignment(
                    fresh_config,
                    plan,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            assigned = ", ".join(execution.assigned_record_ids)
            proposal = self._display_path(fresh_config, execution.plan.proposal_path)
            return RevisionExecution(
                message=(
                    f"Assigned staged card(s) {assigned} to "
                    f"{execution.plan.destination_name} in {proposal}. Canonical "
                    "cards and deck definitions are unchanged."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "delete_content":
            plan = expected.plan
            if isinstance(
                plan,
                assistant_deletion.AssistantCanonicalDeletionPlan,
            ):
                if confirmation.expected_fingerprint != plan.fingerprint:
                    raise RevisionRefusal(
                        "This canonical-card deletion no longer matches the rendered "
                        "plan. Nothing was removed."
                    )
                try:
                    fresh_config = ProjectConfig.load(self.config.root)
                    execution = assistant_deletion.execute_canonical_deletion(
                        fresh_config,
                        plan,
                    )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                removed = ", ".join(execution.removed_record_ids)
                return RevisionExecution(
                    message=(
                        f"Removed canonical card(s) {removed}. Configured deck "
                        "definitions, ledger history, media, staging proposals, and "
                        "existing generated packages remain. Rebuild affected decks "
                        "before using or importing those packages."
                    ),
                    finish=None,
                    complete=True,
                )
            if isinstance(plan, assistant_deletion.AssistantDeckDeletionPlan):
                if confirmation.expected_fingerprint != plan.fingerprint:
                    raise RevisionRefusal(
                        "This configured-deck deletion no longer matches the rendered "
                        "plan. Nothing was removed."
                    )
                try:
                    fresh_config = ProjectConfig.load(self.config.root)
                    execution = assistant_deletion.execute_deck_deletion(
                        fresh_config,
                        plan,
                    )
                except (JankiError, OSError, TypeError, ValueError) as exc:
                    raise RevisionRefusal(str(exc)) from exc
                removed = self._display_path(
                    fresh_config,
                    execution.removed_deck_path,
                )
                return RevisionExecution(
                    message=(
                        f"Removed configured deck definition {removed}. Canonical "
                        "cards, ledger history, media, and staging proposals are "
                        "unchanged. Its existing generated package remains and may "
                        "be stale. This does not uninstall the deck from Anki."
                    ),
                    finish=None,
                    complete=True,
                )
            if (
                not isinstance(
                    plan,
                    assistant_staging_actions.AssistantStagedDeletionPlan,
                )
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This staged-card deletion no longer matches the rendered "
                    "plan. Nothing was removed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_staging_actions.execute_staged_deletion(
                    fresh_config,
                    plan,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            removed = ", ".join(execution.removed_record_ids)
            proposal = self._display_path(fresh_config, execution.plan.proposal_path)
            return RevisionExecution(
                message=(
                    f"Removed staged card(s) {removed} from {proposal}. Canonical "
                    "cards and deck definitions are unchanged."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "reidentify_staged_card":
            plan = expected.plan
            if (
                not isinstance(
                    plan,
                    assistant_staging_actions.AssistantReidentificationPlan,
                )
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This staged-card identity change no longer matches the "
                    "rendered plan. Nothing was changed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_staging_actions.execute_reidentification(
                    fresh_config,
                    plan,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            proposal = self._display_path(fresh_config, execution.plan.proposal_path)
            return RevisionExecution(
                message=(
                    f"Changed staged identity {execution.old_record_id} to "
                    f"{execution.new_record_id} in {proposal}. Japanese examples "
                    "and canonical cards are unchanged."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "approve_coverage":
            plan = expected.plan
            if (
                not isinstance(
                    plan,
                    assistant_staging_actions.AssistantCoverageApprovalPlan,
                )
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This owner coverage approval no longer matches the rendered "
                    "account. Nothing was approved."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_staging_actions.execute_coverage_approval(
                    fresh_config,
                    plan,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            proposal = self._display_path(fresh_config, execution.staging_path)
            return RevisionExecution(
                message=(
                    f"Recorded repository-owner coverage approval in {proposal}. "
                    "No cards were promoted."
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "promote_staging":
            plan = expected.plan
            if (
                not isinstance(plan, assistant_promotion.AssistantPromotionPlan)
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This promotion action no longer matches the rendered plan. "
                    "Nothing was promoted."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_promotion.execute_promotion_action(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            promoted = ", ".join(execution.result.promoted_ids) or "none"
            archive = (
                self._display_path(fresh_config, execution.result.archive_path)
                if execution.result.archive_path is not None
                else "none"
            )
            ledger_note = (
                f" Ledger follow-up is incomplete: {execution.result.ledger_error}."
                if execution.result.ledger_error is not None
                else ""
            )
            return RevisionExecution(
                message=(
                    f"Promotion state {execution.result.state}. Canonical card ids: "
                    f"{promoted}. Review archive: {archive}.{ledger_note}"
                ),
                finish=None,
                complete=True,
            )
        if expected.kind == "manage_operation":
            plan = expected.plan
            if (
                not isinstance(plan, assistant_operations.OperationActionPlan)
                or confirmation.expected_fingerprint != plan.fingerprint
            ):
                raise RevisionRefusal(
                    "This paid-operation action no longer matches the rendered "
                    "plan. Nothing was read or changed."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                execution = assistant_operations.execute_operation_action(
                    fresh_config,
                    plan,
                    progress=progress,
                )
            except (JankiError, OSError, TypeError, ValueError) as exc:
                raise RevisionRefusal(str(exc)) from exc
            operation_id = plan.operation.operation_id
            if plan.action == "recover":
                if (
                    execution.recovery_kind is None
                    or not execution.result_names
                    or len(execution.result_names) != len(execution.result_sha256)
                ):
                    raise RevisionRefusal(
                        "The paid-operation recovery returned no exact durable result."
                    )
                destinations = ", ".join(
                    f"{name} (SHA-256 {digest})"
                    for name, digest in zip(
                        execution.result_names,
                        execution.result_sha256,
                        strict=True,
                    )
                )
                answer = (
                    f"\n\nRecovered decoded Assistant answer:\n\n"
                    f"{execution.assistant_answer}"
                    if execution.assistant_answer is not None
                    else ""
                )
                intent_note = (
                    " The complete Assistant manifest durably preserves "
                    f"{execution.assistant_action_intent_count} typed action intent"
                    f"{'s' if execution.assistant_action_intent_count != 1 else ''}; "
                    "it was not prepared or executed because this recovery cannot "
                    "safely reconstruct a new confirmation card from the current "
                    "chat focus. Do not repeat the paid call."
                    if execution.assistant_action_intent_count
                    else ""
                )
                return RevisionExecution(
                    message=(
                        f"Recovered paid operation {operation_id} through its existing "
                        f"{execution.recovery_kind} recovery service. Durable result: "
                        f"{destinations}. No provider call was made.{intent_note}{answer}"
                    ),
                    finish=None,
                    complete=True,
                    remember_in_chat_context=False,
                )
            if plan.action == "show_reply":
                if execution.reply is None or execution.reply_sha256 is None:
                    raise RevisionRefusal(
                        "The paid-operation reply reader returned no exact bytes."
                    )
                return RevisionExecution(
                    message=self._private_reply_message(
                        operation_id,
                        execution.reply,
                        execution.reply_sha256,
                    ),
                    finish=None,
                    complete=True,
                    remember_in_chat_context=False,
                )
            if plan.action == "end":
                if execution.operation is None:
                    raise RevisionRefusal(
                        "The paid-operation end action returned no durable state."
                    )
                return RevisionExecution(
                    message=(
                        f"Ended paid operation {operation_id}. Its durable state is "
                        f"{execution.operation.state}. No recovery evidence was "
                        "discarded and no new paid call was authorized."
                    ),
                    finish=None,
                    complete=True,
                )
            return RevisionExecution(
                message=(
                    f"Forgot paid operation {operation_id} and retired its exact "
                    "bound recovery evidence. No new paid call was authorized."
                ),
                finish=None,
                complete=True,
            )
        raise RevisionRefusal(
            f"The local service for {expected.kind!r} is not available. Nothing changed."
        )

    @staticmethod
    def _private_reply_message(
        operation_id: str,
        payload: bytes,
        digest: str,
    ) -> str:
        try:
            body = payload.decode("utf-8", errors="strict")
            encoding = "UTF-8"
        except UnicodeError:
            body = base64.b64encode(payload).decode("ascii")
            encoding = "base64 of the exact bytes"
        longest = 0
        run = 0
        for character in body:
            if character == "`":
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        fence = "`" * max(3, longest + 1)
        return (
            f"Exact private recovery reply for paid operation {operation_id} "
            f"({len(payload)} bytes · SHA-256 {digest} · {encoding}). This reply "
            "is shown in the thread but will not be sent to the conversation model.\n\n"
            f"{fence}text\n{body}\n{fence}"
        )

    @staticmethod
    def _display_path(config: ProjectConfig, path: Path) -> str:
        try:
            return path.resolve().relative_to(config.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _review_example(example: ExampleSentence) -> RevisionExampleReview:
        return RevisionExampleReview(
            register=example.register,
            japanese=example.japanese,
            furigana=example.furigana,
            english=example.english,
        )

    def _finish_review(
        self,
        config: ProjectConfig,
        plan: revision_finish.RevisionFinishPlan,
        *,
        preparation_id: str,
    ) -> RevisionFinishReview:
        provider = plan.audio.example_provider
        if provider is None:
            raise RevisionRefusal(
                "The exact example-audio provider is unavailable, so Apply and "
                "finish cannot be reviewed yet. The staged proposal was not applied."
            )
        records = tuple(
            RevisionRecordReview(
                record_id=record_id,
                current_examples=tuple(
                    self._review_example(example)
                    for example in plan.revision.current_drill_examples[record_id]
                ),
                proposed_examples=tuple(
                    self._review_example(example)
                    for example in plan.revision.drill_examples[record_id]
                ),
            )
            for record_id in plan.revision.selected_record_ids
        )
        counts = plan.audio.example_counts
        return RevisionFinishReview(
            preparation_id=preparation_id,
            request_fingerprint=plan.fingerprint,
            target=plan.revision.deck_relative_path,
            current_form_note=plan.revision.current_form_note,
            proposed_form_note=plan.revision.form_note,
            records=records,
            audio_provider=provider.name,
            audio_model=provider.settings.get("model") or "not separately named",
            audio_access=provider.access,
            audio_total=counts.total,
            audio_current=counts.current,
            audio_recoverable=counts.recoverable,
            audio_provider_required=counts.provider_required,
            output_path=self._display_path(config, plan.build.output_path),
            card_count=plan.build.card_count,
        )

    def consume_replan_and_finish(
        self,
        confirmation: RevisionFinishConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> RevisionFinishExecution:
        """Consume, re-plan, compare, and run the shared aggregate finish."""

        with self._plan_lock:
            expected = self._finish_plans.pop(confirmation.preparation_id, None)
        if isinstance(expected, card_revision_finish.CardRevisionFinishPlan):
            target = self._display_path(self.config, expected.deck_path)
            if (
                confirmation.target != target
                or confirmation.expected_fingerprint != expected.fingerprint
            ):
                raise RevisionRefusal(
                    "This Apply and finish review is missing, stale, already used, or "
                    "belongs elsewhere. Nothing was applied. Reopen the existing "
                    "staged proposal; do not repeat its paid writing call."
                )
            try:
                fresh_config = ProjectConfig.load(self.config.root)
                result = card_revision_finish.execute_card_revision_finish(
                    fresh_config,
                    expected,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001 - report durable receipt truth
                problem = str(exc) or type(exc).__name__
                try:
                    result = card_revision_finish.inspect_card_revision_finish(
                        ProjectConfig.load(self.config.root),
                        expected.fingerprint,
                    )
                except Exception as inspect_error:  # noqa: BLE001 - no durable receipt
                    detail = str(inspect_error) or type(inspect_error).__name__
                    raise RevisionRefusal(
                        f"Apply and finish stopped before Janki could prove durable "
                        f"authority: {problem} Receipt inspection also failed: {detail} "
                        "Nothing may be retried or repaid until the staged proposal and "
                        "finish records are inspected."
                    ) from exc
                return self._card_finish_execution(
                    fresh_config,
                    result,
                    problem=problem,
                )
            return self._card_finish_execution(fresh_config, result, problem=None)
        if (
            expected is None
            or confirmation.target != expected.revision.deck_relative_path
            or confirmation.expected_fingerprint != expected.fingerprint
        ):
            raise RevisionRefusal(
                "This Apply and finish review is missing, stale, already used, or "
                "belongs elsewhere. Nothing was applied. Return to the Workbench "
                "and reopen the existing staged proposal for a fresh finish review; "
                "do not repeat the paid revise call."
            )

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            fresh = revision_finish.plan_revision_finish(
                fresh_config,
                expected.revision.staging_path,
            )
        except (JankiError, OSError, TypeError, ValueError) as exc:
            raise RevisionRefusal(
                f"The reviewed Apply and finish plan could not be re-created: {exc} "
                "Nothing was applied. Return to the Workbench and reopen the existing "
                "staged proposal for a fresh finish review; do not repeat the paid "
                "revise call."
            ) from exc
        if not secrets.compare_digest(fresh.fingerprint, expected.fingerprint) or dict(
            fresh.authority
        ) != dict(expected.authority):
            raise RevisionRefusal(
                "The staged content or its audio/build consequences changed after "
                "you reviewed them. Nothing was applied. Return to the Workbench and "
                "reopen the existing staged proposal for a fresh finish review; do "
                "not repeat the paid revise call."
            )

        problem: str | None = None
        try:
            result = revision_finish.execute_revision_finish(
                fresh_config,
                fresh,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 - report durable receipt truth
            problem = str(exc) or type(exc).__name__
            try:
                result = revision_finish.inspect_revision_finish(
                    fresh_config,
                    fresh.fingerprint,
                )
            except Exception as inspect_error:  # noqa: BLE001 - state is genuinely unknown
                detail = str(inspect_error) or type(inspect_error).__name__
                raise RevisionRefusal(
                    f"Apply and finish stopped: {problem} Janki could not prove a "
                    f"durable finish state: {detail} Do not click again or prepare a "
                    "replacement until you inspect the operation journal and finish "
                    "records."
                ) from exc
        return self._finish_execution(fresh_config, result, problem=problem)

    def _card_finish_execution(
        self,
        config: ProjectConfig,
        result: card_revision_finish.CardRevisionFinishResult,
        *,
        problem: str | None,
    ) -> RevisionFinishExecution:
        states = {
            "authorized": (
                "The exact finish authority is durable; the reviewed content has not "
                "yet been applied."
            ),
            "promoted": (
                "The reviewed content is promoted; selected-card audio and the Anki "
                "package still need to finish."
            ),
            "audio_complete": (
                "The reviewed content and selected-card audio are durable; the Anki "
                "package still needs to finish."
            ),
            "complete": (
                "The reviewed content, selected-card audio, and Anki package are complete."
            ),
        }
        summary = states[result.state]
        if problem is not None:
            summary = f"Apply and finish stopped: {problem} {summary}"
        if result.state != "complete":
            summary += (
                f" Resume only receipt {result.receipt_id}; current and recoverable "
                "audio must be reused and no paid call may be repeated."
            )
        else:
            summary += f" Package: {self._display_path(config, result.output_path)}."
        return RevisionFinishExecution(
            message=summary,
            receipt_id=result.receipt_id,
            state=result.state,
            target=self._display_path(config, result.deck_path),
            output_path=self._display_path(config, result.output_path),
            package_sha256=result.package_sha256,
            card_count=result.card_count,
        )

    def _finish_execution(
        self,
        config: ProjectConfig,
        result: revision_finish.RevisionFinishResult,
        *,
        problem: str | None,
    ) -> RevisionFinishExecution:
        states = {
            "authorized": (
                "The exact finish authority is durable, but the reviewed revision "
                "is not yet proven applied."
            ),
            "revision_applied": (
                "The reviewed revision is applied and archived; example audio and "
                "the Anki package still need to finish."
            ),
            "audio_complete": (
                "The reviewed revision and exact example audio are durable; the "
                "Anki package still needs to finish."
            ),
            "complete": ("The reviewed revision, example audio, and Anki package are complete."),
        }
        summary = states[result.state]
        if problem is not None:
            summary = f"Apply and finish stopped: {problem} {summary}"
        if result.state != "complete":
            summary += (
                f" Resume only receipt {result.receipt_id}; do not create a broader "
                "replacement plan."
            )
        else:
            summary += f" Package: {self._display_path(config, result.output_path)}."
        return RevisionFinishExecution(
            message=summary,
            receipt_id=result.receipt_id,
            state=result.state,
            target=self._display_path(config, result.deck_path),
            output_path=self._display_path(config, result.output_path),
            package_sha256=result.package_sha256,
            card_count=result.card_count,
        )


def discover_revision_adapter(
    config: ProjectConfig,
) -> tuple[RevisionAssistantAdapter, tuple[str, ...]]:
    """List every configured deck and allowlist each supported rich drill target."""

    empty = RevisionAssistantAdapter(config=config, deck_choices=(), _targets=())

    try:
        deck_paths = status.deck_files(config)
    except JankiError as exc:
        return empty, (
            f"assistant deck discovery was refused: {exc}. Source intake and "
            "extraction remain available; conversation and revision are disabled.",
        )

    choices: list[AssistantDeckChoice] = []
    targets: list[_RevisionDeckTarget] = []
    warnings: list[str] = []
    used_ids: set[str] = set()
    used_scopes: set[str] = set()
    project_root = config.root.resolve()
    for deck_path in deck_paths:
        try:
            resolved_deck_path = deck_path.resolve()
            scope = resolved_deck_path.relative_to(project_root).as_posix()
        except ValueError:
            warnings.append(
                f"assistant configured deck is outside the project root: {deck_path}. "
                "It was not made selectable."
            )
            continue
        if scope in used_scopes:
            warnings.append(
                f"assistant configured deck {deck_path} resolves to an already "
                f"listed deck ({scope}). It was not made selectable."
            )
            continue
        used_scopes.add(scope)
        label, record_ids, reason, warning = _inspect_deck(resolved_deck_path)
        if warning is not None:
            warnings.append(f"assistant could not safely inspect {deck_path}: {warning}")
        if not label.strip():
            warnings.append(
                f"assistant configured deck has no display name: {deck_path}. "
                "It was not made selectable."
            )
            continue
        deck_id = secrets.token_urlsafe(24)
        while deck_id in used_ids:
            deck_id = secrets.token_urlsafe(24)
        used_ids.add(deck_id)
        choice = AssistantDeckChoice(
            deck_id=deck_id,
            label=label,
            scope=scope,
            chat_supported=warning is None,
            revision_supported=record_ids is not None,
            unavailable_reason=reason,
        )
        choices.append(choice)
        targets.append(
            _RevisionDeckTarget(
                choice=choice,
                path=resolved_deck_path,
                record_ids=record_ids,
            )
        )

    duplicate_labels = {
        choice.label
        for choice in choices
        if sum(candidate.label == choice.label for candidate in choices) > 1
    }
    if duplicate_labels:
        for index, (choice, target) in enumerate(zip(choices, targets, strict=True)):
            if choice.label not in duplicate_labels:
                continue
            distinct_choice = replace(
                choice,
                label=f"{choice.label} ({Path(choice.scope).name})",
            )
            choices[index] = distinct_choice
            targets[index] = replace(target, choice=distinct_choice)

    return (
        RevisionAssistantAdapter(
            config=config,
            deck_choices=tuple(choices),
            _targets=tuple(targets),
        ),
        tuple(warnings),
    )
