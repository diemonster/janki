"""Bind project source intake and an optional exact deck target to ChatKit.

This module is imported only when ``[assistant] enabled = true``. Project-wide
attachment intake and extraction are always available. Discovery never guesses
which deck the owner meant: conversation and revision are enabled only when
exactly one configured conjugation deck carries nonempty rich drill examples,
and every card in that deck is selected in its stored order.
"""

from __future__ import annotations

import hashlib
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
    assistant_chat,
    describe_extraction,
    dispatch_extraction,
    revision,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.exporters.pattern_cards import read_drill_deck_content
from japanese_anki.workbench.assistant import (
    ChatReply,
    RevisionConfirmation,
    RevisionExecution,
    RevisionRefusal,
    SourceExtractionConfirmation,
    SourceExtractionExecution,
    SourceExtractionPlan,
)
from japanese_anki.workbench.assistant import RevisionPlan as AssistantRevisionPlan

__all__ = [
    "RevisionAssistantAdapter",
    "discover_revision_adapter",
]

_PROJECT_SCOPE = "janki-project"


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
    """Project intake plus an optional exact revision target behind ChatKit."""

    config: ProjectConfig
    deck_path: Path | None
    record_ids: tuple[str, ...]
    _plans: dict[str, revision.RevisionPlan] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _extraction_expectations: dict[str, ExtractionDispatchExpectation] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _plan_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def deck_scope(self) -> str:
        if self.deck_path is None:
            return _PROJECT_SCOPE
        return self.deck_path.resolve().relative_to(self.config.root.resolve()).as_posix()

    @property
    def conversation_available(self) -> bool:
        """Whether this adapter has the exact deck scope conversation requires."""

        return self.deck_path is not None

    def _revision_deck(self) -> Path:
        if self.deck_path is None:
            raise RevisionRefusal(
                "Janki has no unique deck selected for conversation or revision. "
                "Source attachment and extraction remain available; configure exactly "
                "one eligible drill deck before asking about or changing a deck."
            )
        return self.deck_path

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
    ) -> ChatReply:
        """Answer one ordinary turn without granting any revision authority."""

        self._revision_deck()
        if deck_scope != self.deck_scope:
            raise RevisionRefusal("The requested deck is outside this assistant's scope.")
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = assistant_chat.plan_chat(
                fresh_config,
                deck_scope=deck_scope,
                history=history,
                message=message,
            )
            result = assistant_chat.run_chat(fresh_config, plan, progress=progress)
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        return ChatReply(text=result.answer)

    def prepare_revision(
        self,
        *,
        deck_scope: str,
        instruction: str,
    ) -> AssistantRevisionPlan:
        """Plan the full ordered deck selection without granting authority."""

        deck_path = self._revision_deck()
        if deck_scope != self.deck_scope:
            raise RevisionRefusal("The requested deck is outside this assistant's scope.")
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = revision.plan_revision(
                fresh_config,
                deck_path,
                self.record_ids,
                instruction,
            )
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        with self._plan_lock:
            self._plans[plan.plan_fingerprint] = plan
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
        request_bytes_sha256 = hashlib.sha256(
            plan.provider_plan.request_bytes
        ).hexdigest()
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
        return AssistantRevisionPlan(
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
            target_display = target.staging_path.relative_to(
                fresh_config.root.resolve()
            ).as_posix()
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
            effects.append(
                "Also send the existing expression list so prose extraction can skip it"
            )
        replaces = consent.replaces is not None
        if replaces:
            card_review = (
                f"{consent.replaces_cards} cards, {consent.replaces_state}"
                if consent.replaces_state
                else "the existing card review"
            )
            grammar = (
                f"; {consent.replaces_grammar}" if consent.replaces_grammar else ""
            )
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
            target_display = outcome.target.relative_to(
                fresh_config.root.resolve()
            ).as_posix()
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
        """Consume the displayed plan and let ``run_revision`` re-plan under lock."""

        self._revision_deck()
        with self._plan_lock:
            expected = self._plans.pop(confirmation.expected_fingerprint, None)
        if (
            expected is None
            or confirmation.deck_scope != self.deck_scope
            or confirmation.target != expected.deck_relative_path
            or confirmation.instruction != expected.owner_instruction
            or confirmation.expected_fingerprint != expected.plan_fingerprint
        ):
            raise RevisionRefusal(
                "This revision plan is missing, stale, already consumed, or belongs "
                "to another deck. Prepare it again; nothing was sent."
            )

        try:
            fresh_config = ProjectConfig.load(self.config.root)
            result = revision.run_revision(fresh_config, expected, progress=progress)
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        try:
            staged = result.staging_path.relative_to(fresh_config.root.resolve()).as_posix()
        except ValueError:
            staged = str(result.staging_path)
        return RevisionExecution(
            message=(
                f"The revision proposal is staged at {staged}. Its exact old/new "
                "content has not been accepted. Review it in the main workbench; "
                "this Assistant will not start another confirmation chain."
            ),
        )


def discover_revision_adapter(
    config: ProjectConfig,
) -> tuple[RevisionAssistantAdapter, tuple[str, ...]]:
    """Always return project intake, with revision only when its target is unique."""

    project_only = RevisionAssistantAdapter(
        config=config,
        deck_path=None,
        record_ids=(),
    )

    try:
        deck_paths = status.deck_files(config)
    except JankiError as exc:
        return project_only, (
            f"assistant deck discovery was refused: {exc}. Source intake and "
            "extraction remain available; conversation and revision are disabled.",
        )

    candidates: list[Path] = []
    for deck_path in deck_paths:
        try:
            deck_config, _records = resolve_deck_records(deck_path)
        except JankiError as exc:
            return project_only, (
                f"assistant could not safely inspect configured deck {deck_path}: "
                f"{exc}. Source intake and extraction remain available; conversation "
                "and revision are disabled.",
            )
        examples = deck_config.get("drill_examples")
        if (
            str(deck_config.get("kind") or "").strip().lower() == "conjugation"
            and bool(examples)
        ):
            candidates.append(deck_path)

    if len(candidates) != 1:
        found = "none" if not candidates else ", ".join(path.name for path in candidates)
        return project_only, (
            "assistant needs exactly one configured conjugation deck with nonempty "
            f"drill_examples; found {found}. No deck was guessed. Source intake and "
            "extraction remain available; conversation and revision are disabled.",
        )

    deck_path = candidates[0]
    try:
        content = read_drill_deck_content(deck_path)
    except JankiError as exc:
        return project_only, (
            f"assistant revision deck {deck_path} is not usable: {exc}. Source intake "
            "and extraction remain available; conversation and revision are disabled.",
        )
    if not content.record_ids or not content.drill_examples:
        return project_only, (
            f"assistant revision deck {deck_path} has no complete drill scope; "
            "no deck was guessed. Source intake and extraction remain available; "
            "conversation and revision are disabled.",
        )
    return (
        RevisionAssistantAdapter(
            config=config,
            deck_path=deck_path.resolve(),
            record_ids=content.record_ids,
        ),
        (),
    )
