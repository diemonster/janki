"""Bind project source intake and explicitly selected deck targets to ChatKit.

This module is imported only when ``[assistant] enabled = true``. Project-wide
attachment intake and extraction are always available. Discovery lists every
configured deck but never guesses which one the owner meant. Every safely read
deck is selectable for conversation. Only a conjugation deck carrying complete
rich drill examples is selectable for revision, and every card in that selected
deck is revised in its stored order.
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


@dataclass(frozen=True, slots=True)
class _RevisionDeckTarget:
    """One startup-allowlisted deck behind an opaque browser-facing id."""

    choice: AssistantDeckChoice
    path: Path
    record_ids: tuple[str, ...] | None


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
    _finish_plans: dict[str, revision_finish.RevisionFinishPlan] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _plan_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def _target_for_scope(self, deck_scope: str) -> _RevisionDeckTarget:
        if not isinstance(deck_scope, str) or not deck_scope.strip():
            raise RevisionRefusal("Choose one supported deck before asking about it.")
        for target in self._targets:
            if target.choice.scope != deck_scope:
                continue
            if not target.choice.revision_supported or target.record_ids is None:
                raise RevisionRefusal(
                    target.choice.unavailable_reason
                    or "This configured deck is not available for deck changes."
                )
            return target
        raise RevisionRefusal("The requested deck is outside this assistant's allowlist.")

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

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
    ) -> ChatReply:
        """Answer one ordinary turn without granting any revision authority."""

        target = self._chat_target_for_scope(deck_scope)
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = assistant_chat.plan_chat(
                fresh_config,
                deck_scope=deck_scope,
                history=history,
                message=message,
                revision_supported=target.choice.revision_supported,
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

        target = self._target_for_scope(deck_scope)
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = revision.plan_revision(
                fresh_config,
                target.path,
                target.record_ids,
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

        with self._plan_lock:
            expected = self._plans.pop(confirmation.expected_fingerprint, None)
        if (
            expected is None
            or confirmation.deck_scope != expected.deck_relative_path
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
