"""Bind ChatKit conversation and revision to one conjugation deck.

This module is imported only when ``[assistant] enabled = true``.  Discovery
never guesses which deck the owner meant: the first vertical slice is exposed
only when exactly one configured conjugation deck carries nonempty rich drill
examples, and every card in that deck is selected in its stored order.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from japanese_anki import status
from japanese_anki.application import assistant_chat, revision
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.exporters.pattern_cards import read_drill_deck_content
from japanese_anki.workbench.assistant import (
    ChatReply,
    RevisionConfirmation,
    RevisionExecution,
    RevisionRefusal,
)
from japanese_anki.workbench.assistant import RevisionPlan as AssistantRevisionPlan

__all__ = [
    "RevisionAssistantAdapter",
    "discover_revision_adapter",
]


@dataclass(slots=True)
class RevisionAssistantAdapter:
    """One chat scope and full-deck target behind ChatKit's authority boundary."""

    config: ProjectConfig
    deck_path: Path
    record_ids: tuple[str, ...]
    _plans: dict[str, revision.RevisionPlan] = field(
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
        return self.deck_path.resolve().relative_to(self.config.root.resolve()).as_posix()

    @property
    def chat_disclosure(self) -> str:
        """Current send boundary, reloaded whenever the Assistant page renders."""

        fresh = ProjectConfig.load(self.config.root)
        billing = (
            "the logged-in Claude Pro/Max subscription via Claude Code"
            if fresh.assistant_provider == "claude-code"
            else "Anthropic API billing"
        )
        try:
            assistant_store = fresh.assistant_dir.relative_to(
                fresh.root.resolve()
            ).as_posix()
        except ValueError:
            assistant_store = str(fresh.assistant_dir)
        return (
            f"Ask uses {fresh.assistant_provider}, {billing}, model "
            f"{fresh.assistant_model}. Each send is one journaled read-only model "
            "call containing only this deck path, the bounded visible conversation "
            "history, and the current message. The sent context and answer are "
            f"stored under {assistant_store} as repository provenance."
        )

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
    ) -> ChatReply:
        """Answer one ordinary turn without granting any revision authority."""

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

        if deck_scope != self.deck_scope:
            raise RevisionRefusal("The requested deck is outside this assistant's scope.")
        try:
            fresh_config = ProjectConfig.load(self.config.root)
            plan = revision.plan_revision(
                fresh_config,
                self.deck_path,
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

    def consume_replan_and_execute(
        self,
        confirmation: RevisionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> RevisionExecution:
        """Consume the displayed plan and let ``run_revision`` re-plan under lock."""

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
) -> tuple[RevisionAssistantAdapter | None, tuple[str, ...]]:
    """Return the sole safe first-slice target, or warnings explaining why not."""

    try:
        deck_paths = status.deck_files(config)
    except JankiError as exc:
        return None, (f"assistant deck discovery was refused: {exc}",)

    candidates: list[Path] = []
    for deck_path in deck_paths:
        try:
            deck_config, _records = resolve_deck_records(deck_path)
        except JankiError as exc:
            return None, (
                f"assistant could not safely inspect configured deck {deck_path}: {exc}",
            )
        examples = deck_config.get("drill_examples")
        if (
            str(deck_config.get("kind") or "").strip().lower() == "conjugation"
            and bool(examples)
        ):
            candidates.append(deck_path)

    if len(candidates) != 1:
        found = "none" if not candidates else ", ".join(path.name for path in candidates)
        return None, (
            "assistant needs exactly one configured conjugation deck with nonempty "
            f"drill_examples; found {found}. No deck was guessed.",
        )

    deck_path = candidates[0]
    try:
        content = read_drill_deck_content(deck_path)
    except JankiError as exc:
        return None, (f"assistant revision deck {deck_path} is not usable: {exc}",)
    if not content.record_ids or not content.drill_examples:
        return None, (
            f"assistant revision deck {deck_path} has no complete drill scope; "
            "no deck was guessed.",
        )
    return (
        RevisionAssistantAdapter(
            config=config,
            deck_path=deck_path.resolve(),
            record_ids=content.record_ids,
        ),
        (),
    )
