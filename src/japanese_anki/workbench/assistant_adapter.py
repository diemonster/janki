"""Bind the deterministic ChatKit controller to conjugation-deck revision.

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
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_build, revision, revision_apply
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.exporters.pattern_cards import read_drill_deck_content
from japanese_anki.workbench.assistant import (
    OwnerActionConfirmation,
    OwnerActionExecution,
    OwnerActionPlan,
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
    """One full-deck revision target behind ChatKit's confirmation protocol."""

    config: ProjectConfig
    deck_path: Path
    record_ids: tuple[str, ...]
    _plans: dict[str, revision.RevisionPlan] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _followup_plans: dict[
        str,
        revision_apply.RevisionApplyPlan
        | audio_application.AudioPlan
        | deck_build.ConjugationDeckBuildPlan,
    ] = field(default_factory=dict, init=False, repr=False)
    _followup_bindings: dict[str, tuple[str, str]] = field(
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
                "Audio generation and deck building are separate owner-authorized actions.",
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
                "content has not been accepted."
            ),
            next_action="apply",
            continuation_context=staged,
        )

    def prepare_followup(
        self,
        kind: str,
        *,
        deck_scope: str,
        continuation_context: str,
    ) -> OwnerActionPlan:
        """Render one exact post-revision plan without granting authority."""

        if deck_scope != self.deck_scope:
            raise RevisionRefusal("The requested deck is outside this assistant's scope.")
        try:
            if kind == "apply":
                staging = self.config.root / continuation_context
                plan = revision_apply.plan_revision_apply(self.config, staging)
                effects = [
                    (
                        f"Apply proposal {plan.staging_path.name} to "
                        f"{plan.deck_relative_path} for {len(plan.selected_record_ids)} cards"
                    ),
                    f"Current form note: {plan.current_form_note}",
                    f"Proposed form note: {plan.form_note}",
                    (
                        f"Use committed revision operation {plan.operation_id}, model "
                        f"{plan.model}, request {plan.request_fingerprint}"
                    ),
                ]
                for record_id in plan.selected_record_ids:
                    effects.append(f"{record_id}:")
                    current_by_register = {
                        example.register: example
                        for example in plan.current_drill_examples[record_id]
                    }
                    for example in plan.current_drill_examples[record_id]:
                        effects.append(
                            f"  current {example.register}: {example.japanese} / "
                            f"{example.furigana} / {example.english}"
                        )
                    for example in plan.drill_examples[record_id]:
                        effects.append(
                            f"  proposed {example.register}: {example.japanese} / "
                            f"{example.furigana} / {example.english}"
                        )
                        current = current_by_register[example.register]
                        if current.japanese != example.japanese:
                            cleared = []
                            if current.audio:
                                cleared.append(f"audio reference {current.audio}")
                            if current.spoken_japanese:
                                cleared.append("spoken-Japanese override")
                            if cleared:
                                effects.append(
                                    f"  changed text clears {' and '.join(cleared)}"
                                )
                effects.append(
                    f"Archive the accepted proposal at {plan.archive_path.name}"
                )
                effects.append(f"Produce deck SHA-256 {plan.intended_deck_sha256}")
                rendered = OwnerActionPlan(
                    kind="apply",
                    fingerprint=plan.plan_fingerprint,
                    target=plan.deck_relative_path,
                    title="Confirm this exact proposal",
                    effects=tuple(effects),
                    disclosures=(
                        "This writes canonical deck content. It does not generate "
                        "audio or build a package.",
                        "No model or audio provider call is made by this action.",
                    ),
                )
            elif kind == "audio":
                current_sha = hashlib.sha256(
                    read_drill_deck_content(self.deck_path).revision.text.encode("utf-8")
                ).hexdigest()
                if current_sha != continuation_context:
                    raise RevisionRefusal(
                        "The deck changed after the confirmed proposal apply. Prepare the "
                        "fresh next action from its durable result."
                    )
                plan = audio_application.plan_deck_audio(
                    self.config,
                    self.deck_path,
                    words=False,
                    examples=True,
                    force=False,
                )
                provider = plan.example_provider
                if provider is None:
                    raise RevisionRefusal("The example-audio plan has no provider.")
                model = provider.settings.get("model", "not reported")
                effects = [
                    f"Voice {plan.example_counts.total} polite/casual example clips",
                    f"Keep {plan.example_counts.current} clips already current",
                    f"Recover {plan.example_counts.recoverable} already-paid staged clips",
                    f"Request {plan.example_counts.provider_required} clips from the provider",
                    (
                        f"Use {provider.name} ({provider.access}), voice {provider.voice}, "
                        f"speed {provider.speed}, model {model}"
                    ),
                    "Save audio references and media through the existing audio WAL transaction",
                ]
                for clip in plan.clips:
                    effects.append(
                        f"{clip.state}: {clip.request_input} → {clip.target} "
                        f"({clip.provider.name}, voice {clip.provider.voice})"
                    )
                paid = plan.example_counts.provider_required
                consequence = (
                    f"Confirming may make {paid} paid provider call{'s' if paid != 1 else ''}."
                    if provider.access == "paid-network" and paid
                    else "This exact plan requires no new paid provider call."
                )
                rendered = OwnerActionPlan(
                    kind="audio",
                    fingerprint=plan.fingerprint,
                    target=self.deck_scope,
                    title="Confirm this exact example-audio plan",
                    effects=tuple(effects),
                    disclosures=(
                        consequence,
                        "Force and media pruning are disabled for this action.",
                        "Building the .apkg remains a separate confirmation.",
                    ),
                )
            elif kind == "build":
                current_sha = hashlib.sha256(
                    read_drill_deck_content(self.deck_path).revision.text.encode("utf-8")
                ).hexdigest()
                if current_sha != continuation_context:
                    raise RevisionRefusal(
                        "The deck changed after the confirmed audio action. Prepare the "
                        "fresh build from its durable result."
                    )
                plan = deck_build.plan_conjugation_deck_build(
                    self.config,
                    self.deck_path,
                )
                try:
                    output = plan.output_path.relative_to(self.config.root.resolve()).as_posix()
                except ValueError:
                    output = str(plan.output_path)
                rendered = OwnerActionPlan(
                    kind="build",
                    fingerprint=plan.fingerprint,
                    target=output,
                    title="Confirm this exact deck build",
                    effects=(
                        f"Build {plan.card_count} cards for {plan.deck_name}",
                        f"Use the {plan.form} conjugation deck definition",
                        f"Write the package to {output}",
                        f"Bind deck SHA-256 {plan.deck_sha256}",
                        f"Bind source SHA-256 {plan.source_sha256}",
                    ),
                    disclosures=(
                        "This makes no provider call.",
                        "The package is a generated artifact; repository deck and "
                        "media remain authoritative.",
                    ),
                )
            else:
                raise RevisionRefusal("That post-revision action is not supported.")
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        with self._plan_lock:
            self._followup_plans[rendered.fingerprint] = plan
            self._followup_bindings[rendered.fingerprint] = (rendered.kind, rendered.target)
        return rendered

    def consume_followup(
        self,
        confirmation: OwnerActionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> OwnerActionExecution:
        """Consume one displayed post-revision plan; each service re-plans itself."""

        with self._plan_lock:
            expected = self._followup_plans.pop(confirmation.expected_fingerprint, None)
            binding = self._followup_bindings.pop(
                confirmation.expected_fingerprint,
                None,
            )
        if (
            expected is None
            or binding != (confirmation.kind, confirmation.target)
            or confirmation.deck_scope != self.deck_scope
        ):
            raise RevisionRefusal(
                "This action plan is missing, stale, already consumed, or belongs to another deck."
            )
        try:
            if confirmation.kind == "apply" and isinstance(
                expected, revision_apply.RevisionApplyPlan
            ):
                result = revision_apply.execute_revision_apply(
                    self.config,
                    expected,
                    progress=progress,
                )
                return OwnerActionExecution(
                    message=(
                        f"The revision is applied and archived at {result.archive_path.name}. "
                        "Audio has not been generated."
                    ),
                    next_action="audio",
                    continuation_context=result.deck_sha256,
                )
            if confirmation.kind == "audio" and isinstance(
                expected, audio_application.AudioPlan
            ):
                result = audio_application.execute_deck_audio(
                    self.config,
                    self.deck_path,
                    words=False,
                    examples=True,
                    expected_fingerprint=expected.fingerprint,
                    force=False,
                    prune=False,
                    progress=progress,
                )
                if not result.succeeded or result.pending_recovery or result.ledger_error:
                    reason = result.stopped_by or result.ledger_error or result.state
                    raise RevisionRefusal(f"Example audio did not complete durably: {reason}")
                deck_sha = hashlib.sha256(
                    read_drill_deck_content(self.deck_path).revision.text.encode("utf-8")
                ).hexdigest()
                return OwnerActionExecution(
                    message=(
                        f"Example audio completed: {result.file_count} files written and "
                        f"{result.up_to_date} already current."
                    ),
                    next_action="build",
                    continuation_context=deck_sha,
                )
            if confirmation.kind == "build" and isinstance(
                expected, deck_build.ConjugationDeckBuildPlan
            ):
                progress("Preparing deck")
                progress("Building package")
                result = deck_build.execute_conjugation_deck_build(self.config, expected)
                progress("Saving package")
                return OwnerActionExecution(
                    message=(
                        f"Built {result.card_count} cards at {result.output_path} "
                        f"(SHA-256 {result.package_sha256})."
                    )
                )
        except JankiError as exc:
            raise RevisionRefusal(str(exc)) from exc
        raise RevisionRefusal("The action confirmation did not match its prepared plan.")


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
