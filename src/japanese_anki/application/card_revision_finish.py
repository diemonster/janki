"""One owner-authorized review and finish for a staged card revision.

The generic ``revise`` pass stages ordinary vocabulary records.  This service
binds their exact visible field diff together with promotion, selected-card
audio, and package consequences behind one durable owner authority receipt.
It persists that receipt before recording review, and it never writes cards or
media itself: the existing review and promotion services remain their writers,
the audio transaction remains the media/ledger writer, and the package service
remains the only ``.apkg`` publisher.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from japanese_anki import ledger, operations, staging, status
from japanese_anki.application import (
    assistant_ai_enrichment_review,
    assistant_card_revision_review,
    assistant_context,
    assistant_promotion,
    card_revision,
    deck_package,
    study_curation,
)
from japanese_anki.application import audio as audio_application
from japanese_anki.application import promotion as promotion_application
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    atomic_write_text_bound,
    exclusive_path_lock,
    prepare_bound_directory,
    read_bytes_bound,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import openai_realtime, sentence_profile_for

__all__ = [
    "CardRevisionFinishError",
    "CardRevisionFinishPlan",
    "CardRevisionFinishResult",
    "execute_card_revision_finish",
    "inspect_card_revision_finish",
    "plan_ai_enrichment_finish",
    "plan_card_revision_finish",
    "resume_card_revision_finish",
]


CardRevisionFinishState = Literal[
    "authorized",
    "promoted",
    "audio_complete",
    "complete",
]
CardRevisionFinishPhase = Literal[
    "Preparing finish",
    "Applying reviewed cards",
    "Creating card audio",
    "Building Anki package",
    "Saving finish receipt",
]
CardRevisionFinishProgress = Callable[[CardRevisionFinishPhase], None]


class CardRevisionFinishError(JankiError):
    """The exact reviewed-card finish authority cannot safely continue."""


@dataclass(frozen=True, slots=True)
class CardRevisionFinishPlan:
    """Display-only consequences of one reviewed generic revision finish."""

    repository_root: Path
    proposal_kind: Literal["card_revision", "ai_enrichment"]
    resource_id: str
    instruction: str
    review: (
        assistant_card_revision_review.AssistantCardRevisionReviewPlan
        | assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan
        | None
    )
    promotion: assistant_promotion.AssistantPromotionPlan | None
    deck_path: Path
    record_ids: tuple[str, ...]
    audio: audio_application.AudioPlan
    build: deck_package.DeckPackagePlan
    projected_canonical_text: str
    projected_audio_text: str
    finish_directory: Path
    record_path: Path
    authority: Mapping[str, Any]
    fingerprint: str

    @property
    def provider_required_count(self) -> int:
        return self.audio.provider_required_count

    @property
    def paid_provider_call_possible(self) -> bool:
        return any(
            clip.state == "provider-required"
            and clip.provider.access == "paid-network"
            for clip in self.audio.clips
        )


@dataclass(frozen=True, slots=True)
class CardRevisionFinishResult:
    """Truthful durable state after an initial or resumed finish execution."""

    receipt_id: str
    state: CardRevisionFinishState
    record_path: Path
    deck_path: Path
    output_path: Path
    max_provider_calls: int
    package_sha256: str | None = None
    card_count: int | None = None
    audio: audio_application.AudioExecutionOutcome | None = None

    @property
    def succeeded(self) -> bool:
        return self.state == "complete"


_STATES: tuple[CardRevisionFinishState, ...] = (
    "authorized",
    "promoted",
    "audio_complete",
    "complete",
)
_STATE_INDEX = {state: position for position, state in enumerate(_STATES)}
_RECORD_KEYS = {
    "schema_version",
    "kind",
    "receipt_id",
    "state",
    "authority",
    "authorized_at",
    "updated_at",
    "promotion_receipt",
    "audio_receipt",
    "build_binding",
    "build_receipt",
    "paid_clip_reservations",
}
_RECEIPT_KEYS = {
    "promotion_receipt",
    "audio_receipt",
    "build_receipt",
}
_ALLOWED_CLIP_EVOLUTION = {
    "current": {"current"},
    "recoverable": {"recoverable", "current"},
    "provider-required": {"provider-required", "recoverable", "current"},
}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CardRevisionFinishError(
            f"Card revision finish authority is not finite JSON: {exc}"
        ) from exc


def _records_text(records: Sequence[VocabularyRecord]) -> str:
    return (
        json.dumps(
            [
                record.to_dict()
                for record in sorted(records, key=lambda item: item.id)
            ],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    root = config.root.resolve()
    target = path.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise CardRevisionFinishError(
            f"Card revision finish {label} escapes the repository: {target}"
        ) from exc
    value = relative.as_posix()
    if not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise CardRevisionFinishError(
            f"Card revision finish {label} is not repository-relative."
        )
    return value


def _authority_path(config: ProjectConfig, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CardRevisionFinishError(f"Card revision finish {label} path is malformed.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise CardRevisionFinishError(
            f"Card revision finish {label} path is not repository-relative."
        )
    path = (config.root.resolve() / Path(*pure.parts)).absolute()
    try:
        path.relative_to(config.root.resolve())
    except ValueError as exc:  # pragma: no cover - PurePosixPath check is stronger
        raise CardRevisionFinishError(
            f"Card revision finish {label} path escapes the repository."
        ) from exc
    return path


def _provider_wire(
    provider: audio_application.AudioProviderPlan | None,
) -> dict[str, Any] | None:
    if provider is None:
        return None
    return {
        "name": provider.name,
        "access": provider.access,
        "voice": provider.voice,
        "speed": provider.speed,
        "suffix": provider.suffix,
        "destination": provider.destination,
        "transport": provider.transport,
        "settings": dict(provider.settings),
    }


def _media_path(config: ProjectConfig, target: str) -> Path:
    pure = PurePosixPath(target)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != target
        or target in {"", ".", ".."}
    ):
        raise CardRevisionFinishError("An authorized card-audio target is malformed.")
    return config.media_dir.absolute() / audio_application.audio_cmd.AUDIO_SUBDIR / target


def _clip_wire(
    config: ProjectConfig,
    clip: audio_application.AudioClipPlan,
) -> dict[str, Any]:
    media_sha256 = None
    if clip.state == "current":
        try:
            media_sha256 = _sha(read_bytes_bound(_media_path(config, clip.target)))
        except (DataError, OSError) as exc:
            raise CardRevisionFinishError(
                f"Could not bind current card audio {clip.target!r}: {exc}"
            ) from exc
    return {
        "record_id": clip.record_id,
        "kind": clip.kind,
        "target": clip.target,
        "request_input": clip.request_input,
        "forced_accent": clip.forced_accent,
        "content_fingerprint": clip.content_fingerprint,
        "provider": _provider_wire(clip.provider),
        "initial_state": clip.state,
        "initial_recovery_key": clip.recovery_key,
        "initial_recovery_sha256": clip.recovery_sha256,
        "initial_recovery_source": clip.recovery_source,
        "initial_media_sha256": media_sha256,
    }


def _audio_wire(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    *,
    projected_input_sha256: str,
    projected_output_sha256: str,
) -> dict[str, Any]:
    return {
        "canonical_path": _relative(config, plan.canonical_path, label="audio owner"),
        "ledger_path": _relative(config, plan.ledger_path, label="audio ledger"),
        "operations_path": _relative(
            config,
            config.operations_file.resolve(),
            label="operation journal",
        ),
        "media_dir": _relative(config, plan.media_dir, label="media directory"),
        "record_ids": list(plan.record_ids),
        "targeted": plan.targeted,
        "words": plan.words,
        "examples": plan.examples,
        "force": plan.force,
        "prune": False,
        "word_provider": _provider_wire(plan.word_provider),
        "example_provider": _provider_wire(plan.example_provider),
        "clips": [_clip_wire(config, clip) for clip in plan.clips],
        "initial_plan_fingerprint": plan.fingerprint,
        "max_provider_calls": sum(
            clip.state == "provider-required" and clip.provider.access == "paid-network"
            for clip in plan.clips
        ),
        "projected_input_sha256": projected_input_sha256,
        "projected_output_sha256": projected_output_sha256,
    }


def _package_input_wire(
    config: ProjectConfig,
    item: deck_package.DeckPackageInput,
) -> dict[str, Any]:
    return {
        "label": item.label,
        "path": _relative(config, item.path, label=item.label),
        "sha256": item.sha256,
    }


def _package_wire(
    config: ProjectConfig,
    plan: deck_package.DeckPackagePlan,
) -> dict[str, Any]:
    return {
        "deck_path": _relative(config, plan.deck_path, label="package deck"),
        "output_path": _relative(config, plan.output_path, label="package output"),
        "kind": plan.kind,
        "deck_name": plan.deck_name,
        "variant": plan.variant,
        "card_types": list(plan.card_types),
        "note_count": plan.note_count,
        "card_count": plan.card_count,
        "record_ids": list(plan.record_ids),
        "deck_input": _package_input_wire(config, plan.deck_input),
        "source_inputs": [
            _package_input_wire(config, item) for item in plan.source_inputs
        ],
        "template_inputs": [
            _package_input_wire(config, item) for item in plan.template_inputs
        ],
        "media_inputs": [
            _package_input_wire(config, item) for item in plan.media_inputs
        ],
        "output_revision": plan.output_revision,
        "output_identity": (
            None if plan.output_identity is None else list(plan.output_identity)
        ),
        "configuration_fingerprint": plan.configuration_fingerprint,
        "plan_fingerprint": plan.fingerprint,
    }


def _promotion_wire(
    config: ProjectConfig,
    plan: assistant_promotion.AssistantPromotionPlan,
    target: card_revision.CardRevisionTarget,
) -> dict[str, Any]:
    return {
        "mode": "durable_review",
        "resource_id": plan.resource_id,
        "instruction": plan.instruction,
        "operation_id": target.operation_id,
        "proposal_path": _relative(config, plan.proposal_path, label="revision proposal"),
        "proposal_sha256": _sha(read_bytes_bound(plan.proposal_path)),
        "promotion_fingerprint": plan.fingerprint,
        "service_fingerprint": plan.service_fingerprint,
        "projection": plan.projection,
        "deck_path": _relative(config, target.deck_path, label="revision deck"),
        "record_ids": [record.id for record in plan.decision.records],
        "projected_canonical_sha256": _sha(
            _records_text(plan.decision.merged).encode("utf-8")
        ),
        "deck_ownership": _deck_ownership_wire(plan.decision.deck_ownership),
    }


def _deck_ownership_wire(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": value.record_id,
            "state": value.state,
            "memberships": [
                {
                    "name": membership.name,
                    "stem": membership.stem,
                    "takes": membership.takes,
                    "refusal": membership.refusal,
                }
                for membership in value.memberships
            ],
            "unreadable_decks": list(value.unreadable_decks),
        }
        for value in values
    ]


def _pending_review_wire(
    config: ProjectConfig,
    review: assistant_card_revision_review.AssistantCardRevisionReviewPlan,
    records: Sequence[VocabularyRecord],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    selected = set(review.record_ids)
    chosen = [record for record in records if record.id in selected]
    if len(chosen) != len(review.record_ids):
        raise CardRevisionFinishError(
            "The pending owner-review selection no longer matches the proposal."
        )
    return {
        "mode": "pending",
        "resource_id": review.resource_id,
        "proposal_path": _relative(
            config, review.proposal_path, label="revision proposal"
        ),
        "proposal_sha256": review.proposal_sha256,
        "review_fingerprint": review.fingerprint,
        "projection": review.projection,
        "record_ids": list(review.record_ids),
        "content_fingerprint": staging.card_revision_review_fingerprint(
            meta, chosen
        ),
    }


def _durable_review_wire(
    config: ProjectConfig,
    promotion: assistant_promotion.AssistantPromotionPlan,
) -> dict[str, Any]:
    marker = promotion.decision.meta.get(staging.CARD_REVISION_REVIEW_KEY)
    if not isinstance(marker, Mapping):
        raise CardRevisionFinishError(
            "The card revision has no exact durable owner-review marker."
        )
    return {
        "mode": "durable",
        "proposal_path": _relative(
            config, promotion.proposal_path, label="revision proposal"
        ),
        "proposal_sha256": _sha(read_bytes_bound(promotion.proposal_path)),
        "record_ids": [record.id for record in promotion.decision.records],
        "marker": dict(marker),
    }


def _pending_ai_review_wire(
    config: ProjectConfig,
    review: assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan,
    records: Sequence[VocabularyRecord],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    selected = set(review.record_ids)
    chosen = [record for record in records if record.id in selected]
    if len(chosen) != len(review.record_ids):
        raise CardRevisionFinishError(
            "The pending AI-enrichment review no longer matches the proposal."
        )
    return {
        "mode": "pending",
        "resource_id": review.resource_id,
        "proposal_path": _relative(
            config, review.proposal_path, label="AI-enrichment proposal"
        ),
        "proposal_sha256": review.proposal_sha256,
        "review_fingerprint": review.fingerprint,
        "projection": review.projection,
        "record_ids": list(review.record_ids),
        "content_fingerprint": staging.ai_enrichment_review_fingerprint(
            meta, chosen
        ),
    }


def _durable_ai_review_wire(
    config: ProjectConfig,
    promotion: assistant_promotion.AssistantPromotionPlan,
) -> dict[str, Any]:
    marker = promotion.decision.meta.get(staging.AI_ENRICHMENT_REVIEW_KEY)
    if not isinstance(marker, Mapping):
        raise CardRevisionFinishError(
            "The AI-enrichment proposal has no exact durable owner-review marker."
        )
    return {
        "mode": "durable",
        "proposal_path": _relative(
            config, promotion.proposal_path, label="AI-enrichment proposal"
        ),
        "proposal_sha256": _sha(read_bytes_bound(promotion.proposal_path)),
        "record_ids": [record.id for record in promotion.decision.records],
        "marker": dict(marker),
    }


def _pending_promotion_wire(
    config: ProjectConfig,
    *,
    resource_id: str,
    instruction: str,
    proposal_path: Path,
    proposal_sha256: str,
    target: card_revision.CardRevisionTarget,
    decision: promotion_application.PromotionDecision,
) -> dict[str, Any]:
    return {
        "mode": "after_review",
        "resource_id": resource_id,
        "instruction": instruction,
        "operation_id": target.operation_id,
        "proposal_path": _relative(
            config, proposal_path, label="revision proposal"
        ),
        "proposal_sha256": proposal_sha256,
        "promotion_fingerprint": None,
        "service_fingerprint": None,
        "projection": None,
        "deck_path": _relative(config, target.deck_path, label="revision deck"),
        "record_ids": [record.id for record in decision.records],
        "projected_canonical_sha256": _sha(
            _records_text(decision.merged).encode("utf-8")
        ),
        "deck_ownership": _deck_ownership_wire(decision.deck_ownership),
    }


def _ai_promotion_wire(
    config: ProjectConfig,
    plan: assistant_promotion.AssistantPromotionPlan,
    *,
    operation_id: str,
    deck_path: Path,
) -> dict[str, Any]:
    return {
        "mode": "durable_review",
        "resource_id": plan.resource_id,
        "instruction": plan.instruction,
        "operation_id": operation_id,
        "proposal_path": _relative(
            config, plan.proposal_path, label="AI-enrichment proposal"
        ),
        "proposal_sha256": _sha(read_bytes_bound(plan.proposal_path)),
        "promotion_fingerprint": plan.fingerprint,
        "service_fingerprint": plan.service_fingerprint,
        "projection": plan.projection,
        "deck_path": _relative(config, deck_path, label="focused deck"),
        "record_ids": [record.id for record in plan.decision.records],
        "projected_canonical_sha256": _sha(
            _records_text(plan.decision.merged).encode("utf-8")
        ),
        "deck_ownership": _deck_ownership_wire(plan.decision.deck_ownership),
    }


def _pending_ai_promotion_wire(
    config: ProjectConfig,
    *,
    resource_id: str,
    instruction: str,
    operation_id: str,
    proposal_path: Path,
    proposal_sha256: str,
    deck_path: Path,
    decision: promotion_application.PromotionDecision,
) -> dict[str, Any]:
    return {
        "mode": "after_review",
        "resource_id": resource_id,
        "instruction": instruction,
        "operation_id": operation_id,
        "proposal_path": _relative(
            config, proposal_path, label="AI-enrichment proposal"
        ),
        "proposal_sha256": proposal_sha256,
        "promotion_fingerprint": None,
        "service_fingerprint": None,
        "projection": None,
        "deck_path": _relative(config, deck_path, label="focused deck"),
        "record_ids": [record.id for record in decision.records],
        "projected_canonical_sha256": _sha(
            _records_text(decision.merged).encode("utf-8")
        ),
        "deck_ownership": _deck_ownership_wire(decision.deck_ownership),
    }


def _project_audio_references(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
) -> tuple[VocabularyRecord, ...]:
    """Return the exact post-audio canonical records implied by the plan."""

    selected = set(plan.record_ids)
    by_record: dict[str, list[audio_application.AudioClipPlan]] = defaultdict(list)
    for clip in plan.clips:
        by_record[clip.record_id].append(clip)
    projected: list[VocabularyRecord] = []
    used: set[tuple[str, str, str]] = set()
    for record in records:
        if record.id not in selected:
            projected.append(record)
            continue
        clips = by_record.get(record.id, [])
        word = [clip for clip in clips if clip.kind == "word"]
        if len(word) > 1:
            raise CardRevisionFinishError(
                f"Audio plan repeats word audio for {record.id!r}."
            )
        audio = record.audio
        if word:
            clip = word[0]
            expected = (
                f"janki-{ledger.word_audio_filename_fingerprint(record)}"
                f"{clip.provider.suffix}"
            )
            if clip.target != expected:
                raise CardRevisionFinishError(
                    f"Audio plan changed the word-audio identity for {record.id!r}."
                )
            audio = audio_application.audio_cmd.media_relative(
                _media_path(config, clip.target),
                plan.media_dir,
            )
            used.add((clip.record_id, clip.kind, clip.target))
        examples: list[ExampleSentence] = []
        for example in record.examples:
            prefix = f"janki-{ledger.example_audio_filename_fingerprint(record, example)}"
            matches = [
                clip
                for clip in clips
                if clip.kind == "example"
                and clip.target == f"{prefix}{clip.provider.suffix}"
            ]
            if len(matches) > 1:
                raise CardRevisionFinishError(
                    f"Audio plan repeats one example for {record.id!r}."
                )
            if not matches:
                examples.append(example)
                continue
            clip = matches[0]
            if (
                clip.request_input != ledger.example_audio_request(example)
                or clip.content_fingerprint
                != ledger.example_audio_content_fingerprint(example)
            ):
                raise CardRevisionFinishError(
                    f"Audio plan changed the spoken identity for {record.id!r}."
                )
            examples.append(
                replace(
                    example,
                    audio=audio_application.audio_cmd.media_relative(
                        _media_path(config, clip.target),
                        plan.media_dir,
                    ),
                )
            )
            used.add((clip.record_id, clip.kind, clip.target))
        projected.append(replace(record, audio=audio, examples=examples))
        selected.remove(record.id)
    expected = {(clip.record_id, clip.kind, clip.target) for clip in plan.clips}
    if selected or used != expected:
        raise CardRevisionFinishError(
            "Audio plan contains a card or clip outside the reviewed revision scope."
        )
    return tuple(projected)


def _projected_package_media(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
    deck_path: Path,
) -> dict[Path, str | None]:
    clip_by_path = {
        _media_path(config, clip.target).absolute(): clip for clip in plan.clips
    }
    allowed = frozenset(clip_by_path)
    try:
        paths = deck_package.project_deck_media_paths(
            deck_path,
            config,
            config.normalized_file.resolve(),
            records,
            allowed_missing_media=allowed,
        )
    except (JankiError, OSError, ValueError) as exc:
        raise CardRevisionFinishError(str(exc)) from exc
    projected: dict[Path, str | None] = {}
    for raw in paths:
        path = raw.absolute()
        clip = clip_by_path.get(path)
        if clip is None or clip.state == "current":
            projected[path] = _sha(read_bytes_bound(path))
        elif clip.state == "recoverable":
            projected[path] = clip.recovery_sha256
        else:
            projected[path] = None
    return projected


def _resolved_providers(
    config: ProjectConfig,
    *,
    chosen_provider: str | None,
    word_provider: Any | None,
    sentence_provider: Any | None,
) -> tuple[Any, Any]:
    words = word_provider or audio_application.resolve_word_provider(
        config, chosen_provider
    )
    sentences = sentence_provider or audio_application.resolve_sentence_provider(
        config, chosen_provider, words
    )
    return words, sentences


def _finish_directory(config: ProjectConfig) -> Path:
    return (config.staging_dir / "done" / "revisions").absolute()


def _focused_ai_deck(
    config: ProjectConfig,
    focus_resource_id: object,
    record_ids: Sequence[str],
) -> tuple[Path, frozenset[str]]:
    if not isinstance(focus_resource_id, str) or not focus_resource_id.strip():
        raise CardRevisionFinishError(
            "AI-enrichment Apply and finish needs the persisted unique focused "
            "deck; this proposal has none, so Janki will not infer a build destination."
        )
    broker = assistant_context.AssistantContextBroker(config)
    try:
        context = broker.deck_context(focus_resource_id)
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise CardRevisionFinishError(
            "The AI-enrichment proposal's persisted focused deck is no longer "
            f"one exact configured deck: {exc}"
        ) from exc
    if context.deck_kind != "vocabulary":
        raise CardRevisionFinishError(
            "AI-enrichment Apply and finish currently requires a vocabulary deck."
        )
    focused_ids = frozenset(record.id for record in context.records)
    outside = [record_id for record_id in record_ids if record_id not in focused_ids]
    if outside:
        raise CardRevisionFinishError(
            "The AI-enrichment proposal contains cards outside its persisted "
            f"focused deck: {', '.join(outside)}."
        )
    matches: list[Path] = []
    for candidate in status.deck_files(config):
        try:
            candidate_id = broker.resource_id_for_deck(candidate)
        except (JankiError, OSError, UnicodeError, ValueError):
            continue
        if candidate_id == focus_resource_id:
            matches.append(candidate.absolute())
    if len(matches) != 1:
        raise CardRevisionFinishError(
            "The AI-enrichment proposal's persisted focus does not resolve to "
            "one unique configured deck; Janki will not infer a destination."
        )
    return matches[0], focused_ids


def _plan_locked(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
    *,
    record_ids: Sequence[str] | None,
    chosen_provider: str | None,
    word_provider: Any,
    sentence_provider: Any,
    proposal_kind: Literal["card_revision", "ai_enrichment"] = "card_revision",
) -> CardRevisionFinishPlan:
    try:
        proposal = assistant_context.AssistantContextBroker(config).proposal_context(
            resource_id
        )
        if proposal.proposal_kind != proposal_kind:
            raise CardRevisionFinishError(
                f"Apply and finish requires one {proposal_kind} proposal."
            )
        proposed, meta = staging.read_staging(proposal.path)
        review: (
            assistant_card_revision_review.AssistantCardRevisionReviewPlan
            | assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan
            | None
        )
        promotion: assistant_promotion.AssistantPromotionPlan | None
        if proposal_kind == "card_revision":
            raw_revision = meta.get(staging.CARD_REVISION_KEY)
            operation_id = (
                raw_revision.get("operation_id")
                if isinstance(raw_revision, Mapping)
                else None
            )
            if not isinstance(operation_id, str):
                raise CardRevisionFinishError(
                    "The reviewed card revision has no durable operation identity."
                )
            target = card_revision.inspect_card_revision_target(config, operation_id)
            deck_path = target.deck_path.absolute()
            allowed_ids = frozenset(target.selected_record_ids)
            if (deck_package.deck_kind(deck_path) or "vocabulary") != "vocabulary":
                raise CardRevisionFinishError(
                    "Generic Apply and finish currently supports vocabulary decks; "
                    "rich conjugation revisions use their existing finish service."
                )
            if staging.CARD_REVISION_REVIEW_KEY in meta:
                promotion = assistant_promotion.plan_promotion_action(
                    config, resource_id, instruction
                )
                review = None
                decision = promotion.decision
                selected_ids = tuple(record.id for record in decision.records)
                if record_ids is not None and tuple(record_ids) != selected_ids:
                    raise CardRevisionFinishError(
                        "The durable owner-review selection differs from this finish request."
                    )
                review_authority = _durable_review_wire(config, promotion)
                promotion_authority = _promotion_wire(config, promotion, target)
            else:
                if record_ids is None:
                    raise CardRevisionFinishError(
                        "Apply and finish needs the exact visible card selection for owner review."
                    )
                review = assistant_card_revision_review.plan_card_revision_review(
                    config,
                    resource_id=resource_id,
                    record_ids=record_ids,
                )
                promotion = None
                decision = promotion_application.project_card_revision_review_promotion(
                    config,
                    review.proposal_path,
                    review.record_ids,
                    expected_revision=review.proposal_sha256,
                )
                selected_ids = tuple(review.record_ids)
                review_authority = _pending_review_wire(
                    config, review, proposed, meta
                )
                promotion_authority = _pending_promotion_wire(
                    config,
                    resource_id=resource_id,
                    instruction=instruction,
                    proposal_path=review.proposal_path,
                    proposal_sha256=review.proposal_sha256,
                    target=target,
                    decision=decision,
                )
        else:
            raw_enrichment = meta.get(staging.AI_ENRICHMENT_KEY)
            operation_id = meta.get("review_run_id")
            focus_resource_id = (
                raw_enrichment.get("focus_resource_id")
                if isinstance(raw_enrichment, Mapping)
                else None
            )
            if not isinstance(operation_id, str) or not operation_id.strip():
                raise CardRevisionFinishError(
                    "The AI-enrichment proposal has no durable operation identity."
                )
            requested_ids = (
                tuple(record_ids)
                if record_ids is not None
                else tuple(record.id for record in proposed)
            )
            deck_path, allowed_ids = _focused_ai_deck(
                config, focus_resource_id, requested_ids
            )
            if staging.AI_ENRICHMENT_REVIEW_KEY in meta:
                promotion = assistant_promotion.plan_promotion_action(
                    config, resource_id, instruction
                )
                review = None
                decision = promotion.decision
                selected_ids = tuple(record.id for record in decision.records)
                if record_ids is not None and tuple(record_ids) != selected_ids:
                    raise CardRevisionFinishError(
                        "The durable owner-review selection differs from this finish request."
                    )
                review_authority = _durable_ai_review_wire(config, promotion)
                promotion_authority = _ai_promotion_wire(
                    config,
                    promotion,
                    operation_id=operation_id,
                    deck_path=deck_path,
                )
            else:
                if record_ids is None:
                    raise CardRevisionFinishError(
                        "Apply and finish needs the exact visible AI-enrichment "
                        "card selection for owner review."
                    )
                review = assistant_ai_enrichment_review.plan_ai_enrichment_review(
                    config,
                    resource_id=resource_id,
                    record_ids=record_ids,
                )
                promotion = None
                decision = promotion_application.project_ai_enrichment_review_promotion(
                    config,
                    review.proposal_path,
                    review.record_ids,
                    expected_revision=review.proposal_sha256,
                )
                selected_ids = tuple(review.record_ids)
                review_authority = _pending_ai_review_wire(
                    config, review, proposed, meta
                )
                promotion_authority = _pending_ai_promotion_wire(
                    config,
                    resource_id=resource_id,
                    instruction=instruction,
                    operation_id=operation_id,
                    proposal_path=review.proposal_path,
                    proposal_sha256=review.proposal_sha256,
                    deck_path=deck_path,
                    decision=decision,
                )
        if decision.state != "lands":
            detail = decision.error or decision.state
            raise CardRevisionFinishError(
                "The selected reviewed proposal cannot produce one exact canonical "
                f"landing: {detail}"
            )
        if (
            not selected_ids
            or len(selected_ids) != len(set(selected_ids))
            or any(
                record_id not in allowed_ids
                for record_id in selected_ids
            )
        ):
            raise CardRevisionFinishError(
                "The reviewed rows no longer match the focused deck's selected cards."
            )
        projected_records = tuple(decision.merged)
        if not projected_records:
            raise CardRevisionFinishError(
                "The promotion plan has no projected canonical collection."
            )
        projected_text = _records_text(projected_records)
        projected_revision = RecordsRevision(
            config.normalized_file.resolve(), projected_text
        )
        audio = audio_application.plan_targeted_audio_revision(
            config,
            projected_records,
            projected_revision,
            selected_ids,
            words=True,
            examples=True,
            force=False,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        final_records = _project_audio_references(config, audio, projected_records)
        final_text = _records_text(final_records)
        final_revision = RecordsRevision(config.normalized_file.resolve(), final_text)
        media = _projected_package_media(
            config, audio, final_records, deck_path
        )
        build = deck_package.plan_vocabulary_deck_package_revision(
            config,
            deck_path,
            final_records,
            final_revision,
            media_sha256=media,
        )
        authority = {
            "version": 3,
            "proposal_kind": proposal_kind,
            "review": review_authority,
            "promotion": promotion_authority,
            "audio": _audio_wire(
                config,
                audio,
                projected_input_sha256=_sha(projected_text.encode("utf-8")),
                projected_output_sha256=_sha(final_text.encode("utf-8")),
            ),
            "build": _package_wire(config, build),
        }
    except CardRevisionFinishError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise CardRevisionFinishError(str(exc)) from exc
    fingerprint = _sha(_canonical(authority).encode("utf-8"))
    directory = _finish_directory(config)
    return CardRevisionFinishPlan(
        repository_root=config.root.resolve(),
        proposal_kind=proposal_kind,
        resource_id=resource_id,
        instruction=instruction,
        review=review,
        promotion=promotion,
        deck_path=deck_path,
        record_ids=selected_ids,
        audio=audio,
        build=build,
        projected_canonical_text=projected_text,
        projected_audio_text=final_text,
        finish_directory=directory,
        record_path=directory / f"card-finish-{fingerprint}.json",
        authority=authority,
        fingerprint=fingerprint,
    )


def plan_card_revision_finish(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
    *,
    record_ids: Sequence[str] | None = None,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> CardRevisionFinishPlan:
    """Plan one exact apply/audio/package consequence without writes or contact."""

    words, sentences = _resolved_providers(
        config,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    # The coordination guard first, before every existing janki lock, and the
    # existing internal order preserved after it. Planning takes it too: a plan
    # computed against a half-applied curation would bind bytes a durable
    # decision is still writing, and one outermost order at every entry is what
    # makes "no code path may invert this ordering" checkable.
    with (
        study_curation.curation_guard(config),
        exclusive_path_lock(config.root / ".janki-audio-operation"),
    ):
        return _plan_locked(
            config,
            resource_id,
            instruction,
            record_ids=record_ids,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
        )


def plan_ai_enrichment_finish(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
    *,
    record_ids: Sequence[str] | None = None,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> CardRevisionFinishPlan:
    """Plan one exact enrichment review/apply/audio/package confirmation."""

    words, sentences = _resolved_providers(
        config,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    with (
        study_curation.curation_guard(config),
        exclusive_path_lock(config.root / ".janki-audio-operation"),
    ):
        return _plan_locked(
            config,
            resource_id,
            instruction,
            record_ids=record_ids,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
            proposal_kind="ai_enrichment",
        )


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise CardRevisionFinishError(
                f"Card revision finish JSON repeats key {key!r}."
            )
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise CardRevisionFinishError(
        f"Card revision finish JSON contains non-finite number {value}."
    )


def _record_text(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _strict_record(wire: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            wire.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except CardRevisionFinishError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CardRevisionFinishError(
            f"Could not parse card revision finish {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise CardRevisionFinishError(
            "Card revision finish has invalid top-level fields."
        )
    receipt_id = value.get("receipt_id")
    authority = value.get("authority")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "card_revision_finish"
        or not _is_sha(receipt_id)
        or path.name != f"card-finish-{receipt_id}.json"
        or not isinstance(authority, Mapping)
        or set(authority)
        != {"version", "proposal_kind", "review", "promotion", "audio", "build"}
        or authority.get("version") != 3
        or authority.get("proposal_kind") not in {"card_revision", "ai_enrichment"}
        or _sha(_canonical(authority).encode("utf-8")) != receipt_id
    ):
        raise CardRevisionFinishError(
            "Card revision finish identity or authority is corrupt."
        )
    state = value.get("state")
    if state not in _STATE_INDEX:
        raise CardRevisionFinishError("Card revision finish state is invalid.")
    for key in ("authorized_at", "updated_at"):
        timestamp = value.get(key)
        if not isinstance(timestamp, str):
            raise CardRevisionFinishError(
                f"Card revision finish {key} is malformed."
            )
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise CardRevisionFinishError(
                f"Card revision finish {key} is malformed."
            ) from exc
    required = _STATE_INDEX[state]
    for index, key in enumerate(
        ("promotion_receipt", "audio_receipt", "build_receipt")
    ):
        present = isinstance(value.get(key), Mapping)
        if (index < required) != present:
            raise CardRevisionFinishError(
                "Card revision finish receipts do not match its durable state."
            )
    binding = value.get("build_binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise CardRevisionFinishError(
            "Card revision finish build binding is malformed."
        )
    if state in {"authorized", "promoted"} and binding is not None:
        raise CardRevisionFinishError(
            "Card revision finish bound a package before audio completed."
        )
    reservations = value.get("paid_clip_reservations")
    if not isinstance(reservations, list):
        raise CardRevisionFinishError(
            "Card revision finish paid reservations are malformed."
        )
    seen_targets: set[str] = set()
    seen_operations: set[str] = set()
    for raw in reservations:
        if not isinstance(raw, Mapping) or set(raw) != {
            "target",
            "operation_id",
            "status",
        }:
            raise CardRevisionFinishError(
                "Card revision finish paid reservation is malformed."
            )
        target = raw.get("target")
        operation_id = raw.get("operation_id")
        try:
            canonical_operation = str(uuid.UUID(str(operation_id)))
        except ValueError as exc:
            raise CardRevisionFinishError(
                "Card revision finish paid operation id is malformed."
            ) from exc
        if (
            not isinstance(target, str)
            or not target
            or not isinstance(operation_id, str)
            or canonical_operation != operation_id
            or target in seen_targets
            or operation_id in seen_operations
            or raw.get("status") not in {"reserved", "failed_before_send"}
        ):
            raise CardRevisionFinishError(
                "Card revision finish paid reservation identity is malformed."
            )
        seen_targets.add(target)
        seen_operations.add(operation_id)
    return value


def _read_record(path: Path) -> tuple[dict[str, Any], str]:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError as exc:
        raise CardRevisionFinishError(
            f"Card revision finish no longer exists: {path}"
        ) from exc
    except (DataError, OSError) as exc:
        raise CardRevisionFinishError(
            f"Could not safely read card revision finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _read_record_optional(path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise CardRevisionFinishError(
            f"Could not safely read card revision finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _new_record(plan: CardRevisionFinishPlan) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "kind": "card_revision_finish",
        "receipt_id": plan.fingerprint,
        "state": "authorized",
        "authority": dict(plan.authority),
        "authorized_at": now,
        "updated_at": now,
        "promotion_receipt": None,
        "audio_receipt": None,
        "build_binding": None,
        "build_receipt": None,
        "paid_clip_reservations": [],
    }


def _write_new(path: Path, record: Mapping[str, Any]) -> str:
    text = _record_text(record)
    _strict_record(text.encode("utf-8"), path)
    try:
        atomic_write_text_bound(path, text, expected_absent=True)
    except (DataError, OSError) as exc:
        raise CardRevisionFinishError(
            f"Could not record finish authority before applying cards: {exc}"
        ) from exc
    return _sha(text.encode("utf-8"))


def _replace_record(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    **changes: Any,
) -> tuple[dict[str, Any], str]:
    updated = dict(record)
    updated.update(changes)
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current != dict(record) or current_revision != revision:
            raise CardRevisionFinishError(
                "Card revision finish changed while a completed phase was recorded."
            )
        atomic_write_text_bound(path, text, expected_revision=revision)
    return updated, _sha(text.encode("utf-8"))


def _advance(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    *,
    from_state: CardRevisionFinishState,
    to_state: CardRevisionFinishState,
    receipt_key: str,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if (
        record.get("state") != from_state
        or _STATE_INDEX[to_state] != _STATE_INDEX[from_state] + 1
        or receipt_key not in _RECEIPT_KEYS
    ):
        raise CardRevisionFinishError(
            f"Card revision finish cannot advance from {record.get('state')!r} "
            f"to {to_state!r}."
        )
    return _replace_record(
        path,
        record,
        revision,
        state=to_state,
        **{receipt_key: dict(receipt)},
    )


def _authority_section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    authority = record.get("authority")
    section = authority.get(key) if isinstance(authority, Mapping) else None
    if not isinstance(section, Mapping):
        raise CardRevisionFinishError(
            f"Card revision finish {key} authority is malformed."
        )
    return section


def _proposal_kind(record: Mapping[str, Any]) -> Literal["card_revision", "ai_enrichment"]:
    authority = record.get("authority")
    kind = authority.get("proposal_kind") if isinstance(authority, Mapping) else None
    if kind not in {"card_revision", "ai_enrichment"}:
        raise CardRevisionFinishError(
            "Card revision finish proposal kind is malformed."
        )
    return kind


def _emit(
    progress: CardRevisionFinishProgress | None,
    phase: CardRevisionFinishPhase,
) -> None:
    if progress is None:
        return
    try:
        progress(phase)
    except Exception:
        return


def _result(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
    *,
    audio: audio_application.AudioExecutionOutcome | None = None,
) -> CardRevisionFinishResult:
    promotion = _authority_section(record, "promotion")
    build = _authority_section(record, "build")
    audio_authority = _authority_section(record, "audio")
    build_receipt = record.get("build_receipt")
    return CardRevisionFinishResult(
        receipt_id=str(record["receipt_id"]),
        state=str(record["state"]),  # type: ignore[arg-type]
        record_path=path,
        deck_path=_authority_path(
            config, promotion.get("deck_path"), label="revision deck"
        ),
        output_path=_authority_path(
            config, build.get("output_path"), label="package output"
        ),
        max_provider_calls=int(audio_authority.get("max_provider_calls", 0)),
        package_sha256=(
            str(build_receipt.get("package_sha256"))
            if isinstance(build_receipt, Mapping)
            else None
        ),
        card_count=(
            int(build_receipt.get("card_count"))
            if isinstance(build_receipt, Mapping)
            and isinstance(build_receipt.get("card_count"), int)
            and not isinstance(build_receipt.get("card_count"), bool)
            else None
        ),
        audio=audio,
    )


def _proposal_path(config: ProjectConfig, record: Mapping[str, Any]) -> Path:
    promotion = _authority_section(record, "promotion")
    path = _authority_path(
        config, promotion.get("proposal_path"), label="revision proposal"
    )
    staging_root = config.staging_dir.absolute()
    if path.parent != staging_root or path.suffix.lower() not in {".yaml", ".yml"}:
        raise CardRevisionFinishError(
            "Card revision finish proposal is outside the direct staging namespace."
        )
    return path


def _current_proposal_resource_id(config: ProjectConfig, path: Path) -> str:
    broker = assistant_context.AssistantContextBroker(config)
    try:
        catalog = json.loads(broker.catalog().wire)
        data = catalog.get("data") if isinstance(catalog, Mapping) else None
        resources = data.get("resources") if isinstance(data, Mapping) else None
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise CardRevisionFinishError(
            f"Could not refresh the reviewed proposal resource: {exc}"
        ) from exc
    if not isinstance(resources, list):
        raise CardRevisionFinishError("The proposal catalog is malformed.")
    matches: list[str] = []
    for item in resources:
        resource_id = item.get("resource_id") if isinstance(item, Mapping) else None
        if not isinstance(resource_id, str):
            continue
        try:
            candidate = broker.proposal_context(resource_id)
        except (JankiError, OSError, UnicodeError, ValueError):
            continue
        if candidate.path.absolute() == path.absolute():
            matches.append(resource_id)
    if len(matches) != 1:
        raise CardRevisionFinishError(
            "The exact reviewed proposal no longer has one current opaque resource."
        )
    return matches[0]


def _ensure_exact_review(
    config: ProjectConfig,
    record: Mapping[str, Any],
) -> None:
    authority = _authority_section(record, "review")
    promotion = _authority_section(record, "promotion")
    proposal = _proposal_path(config, record)
    kind = _proposal_kind(record)
    if kind == "card_revision":
        marker_key = staging.CARD_REVISION_REVIEW_KEY
        marker_version = staging.CARD_REVISION_REVIEW_VERSION
        fingerprint = staging.card_revision_review_fingerprint
    else:
        marker_key = staging.AI_ENRICHMENT_REVIEW_KEY
        marker_version = staging.AI_ENRICHMENT_REVIEW_VERSION
        fingerprint = staging.ai_enrichment_review_fingerprint
    mode = authority.get("mode")
    if mode == "pending":
        resource_id = authority.get("resource_id")
        record_ids = authority.get("record_ids")
        if not isinstance(resource_id, str) or not isinstance(record_ids, list):
            raise CardRevisionFinishError(
                "The pending owner-review authority is malformed."
            )
        _records, current_meta = staging.read_staging(proposal)
        if marker_key not in current_meta:
            if kind == "card_revision":
                fresh = assistant_card_revision_review.plan_card_revision_review(
                    config,
                    resource_id=resource_id,
                    record_ids=record_ids,
                )
            else:
                fresh = assistant_ai_enrichment_review.plan_ai_enrichment_review(
                    config,
                    resource_id=resource_id,
                    record_ids=record_ids,
                )
            if (
                fresh.fingerprint != authority.get("review_fingerprint")
                or fresh.projection != authority.get("projection")
                or fresh.proposal_sha256 != authority.get("proposal_sha256")
                or fresh.proposal_path.absolute() != proposal.absolute()
            ):
                raise CardRevisionFinishError(
                    "The visible content review changed after Apply and finish was confirmed."
                )
            if kind == "card_revision":
                assert isinstance(
                    fresh,
                    assistant_card_revision_review.AssistantCardRevisionReviewPlan,
                )
                assistant_card_revision_review.execute_card_revision_review(config, fresh)
            else:
                assert isinstance(
                    fresh,
                    assistant_ai_enrichment_review.AssistantAiEnrichmentReviewPlan,
                )
                assistant_ai_enrichment_review.execute_ai_enrichment_review(config, fresh)
        reviewed, meta = staging.read_staging(proposal)
        marker = meta.get(marker_key)
        expected_marker = {
            "version": marker_version,
            "authority": "repository-owner",
            "accepted_record_ids": list(record_ids),
            "content_fingerprint": authority.get("content_fingerprint"),
        }
        if (
            marker != expected_marker
            or [item.id for item in reviewed] != list(record_ids)
            or fingerprint(meta, reviewed) != authority.get("content_fingerprint")
        ):
            raise CardRevisionFinishError(
                "The durable content review does not match the exact values the owner confirmed."
            )
    elif mode == "durable":
        reviewed, meta = staging.read_staging(proposal)
        marker = meta.get(marker_key)
        if (
            not isinstance(marker, Mapping)
            or marker != authority.get("marker")
            or _sha(read_bytes_bound(proposal)) != authority.get("proposal_sha256")
            or [item.id for item in reviewed] != authority.get("record_ids")
            or fingerprint(meta, reviewed) != marker.get("content_fingerprint")
        ):
            raise CardRevisionFinishError(
                "The durable content review changed after Apply and finish was confirmed."
            )
    else:
        raise CardRevisionFinishError("The owner-review authority mode is invalid.")
    if promotion.get("record_ids") != authority.get("record_ids"):
        raise CardRevisionFinishError(
            "Review and promotion authority name different card selections."
        )


def _replan_live_promotion(
    config: ProjectConfig,
    record: Mapping[str, Any],
) -> assistant_promotion.AssistantPromotionPlan:
    authority = _authority_section(record, "promotion")
    instruction = authority.get("instruction")
    if not isinstance(instruction, str):
        raise CardRevisionFinishError(
            "Card revision finish promotion identity is malformed."
        )
    proposal = _proposal_path(config, record)
    resource_id = _current_proposal_resource_id(config, proposal)
    fresh = assistant_promotion.plan_promotion_action(
        config, resource_id, instruction
    )
    mode = authority.get("mode")
    kind = _proposal_kind(record)
    fixed_match = (
        fresh.proposal_kind == kind
        and fresh.decision.state == "lands"
        and [item.id for item in fresh.decision.records]
        == authority.get("record_ids")
        and _sha(_records_text(fresh.decision.merged).encode("utf-8"))
        == authority.get("projected_canonical_sha256")
        and _deck_ownership_wire(fresh.decision.deck_ownership)
        == authority.get("deck_ownership")
        and fresh.proposal_path.absolute() == proposal.absolute()
    )
    durable_match = (
        mode == "durable_review"
        and fresh.fingerprint == authority.get("promotion_fingerprint")
        and fresh.service_fingerprint == authority.get("service_fingerprint")
        and fresh.projection == authority.get("projection")
        and _sha(read_bytes_bound(fresh.proposal_path))
        == authority.get("proposal_sha256")
    )
    if not fixed_match or (mode == "durable_review" and not durable_match) or mode not in {
        "durable_review",
        "after_review",
    }:
        raise CardRevisionFinishError(
            "The reviewed promotion changed after Apply and finish was confirmed."
        )
    return fresh


def _promotion_receipt(
    config: ProjectConfig,
    result: promotion_application.PromotionExecutionResult,
    expected_ids: tuple[str, ...],
    projected_sha256: str,
) -> dict[str, Any]:
    if (
        result.state != "landed"
        or result.promoted_ids != expected_ids
        or result.archive_path is None
        or result.receipt_id is None
    ):
        raise CardRevisionFinishError(
            "The reviewed promotion did not completely land its exact card scope."
        )
    canonical = records_revision(config.normalized_file.resolve())
    if canonical.text is None or _sha(canonical.text.encode("utf-8")) != projected_sha256:
        raise CardRevisionFinishError(
            "Promotion wrote canonical bytes outside the confirmed finish plan."
        )
    return {
        "promotion_receipt_id": result.receipt_id,
        "record_ids": list(result.promoted_ids),
        "canonical_sha256": projected_sha256,
        "archive_path": _relative(config, result.archive_path, label="revision archive"),
        "archive_sha256": _sha(read_bytes_bound(result.archive_path)),
    }


def _recover_promotion_receipt(
    config: ProjectConfig,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    promotion = _authority_section(record, "promotion")
    audio = _authority_section(record, "audio")
    record_ids = tuple(promotion.get("record_ids", ()))
    expected_sha = audio.get("projected_input_sha256")
    canonical = records_revision(config.normalized_file.resolve())
    if (
        not record_ids
        or canonical.text is None
        or not _is_sha(expected_sha)
        or _sha(canonical.text.encode("utf-8")) != expected_sha
    ):
        raise CardRevisionFinishError(
            "The proposal disappeared without the confirmed canonical result."
        )
    proposal = _proposal_path(config, record)
    archive = config.staging_dir.absolute() / "done" / proposal.name
    try:
        archived, meta = staging.read_staging(archive)
        if _proposal_kind(record) == "card_revision":
            promotion_application.staged_card_revision(
                meta,
                (),
                archived_ids=record_ids,
                archived_records=archived,
            )
        else:
            promotion_application.staged_ai_enrichment(
                meta,
                (),
                archived_ids=record_ids,
                archived_records=archived,
                require_owner_review=True,
            )
        batches = promotion_application.promotion_batches(
            meta,
            archived=archived,
            archive_file=archive.name,
        )
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise CardRevisionFinishError(
            f"Could not recover the exact promoted revision archive: {exc}"
        ) from exc
    matches = [batch for batch in batches if batch.promoted_ids == record_ids]
    if len(matches) != 1:
        raise CardRevisionFinishError(
            "The revision archive has no single batch matching this finish authority."
        )
    return {
        "promotion_receipt_id": matches[0].receipt_id,
        "record_ids": list(record_ids),
        "canonical_sha256": expected_sha,
        "archive_path": _relative(config, archive, label="revision archive"),
        "archive_sha256": _sha(read_bytes_bound(archive)),
    }


def _validate_audio_evolution(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: audio_application.AudioPlan,
) -> None:
    authority = _authority_section(record, "audio")
    fixed = {
        "canonical_path": _relative(config, fresh.canonical_path, label="audio owner"),
        "ledger_path": _relative(config, fresh.ledger_path, label="audio ledger"),
        "operations_path": _relative(
            config, config.operations_file.resolve(), label="operation journal"
        ),
        "media_dir": _relative(config, fresh.media_dir, label="media directory"),
        "record_ids": list(fresh.record_ids),
        "targeted": fresh.targeted,
        "words": fresh.words,
        "examples": fresh.examples,
        "force": fresh.force,
        "prune": False,
        "word_provider": _provider_wire(fresh.word_provider),
        "example_provider": _provider_wire(fresh.example_provider),
    }
    for key, value in fixed.items():
        if authority.get(key) != value:
            raise CardRevisionFinishError(
                f"The card-audio {key.replace('_', ' ')} changed after confirmation."
            )
    expected_clips = authority.get("clips")
    if not isinstance(expected_clips, list) or len(expected_clips) != len(fresh.clips):
        raise CardRevisionFinishError(
            "The card-audio clip set changed after confirmation."
        )
    for expected, clip in zip(expected_clips, fresh.clips, strict=True):
        if not isinstance(expected, Mapping):
            raise CardRevisionFinishError("Stored card-audio authority is malformed.")
        current = _clip_wire(config, clip)
        for key in (
            "record_id",
            "kind",
            "target",
            "request_input",
            "forced_accent",
            "content_fingerprint",
            "provider",
        ):
            if current.get(key) != expected.get(key):
                raise CardRevisionFinishError(
                    "A card-audio request or provider changed after confirmation."
                )
        initial = expected.get("initial_state")
        allowed = _ALLOWED_CLIP_EVOLUTION.get(str(initial))
        if allowed is None or clip.state not in allowed:
            raise CardRevisionFinishError(
                "A card-audio clip would require authority wider than the confirmed plan."
            )
        if initial == "current" and current.get("initial_media_sha256") != expected.get(
            "initial_media_sha256"
        ):
            raise CardRevisionFinishError(
                "An already-current card-audio file changed after confirmation."
            )
        if initial == "recoverable" and clip.state == "recoverable" and any(
            current.get(key) != expected.get(key)
            for key in (
                "initial_recovery_key",
                "initial_recovery_sha256",
                "initial_recovery_source",
            )
        ):
            raise CardRevisionFinishError(
                "The exact recoverable card-audio clip changed after confirmation."
            )
        if (
            initial == "recoverable"
            and clip.state == "current"
            and current.get("initial_media_sha256")
            != expected.get("initial_recovery_sha256")
        ):
            raise CardRevisionFinishError(
                "A recovered card-audio clip no longer has its authorized bytes."
            )


def _realtime_profile(
    sentence_provider: Any,
    clip: audio_application.AudioClipPlan,
) -> openai_realtime.OpenAiRealtimeProvider:
    try:
        provider = sentence_profile_for(sentence_provider, clip.record_id)
    except JankiError as exc:
        raise CardRevisionFinishError(str(exc)) from exc
    if not isinstance(provider, openai_realtime.OpenAiRealtimeProvider):
        raise CardRevisionFinishError(
            "Paid card audio is supported only through OpenAI Realtime."
        )
    return provider


def _matching_reserved_operation(
    config: ProjectConfig,
    operation_id: str,
    clip: audio_application.AudioClipPlan,
    *,
    sentence_provider: Any,
) -> operations.Operation | None:
    provider = _realtime_profile(sentence_provider, clip)
    operation = operations.OperationJournal.load(config.operations_file).operations.get(
        operation_id
    )
    if operation is None:
        return None
    if operation.cleanup is not None:
        raise CardRevisionFinishError(
            f"Paid audio reservation {operation_id} has unfinished cleanup."
        )
    expected_source = audio_application.audio_cmd.audio_journal_source(
        clip.record_id,
        of=clip.kind,
        target=clip.target,
    )
    if (
        operation.kind != "audio-realtime"
        or operation.source_file != expected_source
        or operation.source_sha256 != clip.content_fingerprint
        or operation.request_fp
        != provider.request_fingerprint(
            clip.request_input, forced_accent=clip.forced_accent
        )
        or operation.model != openai_realtime.MODEL
    ):
        raise CardRevisionFinishError(
            f"Paid audio reservation {operation_id} no longer matches its exact request."
        )
    return operation


def _reconcile_reservations(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    fresh: audio_application.AudioPlan,
    *,
    sentence_provider: Any,
) -> tuple[dict[str, Any], str]:
    reservations = record.get("paid_clip_reservations")
    if not isinstance(reservations, list):
        raise CardRevisionFinishError("Paid audio reservations are malformed.")
    clips = {clip.target: clip for clip in fresh.clips}
    updated = [dict(item) for item in reservations]
    changed = False
    forget: list[str] = []
    for item in updated:
        clip = clips.get(str(item.get("target") or ""))
        if clip is None:
            raise CardRevisionFinishError(
                "A reserved paid clip disappeared from the confirmed audio plan."
            )
        if clip.state != "provider-required":
            continue
        operation_id = str(item.get("operation_id") or "")
        operation = _matching_reserved_operation(
            config,
            operation_id,
            clip,
            sentence_provider=sentence_provider,
        )
        if operation is None:
            if item.get("status") == "reserved":
                raise CardRevisionFinishError(
                    "Paid audio operation evidence was removed; refusing to bill "
                    "the confirmed clip a second time."
                )
            continue
        if operation.state == "authorized":
            operations.OperationJournal.load(config.operations_file).advance(
                operation_id,
                "failed_before_send",
                detail="Card revision finish resumed before provider dispatch began.",
            )
            operation = operations.OperationJournal.load(
                config.operations_file
            ).operations[operation_id]
        if operation.state in {"canceled_before_send", "failed_before_send"}:
            item["status"] = "failed_before_send"
            changed = True
            forget.append(operation_id)
        elif operation.state not in {
            "dispatching",
            "running",
            "outcome_unknown",
            "result_captured",
            "committed",
        }:
            raise CardRevisionFinishError(
                f"Paid audio reservation {operation_id} is {operation.state}."
            )
    current = dict(record)
    current_revision = revision
    if changed:
        current, current_revision = _replace_record(
            path,
            record,
            revision,
            paid_clip_reservations=updated,
        )
    for operation_id in forget:
        operations.OperationJournal.load(config.operations_file).forget([operation_id])
    return current, current_revision


def _reserve_paid_clip(
    config: ProjectConfig,
    path: Path,
    durable: dict[str, Any],
    dispatch: audio_application.PaidAudioDispatch,
    *,
    sentence_provider: Any,
) -> None:
    clip = dispatch.clip
    if clip.kind != "example" or clip.provider.access != "paid-network":
        raise CardRevisionFinishError(
            "Finish may reserve only its paid example-audio clips."
        )
    record = durable["record"]
    revision = durable["revision"]
    authority = _authority_section(record, "audio")
    expected = [
        raw
        for raw in authority.get("clips", ())
        if isinstance(raw, Mapping) and raw.get("target") == clip.target
    ]
    if (
        len(expected) != 1
        or expected[0].get("initial_state") != "provider-required"
        or any(
            _clip_wire(config, clip).get(key) != expected[0].get(key)
            for key in (
                "record_id",
                "kind",
                "target",
                "request_input",
                "forced_accent",
                "content_fingerprint",
                "provider",
                "initial_state",
            )
        )
    ):
        raise CardRevisionFinishError(
            "Paid audio dispatch exceeds the confirmed finish authority."
        )
    operation = _matching_reserved_operation(
        config,
        dispatch.operation_id,
        clip,
        sentence_provider=sentence_provider,
    )
    if operation is None or operation.state != "authorized":
        raise CardRevisionFinishError(
            "Paid audio dispatch has no exact newly authorized operation."
        )
    reservations = record.get("paid_clip_reservations")
    assert isinstance(reservations, list)
    same_target = [
        item
        for item in reservations
        if isinstance(item, Mapping) and item.get("target") == clip.target
    ]
    if len(same_target) > 1 or (
        same_target and same_target[0].get("status") != "failed_before_send"
    ):
        raise CardRevisionFinishError(
            "Paid card audio already consumed its one dispatch reservation."
        )
    updated_reservations = [
        item
        for item in reservations
        if not isinstance(item, Mapping) or item.get("target") != clip.target
    ]
    updated_reservations.append(
        {
            "target": clip.target,
            "operation_id": dispatch.operation_id,
            "status": "reserved",
        }
    )
    updated, updated_revision = _replace_record(
        path,
        record,
        revision,
        paid_clip_reservations=updated_reservations,
    )
    durable["record"] = updated
    durable["revision"] = updated_revision


def _audio_receipt(
    config: ProjectConfig,
    fresh: audio_application.AudioPlan,
    expected_output_sha: str,
) -> dict[str, Any]:
    canonical = records_revision(config.normalized_file.resolve())
    if canonical.text is None or _sha(canonical.text.encode("utf-8")) != expected_output_sha:
        raise CardRevisionFinishError(
            "Card audio wrote canonical bytes outside the confirmed finish plan."
        )
    if any(clip.state != "current" for clip in fresh.clips):
        raise CardRevisionFinishError(
            "Card audio reported success without making every selected clip current."
        )
    clips = [
        {
            "target": clip.target,
            "sha256": _sha(read_bytes_bound(_media_path(config, clip.target))),
        }
        for clip in fresh.clips
    ]
    return {
        "canonical_sha256": expected_output_sha,
        "ledger_sha256": _sha(read_bytes_bound(config.ledger_file.resolve())),
        "clips": clips,
    }


def _input_map(value: object) -> dict[str, str | None] | None:
    if not isinstance(value, list):
        return None
    result: dict[str, str | None] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"label", "path", "sha256"}:
            return None
        path = raw.get("path")
        sha256 = raw.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path in result
            or (sha256 is not None and not _is_sha(sha256))
        ):
            return None
        result[path] = sha256 if isinstance(sha256, str) else None
    return result


def _build_compatible(
    projected: Mapping[str, Any],
    exact: Mapping[str, Any],
    *,
    allow_output_evolution: bool,
) -> bool:
    if set(projected) != set(exact):
        return False
    flexible = {"media_inputs", "plan_fingerprint"}
    if allow_output_evolution:
        flexible |= {"output_revision", "output_identity"}
    if any(projected.get(key) != exact.get(key) for key in set(projected) - flexible):
        return False
    if not _is_sha(exact.get("plan_fingerprint")):
        return False
    projected_media = _input_map(projected.get("media_inputs"))
    exact_media = _input_map(exact.get("media_inputs"))
    if (
        projected_media is None
        or exact_media is None
        or set(projected_media) != set(exact_media)
    ):
        return False
    return all(
        expected is None or exact_media[path] == expected
        for path, expected in projected_media.items()
    ) and all(_is_sha(value) for value in exact_media.values())


def _execute_record_locked(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    chosen_provider: str | None,
    word_provider: Any | None,
    sentence_provider: Any | None,
    progress: CardRevisionFinishProgress | None,
) -> CardRevisionFinishResult:
    promotion_authority = _authority_section(record, "promotion")
    audio_authority = _authority_section(record, "audio")
    expected_ids = tuple(promotion_authority.get("record_ids", ()))
    if (
        not expected_ids
        or any(not isinstance(item, str) or not item for item in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise CardRevisionFinishError(
            "Card revision finish record ids are malformed."
        )

    if record["state"] == "authorized":
        proposal = _proposal_path(config, record)
        _emit(progress, "Applying reviewed cards")
        if proposal.is_symlink():
            raise CardRevisionFinishError(
                "The confirmed card revision proposal became a symlink."
            )
        if proposal.exists():
            _ensure_exact_review(config, record)
            plan = _replan_live_promotion(config, record)
            # The unguarded entry. This runs inside `.janki-audio-operation`
            # and inside the finish record's own compare-and-swap lock, both
            # taken under the coordination guard by this module's outermost
            # entries; re-acquiring that non-reentrant guard here would invert
            # the order and deadlock.
            executed = assistant_promotion.execute_promotion_action_under_guard(
                config, plan
            )
            receipt = _promotion_receipt(
                config,
                executed.result,
                expected_ids,
                str(audio_authority.get("projected_input_sha256")),
            )
        else:
            receipt = _recover_promotion_receipt(config, record)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            from_state="authorized",
            to_state="promoted",
            receipt_key="promotion_receipt",
            receipt=receipt,
        )

    if record["state"] == "promoted":
        try:
            fresh_audio = audio_application.plan_targeted_audio(
                config,
                expected_ids,
                words=True,
                examples=True,
                force=False,
                chosen_provider=chosen_provider,
                word_provider=word_provider,
                sentence_provider=sentence_provider,
            )
            current_text = fresh_audio.canonical_revision.text
            if (
                current_text is None
                or _sha(current_text.encode("utf-8"))
                not in {
                    audio_authority.get("projected_input_sha256"),
                    audio_authority.get("projected_output_sha256"),
                }
            ):
                raise CardRevisionFinishError(
                    "The canonical cards changed after the reviewed promotion landed."
                )
            _validate_audio_evolution(config, record, fresh_audio)
            audio_application.preflight_paid_deck_audio_plan(
                fresh_audio,
                word_provider=word_provider,
                sentence_provider=sentence_provider,
            )
        except (JankiError, OSError, TypeError, ValueError) as exc:
            raise CardRevisionFinishError(str(exc)) from exc
        record, revision = _reconcile_reservations(
            config,
            path,
            record,
            revision,
            fresh_audio,
            sentence_provider=sentence_provider,
        )
        _emit(progress, "Creating card audio")
        durable = {"record": record, "revision": revision}

        def reserve(dispatch: audio_application.PaidAudioDispatch) -> None:
            _reserve_paid_clip(
                config,
                path,
                durable,
                dispatch,
                sentence_provider=sentence_provider,
            )

        outcome = audio_application.execute_targeted_audio_locked(
            config,
            expected_ids,
            words=True,
            examples=True,
            expected_fingerprint=fresh_audio.fingerprint,
            force=False,
            prune=False,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            before_paid_dispatch=reserve,
        )
        record = durable["record"]
        revision = durable["revision"]
        if not outcome.succeeded:
            return _result(config, path, record, audio=outcome)
        exact_audio = audio_application.plan_targeted_audio(
            config,
            expected_ids,
            words=True,
            examples=True,
            force=False,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        _validate_audio_evolution(config, record, exact_audio)
        receipt = _audio_receipt(
            config,
            exact_audio,
            str(audio_authority.get("projected_output_sha256")),
        )
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            from_state="promoted",
            to_state="audio_complete",
            receipt_key="audio_receipt",
            receipt=receipt,
        )

    if record["state"] in {"audio_complete", "complete"}:
        audio_receipt = record.get("audio_receipt")
        if not isinstance(audio_receipt, Mapping):
            raise CardRevisionFinishError(
                "Audio-complete finish has no exact audio receipt."
            )
        canonical = records_revision(config.normalized_file.resolve())
        if canonical.text is None or _sha(canonical.text.encode("utf-8")) != audio_receipt.get(
            "canonical_sha256"
        ):
            raise CardRevisionFinishError(
                "The canonical cards changed after finish audio completed."
            )
        for raw in audio_receipt.get("clips", ()):
            if not isinstance(raw, Mapping):
                raise CardRevisionFinishError("Audio receipt clips are malformed.")
            target = str(raw.get("target") or "")
            if _sha(read_bytes_bound(_media_path(config, target))) != raw.get("sha256"):
                raise CardRevisionFinishError(
                    "Finished card audio no longer matches its exact receipt."
                )
        build_authority = _authority_section(record, "build")
        fresh_build = deck_package.plan_deck_package(
            config,
            _authority_path(
                config,
                promotion_authority.get("deck_path"),
                label="revision deck",
            ),
        )
        existing_binding = record.get("build_binding")
        wire = _package_wire(config, fresh_build)
        if not _build_compatible(
            build_authority,
            wire,
            allow_output_evolution=existing_binding is not None,
        ):
            raise CardRevisionFinishError(
                "The package build widened or changed after finish confirmation."
            )
        if existing_binding != wire:
            record, revision = _replace_record(
                path,
                record,
                revision,
                build_binding=wire,
            )
        if record["state"] == "complete":
            receipt = record.get("build_receipt")
            if isinstance(receipt, Mapping):
                try:
                    current = _sha(
                        read_bytes_bound(
                            _authority_path(
                                config,
                                receipt.get("output_path"),
                                label="package output",
                            )
                        )
                    )
                except FileNotFoundError:
                    current = ""
                if current == receipt.get("package_sha256"):
                    return _result(config, path, record)
        _emit(progress, "Building Anki package")
        built = deck_package.execute_deck_package_locked(config, fresh_build)
        receipt = {
            "output_path": _relative(config, built.output_path, label="package output"),
            "package_sha256": built.package_sha256,
            "note_count": built.note_count,
            "card_count": built.card_count,
            "media_count": built.media_count,
            "card_types": list(built.card_types),
            "plan_fingerprint": fresh_build.fingerprint,
        }
        _emit(progress, "Saving finish receipt")
        if record["state"] == "audio_complete":
            record, revision = _advance(
                path,
                record,
                revision,
                from_state="audio_complete",
                to_state="complete",
                receipt_key="build_receipt",
                receipt=receipt,
            )
        else:
            record, revision = _replace_record(
                path,
                record,
                revision,
                build_receipt=receipt,
            )
    return _result(config, path, record)


def _active_finish_for_operation(
    config: ProjectConfig,
    operation_id: str,
) -> str | None:
    directory = _finish_directory(config)
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CardRevisionFinishError(
            f"Could not inspect card revision finish records: {exc}"
        ) from exc
    found: str | None = None
    for entry in entries:
        if not entry.name.startswith("card-finish-") or entry.suffix != ".json":
            continue
        details = os.lstat(entry)
        if not stat.S_ISREG(details.st_mode):
            raise CardRevisionFinishError(
                f"Card revision finish {entry.name} is not a direct regular file."
            )
        record, _revision = _read_record(entry)
        promotion = _authority_section(record, "promotion")
        if promotion.get("operation_id") != operation_id:
            continue
        receipt_id = str(record["receipt_id"])
        if found is not None and found != receipt_id:
            raise CardRevisionFinishError(
                "More than one finish authority names this card revision."
            )
        found = receipt_id
    return found


def execute_card_revision_finish(
    config: ProjectConfig,
    expected: CardRevisionFinishPlan,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
    progress: CardRevisionFinishProgress | None = None,
) -> CardRevisionFinishResult:
    """Persist, re-plan, and execute one exact owner-confirmed finish."""

    _emit(progress, "Preparing finish")
    if expected.repository_root != config.root.resolve():
        raise CardRevisionFinishError(
            "Card revision finish plan belongs to another repository."
        )
    words, sentences = _resolved_providers(
        config,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    with (
        study_curation.curation_guard(config),
        exclusive_path_lock(config.root / ".janki-audio-operation"),
    ):
        fresh = _plan_locked(
            config,
            expected.resource_id,
            expected.instruction,
            record_ids=(
                expected.record_ids if expected.review is not None else None
            ),
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
            proposal_kind=expected.proposal_kind,
        )
        if fresh.fingerprint != expected.fingerprint or fresh.authority != expected.authority:
            raise CardRevisionFinishError(
                "The card Apply-and-finish plan changed after it was displayed."
            )
        audio_application.preflight_paid_deck_audio_plan(
            fresh.audio,
            word_provider=words,
            sentence_provider=sentences,
        )
        prepare_bound_directory(fresh.finish_directory)
        operation_id = str(
            _authority_section({"authority": fresh.authority}, "promotion").get(
                "operation_id"
            )
        )
        existing = _active_finish_for_operation(config, operation_id)
        if existing is not None and existing != fresh.fingerprint:
            raise CardRevisionFinishError(
                "This revision already has a different durable finish authority; "
                f"resume {existing} instead of widening it."
            )
        with exclusive_path_lock(fresh.record_path):
            held = _read_record_optional(fresh.record_path)
            if held is None:
                record = _new_record(fresh)
                revision = _write_new(fresh.record_path, record)
            else:
                record, revision = held
            if (
                record.get("receipt_id") != fresh.fingerprint
                or record.get("authority") != dict(fresh.authority)
            ):
                raise CardRevisionFinishError(
                    "Existing finish receipt does not match the confirmed authority."
                )
        return _execute_record_locked(
            config,
            fresh.record_path,
            record,
            revision,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
            progress=progress,
        )


def _record_path(config: ProjectConfig, receipt_id: str) -> Path:
    if not _is_sha(receipt_id):
        raise CardRevisionFinishError(
            "Card revision finish receipt id must be lowercase SHA-256."
        )
    return _finish_directory(config) / f"card-finish-{receipt_id}.json"


def resume_card_revision_finish(
    config: ProjectConfig,
    receipt_id: str,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
    progress: CardRevisionFinishProgress | None = None,
) -> CardRevisionFinishResult:
    """Resume only the exact authority already stored in ``receipt_id``."""

    _emit(progress, "Preparing finish")
    path = _record_path(config, receipt_id)
    with (
        study_curation.curation_guard(config),
        exclusive_path_lock(config.root / ".janki-audio-operation"),
    ):
        record, revision = _read_record(path)
        if record["state"] in {"audio_complete", "complete"}:
            words, sentences = None, None
        else:
            words, sentences = _resolved_providers(
                config,
                chosen_provider=chosen_provider,
                word_provider=word_provider,
                sentence_provider=sentence_provider,
            )
        return _execute_record_locked(
            config,
            path,
            record,
            revision,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
            progress=progress,
        )


def inspect_card_revision_finish(
    config: ProjectConfig,
    receipt_id: str,
) -> CardRevisionFinishResult:
    """Read one durable finish state without provider contact or mutation."""

    path = _record_path(config, receipt_id)
    record, _revision = _read_record(path)
    return _result(config, path, record)
