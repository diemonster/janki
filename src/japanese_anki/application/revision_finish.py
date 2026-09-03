"""One owner-authorized revision apply, example-audio, and package build.

The visible plan is computed from the reviewed proposal without publishing it.
Its durable authority survives the proposal's later archive, and recovery may
only move an initially authorized clip toward completion.  It may never turn a
clip that was current or locally recoverable at confirmation into a fresh paid
provider call.
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

from japanese_anki import ledger, operations
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_build as deck_build_application
from japanese_anki.application import revision_apply as revision_apply_application
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records,
    prepare_bound_directory,
    read_bytes_bound,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import openai_realtime, sentence_profile_for

__all__ = [
    "RevisionFinishError",
    "RevisionFinishPhase",
    "RevisionFinishPlan",
    "RevisionFinishProgress",
    "RevisionFinishResult",
    "execute_revision_finish",
    "inspect_revision_finish",
    "plan_revision_finish",
    "resume_revision_finish",
]


RevisionFinishPhase = Literal[
    "Preparing finish",
    "Applying reviewed revision",
    "Creating example audio",
    "Building Anki package",
    "Saving finish receipt",
]
RevisionFinishProgress = Callable[[RevisionFinishPhase], None]
RevisionFinishState = Literal[
    "authorized",
    "revision_applied",
    "audio_complete",
    "complete",
]


class RevisionFinishError(JankiError):
    """The exact reviewed finish authority cannot safely continue."""


@dataclass(frozen=True, slots=True)
class RevisionFinishPlan:
    """Display-only consequences of one reviewed revision finish."""

    repository_root: Path
    revision: revision_apply_application.RevisionApplyPlan
    audio: audio_application.AudioPlan
    build: deck_build_application.ConjugationDeckBuildPlan
    projected_audio_deck_text: str
    projected_audio_deck_sha256: str
    finish_directory: Path
    record_path: Path
    authority: Mapping[str, Any]
    fingerprint: str

    @property
    def provider_required_count(self) -> int:
        return self.audio.provider_required_count

    @property
    def paid_provider_call_possible(self) -> bool:
        provider = self.audio.example_provider
        return (
            self.provider_required_count > 0
            and provider is not None
            and provider.access == "paid-network"
        )


@dataclass(frozen=True, slots=True)
class RevisionFinishResult:
    """Truthful durable state after one initial or recovery execution."""

    receipt_id: str
    state: RevisionFinishState
    record_path: Path
    deck_path: Path
    output_path: Path
    example_provider_access: audio_application.AudioAccess
    max_provider_calls: int
    package_sha256: str | None = None
    card_count: int | None = None
    audio: audio_application.AudioExecutionOutcome | None = None

    @property
    def succeeded(self) -> bool:
        return self.state == "complete"

    @property
    def paid_provider_call_possible(self) -> bool:
        return (
            self.state in {"authorized", "revision_applied"}
            and self.max_provider_calls > 0
            and self.example_provider_access == "paid-network"
        )


_RECORD_KEYS = {
    "schema_version",
    "kind",
    "receipt_id",
    "state",
    "authority",
    "authorized_at",
    "updated_at",
    "apply_receipt",
    "audio_receipt",
    "build_binding",
    "build_receipt",
    "paid_clip_reservations",
}
_AUTHORITY_KEYS = {"version", "revision", "audio", "build"}
_REVISION_AUTHORITY_KEYS = {
    "operation_id",
    "staging_path",
    "archive_path",
    "proposal_sha256",
    "request_fingerprint",
    "provider",
    "billing_class",
    "model",
    "request_bytes_sha256",
    "deck_path",
    "deck_base_sha256",
    "selected_record_ids",
    "canonical_context_fingerprint",
    "intended_deck_sha256",
    "apply_plan_fingerprint",
}
_AUDIO_AUTHORITY_KEYS = {
    "canonical_path",
    "ledger_path",
    "operations_path",
    "media_dir",
    "record_ids",
    "targeted",
    "words",
    "examples",
    "force",
    "prune",
    "word_provider",
    "example_provider",
    "clips",
    "initial_plan_fingerprint",
    "max_provider_calls",
    "projected_deck_sha256",
}
_BUILD_AUTHORITY_KEYS = {
    "deck_path",
    "source_path",
    "output_path",
    "deck_sha256",
    "source_sha256",
    "templates",
    "media",
    "deck_name",
    "form",
    "card_count",
    "plan_fingerprint",
}
_PROVIDER_KEYS = {
    "name",
    "access",
    "voice",
    "speed",
    "suffix",
    "destination",
    "transport",
    "settings",
}
_CLIP_AUTHORITY_KEYS = {
    "record_id",
    "kind",
    "target",
    "request_input",
    "forced_accent",
    "content_fingerprint",
    "provider",
    "initial_state",
    "initial_recovery_key",
    "initial_recovery_sha256",
    "initial_recovery_source",
    "initial_media_sha256",
}
_APPLY_RECEIPT_KEYS = {
    "operation_id",
    "deck_sha256",
    "archive_path",
    "archive_sha256",
}
_AUDIO_RECEIPT_KEYS = {"deck_sha256", "ledger_sha256", "clips"}
_AUDIO_RECEIPT_CLIP_KEYS = {"target", "sha256"}
_BUILD_RECEIPT_KEYS = {
    "output_path",
    "package_sha256",
    "card_count",
    "plan_fingerprint",
}
_PAID_RESERVATION_KEYS = {"target", "operation_id", "status"}
_PAID_RESERVATION_STATUSES = {"reserved", "failed_before_send"}
_STATES: tuple[RevisionFinishState, ...] = (
    "authorized",
    "revision_applied",
    "audio_complete",
    "complete",
)
_STATE_INDEX = {state: index for index, state in enumerate(_STATES)}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(config: ProjectConfig, path: Path, label: str) -> str:
    root = config.root.absolute()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise RevisionFinishError(
            f"Revision finish {label} must stay inside the repository: {path}"
        ) from exc
    value = relative.as_posix()
    if not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise RevisionFinishError(f"Revision finish {label} is not repository-relative.")
    return value


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


def _clip_wire(
    config: ProjectConfig,
    clip: audio_application.AudioClipPlan,
) -> dict[str, Any]:
    media_sha256 = None
    if clip.state == "current":
        try:
            media_sha256 = _sha(read_bytes_bound(_media_path(config, clip.target)))
        except (DataError, OSError) as exc:
            raise RevisionFinishError(
                f"Could not bind current example audio {clip.target!r}: {exc}"
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


def _revision_wire(
    config: ProjectConfig,
    plan: revision_apply_application.RevisionApplyPlan,
) -> dict[str, Any]:
    return {
        "operation_id": plan.operation_id,
        "staging_path": _relative(config, plan.staging_path, "staging path"),
        "archive_path": _relative(config, plan.archive_path, "archive path"),
        "proposal_sha256": plan.proposal_sha256,
        "request_fingerprint": plan.request_fingerprint,
        "provider": plan.provider,
        "billing_class": plan.billing_class,
        "model": plan.model,
        "request_bytes_sha256": plan.request_bytes_sha256,
        "deck_path": plan.deck_relative_path,
        "deck_base_sha256": plan.deck_base_sha256,
        "selected_record_ids": list(plan.selected_record_ids),
        "canonical_context_fingerprint": plan.canonical_context_fingerprint,
        "intended_deck_sha256": plan.intended_deck_sha256,
        "apply_plan_fingerprint": plan.plan_fingerprint,
    }


def _audio_wire(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    *,
    projected_deck_sha256: str,
) -> dict[str, Any]:
    return {
        "canonical_path": _relative(config, plan.canonical_path, "audio owner"),
        "ledger_path": _relative(config, plan.ledger_path, "audio ledger"),
        "operations_path": _relative(config, config.operations_file.resolve(), "operation journal"),
        "media_dir": _relative(config, plan.media_dir, "media directory"),
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
        "max_provider_calls": plan.provider_required_count,
        "projected_deck_sha256": projected_deck_sha256,
    }


def _build_wire(
    config: ProjectConfig,
    plan: deck_build_application.ConjugationDeckBuildPlan,
) -> dict[str, Any]:
    wire = {
        "deck_path": _relative(config, plan.deck_path, "build deck"),
        "source_path": _relative(config, plan.source_path, "build source"),
        "output_path": _relative(config, plan.output_path, "build output"),
        "deck_sha256": plan.deck_sha256,
        "source_sha256": plan.source_sha256,
        "templates": [
            {
                "path": _relative(config, item.path, "build template"),
                "sha256": item.sha256,
            }
            for item in plan.template_inputs
        ],
        "media": [
            {
                "path": _relative(config, item.path, "build media"),
                "sha256": item.sha256,
            }
            for item in plan.media_inputs
        ],
        "deck_name": plan.deck_name,
        "form": plan.form,
        "card_count": plan.card_count,
    }
    wire["plan_fingerprint"] = _sha(
        json.dumps(
            wire,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return wire


def _project_audio_references(
    plan: audio_application.AudioPlan,
    records: Sequence[VocabularyRecord],
) -> tuple[VocabularyRecord, ...]:
    """Return the exact post-audio records implied by every planned target."""

    by_record: dict[str, dict[str, audio_application.AudioClipPlan]] = defaultdict(dict)
    for clip in plan.clips:
        if clip.kind != "example":
            raise RevisionFinishError(
                "Revision finish may authorize deck-authored example audio only."
            )
        existing = by_record[clip.record_id].get(clip.target)
        if existing is not None and existing != clip:
            raise RevisionFinishError("Audio plan has conflicting requests for one example target.")
        by_record[clip.record_id][clip.target] = clip
    selected_ids = set(plan.record_ids)
    projected: list[VocabularyRecord] = []
    for record in records:
        if record.id not in selected_ids:
            projected.append(record)
            continue
        clips = by_record.pop(record.id, {})
        used: set[str] = set()
        examples: list[ExampleSentence] = []
        for example in record.examples:
            request_input = ledger.example_audio_request(example)
            content_fingerprint = ledger.example_audio_content_fingerprint(example)
            target_prefix = f"janki-{ledger.example_audio_filename_fingerprint(record, example)}"
            matches = [
                clip
                for clip in clips.values()
                if clip.target == f"{target_prefix}{clip.provider.suffix}"
            ]
            if len(matches) != 1:
                raise RevisionFinishError(
                    f"Audio plan no longer names every example for {record.id!r}."
                )
            clip = matches[0]
            if (
                clip.request_input != request_input
                or clip.content_fingerprint != content_fingerprint
            ):
                raise RevisionFinishError(
                    f"Audio plan changed the spoken identity for {record.id!r}."
                )
            used.add(clip.target)
            examples.append(
                replace(
                    example,
                    audio=audio_application.audio_cmd.media_relative(
                        plan.media_dir / audio_application.audio_cmd.AUDIO_SUBDIR / clip.target,
                        plan.media_dir,
                    ),
                )
            )
        if used != set(clips):
            raise RevisionFinishError(f"Audio plan contains an unused example for {record.id!r}.")
        projected.append(replace(record, examples=examples))
        selected_ids.remove(record.id)
    if selected_ids:
        raise RevisionFinishError("Audio plan selects an owner outside the revision deck.")
    if by_record:
        raise RevisionFinishError("Audio plan contains an owner outside the revision deck.")
    return tuple(projected)


def _selected_audio_record_ids(
    revision: revision_apply_application.RevisionApplyPlan,
    proposed_revision: RecordsRevision,
    records: Sequence[VocabularyRecord],
) -> tuple[str, ...]:
    """Map reviewed card ids to their structural drill-audio owner ids."""

    assert proposed_revision.text is not None
    section = pattern_cards.conjugation_deck_section(
        revision.deck_path,
        proposed_revision.text,
    )
    deck_id = section.get("deck_id")
    if isinstance(deck_id, bool) or not isinstance(deck_id, int):
        raise RevisionFinishError("Revision finish deck_id is malformed.")
    form = str(section.get("form") or "te_form").strip()
    selected = tuple(
        pattern_cards.drill_audio_owner_id(deck_id, form, record_id)
        for record_id in revision.selected_record_ids
    )
    available = {record.id for record in records}
    missing = [record_id for record_id in selected if record_id not in available]
    if missing:
        raise RevisionFinishError("A reviewed revision card has no matching drill-audio owner.")
    return selected


def _projected_build_media(
    config: ProjectConfig,
    deck_path: Path,
    revision: RecordsRevision,
    audio: audio_application.AudioPlan,
) -> dict[Path, str | None]:
    """Bind selected future clips and every unchanged referenced media byte."""

    selected: dict[Path, str | None] = {}
    for clip in audio.clips:
        path = _media_path(config, clip.target).resolve()
        sha256: str | None = None
        if clip.state == "current":
            sha256 = _sha(read_bytes_bound(path))
        elif clip.state == "recoverable":
            sha256 = clip.recovery_sha256
        existing = selected.get(path)
        if path in selected and existing != sha256:
            raise RevisionFinishError(
                "Selected example-audio targets have conflicting projected bytes."
            )
        selected[path] = sha256

    assert revision.text is not None
    section = pattern_cards.conjugation_deck_section(deck_path, revision.text)
    form = str(section.get("form") or "te_form").strip()
    source = pattern_cards.collection_for_section(deck_path, config, section)
    records = load_records(source)
    paths = pattern_cards.conjugation_media_paths_for_section(
        deck_path,
        config,
        section,
        records,
        form,
        allowed_missing_media=frozenset(selected),
    )
    if not set(selected) <= set(paths):
        raise RevisionFinishError(
            "Selected example-audio targets are missing from the projected deck."
        )
    projected: dict[Path, str | None] = {}
    for path in paths:
        target = path.resolve()
        if target in selected:
            projected[target] = selected[target]
            continue
        projected[target] = _sha(read_bytes_bound(target))
    return projected


def _finish_directory(config: ProjectConfig) -> Path:
    return (config.staging_dir / "done" / "revisions").absolute()


def _resolved_providers(
    config: ProjectConfig,
    *,
    chosen_provider: str | None,
    word_provider: Any | None,
    sentence_provider: Any | None,
) -> tuple[Any, Any]:
    words = word_provider
    if words is None:
        words = audio_application.resolve_word_provider(config, chosen_provider)
    sentences = sentence_provider
    if sentences is None:
        sentences = audio_application.resolve_sentence_provider(
            config,
            chosen_provider,
            words,
        )
    return words, sentences


def _validate_provider_journal_binding(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
    *,
    word_provider: Any,
    sentence_provider: Any,
) -> None:
    expected = config.operations_file.resolve()
    for clip in plan.clips:
        if clip.provider.access != "paid-network":
            continue
        selected = word_provider if clip.kind == "word" else sentence_provider
        try:
            provider = sentence_profile_for(selected, clip.record_id)
        except JankiError as exc:
            raise RevisionFinishError(str(exc)) from exc
        if not isinstance(provider, openai_realtime.OpenAiRealtimeProvider):
            raise RevisionFinishError(
                "Revision finish supports paid audio only through OpenAI Realtime."
            )
        if provider.operations_path is None or Path(provider.operations_path).resolve() != expected:
            raise RevisionFinishError(
                "OpenAI Realtime audio is bound to a different operation journal."
            )


def _project_revision_finish(
    config: ProjectConfig,
    revision: revision_apply_application.RevisionApplyPlan,
    *,
    chosen_provider: str | None,
    word_provider: Any | None,
    sentence_provider: Any | None,
) -> tuple[
    audio_application.AudioPlan,
    str,
    str,
    deck_build_application.ConjugationDeckBuildPlan,
]:
    proposed_revision = RecordsRevision(revision.deck_path.resolve(), revision.intended_deck_text)
    proposed_records = pattern_cards.drill_audio_records_from_revision(
        revision.deck_path,
        config,
        proposed_revision,
    )
    selected_audio_ids = _selected_audio_record_ids(
        revision,
        proposed_revision,
        proposed_records,
    )
    audio = audio_application.plan_deck_audio_revision(
        config,
        revision.deck_path,
        proposed_revision,
        record_ids=selected_audio_ids,
        words=False,
        examples=True,
        force=False,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    final_records = _project_audio_references(audio, proposed_records)
    final_text = pattern_cards.render_drill_audio_records(
        revision.deck_path,
        final_records,
        expected=proposed_revision,
    )
    final_sha = _sha(final_text.encode("utf-8"))
    final_revision = RecordsRevision(revision.deck_path.resolve(), final_text)
    projected_media = _projected_build_media(
        config,
        revision.deck_path,
        final_revision,
        audio,
    )
    build = deck_build_application.plan_conjugation_deck_build_revision(
        config,
        revision.deck_path,
        final_revision,
        projected_media=projected_media,
    )
    return audio, final_text, final_sha, build


def _plan_revision_finish_locked(
    config: ProjectConfig,
    staging_path: Path | str,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> RevisionFinishPlan:
    """Plan while the caller owns ``.janki-audio-operation``."""

    try:
        revision = revision_apply_application.plan_revision_apply(config, staging_path)
        audio, final_text, final_sha, build = _project_revision_finish(
            config,
            revision,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        _validate_provider_journal_binding(
            config,
            audio,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
    except RevisionFinishError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise RevisionFinishError(str(exc)) from exc
    finish_directory = _finish_directory(config)
    authority = {
        "version": 1,
        "revision": _revision_wire(config, revision),
        "audio": _audio_wire(config, audio, projected_deck_sha256=final_sha),
        "build": _build_wire(config, build),
    }
    fingerprint = _sha(_canonical(authority).encode("utf-8"))
    return RevisionFinishPlan(
        repository_root=config.root.resolve(),
        revision=revision,
        audio=audio,
        build=build,
        projected_audio_deck_text=final_text,
        projected_audio_deck_sha256=final_sha,
        finish_directory=finish_directory,
        record_path=finish_directory / f"finish-{fingerprint}.json",
        authority=authority,
        fingerprint=fingerprint,
    )


def plan_revision_finish(
    config: ProjectConfig,
    staging_path: Path | str,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
) -> RevisionFinishPlan:
    """Plan one exact consequence without writes or provider contact."""

    words, sentences = _resolved_providers(
        config,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        return _plan_revision_finish_locked(
            config,
            staging_path,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
        )


def _emit_progress(
    progress: RevisionFinishProgress | None,
    phase: RevisionFinishPhase,
) -> None:
    """A disconnected surface cannot interrupt durable work."""

    if progress is None:
        return
    try:
        progress(phase)
    except Exception:
        return


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RevisionFinishError(f"Revision finish JSON repeats key {key!r}.")
        value[key] = item
    return value


def _constant(value: str) -> Any:
    raise RevisionFinishError(f"Revision finish JSON contains non-finite number {value}.")


def _record_text(record: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _exact_mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RevisionFinishError(f"Revision finish has invalid {label} fields.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RevisionFinishError(f"Revision finish {label} must be nonblank text.")
    return value


def _provider_record(value: Any, label: str) -> None:
    provider = _exact_mapping(value, _PROVIDER_KEYS, label)
    for key in ("name", "access", "suffix", "destination", "transport"):
        _text(provider.get(key), f"{label} {key}")
    if provider.get("access") not in {"local-network", "paid-network"}:
        raise RevisionFinishError(f"Revision finish {label} access is invalid.")
    voice = provider.get("voice")
    speed = provider.get("speed")
    if isinstance(voice, bool) or not isinstance(voice, int | str):
        raise RevisionFinishError(f"Revision finish {label} voice is invalid.")
    if isinstance(speed, bool) or not isinstance(speed, int | float) or speed <= 0:
        raise RevisionFinishError(f"Revision finish {label} speed is invalid.")
    settings = provider.get("settings")
    if not isinstance(settings, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in settings.items()
    ):
        raise RevisionFinishError(f"Revision finish {label} settings are invalid.")


def _validate_authority(value: Any) -> Mapping[str, Any]:
    authority = _exact_mapping(value, _AUTHORITY_KEYS, "authority")
    if authority.get("version") != 1:
        raise RevisionFinishError("Revision finish authority version is unsupported.")

    revision = _exact_mapping(
        authority.get("revision"), _REVISION_AUTHORITY_KEYS, "revision authority"
    )
    for key in (
        "operation_id",
        "staging_path",
        "archive_path",
        "request_fingerprint",
        "provider",
        "billing_class",
        "model",
        "deck_path",
    ):
        _text(revision.get(key), f"revision {key}")
    for key in (
        "proposal_sha256",
        "request_bytes_sha256",
        "deck_base_sha256",
        "canonical_context_fingerprint",
        "intended_deck_sha256",
        "apply_plan_fingerprint",
    ):
        if not _is_sha(revision.get(key)):
            raise RevisionFinishError(f"Revision finish {key} is malformed.")
    selected = revision.get("selected_record_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or not item for item in selected)
        or len(selected) != len(set(selected))
    ):
        raise RevisionFinishError("Revision finish selected record ids are malformed.")

    audio = _exact_mapping(authority.get("audio"), _AUDIO_AUTHORITY_KEYS, "audio authority")
    for key in ("canonical_path", "ledger_path", "operations_path", "media_dir"):
        _text(audio.get(key), f"audio {key}")
    record_ids = audio.get("record_ids")
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or any(not isinstance(item, str) or not item for item in record_ids)
        or len(record_ids) != len(set(record_ids))
    ):
        raise RevisionFinishError("Revision finish audio record ids are malformed.")
    for key, expected in (
        ("targeted", True),
        ("words", False),
        ("examples", True),
        ("force", False),
        ("prune", False),
    ):
        if audio.get(key) is not expected:
            raise RevisionFinishError(f"Revision finish audio {key} is invalid.")
    if audio.get("word_provider") is not None:
        raise RevisionFinishError("Revision finish may not authorize word audio.")
    _provider_record(audio.get("example_provider"), "example provider")
    clips = audio.get("clips")
    if not isinstance(clips, list) or not clips:
        raise RevisionFinishError("Revision finish needs at least one example clip.")
    clip_targets: list[str] = []
    for position, raw_clip in enumerate(clips):
        clip = _exact_mapping(raw_clip, _CLIP_AUTHORITY_KEYS, f"audio clip {position + 1}")
        for key in ("record_id", "target", "request_input", "content_fingerprint"):
            _text(clip.get(key), f"audio clip {position + 1} {key}")
        if clip.get("kind") != "example" or clip.get("forced_accent") is not False:
            raise RevisionFinishError("Revision finish clip kind is invalid.")
        if not _is_sha(clip.get("content_fingerprint")):
            raise RevisionFinishError("Revision finish clip fingerprint is malformed.")
        _provider_record(clip.get("provider"), f"audio clip {position + 1} provider")
        state = clip.get("initial_state")
        if state not in _ALLOWED_CLIP_EVOLUTION:
            raise RevisionFinishError("Revision finish initial clip state is invalid.")
        recovery = (
            clip.get("initial_recovery_key"),
            clip.get("initial_recovery_sha256"),
            clip.get("initial_recovery_source"),
        )
        if state == "recoverable":
            if not all(isinstance(item, str) and item for item in recovery) or not _is_sha(
                recovery[1]
            ):
                raise RevisionFinishError("Revision finish recoverable clip identity is malformed.")
        elif any(item is not None for item in recovery):
            raise RevisionFinishError(
                "Revision finish non-recoverable clip carries recovery identity."
            )
        media_sha256 = clip.get("initial_media_sha256")
        if state == "current":
            if not _is_sha(media_sha256):
                raise RevisionFinishError(
                    "Revision finish current clip lacks its exact media hash."
                )
        elif media_sha256 is not None:
            raise RevisionFinishError(
                "Revision finish non-current clip carries a current-media hash."
            )
        clip_targets.append(str(clip["target"]))
    if len(clip_targets) != len(set(clip_targets)):
        raise RevisionFinishError("Revision finish repeats an example-audio target.")
    if not _is_sha(audio.get("initial_plan_fingerprint")) or not _is_sha(
        audio.get("projected_deck_sha256")
    ):
        raise RevisionFinishError("Revision finish audio fingerprint is malformed.")
    maximum = audio.get("max_provider_calls")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise RevisionFinishError("Revision finish paid-call maximum is malformed.")
    if maximum != sum(clip.get("initial_state") == "provider-required" for clip in clips):
        raise RevisionFinishError("Revision finish paid-call maximum is stale.")

    build = _exact_mapping(authority.get("build"), _BUILD_AUTHORITY_KEYS, "build authority")
    for key in ("deck_path", "source_path", "output_path", "deck_name", "form"):
        _text(build.get(key), f"build {key}")
    for key in ("deck_sha256", "source_sha256", "plan_fingerprint"):
        if not _is_sha(build.get(key)):
            raise RevisionFinishError(f"Revision finish build {key} is malformed.")
    templates = build.get("templates")
    if not isinstance(templates, list) or len(templates) != len(
        pattern_cards.PATTERN_TEMPLATE_FILENAMES
    ):
        raise RevisionFinishError("Revision finish build templates are malformed.")
    template_paths: list[str] = []
    for raw_template in templates:
        template = _exact_mapping(
            raw_template,
            {"path", "sha256"},
            "build template",
        )
        template_paths.append(_text(template.get("path"), "build template path"))
        if not _is_sha(template.get("sha256")):
            raise RevisionFinishError("Revision finish build template hash is malformed.")
    if len(template_paths) != len(set(template_paths)):
        raise RevisionFinishError("Revision finish repeats a build template path.")
    media = build.get("media")
    if not isinstance(media, list):
        raise RevisionFinishError("Revision finish build media is malformed.")
    media_paths: list[str] = []
    media_hashes: dict[str, str | None] = {}
    for raw_media in media:
        item = _exact_mapping(raw_media, {"path", "sha256"}, "build media")
        path = _text(item.get("path"), "build media path")
        sha256 = item.get("sha256")
        if sha256 is not None and not _is_sha(sha256):
            raise RevisionFinishError("Revision finish build media hash is malformed.")
        media_paths.append(path)
        media_hashes[path] = sha256
    if len(media_paths) != len(set(media_paths)):
        raise RevisionFinishError("Revision finish repeats a build media path.")
    media_directory = PurePosixPath(str(audio.get("media_dir")))
    expected_media: dict[str, str | None] = {}
    for raw_clip in clips:
        assert isinstance(raw_clip, Mapping)
        path = (media_directory / "audio" / str(raw_clip.get("target"))).as_posix()
        state = raw_clip.get("initial_state")
        sha256 = None
        if state == "current":
            sha256 = raw_clip.get("initial_media_sha256")
        elif state == "recoverable":
            sha256 = raw_clip.get("initial_recovery_sha256")
        expected_media[path] = sha256 if isinstance(sha256, str) else None
    if any(media_hashes.get(path) != sha256 for path, sha256 in expected_media.items()):
        raise RevisionFinishError(
            "Revision finish build media does not contain its exact audio authority."
        )
    unselected_media = set(media_hashes) - set(expected_media)
    if any(media_hashes[path] is None for path in unselected_media):
        raise RevisionFinishError(
            "Unselected revision build media must bind already-current exact bytes."
        )
    count = build.get("card_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RevisionFinishError("Revision finish build card count is malformed.")
    if build.get("deck_sha256") != audio.get("projected_deck_sha256"):
        raise RevisionFinishError("Revision finish build is not based on the projected audio deck.")
    if build.get("deck_path") != revision.get("deck_path"):
        raise RevisionFinishError("Revision finish deck paths disagree.")
    return authority


def _build_wires_compatible(
    projected: Mapping[str, Any],
    exact: Mapping[str, Any],
    *,
    require_exact_hashes: bool = False,
) -> bool:
    """Whether exact provider output only strengthens projected media hashes."""

    if set(projected) != _BUILD_AUTHORITY_KEYS or set(exact) != _BUILD_AUTHORITY_KEYS:
        return False
    for key in _BUILD_AUTHORITY_KEYS - {"media", "plan_fingerprint"}:
        if exact.get(key) != projected.get(key):
            return False
    if not _is_sha(exact.get("plan_fingerprint")):
        return False

    def media_map(value: Any, *, exact_hashes: bool) -> dict[str, str | None] | None:
        if not isinstance(value, list):
            return None
        result: dict[str, str | None] = {}
        for raw in value:
            if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
                return None
            path = raw.get("path")
            sha256 = raw.get("sha256")
            if not isinstance(path, str) or not path or path in result:
                return None
            if exact_hashes:
                if not _is_sha(sha256):
                    return None
            elif sha256 is not None and not _is_sha(sha256):
                return None
            result[path] = sha256 if isinstance(sha256, str) else None
        return result

    projected_media = media_map(projected.get("media"), exact_hashes=False)
    exact_media = media_map(exact.get("media"), exact_hashes=require_exact_hashes)
    if projected_media is None or exact_media is None:
        return False
    if set(projected_media) != set(exact_media):
        return False
    return all(
        sha256 is None or exact_media[path] == sha256 for path, sha256 in projected_media.items()
    )


def _validate_receipt(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    return _exact_mapping(value, keys, label)


def _validate_receipts(record: Mapping[str, Any]) -> None:
    authority = _exact_mapping(record.get("authority"), _AUTHORITY_KEYS, "authority")
    revision_authority = _exact_mapping(
        authority.get("revision"), _REVISION_AUTHORITY_KEYS, "revision authority"
    )
    audio_authority = _exact_mapping(
        authority.get("audio"), _AUDIO_AUTHORITY_KEYS, "audio authority"
    )
    build_authority = _exact_mapping(
        authority.get("build"), _BUILD_AUTHORITY_KEYS, "build authority"
    )
    apply = record.get("apply_receipt")
    if isinstance(apply, Mapping):
        apply = _validate_receipt(apply, _APPLY_RECEIPT_KEYS, "apply receipt")
        _text(apply.get("operation_id"), "apply receipt operation id")
        _text(apply.get("archive_path"), "apply receipt archive path")
        for key in ("deck_sha256", "archive_sha256"):
            if not _is_sha(apply.get(key)):
                raise RevisionFinishError(f"Revision finish apply receipt {key} is malformed.")
        expected_apply = {
            "operation_id": revision_authority.get("operation_id"),
            "deck_sha256": revision_authority.get("intended_deck_sha256"),
            "archive_path": revision_authority.get("archive_path"),
        }
        if any(apply.get(key) != value for key, value in expected_apply.items()):
            raise RevisionFinishError("Revision finish apply receipt does not match its authority.")
    audio = record.get("audio_receipt")
    if isinstance(audio, Mapping):
        audio = _validate_receipt(audio, _AUDIO_RECEIPT_KEYS, "audio receipt")
        for key in ("deck_sha256", "ledger_sha256"):
            if not _is_sha(audio.get(key)):
                raise RevisionFinishError(f"Revision finish audio receipt {key} is malformed.")
        clips = audio.get("clips")
        if not isinstance(clips, list) or not clips:
            raise RevisionFinishError("Revision finish audio receipt clips are malformed.")
        expected_clips = audio_authority.get("clips")
        if not isinstance(expected_clips, list) or len(clips) != len(expected_clips):
            raise RevisionFinishError(
                "Revision finish audio receipt clip set does not match its authority."
            )
        for raw_clip, expected_clip in zip(clips, expected_clips, strict=True):
            clip = _validate_receipt(raw_clip, _AUDIO_RECEIPT_CLIP_KEYS, "audio receipt clip")
            _text(clip.get("target"), "audio receipt clip target")
            if not _is_sha(clip.get("sha256")):
                raise RevisionFinishError("Revision finish audio receipt hash is malformed.")
            if not isinstance(expected_clip, Mapping) or clip.get("target") != expected_clip.get(
                "target"
            ):
                raise RevisionFinishError(
                    "Revision finish audio receipt target does not match its authority."
                )
            initial_media_sha256 = expected_clip.get("initial_media_sha256")
            if (
                expected_clip.get("initial_state") == "current"
                and clip.get("sha256") != initial_media_sha256
            ):
                raise RevisionFinishError("Revision finish changed an already-current audio file.")
            if expected_clip.get("initial_state") == "recoverable" and clip.get(
                "sha256"
            ) != expected_clip.get("initial_recovery_sha256"):
                raise RevisionFinishError("Revision finish changed a recoverable audio file.")
        if audio.get("deck_sha256") != audio_authority.get("projected_deck_sha256"):
            raise RevisionFinishError(
                "Revision finish audio receipt deck does not match its authority."
            )
    build_binding = record.get("build_binding")
    if isinstance(build_binding, Mapping):
        build_binding = _exact_mapping(build_binding, _BUILD_AUTHORITY_KEYS, "build binding")
        if not _build_wires_compatible(
            build_authority,
            build_binding,
            require_exact_hashes=True,
        ):
            raise RevisionFinishError(
                "Revision finish build binding widens its confirmed authority."
            )
        raw_media = build_binding.get("media")
        if not isinstance(raw_media, list) or any(
            not isinstance(item, Mapping) or not _is_sha(item.get("sha256")) for item in raw_media
        ):
            raise RevisionFinishError("Revision finish build binding lacks exact media hashes.")
    build = record.get("build_receipt")
    if isinstance(build, Mapping):
        build = _validate_receipt(build, _BUILD_RECEIPT_KEYS, "build receipt")
        _text(build.get("output_path"), "build receipt output path")
        for key in ("package_sha256", "plan_fingerprint"):
            if not _is_sha(build.get(key)):
                raise RevisionFinishError(f"Revision finish build receipt {key} is malformed.")
        count = build.get("card_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RevisionFinishError("Revision finish build receipt count is malformed.")
        if not isinstance(build_binding, Mapping):
            raise RevisionFinishError(
                "Revision finish build receipt lacks its exact build binding."
            )
        expected_build = {
            "output_path": build_binding.get("output_path"),
            "card_count": build_binding.get("card_count"),
            "plan_fingerprint": build_binding.get("plan_fingerprint"),
        }
        if any(build.get(key) != value for key, value in expected_build.items()):
            raise RevisionFinishError("Revision finish build receipt does not match its authority.")


def _strict_record(wire: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            wire.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except RevisionFinishError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RevisionFinishError(f"Could not parse revision finish {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise RevisionFinishError("Revision finish has invalid top-level fields.")
    if value.get("schema_version") != 1 or value.get("kind") != "revision_finish":
        raise RevisionFinishError("Unsupported revision finish record.")
    receipt_id = value.get("receipt_id")
    if not _is_sha(receipt_id) or path.name != f"finish-{receipt_id}.json":
        raise RevisionFinishError("Revision finish receipt does not match its filename.")
    authority = _validate_authority(value.get("authority"))
    if _sha(_canonical(authority).encode("utf-8")) != receipt_id:
        raise RevisionFinishError("Revision finish authority fingerprint is corrupt.")
    reservations = value.get("paid_clip_reservations")
    if not isinstance(reservations, list):
        raise RevisionFinishError("Revision finish paid reservations are malformed.")
    audio_authority = _exact_mapping(
        authority.get("audio"), _AUDIO_AUTHORITY_KEYS, "audio authority"
    )
    raw_clips = audio_authority.get("clips")
    assert isinstance(raw_clips, list)  # established by _validate_authority
    paid_targets = {
        str(clip.get("target"))
        for clip in raw_clips
        if isinstance(clip, Mapping)
        and clip.get("initial_state") == "provider-required"
        and isinstance(clip.get("provider"), Mapping)
        and clip["provider"].get("access") == "paid-network"
    }
    seen_targets: set[str] = set()
    seen_operations: set[str] = set()
    for raw_reservation in reservations:
        reservation = _exact_mapping(
            raw_reservation,
            _PAID_RESERVATION_KEYS,
            "paid clip reservation",
        )
        target = _text(reservation.get("target"), "paid reservation target")
        operation_id = _text(reservation.get("operation_id"), "paid reservation operation id")
        if reservation.get("status") not in _PAID_RESERVATION_STATUSES:
            raise RevisionFinishError("Revision finish paid reservation status is malformed.")
        try:
            canonical_operation = str(uuid.UUID(operation_id))
        except (ValueError, AttributeError) as exc:
            raise RevisionFinishError(
                "Revision finish paid reservation operation id is malformed."
            ) from exc
        if canonical_operation != operation_id:
            raise RevisionFinishError(
                "Revision finish paid reservation operation id is not canonical."
            )
        if target not in paid_targets:
            raise RevisionFinishError("Revision finish reserved a clip outside its paid authority.")
        if target in seen_targets or operation_id in seen_operations:
            raise RevisionFinishError("Revision finish repeats a paid clip reservation.")
        seen_targets.add(target)
        seen_operations.add(operation_id)
    state = value.get("state")
    if state not in _STATE_INDEX:
        raise RevisionFinishError("Revision finish state is invalid.")
    for label in ("authorized_at", "updated_at"):
        timestamp = value.get(label)
        if not isinstance(timestamp, str):
            raise RevisionFinishError(f"Revision finish {label} is malformed.")
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise RevisionFinishError(f"Revision finish {label} is malformed.") from exc
    receipts = (
        value.get("apply_receipt"),
        value.get("audio_receipt"),
        value.get("build_receipt"),
    )
    required = _STATE_INDEX[state]
    if any(
        (position < required and not isinstance(receipt, Mapping))
        or (position >= required and receipt is not None)
        for position, receipt in enumerate(receipts)
    ):
        raise RevisionFinishError("Revision finish receipts do not match its durable state.")
    build_binding = value.get("build_binding")
    if state in {"authorized", "revision_applied"} and build_binding is not None:
        raise RevisionFinishError("Revision finish bound build media before audio completed.")
    if state == "complete" and not isinstance(build_binding, Mapping):
        raise RevisionFinishError("Completed revision finish lacks its exact build binding.")
    if build_binding is not None and not isinstance(build_binding, Mapping):
        raise RevisionFinishError("Revision finish build binding is malformed.")
    _validate_receipts(value)
    return value


def _read_record(path: Path) -> tuple[dict[str, Any], str]:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError as exc:
        raise RevisionFinishError(f"Revision finish no longer exists: {path}") from exc
    except (DataError, OSError) as exc:
        raise RevisionFinishError(f"Could not safely read revision finish {path}: {exc}") from exc
    return _strict_record(wire, path), _sha(wire)


def _read_record_optional(path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise RevisionFinishError(f"Could not safely read revision finish {path}: {exc}") from exc
    return _strict_record(wire, path), _sha(wire)


def _new_record(plan: RevisionFinishPlan) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "kind": "revision_finish",
        "receipt_id": plan.fingerprint,
        "state": "authorized",
        "authority": dict(plan.authority),
        "authorized_at": now,
        "updated_at": now,
        "paid_clip_reservations": [],
        "apply_receipt": None,
        "audio_receipt": None,
        "build_binding": None,
        "build_receipt": None,
    }


def _authority_section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    authority = record.get("authority")
    if not isinstance(authority, Mapping):
        raise RevisionFinishError("Revision finish authority is malformed.")
    section = authority.get(key)
    if not isinstance(section, Mapping):
        raise RevisionFinishError(f"Revision finish {key} authority is malformed.")
    return section


def _authority_path(
    config: ProjectConfig,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise RevisionFinishError(f"Revision finish {label} path is malformed.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise RevisionFinishError(f"Revision finish {label} path is not repository-relative.")
    root = config.root.absolute()
    path = (root / Path(*pure.parts)).absolute()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RevisionFinishError(f"Revision finish {label} path escapes the repository.") from exc
    return path


def _record_path(config: ProjectConfig, receipt_id: str) -> Path:
    if not _is_sha(receipt_id):
        raise RevisionFinishError("Revision finish receipt id must be lowercase SHA-256.")
    return _finish_directory(config) / f"finish-{receipt_id}.json"


def _active_finish_for_revision(
    config: ProjectConfig,
    *,
    operation_id: str,
    staging_path: str,
) -> str | None:
    """Return an existing exact authority for this revision, if one exists."""

    directory = _finish_directory(config)
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RevisionFinishError(f"Could not inspect revision finish records: {exc}") from exc
    found: str | None = None
    for entry in entries:
        if not entry.name.startswith("finish-") or entry.suffix != ".json":
            continue
        try:
            details = os.lstat(entry)
        except OSError as exc:
            raise RevisionFinishError(
                f"Could not inspect revision finish {entry.name}: {exc}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            raise RevisionFinishError(f"Revision finish {entry.name} is not a direct regular file.")
        record, _revision = _read_record(entry)
        revision = _authority_section(record, "revision")
        if (
            revision.get("operation_id") == operation_id
            or revision.get("staging_path") == staging_path
        ):
            receipt = str(record["receipt_id"])
            if found is not None and found != receipt:
                raise RevisionFinishError(
                    "More than one durable finish authority names this revision."
                )
            found = receipt
    return found


def _advance_record(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    *,
    from_state: RevisionFinishState,
    to_state: RevisionFinishState,
    receipt_key: Literal["apply_receipt", "audio_receipt", "build_receipt"],
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if record.get("state") != from_state:
        raise RevisionFinishError(
            f"Revision finish cannot advance from {record.get('state')!r} to {to_state!r}."
        )
    if _STATE_INDEX[to_state] != _STATE_INDEX[from_state] + 1:
        raise RevisionFinishError("Revision finish states may advance only one phase.")
    updated = dict(record)
    updated["state"] = to_state
    updated[receipt_key] = dict(receipt)
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current_revision != revision or current != dict(record):
            raise RevisionFinishError(
                "Revision finish changed while its completed phase was recorded."
            )
        atomic_write_text_bound(path, text, expected_revision=revision)
    return updated, _sha(text.encode("utf-8"))


def _bind_executable_build(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    plan: deck_build_application.ConjugationDeckBuildPlan,
) -> tuple[dict[str, Any], str]:
    """CAS-bind exact provider output before packaging may read it."""

    if record.get("state") not in {"audio_complete", "complete"}:
        raise RevisionFinishError(
            "Revision finish may bind package media only after audio completes."
        )
    if any(item.sha256 is None for item in plan.media_inputs):
        raise RevisionFinishError("Revision finish cannot bind a build with unproduced media.")
    wire = _build_wire(config, plan)
    if not _build_wires_compatible(
        _authority_section(record, "build"),
        wire,
        require_exact_hashes=True,
    ):
        raise RevisionFinishError("The package build widens the confirmed finish authority.")
    existing = record.get("build_binding")
    if existing is not None:
        if not isinstance(existing, Mapping) or dict(existing) != wire:
            raise RevisionFinishError("The package build changed after its exact media was bound.")
        return dict(record), revision
    if record.get("state") == "complete":
        raise RevisionFinishError("Completed finish lacks its exact build binding.")
    updated = dict(record)
    updated["build_binding"] = wire
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current_revision != revision or current != dict(record):
            raise RevisionFinishError("Revision finish changed while its package media was bound.")
        atomic_write_text_bound(path, text, expected_revision=revision)
    return updated, _sha(text.encode("utf-8"))


_RESERVED_OPERATION_STATES = {
    "authorized",
    "canceled_before_send",
    "dispatching",
    "running",
    "outcome_unknown",
    "result_captured",
    "committed",
    "failed_before_send",
}
_PROVEN_UNSENT_OPERATION_STATES = {
    "authorized",
    "canceled_before_send",
    "failed_before_send",
}


def _realtime_profile(
    sentence_provider: Any,
    clip: audio_application.AudioClipPlan,
) -> openai_realtime.OpenAiRealtimeProvider:
    try:
        provider = sentence_profile_for(sentence_provider, clip.record_id)
    except JankiError as exc:
        raise RevisionFinishError(str(exc)) from exc
    if not isinstance(provider, openai_realtime.OpenAiRealtimeProvider):
        raise RevisionFinishError("Reserved paid audio no longer selects OpenAI Realtime.")
    return provider


def _validate_reserved_operation(
    config: ProjectConfig,
    operation_id: str,
    clip: audio_application.AudioClipPlan,
    *,
    sentence_provider: Any,
    allow_cleanup: bool = False,
) -> operations.Operation:
    provider = _realtime_profile(sentence_provider, clip)
    try:
        operation = operations.OperationJournal.load(config.operations_file).operations.get(
            operation_id
        )
        request_fingerprint = provider.request_fingerprint(
            clip.request_input,
            forced_accent=clip.forced_accent,
        )
    except JankiError as exc:
        raise RevisionFinishError(str(exc)) from exc
    if operation is None:
        raise RevisionFinishError(
            f"Paid audio reservation {operation_id} no longer has its operation "
            "evidence; refusing a second provider call."
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
        or operation.request_fp != request_fingerprint
        or operation.model != openai_realtime.MODEL
        or (operation.cleanup is not None and not allow_cleanup)
        or operation.response_spool is None
        or operation.state not in _RESERVED_OPERATION_STATES
    ):
        raise RevisionFinishError(
            f"Paid audio reservation {operation_id} no longer matches its exact "
            "recoverable provider operation."
        )
    return operation


def _proven_unsent_operations_for_clip(
    config: ProjectConfig,
    clip: audio_application.AudioClipPlan,
    *,
    sentence_provider: Any,
) -> list[operations.Operation]:
    provider = _realtime_profile(sentence_provider, clip)
    try:
        request_fingerprint = provider.request_fingerprint(
            clip.request_input,
            forced_accent=clip.forced_accent,
        )
        journal = operations.OperationJournal.load(config.operations_file)
    except JankiError as exc:
        raise RevisionFinishError(str(exc)) from exc
    expected_source = audio_application.audio_cmd.audio_journal_source(
        clip.record_id,
        of=clip.kind,
        target=clip.target,
    )
    return [
        operation
        for operation in journal.operations.values()
        if operation.kind == "audio-realtime"
        and operation.state in _PROVEN_UNSENT_OPERATION_STATES
        and operation.source_file == expected_source
        and operation.source_sha256 == clip.content_fingerprint
        and operation.request_fp == request_fingerprint
        and operation.model == openai_realtime.MODEL
    ]


def _reserve_paid_clip(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    dispatch: audio_application.PaidAudioDispatch,
    *,
    sentence_provider: Any,
) -> tuple[dict[str, Any], str]:
    clip = dispatch.clip
    if clip.kind != "example" or clip.provider.access != "paid-network":
        raise RevisionFinishError("Finish may reserve only its paid example-audio clips.")
    audio_authority = _authority_section(record, "audio")
    expected_clips = audio_authority.get("clips")
    if not isinstance(expected_clips, list):
        raise RevisionFinishError("Stored example-audio authority is malformed.")
    expected = [
        item
        for item in expected_clips
        if isinstance(item, Mapping) and item.get("target") == clip.target
    ]
    if (
        len(expected) != 1
        or expected[0].get("initial_state") != "provider-required"
        or dict(expected[0]) != _clip_wire(config, clip)
    ):
        raise RevisionFinishError("Paid audio dispatch exceeds the confirmed finish authority.")
    _validate_reserved_operation(
        config,
        dispatch.operation_id,
        clip,
        sentence_provider=sentence_provider,
    )
    reservations = record.get("paid_clip_reservations")
    if not isinstance(reservations, list):
        raise RevisionFinishError("Revision finish paid reservations are malformed.")
    same_target = [
        item
        for item in reservations
        if isinstance(item, Mapping) and item.get("target") == clip.target
    ]
    if (
        len(same_target) > 1
        or (same_target and same_target[0].get("status") != "failed_before_send")
        or any(
            isinstance(item, Mapping) and item.get("operation_id") == dispatch.operation_id
            for item in reservations
        )
    ):
        raise RevisionFinishError(
            "Paid example audio already consumed its one dispatch reservation."
        )
    updated = dict(record)
    updated["paid_clip_reservations"] = [
        *[
            item
            for item in reservations
            if not isinstance(item, Mapping) or item.get("target") != clip.target
        ],
        {
            "target": clip.target,
            "operation_id": dispatch.operation_id,
            "status": "reserved",
        },
    ]
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current_revision != revision or current != dict(record):
            raise RevisionFinishError("Revision finish changed while paid audio was reserved.")
        atomic_write_text_bound(path, text, expected_revision=revision)
    return updated, _sha(text.encode("utf-8"))


def _revision_plan_for_record(
    config: ProjectConfig,
    record: Mapping[str, Any],
) -> revision_apply_application.RevisionApplyPlan:
    authority = _authority_section(record, "revision")
    staging = _authority_path(config, authority.get("staging_path"), label="staging")
    archive = _authority_path(config, authority.get("archive_path"), label="revision archive")
    state = str(record.get("state"))
    if staging.exists():
        if state != "authorized":
            raise RevisionFinishError(
                "Applied revision finish unexpectedly regained a live proposal."
            )
        plan = revision_apply_application.plan_revision_apply(config, staging)
    elif state == "authorized":
        plan = revision_apply_application.plan_revision_apply_recovery(
            config,
            staging,
            archive_path=archive,
            plan_fingerprint=str(authority.get("apply_plan_fingerprint") or ""),
        )
    else:
        plan = revision_apply_application.plan_revision_apply_archived_authority(
            config,
            staging,
            archive_path=archive,
            plan_fingerprint=str(authority.get("apply_plan_fingerprint") or ""),
        )
    if _revision_wire(config, plan) != dict(authority):
        raise RevisionFinishError(
            "The reviewed revision no longer matches its durable finish authority."
        )
    if state != "authorized":
        receipt = record.get("apply_receipt")
        if not isinstance(receipt, Mapping):
            raise RevisionFinishError("Applied revision finish lacks its apply receipt.")
        try:
            archive_sha256 = _sha(read_bytes_bound(archive))
        except (DataError, OSError) as exc:
            raise RevisionFinishError(f"Could not verify accepted revision archive: {exc}") from exc
        if archive_sha256 != receipt.get("archive_sha256"):
            raise RevisionFinishError("Accepted revision archive changed after it was applied.")
    return plan


_CLIP_IDENTITY_KEYS = {
    "record_id",
    "kind",
    "target",
    "request_input",
    "forced_accent",
    "content_fingerprint",
    "provider",
}
_ALLOWED_CLIP_EVOLUTION = {
    "current": {"current"},
    "recoverable": {"recoverable", "current"},
    "provider-required": {"provider-required", "recoverable", "current"},
}


def _validate_audio_evolution(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: audio_application.AudioPlan,
) -> None:
    authority = _authority_section(record, "audio")
    fixed = {
        "canonical_path": _relative(config, fresh.canonical_path, "audio owner"),
        "ledger_path": _relative(config, fresh.ledger_path, "audio ledger"),
        "operations_path": _relative(config, config.operations_file.resolve(), "operation journal"),
        "media_dir": _relative(config, fresh.media_dir, "media directory"),
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
            raise RevisionFinishError(
                f"The example-audio {key.replace('_', ' ')} changed after confirmation."
            )
    expected_clips = authority.get("clips")
    if not isinstance(expected_clips, list) or len(expected_clips) != len(fresh.clips):
        raise RevisionFinishError("The example-audio clip set changed after confirmation.")
    provider_required = 0
    for expected, clip in zip(expected_clips, fresh.clips, strict=True):
        if not isinstance(expected, Mapping):
            raise RevisionFinishError("Stored example-audio clip authority is malformed.")
        current = _clip_wire(config, clip)
        if any(expected.get(key) != current.get(key) for key in _CLIP_IDENTITY_KEYS):
            raise RevisionFinishError(
                "An example-audio request or provider profile changed after confirmation."
            )
        initial_state = expected.get("initial_state")
        allowed = _ALLOWED_CLIP_EVOLUTION.get(str(initial_state))
        if allowed is None or clip.state not in allowed:
            raise RevisionFinishError(
                "An example clip would require authority wider than the confirmed plan."
            )
        if (
            initial_state == "recoverable"
            and clip.state == "recoverable"
            and (
                expected.get("initial_recovery_key") != clip.recovery_key
                or expected.get("initial_recovery_sha256") != clip.recovery_sha256
                or expected.get("initial_recovery_source") != clip.recovery_source
            )
        ):
            raise RevisionFinishError(
                "The exact recoverable example clip changed after confirmation."
            )
        if (
            initial_state == "recoverable"
            and clip.state == "current"
            and current.get("initial_media_sha256") != expected.get("initial_recovery_sha256")
        ):
            raise RevisionFinishError(
                "A recovered example clip no longer has its authorized bytes."
            )
        if initial_state == "current" and expected.get("initial_media_sha256") != current.get(
            "initial_media_sha256"
        ):
            raise RevisionFinishError(
                "An already-current example-audio file changed after confirmation."
            )
        if clip.state == "provider-required":
            provider_required += 1
    maximum = authority.get("max_provider_calls")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or provider_required > maximum:
        raise RevisionFinishError(
            "The example-audio run would exceed the confirmed paid-call maximum."
        )


def _validate_paid_reservations(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: audio_application.AudioPlan,
    *,
    sentence_provider: Any,
) -> None:
    reservations = record.get("paid_clip_reservations")
    if not isinstance(reservations, list):
        raise RevisionFinishError("Revision finish paid reservations are malformed.")
    by_target = {clip.target: clip for clip in fresh.clips}
    for reservation in reservations:
        if not isinstance(reservation, Mapping):
            raise RevisionFinishError("Revision finish paid reservation is malformed.")
        target = str(reservation.get("target") or "")
        operation_id = str(reservation.get("operation_id") or "")
        clip = by_target.get(target)
        if clip is None:
            raise RevisionFinishError("Reserved paid audio disappeared from the finish plan.")
        if clip.state == "provider-required" and reservation.get("status") == "reserved":
            _validate_reserved_operation(
                config,
                operation_id,
                clip,
                sentence_provider=sentence_provider,
            )


def _reconcile_proven_unsent_reservations(
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
        raise RevisionFinishError("Revision finish paid reservations are malformed.")
    clips = {clip.target: clip for clip in fresh.clips}
    changed = False
    updated_reservations: list[dict[str, Any]] = []
    forget_after_record: list[str] = []
    for raw in reservations:
        if not isinstance(raw, Mapping):
            raise RevisionFinishError("Revision finish paid reservation is malformed.")
        item = dict(raw)
        target = str(item.get("target") or "")
        operation_id = str(item.get("operation_id") or "")
        clip = clips.get(target)
        if clip is None:
            raise RevisionFinishError("Reserved paid audio disappeared from the finish plan.")
        operation = operations.OperationJournal.load(config.operations_file).operations.get(
            operation_id
        )
        if item.get("status") == "failed_before_send":
            if operation is not None:
                operation = _validate_reserved_operation(
                    config,
                    operation_id,
                    clip,
                    sentence_provider=sentence_provider,
                    allow_cleanup=True,
                )
                if operation.state not in {
                    "canceled_before_send",
                    "failed_before_send",
                }:
                    raise RevisionFinishError(
                        "A released paid reservation regained a billable operation."
                    )
                forget_after_record.append(operation_id)
            updated_reservations.append(item)
            continue
        if operation is None:
            updated_reservations.append(item)
            continue
        if operation.state in _PROVEN_UNSENT_OPERATION_STATES:
            operation = _validate_reserved_operation(
                config,
                operation_id,
                clip,
                sentence_provider=sentence_provider,
                allow_cleanup=operation.state != "authorized",
            )
        elif operation.state in {"result_captured", "committed"} and (
            clip.state != "provider-required"
        ):
            operation = _validate_reserved_operation(
                config,
                operation_id,
                clip,
                sentence_provider=sentence_provider,
                allow_cleanup=True,
            )
            if operation.cleanup is not None:
                # The journal already holds the exact durable forget decision.
                # Resume it here so ordinary audio recovery never wedges on a
                # successful call whose automatic cleanup was interrupted.
                forget_after_record.append(operation_id)
            updated_reservations.append(item)
            continue
        else:
            _validate_reserved_operation(
                config,
                operation_id,
                clip,
                sentence_provider=sentence_provider,
            )
            updated_reservations.append(item)
            continue
        if operation.state == "authorized":
            operations.OperationJournal.load(config.operations_file).advance(
                operation_id,
                "failed_before_send",
                detail="Revision finish resumed before provider dispatch began.",
            )
            operation = operations.OperationJournal.load(config.operations_file).operations[
                operation_id
            ]
        if operation.state in {"canceled_before_send", "failed_before_send"}:
            item["status"] = "failed_before_send"
            changed = True
            forget_after_record.append(operation_id)
        updated_reservations.append(item)
    updated = dict(record)
    updated_revision = revision
    if changed:
        updated["paid_clip_reservations"] = updated_reservations
        updated["updated_at"] = datetime.now(UTC).isoformat()
        text = _record_text(updated)
        _strict_record(text.encode("utf-8"), path)
        with exclusive_path_lock(path):
            current, current_revision = _read_record(path)
            if current_revision != revision or current != dict(record):
                raise RevisionFinishError(
                    "Revision finish changed while a proven-unsent call was released."
                )
            atomic_write_text_bound(path, text, expected_revision=revision)
        updated_revision = _sha(text.encode("utf-8"))
    for operation_id in forget_after_record:
        operations.OperationJournal.load(config.operations_file).forget([operation_id])
    reserved_operation_ids = {
        str(item.get("operation_id") or "")
        for item in updated_reservations
        if isinstance(item, Mapping) and item.get("status") == "reserved"
    }
    for clip in fresh.clips:
        if clip.state != "provider-required" or clip.provider.access != "paid-network":
            continue
        proven_unsent = _proven_unsent_operations_for_clip(
            config,
            clip,
            sentence_provider=sentence_provider,
        )
        for operation in proven_unsent:
            if operation.operation_id in reserved_operation_ids:
                continue
            if operation.state == "authorized":
                operations.OperationJournal.load(config.operations_file).advance(
                    operation.operation_id,
                    "failed_before_send",
                    detail=("Revision finish resumed before its paid reservation became durable."),
                )
            operations.OperationJournal.load(config.operations_file).forget(
                [operation.operation_id]
            )
    return updated, updated_revision


def _apply_receipt(
    config: ProjectConfig,
    result: revision_apply_application.RevisionApplyResult,
) -> dict[str, Any]:
    archive = read_bytes_bound(result.archive_path)
    return {
        "operation_id": result.operation_id,
        "deck_sha256": result.deck_sha256,
        "archive_path": _relative(config, result.archive_path, "revision archive"),
        "archive_sha256": _sha(archive),
    }


def _media_path(config: ProjectConfig, target: str) -> Path:
    pure = PurePosixPath(target)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != target
        or target in {"", ".", ".."}
    ):
        raise RevisionFinishError("An authorized example-audio target is malformed.")
    audio_root = config.media_dir.absolute() / audio_application.audio_cmd.AUDIO_SUBDIR
    return audio_root / target


def _audio_receipt(
    config: ProjectConfig,
    plan: audio_application.AudioPlan,
) -> dict[str, Any]:
    deck = records_revision(plan.canonical_path)
    if deck.text is None:
        raise RevisionFinishError("The revised deck disappeared after audio finished.")
    ledger_wire = read_bytes_bound(config.ledger_file.resolve())
    clips: list[dict[str, str]] = []
    for clip in plan.clips:
        path = _media_path(config, clip.target)
        clips.append(
            {
                "target": clip.target,
                "sha256": _sha(read_bytes_bound(path)),
            }
        )
    return {
        "deck_sha256": _sha(deck.text.encode("utf-8")),
        "ledger_sha256": _sha(ledger_wire),
        "clips": clips,
    }


def _build_receipt(
    config: ProjectConfig,
    result: deck_build_application.ConjugationDeckBuildResult,
    plan: deck_build_application.ConjugationDeckBuildPlan,
) -> dict[str, Any]:
    return {
        "output_path": _relative(config, result.output_path, "build output"),
        "package_sha256": result.package_sha256,
        "card_count": result.card_count,
        "plan_fingerprint": _build_wire(config, plan)["plan_fingerprint"],
    }


def _same_build_authority(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: deck_build_application.ConjugationDeckBuildPlan,
) -> bool:
    return _build_wires_compatible(
        _authority_section(record, "build"),
        _build_wire(config, fresh),
    )


def _validate_canonical_audio_evolution(
    config: ProjectConfig,
    record: Mapping[str, Any],
    revision: revision_apply_application.RevisionApplyPlan,
    audio: audio_application.AudioPlan,
    final_text: str,
) -> None:
    current = records_revision(revision.deck_path)
    if current.text is None:
        raise RevisionFinishError("The revision deck disappeared during finish.")
    state = str(record.get("state"))
    if state in {"audio_complete", "complete"}:
        if current.text != final_text:
            raise RevisionFinishError(
                "The audio-complete deck no longer matches the confirmed final bytes."
            )
        return
    if state == "authorized":
        return
    if state != "revision_applied":
        raise RevisionFinishError("Revision finish has an invalid recovery state.")

    baseline_revision = RecordsRevision(revision.deck_path.resolve(), revision.intended_deck_text)
    baseline = pattern_cards.drill_audio_records_from_revision(
        revision.deck_path,
        config,
        baseline_revision,
    )
    projected = _project_audio_references(audio, baseline)
    actual = pattern_cards.drill_audio_records_from_revision(
        revision.deck_path,
        config,
        current,
    )
    if len(actual) != len(baseline) or len(projected) != len(baseline):
        raise RevisionFinishError("The revision deck changed outside its authorized audio fields.")
    for before, after, complete in zip(baseline, actual, projected, strict=True):
        if after.id != before.id or complete.id != before.id:
            raise RevisionFinishError("The revision deck changed its audio-owner identities.")
        if len(after.examples) != len(before.examples) or len(complete.examples) != len(
            before.examples
        ):
            raise RevisionFinishError(
                "The revision deck changed outside its authorized audio fields."
            )
        normalized_examples: list[ExampleSentence] = []
        for old_example, current_example, final_example in zip(
            before.examples,
            after.examples,
            complete.examples,
            strict=True,
        ):
            if replace(current_example, audio=old_example.audio) != old_example:
                raise RevisionFinishError(
                    "The revision deck changed outside its authorized audio fields."
                )
            if current_example.audio not in {old_example.audio, final_example.audio}:
                raise RevisionFinishError(
                    "The revision deck names audio outside the confirmed finish plan."
                )
            normalized_examples.append(old_example)
        if replace(after, examples=normalized_examples) != before:
            raise RevisionFinishError(
                "The revision deck changed outside its authorized audio fields."
            )
    allowed_text = pattern_cards.render_drill_audio_records(
        revision.deck_path,
        actual,
        expected=baseline_revision,
    )
    if current.text != allowed_text:
        raise RevisionFinishError("The revision deck changed outside its authorized audio fields.")


def _validate_completed_audio_receipt(
    config: ProjectConfig,
    record: Mapping[str, Any],
    fresh: audio_application.AudioPlan,
) -> None:
    if str(record.get("state")) not in {"audio_complete", "complete"}:
        return
    receipt = record.get("audio_receipt")
    if not isinstance(receipt, Mapping):
        raise RevisionFinishError("Audio-complete finish lacks its audio receipt.")
    raw_clips = receipt.get("clips")
    if not isinstance(raw_clips, list):
        raise RevisionFinishError("Audio-complete finish has a malformed receipt.")
    receipt_hashes = {
        str(item.get("target")): str(item.get("sha256"))
        for item in raw_clips
        if isinstance(item, Mapping)
    }
    if len(receipt_hashes) != len(fresh.clips):
        raise RevisionFinishError("Audio-complete finish no longer has its exact clip receipt set.")
    for clip in fresh.clips:
        if clip.state != "current":
            raise RevisionFinishError(
                "Audio-complete finish regressed to missing or recoverable media."
            )
        expected_sha256 = receipt_hashes.get(clip.target)
        try:
            actual_sha256 = _sha(read_bytes_bound(_media_path(config, clip.target)))
        except (DataError, OSError) as exc:
            raise RevisionFinishError(
                f"Could not verify completed audio {clip.target!r}: {exc}"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise RevisionFinishError("Audio-complete media no longer matches its finish receipt.")


def _prevalidate_record_execution(
    config: ProjectConfig,
    record: Mapping[str, Any],
    revision: revision_apply_application.RevisionApplyPlan,
    *,
    chosen_provider: str | None,
    word_provider: Any,
    sentence_provider: Any,
) -> tuple[
    audio_application.AudioPlan,
    str,
    deck_build_application.ConjugationDeckBuildPlan,
]:
    try:
        projected_audio, final_text, final_sha, fresh_build = _project_revision_finish(
            config,
            revision,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        fresh_audio = projected_audio
        if str(record.get("state")) != "authorized":
            audio_authority = _authority_section(record, "audio")
            fresh_audio = audio_application.plan_deck_audio(
                config,
                revision.deck_path,
                record_ids=tuple(audio_authority["record_ids"]),
                words=False,
                examples=True,
                force=False,
                chosen_provider=chosen_provider,
                word_provider=word_provider,
                sentence_provider=sentence_provider,
            )
        _validate_provider_journal_binding(
            config,
            fresh_audio,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        _validate_audio_evolution(config, record, fresh_audio)
        audio_authority = _authority_section(record, "audio")
        if final_sha != audio_authority.get("projected_deck_sha256"):
            raise RevisionFinishError("The projected final deck changed after confirmation.")
        if not _same_build_authority(config, record, fresh_build):
            raise RevisionFinishError(
                "The package build changed after the finish plan was confirmed."
            )
        _validate_canonical_audio_evolution(
            config,
            record,
            revision,
            projected_audio,
            final_text,
        )
        _validate_completed_audio_receipt(config, record, fresh_audio)
    except RevisionFinishError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise RevisionFinishError(str(exc)) from exc
    return fresh_audio, final_text, fresh_build


def _result_from_record(
    config: ProjectConfig,
    record: Mapping[str, Any],
    *,
    path: Path,
    audio: audio_application.AudioExecutionOutcome | None = None,
) -> RevisionFinishResult:
    revision = _authority_section(record, "revision")
    audio_authority = _authority_section(record, "audio")
    example_provider = audio_authority["example_provider"]
    assert isinstance(example_provider, Mapping)
    build = _authority_section(record, "build")
    build_receipt = record.get("build_receipt")
    return RevisionFinishResult(
        receipt_id=str(record["receipt_id"]),
        state=str(record["state"]),  # type: ignore[arg-type]
        record_path=path,
        deck_path=_authority_path(config, revision.get("deck_path"), label="deck"),
        output_path=_authority_path(config, build.get("output_path"), label="build output"),
        example_provider_access=str(example_provider["access"]),  # type: ignore[arg-type]
        max_provider_calls=int(audio_authority["max_provider_calls"]),
        package_sha256=(
            str(build_receipt.get("package_sha256")) if isinstance(build_receipt, Mapping) else None
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


def _replace_complete_build_receipt(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if record.get("state") != "complete":
        raise RevisionFinishError("Only a completed finish can refresh its package.")
    updated = dict(record)
    updated["build_receipt"] = dict(receipt)
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current_revision != revision or current != dict(record):
            raise RevisionFinishError(
                "Revision finish changed while its package receipt was refreshed."
            )
        atomic_write_text_bound(path, text, expected_revision=revision)
    return updated, _sha(text.encode("utf-8"))


def _validate_receipted_audio_without_provider(
    config: ProjectConfig,
    record: Mapping[str, Any],
    revision: revision_apply_application.RevisionApplyPlan,
) -> None:
    audio_authority = _authority_section(record, "audio")
    current = records_revision(revision.deck_path)
    if current.text is None or _sha(current.text.encode("utf-8")) != audio_authority.get(
        "projected_deck_sha256"
    ):
        raise RevisionFinishError(
            "The audio-complete deck no longer matches its confirmed final bytes."
        )
    receipt = record.get("audio_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("deck_sha256") != audio_authority.get(
        "projected_deck_sha256"
    ):
        raise RevisionFinishError("Audio-complete finish lacks its exact deck receipt.")
    raw_receipts = receipt.get("clips")
    raw_clips = audio_authority.get("clips")
    if not isinstance(raw_receipts, list) or not isinstance(raw_clips, list):
        raise RevisionFinishError("Audio-complete finish has malformed clip receipts.")
    hashes = {
        str(item.get("target")): str(item.get("sha256"))
        for item in raw_receipts
        if isinstance(item, Mapping)
    }
    if len(hashes) != len(raw_clips):
        raise RevisionFinishError("Audio-complete finish lost a clip receipt.")
    for raw_clip in raw_clips:
        if not isinstance(raw_clip, Mapping):
            raise RevisionFinishError("Audio-complete finish has malformed authority.")
        target = str(raw_clip.get("target") or "")
        try:
            actual = _sha(read_bytes_bound(_media_path(config, target)))
        except (DataError, OSError) as exc:
            raise RevisionFinishError(
                f"Could not verify completed audio {target!r}: {exc}"
            ) from exc
        if actual != hashes.get(target):
            raise RevisionFinishError("Audio-complete media no longer matches its finish receipt.")


def _execute_receipted_build_locked(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    revision_plan: revision_apply_application.RevisionApplyPlan,
    *,
    progress: RevisionFinishProgress | None,
) -> RevisionFinishResult:
    _validate_receipted_audio_without_provider(config, record, revision_plan)
    try:
        fresh_build = deck_build_application.plan_conjugation_deck_build(
            config,
            revision_plan.deck_path,
        )
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise RevisionFinishError(str(exc)) from exc
    if not _same_build_authority(config, record, fresh_build):
        raise RevisionFinishError("The package build changed after the finish plan was confirmed.")
    record, revision = _bind_executable_build(
        config,
        path,
        record,
        revision,
        fresh_build,
    )
    if record["state"] == "audio_complete":
        _emit_progress(progress, "Building Anki package")
        built = deck_build_application.execute_conjugation_deck_build_locked(
            config,
            fresh_build,
        )
        _emit_progress(progress, "Saving finish receipt")
        record, revision = _advance_record(
            path,
            record,
            revision,
            from_state="audio_complete",
            to_state="complete",
            receipt_key="build_receipt",
            receipt=_build_receipt(config, built, fresh_build),
        )
    else:
        build_receipt = record.get("build_receipt")
        assert isinstance(build_receipt, Mapping)  # strict record state contract
        output = _authority_path(
            config,
            build_receipt.get("output_path"),
            label="build output",
        )
        rebuild = False
        try:
            rebuild = _sha(read_bytes_bound(output)) != build_receipt.get("package_sha256")
        except FileNotFoundError:
            rebuild = True
        except (DataError, OSError) as exc:
            raise RevisionFinishError(
                f"Could not safely inspect the completed package: {exc}"
            ) from exc
        if rebuild:
            _emit_progress(progress, "Building Anki package")
            built = deck_build_application.execute_conjugation_deck_build_locked(
                config,
                fresh_build,
            )
            _emit_progress(progress, "Saving finish receipt")
            record, revision = _replace_complete_build_receipt(
                path,
                record,
                revision,
                _build_receipt(config, built, fresh_build),
            )
    return _result_from_record(config, record, path=path)


def _execute_record_locked(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    chosen_provider: str | None,
    word_provider: Any | None,
    sentence_provider: Any | None,
    progress: RevisionFinishProgress | None,
) -> RevisionFinishResult:
    revision_plan = _revision_plan_for_record(config, record)
    if str(record.get("state")) in {"audio_complete", "complete"}:
        return _execute_receipted_build_locked(
            config,
            path,
            record,
            revision,
            revision_plan,
            progress=progress,
        )
    planned_audio, _final_text, _projected_build = _prevalidate_record_execution(
        config,
        record,
        revision_plan,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    record, revision = _reconcile_proven_unsent_reservations(
        config,
        path,
        record,
        revision,
        planned_audio,
        sentence_provider=sentence_provider,
    )
    _validate_paid_reservations(
        config,
        record,
        planned_audio,
        sentence_provider=sentence_provider,
    )
    audio_application.preflight_paid_deck_audio_plan(
        planned_audio,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    if record["state"] == "authorized":
        _emit_progress(progress, "Applying reviewed revision")
        applied = revision_apply_application.execute_revision_apply_locked(
            config,
            revision_plan,
        )
        record, revision = _advance_record(
            path,
            record,
            revision,
            from_state="authorized",
            to_state="revision_applied",
            receipt_key="apply_receipt",
            receipt=_apply_receipt(config, applied),
        )

    deck = revision_plan.deck_path.resolve()
    audio_authority = _authority_section(record, "audio")
    authorized_record_ids = tuple(audio_authority["record_ids"])
    fresh_audio = audio_application.plan_deck_audio(
        config,
        deck,
        record_ids=authorized_record_ids,
        words=False,
        examples=True,
        force=False,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    _validate_audio_evolution(config, record, fresh_audio)
    _validate_paid_reservations(
        config,
        record,
        fresh_audio,
        sentence_provider=sentence_provider,
    )
    _validate_provider_journal_binding(
        config,
        fresh_audio,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    audio_application.preflight_paid_deck_audio_plan(
        fresh_audio,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    if _STATE_INDEX[str(record["state"])] < _STATE_INDEX["audio_complete"]:
        _emit_progress(progress, "Creating example audio")
        durable = {"record": record, "revision": revision}

        def reserve_paid(dispatch: audio_application.PaidAudioDispatch) -> None:
            updated, updated_revision = _reserve_paid_clip(
                config,
                path,
                durable["record"],
                durable["revision"],
                dispatch,
                sentence_provider=sentence_provider,
            )
            durable["record"] = updated
            durable["revision"] = updated_revision

        audio_result = audio_application.execute_deck_audio_locked(
            config,
            deck,
            record_ids=authorized_record_ids,
            words=False,
            examples=True,
            expected_fingerprint=fresh_audio.fingerprint,
            force=False,
            prune=False,
            chosen_provider=chosen_provider,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            before_paid_dispatch=reserve_paid,
        )
        record = durable["record"]
        revision = durable["revision"]
        if not audio_result.succeeded:
            return _result_from_record(
                config,
                record,
                path=path,
                audio=audio_result,
            )
        audio_authority = _authority_section(record, "audio")
        receipt = _audio_receipt(config, fresh_audio)
        if receipt["deck_sha256"] != audio_authority.get("projected_deck_sha256"):
            raise RevisionFinishError(
                "Example audio produced deck bytes outside the confirmed finish plan."
            )
        record, revision = _advance_record(
            path,
            record,
            revision,
            from_state="revision_applied",
            to_state="audio_complete",
            receipt_key="audio_receipt",
            receipt=receipt,
        )

    fresh_build = deck_build_application.plan_conjugation_deck_build(config, deck)
    if not _same_build_authority(config, record, fresh_build):
        raise RevisionFinishError("The package build changed after the finish plan was confirmed.")
    record, revision = _bind_executable_build(
        config,
        path,
        record,
        revision,
        fresh_build,
    )
    if record["state"] == "audio_complete":
        _emit_progress(progress, "Building Anki package")
        built = deck_build_application.execute_conjugation_deck_build_locked(
            config,
            fresh_build,
        )
        _emit_progress(progress, "Saving finish receipt")
        record, revision = _advance_record(
            path,
            record,
            revision,
            from_state="audio_complete",
            to_state="complete",
            receipt_key="build_receipt",
            receipt=_build_receipt(config, built, fresh_build),
        )

    if record["state"] == "complete":
        build_receipt = record.get("build_receipt")
        if not isinstance(build_receipt, Mapping):
            raise RevisionFinishError("Completed finish lacks its package receipt.")
        output = _authority_path(
            config,
            build_receipt.get("output_path"),
            label="build output",
        )
        if _sha(read_bytes_bound(output)) != build_receipt.get("package_sha256"):
            raise RevisionFinishError(
                "The completed Anki package no longer matches its finish receipt."
            )
    return _result_from_record(config, record, path=path)


def execute_revision_finish(
    config: ProjectConfig,
    expected: RevisionFinishPlan,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
    progress: RevisionFinishProgress | None = None,
) -> RevisionFinishResult:
    """Persist and execute one exact owner-confirmed finish plan."""

    _emit_progress(progress, "Preparing finish")
    if expected.repository_root != config.root.resolve():
        raise RevisionFinishError("Revision finish plan belongs to another repository.")
    words, sentences = _resolved_providers(
        config,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    operation_lock = config.root / ".janki-audio-operation"
    with exclusive_path_lock(operation_lock):
        fresh = _plan_revision_finish_locked(
            config,
            expected.revision.staging_path,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
        )
        if fresh.fingerprint != expected.fingerprint or dict(fresh.authority) != dict(
            expected.authority
        ):
            raise RevisionFinishError("The apply-and-finish plan changed after it was displayed.")
        # Provider availability is a local/no-contact fact. Check it before
        # accepting or applying the reviewed proposal where the audio service
        # can do so without adopting recovery state.
        audio_application.preflight_paid_deck_audio_plan(
            fresh.audio,
            word_provider=words,
            sentence_provider=sentences,
        )
        prepare_bound_directory(fresh.finish_directory)
        existing = _active_finish_for_revision(
            config,
            operation_id=fresh.revision.operation_id,
            staging_path=_relative(config, fresh.revision.staging_path, "staging path"),
        )
        if existing is not None and existing != fresh.fingerprint:
            raise RevisionFinishError(
                "This revision already has a different durable finish authority; "
                f"resume {existing} instead of widening it."
            )
        with exclusive_path_lock(fresh.record_path):
            existing_record = _read_record_optional(fresh.record_path)
            if existing_record is None:
                record = _new_record(fresh)
                text = _record_text(record)
                _strict_record(text.encode("utf-8"), fresh.record_path)
                try:
                    atomic_write_text_bound(
                        fresh.record_path,
                        text,
                        expected_absent=True,
                    )
                except (DataError, OSError) as write_error:
                    raise RevisionFinishError(
                        f"Could not record finish authority before applying: {write_error}"
                    ) from write_error
                record_revision = _sha(text.encode("utf-8"))
            else:
                record, record_revision = existing_record
            if record.get("receipt_id") != fresh.fingerprint or record.get("authority") != dict(
                fresh.authority
            ):
                raise RevisionFinishError(
                    "Existing finish record does not match the confirmed authority."
                )
        return _execute_record_locked(
            config,
            fresh.record_path,
            record,
            record_revision,
            chosen_provider=chosen_provider,
            word_provider=words,
            sentence_provider=sentences,
            progress=progress,
        )


def resume_revision_finish(
    config: ProjectConfig,
    receipt_id: str,
    *,
    chosen_provider: str | None = None,
    word_provider: Any | None = None,
    sentence_provider: Any | None = None,
    progress: RevisionFinishProgress | None = None,
) -> RevisionFinishResult:
    """Resume only the durable authority already recorded for ``receipt_id``."""

    _emit_progress(progress, "Preparing finish")
    path = _record_path(config, receipt_id)
    operation_lock = config.root / ".janki-audio-operation"
    with exclusive_path_lock(operation_lock):
        record, revision = _read_record(path)
        if str(record.get("state")) in {"audio_complete", "complete"}:
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


def inspect_revision_finish(
    config: ProjectConfig,
    receipt_id: str,
) -> RevisionFinishResult:
    """Read one durable finish state without replanning or provider contact."""

    path = _record_path(config, receipt_id)
    record, _revision = _read_record(path)
    return _result_from_record(config, record, path=path)
