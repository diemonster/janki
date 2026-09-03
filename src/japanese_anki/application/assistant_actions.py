"""Resolve closed Assistant intents into existing Janki application services.

The Assistant provider supplies opaque resource ids, never paths.  This broker
freshly resolves one configured deck through :mod:`assistant_context`, asks the
existing audio or build service for its complete display plan, and wraps that
plan in a small canonical projection suitable for a confirmation card.

Execution is deliberately a second operation.  It resolves the opaque target
again, re-plans, compares the exact projection, and only then calls the same
plan-bound executor used by the CLI/workbench.  This module owns no Japanese
and no writer of its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from japanese_anki import audio_cmd, ledger, status
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_package as deck_package_application
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    DeckContext,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.io import (
    exclusive_path_lock,
    load_records,
    read_bytes_bound_snapshot,
    records_revision,
)
from japanese_anki.tts import SpeechProvider

__all__ = [
    "AssistantActionError",
    "AssistantActionExecution",
    "AssistantActionIntentLike",
    "AssistantActionPlan",
    "AudioActionChoices",
    "execute_action",
    "plan_action",
]


ActionKind = Literal["generate_audio", "build_deck"]
ActionServicePlan: TypeAlias = (
    audio_application.AudioPlan | deck_package_application.DeckPackagePlan
)
ActionServiceResult: TypeAlias = (
    audio_application.AudioExecutionOutcome | deck_package_application.DeckPackageResult
)


class AssistantActionError(JankiError):
    """A typed Assistant intent cannot be planned or executed exactly."""


class AssistantActionIntentLike(Protocol):
    """The closed portion of an ordinary Assistant answer used by this broker."""

    kind: str
    resource_ids: Sequence[str]
    record_ids: Sequence[str]
    instruction: str

    @property
    def options(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AudioActionChoices:
    """Exact safe defaults or explicit owner choices for one audio action."""

    words: bool
    examples: bool
    force: bool
    prune: bool
    clip_classes_source: Literal["deck-default", "owner-explicit"]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.words, self.examples, self.force, self.prune)
        ):
            raise ValueError("Assistant audio choices must be boolean")
        if not self.words and not self.examples:
            raise ValueError("Assistant audio choices need at least one clip class")

    def intent_options(self) -> Mapping[str, bool]:
        """Recreate only the closed options required for an exact re-plan."""

        values: dict[str, bool] = {}
        if self.clip_classes_source == "owner-explicit":
            values.update(
                audio_words=self.words,
                audio_examples=self.examples,
            )
        if self.force:
            values["audio_force"] = True
        if self.prune:
            values["audio_prune"] = True
        return values


@dataclass(frozen=True, slots=True)
class AssistantActionPlan:
    """One local service plan plus its exact browser-safe projection."""

    repository_root: Path
    kind: ActionKind
    resource_id: str
    instruction: str
    deck_path: Path
    deck_name: str
    deck_kind: str
    requested_record_ids: tuple[str, ...]
    service_fingerprint: str
    projection_wire: str
    fingerprint: str
    service_plan: ActionServicePlan
    audio_choices: AudioActionChoices | None = None

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute() or not self.deck_path.is_absolute():
            raise ValueError("Assistant action paths must be absolute")
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Assistant action repository root must be canonical")
        if self.deck_path != _lexical_path(self.deck_path):
            raise ValueError("Assistant action deck path must be lexical absolute")
        try:
            self.deck_path.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("Assistant action deck must stay in its repository") from exc
        for label, value in (
            ("resource id", self.resource_id),
            ("instruction", self.instruction),
            ("deck name", self.deck_name),
            ("deck kind", self.deck_kind),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Assistant action {label} must be nonblank")
        if len(set(self.requested_record_ids)) != len(self.requested_record_ids):
            raise ValueError("Assistant action record ids must be unique")
        if any(not item for item in self.requested_record_ids):
            raise ValueError("Assistant action record ids must be nonblank")
        if not _is_sha256(self.service_fingerprint):
            raise ValueError("Assistant action service fingerprint must be SHA-256")
        if getattr(self.service_plan, "fingerprint", None) != self.service_fingerprint:
            raise ValueError("Assistant action service plan does not match its fingerprint")
        if (
            isinstance(self.service_plan, deck_package_application.DeckPackagePlan)
            and self.service_plan.deck_path != self.deck_path
        ):
            raise ValueError(
                "Assistant action and build service must target the same configured deck"
            )
        try:
            parsed = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Assistant action projection must be JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("Assistant action projection must be a JSON object")
        if _canonical_json(parsed) != self.projection_wire:
            raise ValueError("Assistant action projection must use canonical JSON")
        if _sha256(self.projection_wire.encode("utf-8")) != self.fingerprint:
            raise ValueError("Assistant action fingerprint does not bind its projection")
        if self.kind == "generate_audio":
            if not isinstance(self.service_plan, audio_application.AudioPlan) or not isinstance(
                self.audio_choices, AudioActionChoices
            ):
                raise ValueError("Assistant audio action needs its exact audio choices")
            if (
                self.service_plan.words != self.audio_choices.words
                or self.service_plan.examples != self.audio_choices.examples
                or self.service_plan.force != self.audio_choices.force
            ):
                raise ValueError("Assistant audio choices do not match its service plan")
            expected_options = {
                "words": self.audio_choices.words,
                "examples": self.audio_choices.examples,
                "force": self.audio_choices.force,
                "prune": self.audio_choices.prune,
                "clip_classes_source": self.audio_choices.clip_classes_source,
            }
            if parsed.get("options") != expected_options:
                raise ValueError("Assistant audio projection does not match its choices")
        elif self.audio_choices is not None:
            raise ValueError("A non-audio Assistant action cannot carry audio choices")

    @property
    def projection(self) -> Mapping[str, Any]:
        """The exact parsed value a browser confirmation card may render."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class AssistantActionExecution:
    """The freshly re-planned action and the existing service's result."""

    plan: AssistantActionPlan
    result: ActionServiceResult


@dataclass(frozen=True, slots=True)
class _ResolvedDeck:
    path: Path
    context: DeckContext


@dataclass(frozen=True, slots=True)
class _ReplayIntent:
    kind: str
    resource_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    instruction: str
    options: Mapping[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical_path(path: Path | str) -> Path:
    """Return an absolute normalized name without following symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


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
        raise AssistantActionError(f"Assistant action plan cannot be fingerprinted: {exc}") from exc


def _exact_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise AssistantActionError(f"Assistant action {label} must be a list of text.")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise AssistantActionError(f"Every Assistant action {label[:-1]} must be nonblank text.")
    repeated = [item for item, count in Counter(items).items() if count > 1]
    if repeated:
        raise AssistantActionError(
            f"Assistant action {label[:-1]} {repeated[0]!r} was supplied twice."
        )
    return items


def _intent_parts(
    intent: AssistantActionIntentLike,
) -> tuple[ActionKind, str, tuple[str, ...], str, Mapping[str, Any]]:
    kind = getattr(intent, "kind", None)
    if kind not in {"generate_audio", "build_deck"}:
        raise AssistantActionError(
            f"Assistant action kind {kind!r} is not handled by the audio/build broker."
        )
    resources = _exact_sequence(getattr(intent, "resource_ids", None), label="resource ids")
    if len(resources) != 1:
        raise AssistantActionError(
            "Assistant audio/build actions need exactly one configured deck resource."
        )
    record_ids = _exact_sequence(getattr(intent, "record_ids", None), label="record ids")
    instruction = getattr(intent, "instruction", None)
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantActionError(
            "Assistant audio/build actions need the owner's nonblank instruction."
        )
    raw_options = getattr(intent, "options", {})
    if not isinstance(raw_options, Mapping):
        raise AssistantActionError("Assistant audio/build options must be one object.")
    options = dict(raw_options)
    if any(not isinstance(key, str) for key in options):
        raise AssistantActionError("Every Assistant audio/build option name must be text.")
    return kind, resources[0], record_ids, instruction, options


_AUDIO_OPTION_NAMES = frozenset(
    {"audio_words", "audio_examples", "audio_force", "audio_prune"}
)


def _strict_audio_bool(options: Mapping[str, Any], name: str) -> bool:
    value = options.get(name, False)
    if not isinstance(value, bool):
        raise AssistantActionError(
            f"Assistant audio option {name} must be true or false."
        )
    return value


def _audio_choices(
    deck_kind: str,
    options: Mapping[str, Any],
) -> AudioActionChoices:
    unknown = sorted(set(options) - _AUDIO_OPTION_NAMES)
    if unknown:
        raise AssistantActionError(
            f"Assistant generate_audio has unsupported option {unknown[0]!r}."
        )
    has_words = "audio_words" in options
    has_examples = "audio_examples" in options
    if has_words != has_examples:
        raise AssistantActionError(
            "Assistant audio clip selection needs both audio_words and "
            "audio_examples, or neither to use the displayed deck default."
        )
    if has_words:
        words = _strict_audio_bool(options, "audio_words")
        examples = _strict_audio_bool(options, "audio_examples")
        source: Literal["deck-default", "owner-explicit"] = "owner-explicit"
    else:
        words = deck_kind != "conjugation"
        examples = True
        source = "deck-default"
    if not words and not examples:
        raise AssistantActionError(
            "Assistant generate_audio needs at least one clip class."
        )
    force = _strict_audio_bool(options, "audio_force")
    prune = _strict_audio_bool(options, "audio_prune")
    if deck_kind == "conjugation" and (words or not examples):
        raise AssistantActionError(
            "Deck-authored conjugation audio supports examples only."
        )
    if deck_kind == "conjugation" and prune:
        raise AssistantActionError(
            "Deck-authored conjugation audio cannot perform the repository-wide prune."
        )
    return AudioActionChoices(
        words=words,
        examples=examples,
        force=force,
        prune=prune,
        clip_classes_source=source,
    )


def _require_safe_audio_directory(config: ProjectConfig) -> None:
    """Refuse an existing audio child that is not one lexical directory."""

    audio_dir = config.media_dir.resolve() / audio_cmd.AUDIO_SUBDIR
    try:
        details = os.lstat(audio_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AssistantActionError(
            f"Could not inspect the Assistant audio directory: {exc.strerror or exc}"
        ) from exc
    if stat.S_ISLNK(details.st_mode):
        raise AssistantActionError(
            "The Assistant audio directory must not be a symbolic link."
        )
    if not stat.S_ISDIR(details.st_mode):
        raise AssistantActionError(
            "The Assistant audio directory exists but is not a directory."
        )


def _resolve_deck(config: ProjectConfig, resource_id: str) -> _ResolvedDeck:
    """Resolve only an opaque id minted for the current configured deck census."""

    try:
        broker = AssistantContextBroker(config)
        matches = [
            _lexical_path(path)
            for path in status.deck_files(config)
            if broker.resource_id_for_deck(path) == resource_id
        ]
        if not matches:
            raise AssistantActionError(
                f"Unknown configured deck resource {resource_id!r}; choose it from "
                "a fresh Janki deck list."
            )
        if len(matches) != 1:
            raise AssistantActionError(
                f"Configured deck resource {resource_id!r} is ambiguous; refresh the "
                "project before planning an action."
            )
        context = broker.deck_context(resource_id)
    except AssistantActionError:
        raise
    except AssistantContextError as exc:
        raise AssistantActionError(
            f"Could not resolve configured deck resource {resource_id!r}: {exc}"
        ) from exc
    return _ResolvedDeck(path=matches[0], context=context)


def _relative_path(config: ProjectConfig, path: Path, *, label: str) -> str:
    target = _lexical_path(path)
    try:
        return target.relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantActionError(f"Assistant {label} escapes the configured repository.") from exc


def _deck_name(deck: _ResolvedDeck) -> str:
    value = deck.context.configuration.get("name")
    return str(value).strip() if str(value or "").strip() else deck.path.stem


def _selected_ids(deck: _ResolvedDeck, requested: tuple[str, ...]) -> tuple[str, ...]:
    available = tuple(record.id for record in deck.context.records)
    if not available:
        raise AssistantActionError(
            f"Deck {_deck_name(deck)!r} has no card records that can own audio."
        )
    if not requested:
        return available
    allowed = set(available)
    missing = [record_id for record_id in requested if record_id not in allowed]
    if missing:
        raise AssistantActionError(
            f"Record {missing[0]!r} is not a card in deck {_deck_name(deck)!r}."
        )
    return requested


def _drill_owner_ids(deck: _ResolvedDeck, requested: tuple[str, ...]) -> tuple[str, ...] | None:
    if not requested:
        return None
    selected = _selected_ids(deck, requested)
    revision = records_revision(deck.path)
    if revision.text is None:
        raise AssistantActionError("The selected conjugation deck no longer exists.")
    section = pattern_cards.conjugation_deck_section(deck.path, revision.text)
    deck_id = section.get("deck_id")
    if isinstance(deck_id, bool) or not isinstance(deck_id, int):
        raise AssistantActionError("The selected conjugation deck has no valid integer deck_id.")
    form = str(section.get("form") or "te_form").strip()
    return tuple(
        pattern_cards.drill_audio_owner_id(deck_id, form, record_id) for record_id in selected
    )


def _assert_canonical_audio_scope(
    config: ProjectConfig,
    deck: _ResolvedDeck,
    selected: tuple[str, ...],
) -> None:
    """Refuse a deck version the existing canonical audio writer cannot update."""

    deck_config, resolved = resolve_deck_records(deck.path)
    source = deck_config.get("source")
    source_path = (deck.path.parent / str(source)).resolve() if source else None
    if source_path != config.normalized_file.resolve():
        raise AssistantActionError(
            f"Deck {_deck_name(deck)!r} does not read the canonical vocabulary "
            "store. Its audio needs a deck-owned transaction that does not exist yet."
        )
    by_id = {record.id: record for record in resolved}
    canonical_records = load_records(config.normalized_file)
    counts = Counter(record.id for record in canonical_records)
    canonical = {record.id: record for record in canonical_records}
    for record_id in selected:
        if counts[record_id] != 1 or by_id.get(record_id) != canonical.get(record_id):
            raise AssistantActionError(
                f"Card {record_id!r} has a deck-specific version. The canonical "
                "audio transaction cannot safely write that override."
            )


def _counts(counts: audio_application.AudioClipCounts) -> dict[str, int]:
    return {
        "total": counts.total,
        "current": counts.current,
        "recoverable": counts.recoverable,
        "provider_required": counts.provider_required,
    }


def _provider(
    clip_kind: str,
    provider: audio_application.AudioProviderPlan | None,
    counts: audio_application.AudioClipCounts,
) -> dict[str, object] | None:
    if provider is None:
        return None
    return {
        "clip_kind": clip_kind,
        "name": provider.name,
        "access": provider.access,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": dict(provider.settings),
        "provider_required": counts.provider_required,
    }


def _audio_reference_names(records: Sequence[Any]) -> set[str]:
    referenced: set[str] = set()
    for record in records:
        if record.audio:
            referenced.add(Path(record.audio).name.casefold())
        for example in record.examples:
            if example.audio:
                referenced.add(Path(example.audio).name.casefold())
    return referenced


def _entry_projection(
    config: ProjectConfig,
    path: Path,
) -> dict[str, object] | None:
    """Bind one existing removal/replacement target without following symlinks."""

    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AssistantActionError(
            f"Could not inspect planned audio target {path.name!r}: {exc.strerror or exc}"
        ) from exc
    identity = [
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    ]
    digest: str | None = None
    if stat.S_ISREG(details.st_mode):
        try:
            state, digest, _payload = read_bytes_bound_snapshot(path)
        except JankiError as exc:
            raise AssistantActionError(
                f"Could not bind planned audio target {path.name!r}: {exc}"
            ) from exc
        identity = list(state)
        entry_type = "regular-file"
    elif stat.S_ISLNK(details.st_mode):
        entry_type = "symlink"
    elif stat.S_ISDIR(details.st_mode):
        entry_type = "directory"
    else:
        entry_type = "other"
    return {
        "file": _relative_path(config, path, label="audio file"),
        "entry_type": entry_type,
        "identity": identity,
        "bytes": identity[2],
        "sha256": digest,
    }


def _ledger_audio_files(book: ledger.Ledger) -> set[str]:
    return {
        str(item.get("file") or "")
        for record_entry in book.records.values()
        if isinstance(record_entry, dict)
        for item in (
            record_entry.get("audio")
            if isinstance(record_entry.get("audio"), list)
            else []
        )
        if isinstance(item, dict) and str(item.get("file") or "")
    }


def _audio_cleanup_projection(
    config: ProjectConfig,
    service: audio_application.AudioPlan,
    *,
    prune: bool,
) -> dict[str, object]:
    if not prune:
        return {
            "enabled": False,
            "scope": "none",
            "media_files_removed": [],
            "ledger_audio_files_forgotten": [],
            "missing_record_pending_audio_removed": [],
        }

    book = ledger.load(service.ledger_path)
    present_ids = {record.id for record in service.protected_records}
    missing_pending: list[dict[str, object]] = []
    retained_pending_targets: set[str] = set()
    for key in sorted(book.pending_audio):
        entry = book.pending_audio_entry(key)
        if entry is None:  # pragma: no cover - validated ledger owns this invariant
            continue
        if str(entry.get("record_id") or "") not in present_ids:
            missing_pending.append(
                {
                    "key": key,
                    "record_id": str(entry.get("record_id") or ""),
                    "target": str(entry.get("target") or ""),
                    "staged_file": str(entry.get("staged_file") or ""),
                }
            )
        else:
            retained_pending_targets.add(str(entry.get("target") or "").casefold())

    referenced = _audio_reference_names(service.protected_records)
    referenced.update(retained_pending_targets)
    # A selected missing/recoverable target becomes durable before ordinary
    # pruning. It must not be described as an unrelated removal merely because
    # its current record has not acquired the reference yet.
    referenced.update(clip.target.casefold() for clip in service.clips)
    audio_dir = service.media_dir / audio_cmd.AUDIO_SUBDIR
    candidates: list[dict[str, object]] = []
    candidate_names: set[str] = set()
    if audio_dir.is_dir():
        for path in sorted(audio_dir.glob("janki-*")):
            if path.name.casefold() in referenced:
                continue
            projected = _entry_projection(config, path)
            if projected is None:
                raise AssistantActionError(
                    "The audio cleanup candidates changed while the plan was prepared; "
                    "retry the action."
                )
            candidates.append(projected)
            candidate_names.add(path.name)

    forgotten = sorted(
        name
        for name in (_ledger_audio_files(book) | candidate_names)
        if name.casefold() not in referenced
    )
    return {
        "enabled": True,
        "scope": "repository-wide-unreferenced-janki-audio",
        "media_files_removed": candidates,
        "ledger_audio_files_forgotten": forgotten,
        "missing_record_pending_audio_removed": missing_pending,
    }


def _force_replacements_projection(
    config: ProjectConfig,
    service: audio_application.AudioPlan,
) -> list[dict[str, object]]:
    if not service.force:
        return []
    audio_dir = service.media_dir / audio_cmd.AUDIO_SUBDIR
    replacements: list[dict[str, object]] = []
    for clip in service.clips:
        if Path(clip.target).name != clip.target:
            raise AssistantActionError(
                "The existing audio planner returned a non-local clip target."
            )
        existing = _entry_projection(config, audio_dir / clip.target)
        if existing is None:
            continue
        replacements.append(
            {
                "record_id": clip.record_id,
                "kind": clip.kind,
                "target": clip.target,
                "existing": existing,
            }
        )
    return replacements


def _audio_projection(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
    deck: _ResolvedDeck,
    requested_record_ids: tuple[str, ...],
    service: audio_application.AudioPlan,
    choices: AudioActionChoices,
) -> dict[str, object]:
    providers = [
        item
        for item in (
            _provider("word", service.word_provider, service.word_counts),
            _provider("example", service.example_provider, service.example_counts),
        )
        if item is not None
    ]
    return {
        "schema_version": 1,
        "kind": "generate_audio",
        "instruction": instruction,
        "target": {
            "resource_id": resource_id,
            "deck": _deck_name(deck),
            "deck_kind": deck.context.deck_kind or "vocabulary",
            "configured_file": _relative_path(config, deck.path, label="deck target"),
            "requested_record_ids": list(requested_record_ids),
            "audio_owner_ids": list(service.record_ids),
        },
        "options": {
            "words": choices.words,
            "examples": choices.examples,
            "force": choices.force,
            "prune": choices.prune,
            "clip_classes_source": choices.clip_classes_source,
        },
        "counts": {
            "words": _counts(service.word_counts),
            "examples": _counts(service.example_counts),
        },
        "providers": providers,
        "clips": [
            {
                "record_id": clip.record_id,
                "kind": clip.kind,
                "target": clip.target,
                "request_input": clip.request_input,
                "content_fingerprint": clip.content_fingerprint,
                "state": clip.state,
                "provider": clip.provider.name,
                "billing_class": clip.provider.access,
                "recovery_sha256": clip.recovery_sha256,
            }
            for clip in service.clips
        ],
        "recovery": {
            "saved_clips_adopted_without_dispatch": (
                service.word_counts.recoverable + service.example_counts.recoverable
            ),
            "paid_calls_are_journaled_by_audio_service": True,
        },
        "force_replacements": _force_replacements_projection(config, service),
        "cleanup": _audio_cleanup_projection(
            config,
            service,
            prune=choices.prune,
        ),
        "writes": {
            "media_directory": _relative_path(
                config,
                service.media_dir / "audio",
                label="audio output",
            ),
            "ledger": _relative_path(config, service.ledger_path, label="audio ledger"),
        },
        "service_fingerprint": service.fingerprint,
    }


def _build_projection(
    config: ProjectConfig,
    resource_id: str,
    instruction: str,
    deck: _ResolvedDeck,
    service: deck_package_application.DeckPackagePlan,
) -> dict[str, object]:
    sources = [
        {
            "label": item.label,
            "file": _relative_path(config, item.path, label="build source"),
            "sha256": item.sha256,
        }
        for item in service.source_inputs
    ]
    primary_source = next(
        (item for item in sources if isinstance(item.get("sha256"), str)),
        {
            "label": "configured deck",
            "file": _relative_path(config, service.deck_path, label="deck source"),
            "sha256": service.deck_input.sha256,
        },
    )
    return {
        "schema_version": 1,
        "kind": "build_deck",
        "instruction": instruction,
        "target": {
            "resource_id": resource_id,
            "deck": service.deck_name,
            "deck_kind": service.kind,
            "configured_file": _relative_path(config, deck.path, label="deck target"),
            "variant": service.variant,
            "note_count": service.note_count,
            "card_count": service.card_count,
            "card_types": list(service.card_types),
        },
        "inputs": {
            "deck": {
                "file": _relative_path(config, service.deck_path, label="deck input"),
                "sha256": service.deck_input.sha256,
            },
            "source": primary_source,
            "sources": sources,
            "templates": [
                {
                    "label": item.label,
                    "file": _relative_path(config, item.path, label="build template"),
                    "sha256": item.sha256,
                }
                for item in service.template_inputs
            ],
            "media": [
                {
                    "label": item.label,
                    "file": _relative_path(config, item.path, label="build media"),
                    "sha256": item.sha256,
                }
                for item in service.media_inputs
            ],
        },
        "provider": None,
        "billing_class": "local",
        "writes": {
            "package": _relative_path(config, service.output_path, label="deck package"),
            "package_precondition": {
                "state": (
                    "absent" if service.output_revision is None else "replace_exact"
                ),
                "sha256": service.output_revision,
                "identity": (
                    None
                    if service.output_identity is None
                    else list(service.output_identity)
                ),
            },
            **(
                {
                    "export_history": _relative_path(
                        config, config.ledger_file, label="export history"
                    )
                }
                if service.kind == "vocabulary"
                else {}
            ),
        },
        "service_fingerprint": service.fingerprint,
    }


def _wrap_plan(
    config: ProjectConfig,
    *,
    kind: ActionKind,
    resource_id: str,
    instruction: str,
    deck: _ResolvedDeck,
    requested_record_ids: tuple[str, ...],
    service_plan: ActionServicePlan,
    projection: Mapping[str, object],
    audio_choices: AudioActionChoices | None = None,
) -> AssistantActionPlan:
    wire = _canonical_json(projection)
    return AssistantActionPlan(
        repository_root=config.root.resolve(),
        kind=kind,
        resource_id=resource_id,
        instruction=instruction,
        deck_path=deck.path,
        deck_name=_deck_name(deck),
        deck_kind=deck.context.deck_kind or "vocabulary",
        requested_record_ids=requested_record_ids,
        service_fingerprint=service_plan.fingerprint,
        projection_wire=wire,
        fingerprint=_sha256(wire.encode("utf-8")),
        service_plan=service_plan,
        audio_choices=audio_choices,
    )


def plan_action(
    config: ProjectConfig,
    intent: AssistantActionIntentLike,
    *,
    word_provider: SpeechProvider | None = None,
    sentence_provider: audio_application.SentenceProvider | None = None,
) -> AssistantActionPlan:
    """Resolve one closed intent and produce a display-only exact plan."""

    kind, resource_id, requested_record_ids, instruction, options = _intent_parts(intent)
    deck = _resolve_deck(config, resource_id)
    deck_kind = deck.context.deck_kind or "vocabulary"

    try:
        if kind == "generate_audio":
            if deck_kind == "pattern":
                raise AssistantActionError(
                    f"Pattern deck {_deck_name(deck)!r} has no word or sentence "
                    "audio transaction to generate."
                )
            choices = _audio_choices(deck_kind, options)
            _require_safe_audio_directory(config)
            if deck_kind == "conjugation":
                service = audio_application.plan_deck_audio(
                    config,
                    deck.path,
                    record_ids=_drill_owner_ids(deck, requested_record_ids),
                    words=choices.words,
                    examples=choices.examples,
                    force=choices.force,
                    word_provider=word_provider,
                    sentence_provider=sentence_provider,
                )
            else:
                selected = _selected_ids(deck, requested_record_ids)
                _assert_canonical_audio_scope(config, deck, selected)
                service = audio_application.plan_targeted_audio(
                    config,
                    selected,
                    words=choices.words,
                    examples=choices.examples,
                    force=choices.force,
                    word_provider=word_provider,
                    sentence_provider=sentence_provider,
                )
            projection = _audio_projection(
                config,
                resource_id,
                instruction,
                deck,
                requested_record_ids,
                service,
                choices,
            )
            return _wrap_plan(
                config,
                kind=kind,
                resource_id=resource_id,
                instruction=instruction,
                deck=deck,
                requested_record_ids=requested_record_ids,
                service_plan=service,
                projection=projection,
                audio_choices=choices,
            )

        if options:
            raise AssistantActionError(
                "Assistant build_deck does not accept audio or other action options."
            )
        if requested_record_ids:
            raise AssistantActionError(
                "A deck build always packages the complete configured deck; "
                "record ids cannot narrow it."
            )
        service = deck_package_application.plan_deck_package(
            config,
            deck.path,
            record_ids=requested_record_ids,
        )
        projection = _build_projection(config, resource_id, instruction, deck, service)
        return _wrap_plan(
            config,
            kind=kind,
            resource_id=resource_id,
            instruction=instruction,
            deck=deck,
            requested_record_ids=requested_record_ids,
            service_plan=service,
            projection=projection,
        )
    except AssistantActionError:
        raise
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantActionError(
            f"Could not plan {kind.replace('_', ' ')} for deck {_deck_name(deck)!r}: {exc}"
        ) from exc


def _assert_fresh(expected: AssistantActionPlan, fresh: AssistantActionPlan) -> None:
    if (
        fresh.fingerprint != expected.fingerprint
        or fresh.projection_wire != expected.projection_wire
    ):
        raise AssistantActionError(
            "The Assistant action plan changed after it was displayed; reload and "
            "review the fresh plan before confirming it."
        )


def execute_action(
    config: ProjectConfig,
    expected: AssistantActionPlan,
    *,
    progress: Callable[[str], None] | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: audio_application.SentenceProvider | None = None,
) -> AssistantActionExecution:
    """Freshly re-plan one confirmed action, then call its existing executor."""

    if expected.repository_root != config.root.resolve():
        raise AssistantActionError("The Assistant action plan belongs to another repository.")

    choices = expected.audio_choices
    replay = _ReplayIntent(
        kind=expected.kind,
        resource_ids=(expected.resource_id,),
        record_ids=expected.requested_record_ids,
        instruction=expected.instruction,
        options={} if choices is None else choices.intent_options(),
    )

    if expected.kind == "generate_audio":
        if not isinstance(choices, AudioActionChoices):
            raise AssistantActionError(
                "The confirmed audio action does not contain its exact choices."
            )
        # Keep the audio-operation lock from the fresh projection through the
        # existing locked writer. No other Janki audio run can change a forced
        # replacement or prune candidate between the click check and execution.
        with exclusive_path_lock(config.root / ".janki-audio-operation"):
            fresh = plan_action(
                config,
                replay,
                word_provider=word_provider,
                sentence_provider=sentence_provider,
            )
            _assert_fresh(expected, fresh)
            service = fresh.service_plan
            if not isinstance(service, audio_application.AudioPlan):
                raise AssistantActionError(
                    "The confirmed audio action does not contain an audio service plan."
                )
            if fresh.deck_kind == "conjugation":
                result = audio_application.execute_deck_audio_locked(
                    config,
                    fresh.deck_path,
                    record_ids=service.record_ids,
                    words=choices.words,
                    examples=choices.examples,
                    expected_fingerprint=service.fingerprint,
                    force=choices.force,
                    prune=choices.prune,
                    progress=progress,
                    word_provider=word_provider,
                    sentence_provider=sentence_provider,
                )
            else:
                result = audio_application.execute_targeted_audio_locked(
                    config,
                    service.record_ids,
                    words=choices.words,
                    examples=choices.examples,
                    expected_fingerprint=service.fingerprint,
                    force=choices.force,
                    prune=choices.prune,
                    progress=progress,
                    word_provider=word_provider,
                    sentence_provider=sentence_provider,
                )
            return AssistantActionExecution(plan=fresh, result=result)

    fresh = plan_action(
        config,
        replay,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    _assert_fresh(expected, fresh)

    service = fresh.service_plan
    if not isinstance(service, deck_package_application.DeckPackagePlan):
        raise AssistantActionError(
            "The confirmed build action does not contain a deck build service plan."
        )
    result = deck_package_application.execute_deck_package(config, service)
    return AssistantActionExecution(plan=fresh, result=result)
