"""Plan, dispatch, recover, and stage the generic existing-card ``revise`` pass.

The ordinary Assistant may select this application operation, but it never
authors card values.  This module sends the exact selected canonical records
to the configured revision provider and turns the captured structured answer
into one unapproved YAML review.  Promotion remains the sole canonical writer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from japanese_anki import ai_schema, claude_client, operations, prompts
from japanese_anki import status as status_module
from japanese_anki.application import card_change_staging, revision_provider
from japanese_anki.application.extraction import classify_dispatch_failure
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import deck_kind, resolve_deck_records
from japanese_anki.io import (
    MERGEABLE_FIELDS,
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records_snapshot,
    prepare_bound_directory,
    read_bytes_bound,
    validate_prefer_incoming,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "CardRevisionError",
    "CardRevisionPlan",
    "CardRevisionRunError",
    "CardRevisionRunResult",
    "CardRevisionTarget",
    "inspect_card_revision_target",
    "plan_card_revision",
    "recover_card_revision",
    "run_card_revision",
]


class CardRevisionError(JankiError):
    """A generic existing-card revision cannot be planned or staged safely."""


class CardRevisionRunError(CardRevisionError):
    """A revision stopped after its paid operation identity was allocated."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        provider_dispatched: bool,
    ) -> None:
        self.operation_id = operation_id
        self.provider_dispatched = provider_dispatched
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CardRevisionPlan:
    """The exact request and repository state one confirmation can consume."""

    repository_root: Path
    operation_id: str
    deck_path: Path
    deck_relative_path: str
    deck_sha256: str
    canonical_path: Path
    canonical_relative_path: str
    canonical_sha256: str
    selected_record_ids: tuple[str, ...]
    current_records: tuple[VocabularyRecord, ...]
    owner_instruction: str
    focus_resource_id: str | None
    provider_plan: revision_provider.RevisionProviderPlan
    style_guide: str
    task_template: str
    user_turn: str
    request_manifest_path: Path
    request_manifest_revision: str | None
    staging_path: Path
    staging_revision: str | None
    plan_fingerprint: str

    @property
    def provider(self) -> str:
        return self.provider_plan.provider

    @property
    def model(self) -> str:
        return self.provider_plan.model

    @property
    def request_fingerprint(self) -> str:
        return self.provider_plan.request_fingerprint

    @property
    def billing_class(self) -> str:
        return self.provider_plan.billing_class

    @property
    def billing_display(self) -> str:
        return self.provider_plan.billing_display

    @property
    def can_dispatch(self) -> bool:
        return self.request_manifest_revision is None and self.staging_revision is None


@dataclass(frozen=True, slots=True)
class CardRevisionRunResult:
    """One captured paid answer now represented by an unapproved YAML review."""

    operation_id: str
    request_manifest_path: Path
    staging_path: Path
    request_fingerprint: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class CardRevisionTarget:
    """The exact configured deck and card scope behind a staged revision."""

    operation_id: str
    deck_path: Path
    deck_relative_path: str
    selected_record_ids: tuple[str, ...]
    focus_resource_id: str | None


@dataclass(frozen=True, slots=True)
class _DecodedRevision:
    summary: str
    replacements: Mapping[str, Mapping[str, Any]]
    manifest_value: Mapping[str, Any]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise CardRevisionError(f"Revision values must be finite JSON data: {exc}") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _operation_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        canonical = str(uuid.UUID(value))
    except (AttributeError, ValueError) as exc:
        raise CardRevisionError("Revision operation id must be canonical UUID text.") from exc
    if canonical != value or uuid.UUID(value).version != 4:
        raise CardRevisionError("Revision operation id must be a canonical UUIDv4.")
    return canonical


def _selected_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise CardRevisionError("Choose at least one canonical record; nothing was sent.")
    selected = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in selected):
        raise CardRevisionError("Every selected record id must be nonblank text.")
    if len(selected) != len(set(selected)):
        raise CardRevisionError("Each selected record id may appear only once.")
    return selected


def _focus_resource(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CardRevisionError("A focus resource id must be nonblank text when supplied.")
    return value.strip()


def _configured_deck(config: ProjectConfig, deck: Path | str) -> Path:
    requested = Path(deck)
    target = (
        requested if requested.is_absolute() else config.deck_dir / requested
    ).absolute()
    root = config.deck_dir.absolute()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CardRevisionError(
            f"Revision target must be a configured deck under {root}; nothing was sent."
        ) from exc
    try:
        configured = {path.absolute() for path in status_module.deck_files(config)}
    except JankiError as exc:
        raise CardRevisionError(f"Could not enumerate configured decks: {exc}") from exc
    if target not in configured:
        raise CardRevisionError(
            f"Revision target is not an exact configured deck: {target}; nothing was sent."
        )
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise CardRevisionError(f"{label} is outside this repository: {path}") from exc


def _optional_revision(path: Path) -> str | None:
    try:
        return _sha(read_bytes_bound(path))
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise CardRevisionError(f"Could not inspect revision target {path}: {exc}") from exc


def _snapshot_selection(
    config: ProjectConfig,
    deck_path: Path,
    selected: tuple[str, ...],
    *,
    acquire_locks: bool,
) -> tuple[bytes, bytes, tuple[VocabularyRecord, ...]]:
    paths = sorted({deck_path, config.normalized_file.absolute()}, key=lambda item: str(item))
    with contextlib.ExitStack() as locks:
        if acquire_locks:
            for path in paths:
                locks.enter_context(exclusive_path_lock(path))
        try:
            deck_before = read_bytes_bound(deck_path)
            canonical, canonical_revision = load_records_snapshot(config.normalized_file)
            if canonical_revision.text is None:
                raise CardRevisionError("The canonical collection does not exist.")
            canonical_wire = canonical_revision.text.encode("utf-8")
            kind = deck_kind(deck_path)
            if kind == "conjugation":
                deck_config = pattern_cards.conjugation_deck_section(
                    deck_path,
                    deck_before.decode("utf-8", errors="strict"),
                )
                declared_source = pattern_cards.collection_for_section(
                    deck_path,
                    config,
                    deck_config,
                )
                deck_records = pattern_cards.shipping_records_for_section(
                    deck_path,
                    deck_config,
                    canonical,
                )
            else:
                deck_config, deck_records = resolve_deck_records(deck_path)
                source_value = deck_config.get("source")
                if not isinstance(source_value, str) or not source_value.strip():
                    raise CardRevisionError(
                        f"Configured deck {deck_path.name} has no canonical vocabulary "
                        "source."
                    )
                declared_source = (deck_path.parent / source_value).resolve()
            deck_after = read_bytes_bound(deck_path)
            canonical_after = read_bytes_bound(config.normalized_file)
        except CardRevisionError:
            raise
        except (JankiError, OSError, UnicodeError) as exc:
            raise CardRevisionError(
                f"Could not read the selected deck and canonical records: {exc}"
            ) from exc
        if deck_before != deck_after or canonical_wire != canonical_after:
            raise CardRevisionError(
                "The selected deck or canonical collection changed while it was read."
            )

    if declared_source != config.normalized_file.resolve():
        raise CardRevisionError(
            f"Configured deck {deck_path.name} does not read this project's canonical "
            "vocabulary collection."
        )

    canonical_by_id: dict[str, VocabularyRecord] = {}
    duplicates: set[str] = set()
    for record in canonical:
        if record.id in canonical_by_id:
            duplicates.add(record.id)
        canonical_by_id[record.id] = record
    duplicated_selection = [record_id for record_id in selected if record_id in duplicates]
    if duplicated_selection:
        raise CardRevisionError(
            "Selected canonical ids are duplicated in the collection: "
            + ", ".join(duplicated_selection)
        )
    missing = [record_id for record_id in selected if record_id not in canonical_by_id]
    if missing:
        raise CardRevisionError(
            "Selected record(s) are missing from the canonical collection: "
            + ", ".join(missing)
        )
    members = {record.id for record in deck_records}
    outside = [record_id for record_id in selected if record_id not in members]
    if outside:
        raise CardRevisionError(
            f"Selected canonical record(s) are not members of {deck_path.name}: "
            + ", ".join(outside)
        )
    return (
        deck_before,
        canonical_wire,
        tuple(canonical_by_id[record_id] for record_id in selected),
    )


def _user_turn(
    *,
    deck_relative_path: str,
    focus_resource_id: str | None,
    owner_instruction: str,
    records: Sequence[VocabularyRecord],
) -> str:
    return _canonical_json(
        {
            "owner_instruction": owner_instruction,
            "deck_focus": {
                "path": deck_relative_path,
                "resource_id": focus_resource_id,
            },
            "selected_canonical_records": [record.to_dict() for record in records],
        },
        pretty=True,
    ) + "\n"


def _plan_identity(plan: CardRevisionPlan) -> str:
    return prompts.fingerprint(
        _canonical_json(
            {
                "version": 1,
                "operation_id": plan.operation_id,
                "deck_path": plan.deck_relative_path,
                "deck_sha256": plan.deck_sha256,
                "canonical_path": plan.canonical_relative_path,
                "canonical_sha256": plan.canonical_sha256,
                "selected_record_ids": list(plan.selected_record_ids),
                "owner_instruction_sha256": prompts.fingerprint(plan.owner_instruction),
                "focus_resource_id": plan.focus_resource_id,
                "provider": plan.provider,
                "model": plan.model,
                "request_fingerprint": plan.request_fingerprint,
                "request_manifest_path": _relative(
                    plan.repository_root,
                    plan.request_manifest_path,
                    label="Revision request manifest",
                ),
                "request_manifest_revision": plan.request_manifest_revision,
                "staging_path": _relative(
                    plan.repository_root,
                    plan.staging_path,
                    label="Revision staging target",
                ),
                "staging_revision": plan.staging_revision,
            }
        )
    )


def _plan_card_revision(
    config: ProjectConfig,
    deck: Path | str,
    selected_record_ids: Sequence[str],
    owner_instruction: str,
    *,
    operation_id: str | None = None,
    focus_resource_id: str | None = None,
    bindings_locked: bool,
) -> CardRevisionPlan:
    if not isinstance(owner_instruction, str) or not owner_instruction.strip():
        raise CardRevisionError("Describe the exact card change; nothing was sent.")
    selected = _selected_ids(selected_record_ids)
    operation = _operation_id(operation_id)
    focus = _focus_resource(focus_resource_id)
    deck_path = _configured_deck(config, deck)
    deck_wire, canonical_wire, records = _snapshot_selection(
        config,
        deck_path,
        selected,
        acquire_locks=not bindings_locked,
    )
    repository_root = config.root.resolve()
    deck_relative = _relative(repository_root, deck_path, label="Revision deck")
    canonical_path = config.normalized_file.resolve()
    canonical_relative = _relative(
        repository_root, canonical_path, label="Canonical collection"
    )

    style_guide = claude_client.read_style_guide(config.root)
    task_template = prompts.load(config.root, "revise-cards")
    schema = ai_schema.card_revision_schema()
    system_blocks = tuple(claude_client.system_blocks(style_guide, task_template))
    user_turn = _user_turn(
        deck_relative_path=deck_relative,
        focus_resource_id=focus,
        owner_instruction=owner_instruction,
        records=records,
    )
    provider_plan = revision_provider.plan_provider(
        str(config.revise_provider).strip().lower(),
        model=str(config.revise_model).strip(),
        style_guide=style_guide,
        task_template=task_template,
        system_blocks=system_blocks,
        user_turn=user_turn,
        schema=schema,
    )
    stem = f"card-revision-{operation}"
    manifest_path = config.staging_dir.resolve() / f"{stem}.request.json"
    staging_path = config.staging_dir.resolve() / f"{stem}.yaml"
    draft = CardRevisionPlan(
        repository_root=repository_root,
        operation_id=operation,
        deck_path=deck_path,
        deck_relative_path=deck_relative,
        deck_sha256=_sha(deck_wire),
        canonical_path=canonical_path,
        canonical_relative_path=canonical_relative,
        canonical_sha256=_sha(canonical_wire),
        selected_record_ids=selected,
        current_records=records,
        owner_instruction=owner_instruction,
        focus_resource_id=focus,
        provider_plan=provider_plan,
        style_guide=style_guide,
        task_template=task_template,
        user_turn=user_turn,
        request_manifest_path=manifest_path,
        request_manifest_revision=_optional_revision(manifest_path),
        staging_path=staging_path,
        staging_revision=_optional_revision(staging_path),
        plan_fingerprint="",
    )
    return replace(draft, plan_fingerprint=_plan_identity(draft))


def plan_card_revision(
    config: ProjectConfig,
    deck: Path | str,
    selected_record_ids: Sequence[str],
    owner_instruction: str,
    *,
    operation_id: str | None = None,
    focus_resource_id: str | None = None,
) -> CardRevisionPlan:
    """Plan one exact generic paid revision without writing repository state."""

    return _plan_card_revision(
        config,
        deck,
        selected_record_ids,
        owner_instruction,
        operation_id=operation_id,
        focus_resource_id=focus_resource_id,
        bindings_locked=False,
    )


def _fresh_plan(
    config: ProjectConfig,
    expected: CardRevisionPlan,
    *,
    bindings_locked: bool = False,
) -> CardRevisionPlan:
    if expected.repository_root != config.root.resolve():
        raise CardRevisionError(
            "[card-revision-config-mismatch] this plan belongs to another repository."
        )
    fresh = _plan_card_revision(
        config,
        expected.deck_path,
        expected.selected_record_ids,
        expected.owner_instruction,
        operation_id=expected.operation_id,
        focus_resource_id=expected.focus_resource_id,
        bindings_locked=bindings_locked,
    )
    if fresh.request_fingerprint != expected.request_fingerprint:
        raise CardRevisionError(
            "[card-revision-request-stale] selected records, prompt, provider, model, "
            "or instruction changed; reload the plan. Nothing was sent."
        )
    if fresh.plan_fingerprint != expected.plan_fingerprint:
        raise CardRevisionError(
            "[card-revision-plan-stale] deck, canonical, or target state changed; "
            "reload the plan. Nothing was sent."
        )
    return fresh


def _strict_json_value(raw: str, *, record_id: str, field: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CardRevisionError(
                    f"Revision value for {record_id}.{field} repeats JSON key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise CardRevisionError(
            f"Revision value for {record_id}.{field} contains {value}."
        )

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except CardRevisionError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise CardRevisionError(
            f"Revision value for {record_id}.{field} is not complete JSON: {exc}"
        ) from exc


def _decode_revision(parsed: Any, selected: tuple[str, ...]) -> _DecodedRevision:
    summary = str(getattr(parsed, "summary", "") or "").strip()
    if not summary:
        raise CardRevisionError("The captured revision has no nonblank summary.")
    raw_changes = list(getattr(parsed, "card_changes", []) or [])
    if not raw_changes:
        raise CardRevisionError("The captured revision proposes no card changes.")
    selected_set = set(selected)
    seen: set[str] = set()
    replacements: dict[str, dict[str, Any]] = {}
    manifest_changes: list[dict[str, Any]] = []
    for raw_change in raw_changes:
        record_id = str(getattr(raw_change, "record_id", "") or "")
        if record_id not in selected_set:
            raise CardRevisionError(
                f"The captured revision names unselected record {record_id!r}."
            )
        if record_id in seen:
            raise CardRevisionError(
                f"The captured revision repeats selected record {record_id!r}."
            )
        seen.add(record_id)
        reason = str(getattr(raw_change, "reason", "") or "").strip()
        if not reason:
            raise CardRevisionError(
                f"The captured revision has no reason for {record_id!r}."
            )
        updates = list(getattr(raw_change, "updates", []) or [])
        if not updates:
            raise CardRevisionError(
                f"The captured revision has no field updates for {record_id!r}."
            )
        record_fields: dict[str, Any] = {}
        manifest_updates: list[dict[str, str]] = []
        for raw_update in updates:
            field = str(getattr(raw_update, "field", "") or "")
            if field in record_fields:
                raise CardRevisionError(
                    f"The captured revision repeats field {record_id}.{field}."
                )
            try:
                validated = validate_prefer_incoming((field,))
            except JankiError as exc:
                raise CardRevisionError(
                    f"The captured revision cannot replace {record_id}.{field}: {exc}"
                ) from exc
            if not validated or field not in MERGEABLE_FIELDS:
                raise CardRevisionError(
                    f"The captured revision names unsupported field {record_id}.{field}."
                )
            value_json = str(getattr(raw_update, "value_json", "") or "")
            value = _strict_json_value(
                value_json,
                record_id=record_id,
                field=field,
            )
            record_fields[field] = value
            manifest_updates.append({"field": field, "value_json": value_json})
        replacements[record_id] = record_fields
        manifest_changes.append(
            {
                "record_id": record_id,
                "reason": reason,
                "updates": manifest_updates,
            }
        )
    ordered = {
        record_id: replacements[record_id]
        for record_id in selected
        if record_id in replacements
    }
    return _DecodedRevision(
        summary=summary,
        replacements=ordered,
        manifest_value={"summary": summary, "card_changes": manifest_changes},
    )


def _manifest_text(value: Mapping[str, Any]) -> str:
    return _canonical_json(value, pretty=True) + "\n"


def _request_manifest(plan: CardRevisionPlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "canonical_card_revision",
        "state": "request",
        "operation_id": plan.operation_id,
        "target": {
            "deck_path": plan.deck_relative_path,
            "deck_sha256": plan.deck_sha256,
            "canonical_path": plan.canonical_relative_path,
            "canonical_sha256": plan.canonical_sha256,
            "selected_record_ids": list(plan.selected_record_ids),
            "focus_resource_id": plan.focus_resource_id,
            "request_manifest_path": _relative(
                plan.repository_root,
                plan.request_manifest_path,
                label="Revision request manifest",
            ),
            "staging_path": _relative(
                plan.repository_root,
                plan.staging_path,
                label="Revision staging target",
            ),
        },
        "request": {
            "provider_plan": plan.provider_plan.persistent_manifest(),
            "owner_instruction": plan.owner_instruction,
            "style_guide": plan.style_guide,
            "task_template": plan.task_template,
            "system_blocks": _plain(plan.provider_plan.system_blocks),
            "user_turn": plan.user_turn,
            "plan_fingerprint": plan.plan_fingerprint,
        },
        "current_records": [record.to_dict() for record in plan.current_records],
    }


def _result_manifest(
    request_manifest: Mapping[str, Any],
    decoded: _DecodedRevision,
    *,
    staging_plan_fingerprint: str,
    staging_sha256: str,
) -> dict[str, Any]:
    result = dict(request_manifest)
    result["state"] = "result"
    result["result"] = {
        "answer": _plain(decoded.manifest_value),
        "staging_plan_fingerprint": staging_plan_fingerprint,
        "staging_sha256": staging_sha256,
    }
    return result


def _staging_plan(
    config: ProjectConfig,
    *,
    operation_id: str,
    request_fingerprint: str,
    provider: str,
    model: str,
    focus_resource_id: str | None,
    current_records: Sequence[VocabularyRecord],
    decoded: _DecodedRevision,
) -> card_change_staging.CardChangeStagingPlan:
    by_id = {record.id: record for record in current_records}
    changed_records = tuple(by_id[record_id] for record_id in decoded.replacements)
    return card_change_staging.plan_card_change_staging(
        config,
        changed_records,
        provenance=card_change_staging.CardRevisionProvenance(
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            provider=provider,
            model=model,
        ),
        field_replacements=decoded.replacements,
        focus_resource_id=focus_resource_id,
    )


def _persist_result_under_locks(
    config: ProjectConfig,
    *,
    manifest_path: Path,
    manifest_revision: str,
    request_manifest: Mapping[str, Any],
    decoded: _DecodedRevision,
    staging_plan: card_change_staging.CardChangeStagingPlan,
) -> None:
    card_change_staging.stage_card_change_staging_under_lock(
        config,
        staging_plan,
        expected_fingerprint=staging_plan.fingerprint,
    )
    staging_sha256 = _sha(read_bytes_bound(staging_plan.staging_path))
    final_text = _manifest_text(
        _result_manifest(
            request_manifest,
            decoded,
            staging_plan_fingerprint=staging_plan.fingerprint,
            staging_sha256=staging_sha256,
        )
    )
    current = read_bytes_bound(manifest_path)
    if current == final_text.encode("utf-8"):
        return
    if _sha(current) != manifest_revision:
        raise CardRevisionError(
            "The durable card-revision request manifest changed before its result "
            "could be recorded."
        )
    atomic_write_text_bound(
        manifest_path,
        final_text,
        expected_revision=manifest_revision,
    )


def _report(progress: Callable[[str], None] | None, label: str) -> None:
    if progress is not None:
        progress(label)


def _cancel_before_send(
    config: ProjectConfig,
    operation_id: str,
    exc: BaseException,
) -> CardRevisionRunError:
    try:
        operations.OperationJournal.load(config.operations_file).advance(
            operation_id,
            "canceled_before_send",
            detail="The durable card-revision request manifest could not be prepared.",
        )
    except JankiError as journal_error:
        return CardRevisionRunError(
            f"Revision {operation_id} was not sent, but request publication failed "
            f"and its authority could not be retired: {exc}; {journal_error}. "
            "Inspect janki operations before retrying.",
            operation_id=operation_id,
            provider_dispatched=False,
        )
    return CardRevisionRunError(
        f"Revision {operation_id} was canceled before send because its exact request "
        f"manifest could not be made durable: {exc}",
        operation_id=operation_id,
        provider_dispatched=False,
    )


def run_card_revision(
    config: ProjectConfig,
    expected: CardRevisionPlan,
    *,
    client: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    api_call: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CardRevisionRunResult:
    """Consume one exact owner-confirmed plan and stage its paid answer."""

    _report(progress, "Reading the source")
    fresh = _fresh_plan(config, expected)
    if not fresh.can_dispatch:
        raise CardRevisionError(
            "This revision operation's manifest or staging target is already occupied."
        )
    provider = revision_provider.provider_for(fresh.provider)
    prepared = provider.prepare(
        fresh.provider_plan,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
        client=client,
    )

    prompt_paths = (
        config.root / prompts.DIRECTORY / "style-guide.md",
        config.root / prompts.DIRECTORY / "revise-cards.md",
    )
    with contextlib.ExitStack() as locks:
        bound_paths = {
            fresh.deck_path,
            fresh.canonical_path,
            fresh.request_manifest_path,
            fresh.staging_path,
            *prompt_paths,
        }
        for path in sorted(bound_paths, key=lambda item: str(item.absolute())):
            locks.enter_context(exclusive_path_lock(path))
        fresh = _fresh_plan(config, expected, bindings_locked=True)
        if not fresh.can_dispatch:
            raise CardRevisionError(
                "This revision operation's manifest or staging target is already occupied."
            )
        operations.prepare_artifact_store(config.operations_file)
        prepare_bound_directory(config.staging_dir)
        journal = operations.OperationJournal.load(config.operations_file)
        try:
            journal.authorize(
                fresh.operation_id,
                kind="revise",
                source_file=fresh.deck_relative_path,
                source_sha256=fresh.deck_sha256,
                request_fp=fresh.request_fingerprint,
                model=fresh.model,
            )
        except Exception as exc:  # noqa: BLE001 - authority may have landed
            raise CardRevisionRunError(
                f"Revision authority could not be recorded safely: {exc} Inspect "
                "janki operations before retrying.",
                operation_id=fresh.operation_id,
                provider_dispatched=False,
            ) from exc
        request_manifest = _request_manifest(fresh)
        request_text = _manifest_text(request_manifest)
        manifest_revision = _sha(request_text.encode("utf-8"))
        try:
            atomic_write_text_bound(
                fresh.request_manifest_path,
                request_text,
                expected_absent=True,
            )
        except Exception as exc:  # noqa: BLE001 - publication may have landed
            raise _cancel_before_send(config, fresh.operation_id, exc) from exc

    try:
        journal.advance(fresh.operation_id, "dispatching")
    except Exception as exc:  # noqa: BLE001 - transition may have landed
        try:
            current = operations.OperationJournal.load(config.operations_file)
            held = current.operations.get(fresh.operation_id)
            if held is not None and held.state == "authorized":
                current.advance(
                    fresh.operation_id,
                    "canceled_before_send",
                    detail="The dispatch boundary could not be recorded.",
                )
        except JankiError:
            pass
        raise CardRevisionRunError(
            f"Revision {fresh.operation_id} was not dispatched because its dispatch "
            f"boundary could not be recorded: {exc}",
            operation_id=fresh.operation_id,
            provider_dispatched=False,
        ) from exc

    try:
        def capture(raw_reply: bytes) -> None:
            if not isinstance(raw_reply, bytes):
                raise operations.OperationError(
                    "Revision provider capture must supply exact response bytes."
                )
            journal.capture_result(
                fresh.operation_id,
                lambda: operations.capture_artifact(
                    config.operations_file,
                    fresh.operation_id,
                    raw_reply,
                ),
            )
            _report(progress, "Checking the answer's shape")

        call_result = provider.dispatch(
            prepared,
            capture=capture,
            runner=provider_runner,
            api_call=api_call,
        )
        captured = operations.OperationJournal.load(config.operations_file).operations.get(
            fresh.operation_id
        )
        if captured is None or captured.state != "result_captured":
            raise operations.OperationError(
                "The revision provider returned data without durably capturing its "
                "exact reply."
            )
        parsed = getattr(call_result, "parsed", None)
        if parsed is None:
            reason = str(getattr(call_result, "stop_reason", "") or "unknown")
            raise CardRevisionError(
                f"{fresh.model} returned no complete card revision ({reason}). Its "
                "exact reply was captured, but no proposal was staged."
            )
        decoded = _decode_revision(parsed, fresh.selected_record_ids)
        staging_plan = _staging_plan(
            config,
            operation_id=fresh.operation_id,
            request_fingerprint=fresh.request_fingerprint,
            provider=fresh.provider,
            model=fresh.model,
            focus_resource_id=fresh.focus_resource_id,
            current_records=fresh.current_records,
            decoded=decoded,
        )
        _report(progress, "Saving proposals")
        with contextlib.ExitStack() as output_locks:
            for path in sorted(
                {fresh.request_manifest_path, fresh.staging_path},
                key=lambda item: str(item.absolute()),
            ):
                output_locks.enter_context(exclusive_path_lock(path))
            journal.commit_result(
                fresh.operation_id,
                lambda: _persist_result_under_locks(
                    config,
                    manifest_path=fresh.request_manifest_path,
                    manifest_revision=manifest_revision,
                    request_manifest=request_manifest,
                    decoded=decoded,
                    staging_plan=staging_plan,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - post-dispatch state must be truthful
        try:
            classify_dispatch_failure(config, journal, fresh.operation_id, exc)
        except JankiError as journal_error:
            raise CardRevisionRunError(
                f"Revision {fresh.operation_id} failed after dispatch: {exc} Janki "
                f"could not settle its journal entry: {journal_error}. This call may "
                "have been billed; inspect janki operations before retrying.",
                operation_id=fresh.operation_id,
                provider_dispatched=True,
            ) from exc
        raise CardRevisionRunError(
            f"Revision {fresh.operation_id} failed after dispatch: {exc}",
            operation_id=fresh.operation_id,
            provider_dispatched=True,
        ) from exc

    return CardRevisionRunResult(
        operation_id=fresh.operation_id,
        request_manifest_path=fresh.request_manifest_path,
        staging_path=fresh.staging_path,
        request_fingerprint=fresh.request_fingerprint,
        plan_fingerprint=fresh.plan_fingerprint,
    )


def _strict_manifest(raw: bytes, path: Path) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CardRevisionError(
                    f"Revision manifest {path} repeats key {key!r}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CardRevisionError(
                    f"Revision manifest {path} contains invalid JSON value {value}."
                )
            ),
        )
    except CardRevisionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CardRevisionError(f"Could not parse revision manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CardRevisionError(f"Revision manifest {path} must contain one object.")
    if _manifest_text(value).encode("utf-8") != raw:
        raise CardRevisionError(
            f"Revision manifest {path} is not its canonical durable JSON form."
        )
    return value


def _mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CardRevisionError(f"Revision manifest has invalid {label} fields.")
    return value


def _recovery_request(
    config: ProjectConfig,
    operation_id: str,
) -> tuple[
    operations.OperationJournal,
    Any,
    Path,
    Mapping[str, Any],
    str,
    tuple[VocabularyRecord, ...],
    revision_provider.RevisionProviderPlan,
]:
    operation = _operation_id(operation_id)
    journal = operations.OperationJournal.load(config.operations_file)
    held = journal.operations.get(operation)
    if held is None or held.kind != "revise":
        raise CardRevisionError(f"No generic revise operation {operation!r} to recover.")
    if held.state not in {"result_captured", "committed"} or held.artifact is None:
        raise CardRevisionError(
            f"Revision {operation!r} has no captured reply ready for recovery; "
            f"its state is {held.state!r}."
        )
    manifest_path = config.staging_dir.resolve() / (
        f"card-revision-{operation}.request.json"
    )
    try:
        wire = read_bytes_bound(manifest_path)
    except (FileNotFoundError, JankiError, OSError) as exc:
        raise CardRevisionError(
            f"Could not read the durable request manifest for revision {operation}: {exc}"
        ) from exc
    manifest = _strict_manifest(wire, manifest_path)
    state = manifest.get("state")
    top = {
        "schema_version",
        "kind",
        "state",
        "operation_id",
        "target",
        "request",
        "current_records",
    }
    if state == "result":
        top.add("result")
    _mapping(manifest, top, label="top-level")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "canonical_card_revision"
        or state not in {"request", "result"}
        or manifest.get("operation_id") != operation
    ):
        raise CardRevisionError("Revision request manifest has the wrong identity.")
    if held.state == "committed" and state != "result":
        raise CardRevisionError("Committed revision has no durable result manifest.")
    target = _mapping(
        manifest.get("target"),
        {
            "deck_path",
            "deck_sha256",
            "canonical_path",
            "canonical_sha256",
            "selected_record_ids",
            "focus_resource_id",
            "request_manifest_path",
            "staging_path",
        },
        label="target",
    )
    request = _mapping(
        manifest.get("request"),
        {
            "provider_plan",
            "owner_instruction",
            "style_guide",
            "task_template",
            "system_blocks",
            "user_turn",
            "plan_fingerprint",
        },
        label="request",
    )
    text_target = (
        "deck_path",
        "deck_sha256",
        "canonical_path",
        "canonical_sha256",
        "request_manifest_path",
        "staging_path",
    )
    text_request = (
        "owner_instruction",
        "style_guide",
        "task_template",
        "user_turn",
        "plan_fingerprint",
    )
    if any(not isinstance(target.get(name), str) for name in text_target) or any(
        not isinstance(request.get(name), str) for name in text_request
    ):
        raise CardRevisionError("Revision request manifest has non-text identity fields.")
    selected_raw = target.get("selected_record_ids")
    selected = _selected_ids(selected_raw if isinstance(selected_raw, list) else [])
    focus = _focus_resource(target.get("focus_resource_id"))
    raw_records = manifest.get("current_records")
    if not isinstance(raw_records, list):
        raise CardRevisionError("Revision request manifest has no current record list.")
    try:
        records = tuple(VocabularyRecord.from_dict(value) for value in raw_records)
    except (JankiError, TypeError) as exc:
        raise CardRevisionError(
            f"Revision request manifest has invalid current records: {exc}"
        ) from exc
    if tuple(record.id for record in records) != selected:
        raise CardRevisionError(
            "Revision request manifest records do not match its ordered selection."
        )
    expected_manifest_relative = _relative(
        config.root.resolve(), manifest_path, label="Revision request manifest"
    )
    expected_staging = config.staging_dir.resolve() / f"card-revision-{operation}.yaml"
    if (
        target["request_manifest_path"] != expected_manifest_relative
        or target["staging_path"]
        != _relative(config.root.resolve(), expected_staging, label="Revision staging")
        or target["deck_path"] != held.source_file
        or target["deck_sha256"] != held.source_sha256
    ):
        raise CardRevisionError("Revision manifest does not match its journaled target.")
    expected_turn = _user_turn(
        deck_relative_path=target["deck_path"],
        focus_resource_id=focus,
        owner_instruction=request["owner_instruction"],
        records=records,
    )
    if expected_turn != request["user_turn"]:
        raise CardRevisionError("Revision manifest does not reproduce its exact user turn.")
    expected_blocks = claude_client.system_blocks(
        request["style_guide"], request["task_template"]
    )
    if request["system_blocks"] != expected_blocks:
        raise CardRevisionError("Revision manifest has inconsistent system prompts.")
    provider_manifest = request.get("provider_plan")
    if not isinstance(provider_manifest, Mapping):
        raise CardRevisionError("Revision manifest has invalid provider metadata.")
    provider_plan = revision_provider.provider_plan_from_manifest(
        provider_manifest,
        model=str(provider_manifest.get("model", "")),
        style_guide=request["style_guide"],
        task_template=request["task_template"],
        system_blocks=request["system_blocks"],
        user_turn=request["user_turn"],
        schema=ai_schema.card_revision_schema(),
    )
    if provider_plan.request_fingerprint != held.request_fp or provider_plan.model != held.model:
        raise CardRevisionError("Revision provider provenance differs from its journal entry.")
    provisional = CardRevisionPlan(
        repository_root=config.root.resolve(),
        operation_id=operation,
        deck_path=config.root.resolve() / target["deck_path"],
        deck_relative_path=target["deck_path"],
        deck_sha256=target["deck_sha256"],
        canonical_path=config.root.resolve() / target["canonical_path"],
        canonical_relative_path=target["canonical_path"],
        canonical_sha256=target["canonical_sha256"],
        selected_record_ids=selected,
        current_records=records,
        owner_instruction=request["owner_instruction"],
        focus_resource_id=focus,
        provider_plan=provider_plan,
        style_guide=request["style_guide"],
        task_template=request["task_template"],
        user_turn=request["user_turn"],
        request_manifest_path=manifest_path,
        request_manifest_revision=None,
        staging_path=expected_staging,
        staging_revision=None,
        plan_fingerprint="",
    )
    if _plan_identity(provisional) != request["plan_fingerprint"]:
        raise CardRevisionError("Revision manifest does not reproduce its confirmed plan.")
    reply = journal.read_reply(operation)
    if _sha(reply) != held.artifact.content_sha256:
        raise CardRevisionError("Captured revision reply does not match its receipt.")
    return (
        journal,
        held,
        manifest_path,
        manifest,
        _sha(wire),
        records,
        provider_plan,
    )


def inspect_card_revision_target(
    config: ProjectConfig,
    operation_id: str,
) -> CardRevisionTarget:
    """Return the validated deck/card identity of one captured revision.

    The request manifest is the durable link the later reviewed proposal no
    longer carries directly.  Reusing the recovery validator keeps this read
    bound to the paid operation and the exact manifest that produced staging;
    no reply bytes are exposed to the caller.
    """

    (
        _journal,
        _held,
        _manifest_path,
        manifest,
        _manifest_revision,
        _records,
        _provider_plan,
    ) = _recovery_request(config, operation_id)
    target = manifest["target"]
    assert isinstance(target, Mapping)
    deck_relative = str(target["deck_path"])
    return CardRevisionTarget(
        operation_id=operation_id,
        deck_path=(config.root.resolve() / deck_relative).absolute(),
        deck_relative_path=deck_relative,
        selected_record_ids=tuple(target["selected_record_ids"]),
        focus_resource_id=target["focus_resource_id"],
    )


def recover_card_revision(
    config: ProjectConfig,
    operation_id: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> CardRevisionRunResult:
    """Finish one captured generic revise reply without redispatching it."""

    _report(progress, "Reading the source")
    (
        journal,
        held,
        manifest_path,
        manifest,
        manifest_revision,
        records,
        provider_plan,
    ) = _recovery_request(config, operation_id)
    reply = journal.read_reply(operation_id)
    _report(progress, "Checking the answer's shape")
    call_result = revision_provider.provider_for(provider_plan.provider).recover(
        provider_plan,
        reply,
    )
    parsed = getattr(call_result, "parsed", None)
    if parsed is None:
        reason = str(getattr(call_result, "stop_reason", "") or "unknown")
        raise CardRevisionError(
            f"Captured revision {operation_id} has no complete answer ({reason})."
        )
    selected = tuple(manifest["target"]["selected_record_ids"])
    decoded = _decode_revision(parsed, selected)
    staging_plan = _staging_plan(
        config,
        operation_id=operation_id,
        request_fingerprint=held.request_fp,
        provider=provider_plan.provider,
        model=provider_plan.model,
        focus_resource_id=manifest["target"]["focus_resource_id"],
        current_records=records,
        decoded=decoded,
    )
    request_manifest = dict(manifest)
    request_manifest.pop("result", None)
    request_manifest["state"] = "request"
    _report(progress, "Saving proposals")
    with contextlib.ExitStack() as locks:
        for path in sorted(
            {manifest_path, staging_plan.staging_path},
            key=lambda item: str(item.absolute()),
        ):
            locks.enter_context(exclusive_path_lock(path))
        if held.state == "committed":
            if manifest.get("state") != "result" or not staging_plan.staging_path.exists():
                raise CardRevisionError("Committed revision output is incomplete.")
            staging_sha = _sha(read_bytes_bound(staging_plan.staging_path))
            expected = _manifest_text(
                _result_manifest(
                    request_manifest,
                    decoded,
                    staging_plan_fingerprint=staging_plan.fingerprint,
                    staging_sha256=staging_sha,
                )
            )
            if expected.encode("utf-8") != read_bytes_bound(manifest_path):
                raise CardRevisionError("Committed revision output has diverged.")
        else:
            journal.commit_result(
                operation_id,
                lambda: _persist_result_under_locks(
                    config,
                    manifest_path=manifest_path,
                    manifest_revision=manifest_revision,
                    request_manifest=request_manifest,
                    decoded=decoded,
                    staging_plan=staging_plan,
                ),
            )
    return CardRevisionRunResult(
        operation_id=operation_id,
        request_manifest_path=manifest_path,
        staging_path=staging_plan.staging_path,
        request_fingerprint=held.request_fp,
        plan_fingerprint=manifest["request"]["plan_fingerprint"],
    )
