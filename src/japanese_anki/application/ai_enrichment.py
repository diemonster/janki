"""Plan, journal, and stage the existing bare-card ``enrich --ai`` pass.

The legacy command still owns terminal presentation and its small-run direct
write.  This application boundary is deliberately stricter: every selected
record is an independently fingerprinted paid request, every exact reply is
captured before it is decoded, and model-authored content can only become an
``ai_enrichment`` staging proposal.  Canonical promotion remains a later owner
decision through the shared promotion transaction.

Requests in one plan are independent rather than carrying the live command's
``Recent examples from this run`` values.  A later request cannot be bound at
confirmation time if its bytes depend on an earlier, unseen model answer.  This
is the same independent request shape used by the existing Message Batches
variant of ``enrich --ai``; the substantive prompt, schema, merge discipline,
and staging provenance remain the existing enrichment path's.
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
from datetime import date
from pathlib import Path
from typing import Any, Literal

from japanese_anki import claude_client, enrich, operations, patterns, prompts, staging
from japanese_anki.application import revision_provider
from japanese_anki.application.extraction import classify_dispatch_failure
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records_snapshot,
    prepare_bound_directory,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "AiEnrichmentCallPlan",
    "AiEnrichmentError",
    "AiEnrichmentItemResult",
    "AiEnrichmentPlan",
    "AiEnrichmentRunError",
    "AiEnrichmentRunResult",
    "plan_ai_enrichment",
    "run_ai_enrichment",
]


class AiEnrichmentError(JankiError):
    """Bare-card AI enrichment could not be planned or staged safely."""


class AiEnrichmentRunError(AiEnrichmentError):
    """One item stopped after its paid operation identity was allocated."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        provider_dispatched: bool,
        completed_results: Sequence[AiEnrichmentItemResult] = (),
    ) -> None:
        self.operation_id = operation_id
        self.provider_dispatched = provider_dispatched
        self.completed_results = tuple(completed_results)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AiEnrichmentCallPlan:
    """One exact record request within an owner-confirmable enrichment batch."""

    operation_id: str
    record_id: str
    record: VocabularyRecord
    record_sha256: str
    input_fingerprint: str
    user_turn: str
    provider_plan: revision_provider.RevisionProviderPlan
    request_manifest_path: Path
    request_manifest_revision: str | None
    staging_path: Path
    staging_revision: str | None

    @property
    def request_fingerprint(self) -> str:
        return self.provider_plan.request_fingerprint

    @property
    def can_dispatch(self) -> bool:
        return self.request_manifest_revision is None and self.staging_revision is None


@dataclass(frozen=True, slots=True)
class AiEnrichmentPlan:
    """Every exact request and repository input covered by one confirmation."""

    repository_root: Path
    canonical_path: Path
    canonical_relative_path: str
    canonical_sha256: str
    patterns_path: Path
    patterns_sha256: str
    style_guide: str
    task_template: str
    taught_patterns: str
    force_fields: tuple[str, ...]
    focus_resource_id: str | None
    provider: str
    model: str
    calls: tuple[AiEnrichmentCallPlan, ...]
    plan_fingerprint: str

    @property
    def billing_display(self) -> str:
        return self.calls[0].provider_plan.billing_display


@dataclass(frozen=True, slots=True)
class AiEnrichmentItemResult:
    """One paid answer's durable destination."""

    operation_id: str
    record_id: str
    request_manifest_path: Path
    staging_path: Path | None
    state: Literal["staged", "no_changes"]
    changed_fields: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AiEnrichmentRunResult:
    """The prefix of one batch that reached durable destinations."""

    plan_fingerprint: str
    results: tuple[AiEnrichmentItemResult, ...]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    options["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise AiEnrichmentError(f"AI-enrichment values must be finite JSON data: {exc}") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    return value


def _operation_ids(
    count: int,
    supplied: Sequence[str] | None,
) -> tuple[str, ...]:
    values = tuple(str(uuid.uuid4()) for _ in range(count)) if supplied is None else tuple(supplied)
    if len(values) != count:
        raise AiEnrichmentError("AI enrichment needs exactly one operation id per selected card.")
    canonical: list[str] = []
    for value in values:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, ValueError) as exc:
            raise AiEnrichmentError(
                "AI-enrichment operation ids must be canonical UUIDv4 text."
            ) from exc
        if parsed.version != 4 or str(parsed) != value:
            raise AiEnrichmentError("AI-enrichment operation ids must be canonical UUIDv4 text.")
        canonical.append(value)
    if len(canonical) != len(set(canonical)):
        raise AiEnrichmentError("AI-enrichment operation ids must be unique.")
    return tuple(canonical)


def _record_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise AiEnrichmentError("Choose an explicit list of canonical cards to enrich.")
    selected = tuple(values)
    if not selected:
        raise AiEnrichmentError("Choose at least one canonical card to enrich.")
    if any(not isinstance(value, str) or not value.strip() for value in selected):
        raise AiEnrichmentError("Every AI-enrichment record id must be nonblank text.")
    if len(selected) != len(set(selected)):
        raise AiEnrichmentError("Each AI-enrichment record id may appear only once.")
    return selected


def _force_fields(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise AiEnrichmentError("AI-enrichment force fields must be an explicit list.")
    fields = tuple(values)
    if len(fields) != len(set(fields)):
        raise AiEnrichmentError("Each AI-enrichment force field may appear only once.")
    invalid = [field for field in fields if field not in enrich.AI_FIELDS]
    if invalid:
        raise AiEnrichmentError(
            "AI enrichment can force only meanings, examples, and usage_notes; "
            f"got {', '.join(repr(item) for item in invalid)}."
        )
    return fields


def _focus(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AiEnrichmentError("AI-enrichment focus must be nonblank when supplied.")
    return value.strip()


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise AiEnrichmentError(f"{label} is outside this repository: {path}") from exc


def _optional_bytes(path: Path) -> bytes:
    try:
        return read_bytes_bound(path)
    except FileNotFoundError:
        return b""
    except (DataError, OSError) as exc:
        raise AiEnrichmentError(f"Could not inspect {path}: {exc}") from exc


def _optional_revision(path: Path) -> str | None:
    try:
        return _sha(read_bytes_bound(path))
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise AiEnrichmentError(f"Could not inspect output target {path}: {exc}") from exc


def _record_digest(record: VocabularyRecord) -> str:
    return _sha(_canonical_json(record.to_dict()).encode("utf-8"))


def _snapshot_inputs(
    config: ProjectConfig,
    selected: tuple[str, ...],
    *,
    acquire_locks: bool,
) -> tuple[
    bytes,
    bytes,
    tuple[VocabularyRecord, ...],
    str,
    str,
    str,
]:
    style_path = prompts.path_for(config.root, "style-guide").absolute()
    task_path = prompts.path_for(config.root, "enrich-bare-word").absolute()
    paths = {
        config.normalized_file.absolute(),
        config.patterns_file.absolute(),
        style_path,
        task_path,
    }
    with contextlib.ExitStack() as locks:
        if acquire_locks:
            for path in sorted(paths, key=lambda item: str(item)):
                locks.enter_context(exclusive_path_lock(path))
        try:
            records, revision = load_records_snapshot(config.normalized_file)
            if revision.text is None:
                raise AiEnrichmentError("The canonical vocabulary collection does not exist.")
            canonical_wire = revision.text.encode("utf-8")
            patterns_wire = _optional_bytes(config.patterns_file.absolute())
            style_guide = prompts.load(config.root, "style-guide")
            task_template = prompts.load(config.root, "enrich-bare-word")
        except AiEnrichmentError:
            raise
        except (JankiError, OSError, UnicodeError) as exc:
            raise AiEnrichmentError(f"Could not read AI-enrichment inputs safely: {exc}") from exc

        if acquire_locks:
            # Bound reads already occurred while every writer lock was held.
            pass

    by_id: dict[str, VocabularyRecord] = {}
    duplicates: set[str] = set()
    for record in records:
        if record.id in by_id:
            duplicates.add(record.id)
        by_id[record.id] = record
    repeated = [record_id for record_id in selected if record_id in duplicates]
    if repeated:
        raise AiEnrichmentError(
            "Selected canonical ids are duplicated in the collection: " + ", ".join(repeated)
        )
    missing = [record_id for record_id in selected if record_id not in by_id]
    if missing:
        raise AiEnrichmentError("Selected canonical cards are missing: " + ", ".join(missing))
    if patterns_wire:
        try:
            store = patterns.load_store_text(
                patterns_wire.decode("utf-8", errors="strict"),
                source=str(config.patterns_file),
            )
        except (JankiError, UnicodeError) as exc:
            raise AiEnrichmentError(f"Could not read reviewed lesson patterns: {exc}") from exc
    else:
        store = {}
    taught = patterns.format_patterns(patterns.reviewed_patterns(store))
    return (
        canonical_wire,
        patterns_wire,
        tuple(by_id[record_id] for record_id in selected),
        style_guide,
        task_template,
        taught,
    )


def _plan_identity(plan: AiEnrichmentPlan) -> str:
    return _sha(
        _canonical_json(
            {
                "version": 1,
                "repository_root": str(plan.repository_root),
                "canonical_path": plan.canonical_relative_path,
                "canonical_sha256": plan.canonical_sha256,
                "patterns_path": _relative(
                    plan.repository_root,
                    plan.patterns_path,
                    label="Pattern store",
                ),
                "patterns_sha256": plan.patterns_sha256,
                "style_guide_sha256": _sha(plan.style_guide.encode("utf-8")),
                "task_template_sha256": _sha(plan.task_template.encode("utf-8")),
                "taught_patterns_sha256": _sha(plan.taught_patterns.encode("utf-8")),
                "force_fields": list(plan.force_fields),
                "focus_resource_id": plan.focus_resource_id,
                "provider": plan.provider,
                "model": plan.model,
                "calls": [
                    {
                        "operation_id": call.operation_id,
                        "record_id": call.record_id,
                        "record_sha256": call.record_sha256,
                        "input_fingerprint": call.input_fingerprint,
                        "request_fingerprint": call.request_fingerprint,
                        "request_manifest_path": _relative(
                            plan.repository_root,
                            call.request_manifest_path,
                            label="AI-enrichment request manifest",
                        ),
                        "request_manifest_revision": call.request_manifest_revision,
                        "staging_path": _relative(
                            plan.repository_root,
                            call.staging_path,
                            label="AI-enrichment staging target",
                        ),
                        "staging_revision": call.staging_revision,
                    }
                    for call in plan.calls
                ],
            }
        ).encode("utf-8")
    )


def _plan_ai_enrichment(
    config: ProjectConfig,
    record_ids: Sequence[str],
    *,
    force_fields: Sequence[str],
    operation_ids: Sequence[str] | None,
    focus_resource_id: str | None,
    acquire_locks: bool,
) -> AiEnrichmentPlan:
    selected = _record_ids(record_ids)
    forced = _force_fields(force_fields)
    operations_ids = _operation_ids(len(selected), operation_ids)
    focus = _focus(focus_resource_id)
    if config.enrich_provider != "anthropic":
        raise AiEnrichmentError(
            "Journaled Assistant enrichment currently requires [ai] "
            'enrich_provider = "anthropic". The existing Codex enrichment '
            "transport has no exact reply capture boundary, so Janki will not "
            "send it from the Assistant until that gap is closed."
        )
    repository_root = config.root.resolve()
    canonical_path = config.normalized_file.resolve()
    patterns_path = config.patterns_file.absolute()
    (
        canonical_wire,
        patterns_wire,
        records,
        style_guide,
        task_template,
        taught,
    ) = _snapshot_inputs(config, selected, acquire_locks=acquire_locks)
    schema = enrich.ai_schema()
    system_blocks = tuple(claude_client.system_blocks(style_guide, task_template))
    calls: list[AiEnrichmentCallPlan] = []
    for record, operation_id in zip(records, operations_ids, strict=True):
        user_turn = enrich.ai_prompt(record, (), taught)
        provider_plan = revision_provider.plan_provider(
            "anthropic-api",
            model=config.enrich_model,
            style_guide=style_guide,
            task_template=task_template,
            system_blocks=system_blocks,
            user_turn=user_turn,
            schema=schema,
            effort=claude_client.effort_for(config.enrich_model),
        )
        stem = f"ai-enrichment-{operation_id}"
        request_manifest_path = config.staging_dir.absolute() / f"{stem}.request.json"
        staging_path = config.staging_dir.absolute() / f"{stem}.yaml"
        calls.append(
            AiEnrichmentCallPlan(
                operation_id=operation_id,
                record_id=record.id,
                record=record,
                record_sha256=_record_digest(record),
                input_fingerprint=enrich.ai_input_fingerprint(record, (), taught),
                user_turn=user_turn,
                provider_plan=provider_plan,
                request_manifest_path=request_manifest_path,
                request_manifest_revision=_optional_revision(request_manifest_path),
                staging_path=staging_path,
                staging_revision=_optional_revision(staging_path),
            )
        )
    draft = AiEnrichmentPlan(
        repository_root=repository_root,
        canonical_path=canonical_path,
        canonical_relative_path=_relative(
            repository_root,
            canonical_path,
            label="Canonical collection",
        ),
        canonical_sha256=_sha(canonical_wire),
        patterns_path=patterns_path,
        patterns_sha256=_sha(patterns_wire),
        style_guide=style_guide,
        task_template=task_template,
        taught_patterns=taught,
        force_fields=forced,
        focus_resource_id=focus,
        provider="anthropic",
        model=config.enrich_model,
        calls=tuple(calls),
        plan_fingerprint="",
    )
    return replace(draft, plan_fingerprint=_plan_identity(draft))


def plan_ai_enrichment(
    config: ProjectConfig,
    record_ids: Sequence[str],
    *,
    force_fields: Sequence[str] = (),
    operation_ids: Sequence[str] | None = None,
    focus_resource_id: str | None = None,
) -> AiEnrichmentPlan:
    """Plan an exact batch without writing authority, manifests, or proposals."""

    return _plan_ai_enrichment(
        config,
        record_ids,
        force_fields=force_fields,
        operation_ids=operation_ids,
        focus_resource_id=focus_resource_id,
        acquire_locks=True,
    )


def _fresh_call(
    config: ProjectConfig,
    expected_plan: AiEnrichmentPlan,
    expected_call: AiEnrichmentCallPlan,
    *,
    acquire_locks: bool,
) -> AiEnrichmentCallPlan:
    if expected_plan.repository_root != config.root.resolve():
        raise AiEnrichmentError(
            "[ai-enrichment-config-mismatch] this plan belongs to another repository."
        )
    fresh = _plan_ai_enrichment(
        config,
        [expected_call.record_id],
        force_fields=expected_plan.force_fields,
        operation_ids=[expected_call.operation_id],
        focus_resource_id=expected_plan.focus_resource_id,
        acquire_locks=acquire_locks,
    ).calls[0]
    if fresh.request_fingerprint != expected_call.request_fingerprint:
        raise AiEnrichmentError(
            "[ai-enrichment-request-stale] the card, reviewed patterns, prompt, "
            "provider, model, or schema changed; reload the plan. Nothing was sent."
        )
    comparable = (
        fresh.record_sha256,
        fresh.input_fingerprint,
        fresh.request_manifest_path,
        fresh.request_manifest_revision,
        fresh.staging_path,
        fresh.staging_revision,
    )
    expected = (
        expected_call.record_sha256,
        expected_call.input_fingerprint,
        expected_call.request_manifest_path,
        expected_call.request_manifest_revision,
        expected_call.staging_path,
        expected_call.staging_revision,
    )
    if comparable != expected:
        raise AiEnrichmentError(
            "[ai-enrichment-plan-stale] the selected card or output targets changed; "
            "reload the plan. Nothing was sent."
        )
    return fresh


def _manifest_text(value: Mapping[str, Any]) -> str:
    return _canonical_json(value, pretty=True) + "\n"


def _request_manifest(
    plan: AiEnrichmentPlan,
    call: AiEnrichmentCallPlan,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ai_enrichment",
        "state": "request",
        "operation_id": call.operation_id,
        "target": {
            "canonical_path": plan.canonical_relative_path,
            "canonical_sha256": plan.canonical_sha256,
            "patterns_sha256": plan.patterns_sha256,
            "record_id": call.record_id,
            "record_sha256": call.record_sha256,
            "focus_resource_id": plan.focus_resource_id,
            "request_manifest_path": _relative(
                plan.repository_root,
                call.request_manifest_path,
                label="AI-enrichment request manifest",
            ),
            "staging_path": _relative(
                plan.repository_root,
                call.staging_path,
                label="AI-enrichment staging target",
            ),
        },
        "request": {
            "provider_plan": call.provider_plan.persistent_manifest(),
            "attribution_provider": plan.provider,
            "style_guide": plan.style_guide,
            "task_template": plan.task_template,
            "taught_patterns": plan.taught_patterns,
            "user_turn": call.user_turn,
            "input_fingerprint": call.input_fingerprint,
            "force_fields": list(plan.force_fields),
            "batch_plan_fingerprint": plan.plan_fingerprint,
        },
        "current_record": call.record.to_dict(),
    }


def _result_manifest(
    request: Mapping[str, Any],
    *,
    parsed: Any,
    state: str,
    changed_fields: Sequence[str],
    warnings: Sequence[str],
    staging_sha256: str | None,
) -> dict[str, Any]:
    result = dict(request)
    result["state"] = "result"
    result["result"] = {
        "answer": _plain(parsed),
        "outcome": state,
        "changed_fields": list(changed_fields),
        "warnings": list(warnings),
        "staging_sha256": staging_sha256,
    }
    return result


def _meta(
    plan: AiEnrichmentPlan,
    call: AiEnrichmentCallPlan,
    outcome: enrich.AiOutcome,
) -> dict[str, Any]:
    fields = sorted(outcome.changes)
    provenance: dict[str, Any] = {
        "version": 1,
        "model": plan.model,
        "provider": plan.provider,
        "request_fingerprints": {
            call.record_id: call.request_fingerprint,
        },
        "input_fingerprints": {
            call.record_id: call.input_fingerprint,
        },
        "fields": {call.record_id: fields},
    }
    if plan.focus_resource_id is not None:
        provenance["focus_resource_id"] = plan.focus_resource_id
    return {
        "source_file": plan.canonical_path.name,
        "extracted_at": date.today().isoformat(),
        "model": plan.model,
        "provider": plan.provider,
        "review_run_id": call.operation_id,
        staging.AI_ENRICHMENT_KEY: provenance,
        "field_replacements": staging.field_replacement_block(
            [call.record],
            {call.record_id: outcome.changes},
        ),
        "review_notes": (
            f"1 record enriched by {plan.model}. This record already exists; "
            "promoting merges only the reviewed proposed fields into it."
        ),
    }


def _same_staging(
    path: Path,
    record: VocabularyRecord,
    meta: Mapping[str, Any],
) -> bool:
    try:
        records, observed = staging.read_staging(path)
    except (FileNotFoundError, JankiError, OSError, UnicodeError):
        return False
    return [item.to_dict() for item in records] == [record.to_dict()] and observed == meta


def _persist_result(
    plan: AiEnrichmentPlan,
    call: AiEnrichmentCallPlan,
    *,
    request_manifest: Mapping[str, Any],
    manifest_revision: str,
    parsed: Any,
    outcome: enrich.AiOutcome,
    warnings: Sequence[str],
) -> AiEnrichmentItemResult:
    changed_fields = tuple(sorted(outcome.changes))
    state: Literal["staged", "no_changes"] = "staged" if changed_fields else "no_changes"
    staging_sha256: str | None = None
    if changed_fields:
        meta = _meta(plan, call, outcome)
        if call.staging_path.exists() or call.staging_path.is_symlink():
            if not _same_staging(call.staging_path, outcome.record, meta):
                raise AiEnrichmentError(
                    f"AI-enrichment staging target {call.staging_path} is occupied "
                    "by different content."
                )
        else:
            staging.write_staging_under_lock(
                call.staging_path,
                [outcome.record],
                meta,
                expected_absent=True,
            )
        staging_sha256 = _sha(read_bytes_bound(call.staging_path))

    final = _manifest_text(
        _result_manifest(
            request_manifest,
            parsed=parsed,
            state=state,
            changed_fields=changed_fields,
            warnings=warnings,
            staging_sha256=staging_sha256,
        )
    )
    current = read_bytes_bound(call.request_manifest_path)
    if current != final.encode("utf-8"):
        if _sha(current) != manifest_revision:
            raise AiEnrichmentError(
                "The durable AI-enrichment request manifest changed before its "
                "result could be recorded."
            )
        atomic_write_text_bound(
            call.request_manifest_path,
            final,
            expected_revision=manifest_revision,
        )
    return AiEnrichmentItemResult(
        operation_id=call.operation_id,
        record_id=call.record_id,
        request_manifest_path=call.request_manifest_path,
        staging_path=call.staging_path if changed_fields else None,
        state=state,
        changed_fields=changed_fields,
        warnings=tuple(warnings),
    )


def _require_current_record(
    config: ProjectConfig,
    expected: AiEnrichmentCallPlan,
) -> None:
    """Revalidate the one canonical record while its path lock is held."""

    try:
        records, revision = load_records_snapshot(config.normalized_file)
    except JankiError as exc:
        raise AiEnrichmentError(f"Could not re-read the selected canonical card: {exc}") from exc
    if revision.text is None:
        raise AiEnrichmentError("The canonical collection disappeared while the provider answered.")
    matches = [record for record in records if record.id == expected.record_id]
    if len(matches) != 1 or matches[0].to_dict() != expected.record.to_dict():
        raise AiEnrichmentError(
            "The selected card changed while the provider answered; the exact "
            "reply was captured and no proposal was staged."
        )


def _run_call(
    config: ProjectConfig,
    plan: AiEnrichmentPlan,
    expected: AiEnrichmentCallPlan,
    *,
    client: Any | None,
    provider_env: Mapping[str, str] | None,
    provider_runner: Callable[..., Any],
    provider_which: Callable[..., str | None],
    provider_spawn: revision_provider.Spawn,
    api_call: Callable[..., Any] | None,
    progress: Callable[[str], None] | None,
) -> AiEnrichmentItemResult:
    if progress is not None:
        progress("Reading the source")
    fresh = _fresh_call(config, plan, expected, acquire_locks=True)
    if not fresh.can_dispatch:
        raise AiEnrichmentError(
            "This AI-enrichment operation's manifest or staging target is occupied."
        )
    provider = revision_provider.provider_for(fresh.provider_plan.provider)
    prepared = provider.prepare(
        fresh.provider_plan,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
        client=client,
    )

    prompt_paths = (
        prompts.path_for(config.root, "style-guide").absolute(),
        prompts.path_for(config.root, "enrich-bare-word").absolute(),
    )
    with contextlib.ExitStack() as locks:
        bound = {
            config.normalized_file.absolute(),
            config.patterns_file.absolute(),
            fresh.request_manifest_path,
            fresh.staging_path,
            *prompt_paths,
        }
        for path in sorted(bound, key=lambda item: str(item)):
            locks.enter_context(exclusive_path_lock(path))
        fresh = _fresh_call(config, plan, expected, acquire_locks=False)
        if not fresh.can_dispatch:
            raise AiEnrichmentError(
                "This AI-enrichment operation's manifest or staging target is occupied."
            )
        operations.prepare_artifact_store(config.operations_file)
        prepare_bound_directory(config.staging_dir)
        journal = operations.OperationJournal.load(config.operations_file)
        try:
            journal.authorize(
                fresh.operation_id,
                kind="enrich",
                source_file=f"{plan.canonical_relative_path}#{fresh.record_id}",
                source_sha256=fresh.record_sha256,
                request_fp=fresh.request_fingerprint,
                model=plan.model,
            )
        except Exception as exc:  # noqa: BLE001 - authority may have landed
            raise AiEnrichmentRunError(
                f"AI-enrichment authority could not be recorded safely: {exc} "
                "Inspect janki operations before retrying.",
                operation_id=fresh.operation_id,
                provider_dispatched=False,
            ) from exc
        request_manifest = _request_manifest(plan, fresh)
        request_text = _manifest_text(request_manifest)
        manifest_revision = _sha(request_text.encode("utf-8"))
        try:
            atomic_write_text_bound(
                fresh.request_manifest_path,
                request_text,
                expected_absent=True,
            )
        except Exception as exc:  # noqa: BLE001 - publication may have landed
            raise operations.cancel_before_send(
                config.operations_file,
                fresh.operation_id,
                error=AiEnrichmentRunError,
                label="AI enrichment",
                detail=(
                    "The durable AI-enrichment request manifest could not be prepared."
                ),
                cause=exc,
            ) from exc

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
                    detail="The AI-enrichment dispatch boundary could not be recorded.",
                )
        except JankiError:
            pass
        raise AiEnrichmentRunError(
            f"AI enrichment {fresh.operation_id} was not dispatched because its "
            f"dispatch boundary could not be recorded: {exc}",
            operation_id=fresh.operation_id,
            provider_dispatched=False,
        ) from exc

    try:

        def capture(raw_reply: bytes) -> None:
            if not isinstance(raw_reply, bytes):
                raise operations.OperationError(
                    "AI-enrichment provider capture must supply exact response bytes."
                )
            journal.capture_result(
                fresh.operation_id,
                lambda: operations.capture_artifact(
                    config.operations_file,
                    fresh.operation_id,
                    raw_reply,
                ),
            )
            if progress is not None:
                progress("Checking the answer's shape")

        call_result = provider.dispatch(
            prepared,
            capture=capture,
            spawn=provider_spawn,
            api_call=api_call,
        )
        captured = operations.OperationJournal.load(config.operations_file).operations.get(
            fresh.operation_id
        )
        if captured is None or captured.state != "result_captured":
            raise operations.OperationError(
                "The AI-enrichment provider returned data without durably "
                "capturing its exact reply."
            )
        parsed = getattr(call_result, "parsed", None)
        if parsed is None:
            reason = str(getattr(call_result, "stop_reason", "") or "unknown")
            raise AiEnrichmentError(
                f"{plan.model} returned no complete enrichment ({reason}). Its "
                "exact reply was captured, but no proposal was staged."
            )
        outcome = enrich.apply_ai_result(
            fresh.record,
            parsed,
            force_fields=plan.force_fields,
        )
        warnings = tuple(outcome.romaji_rejected)
        if outcome.preserved:
            warnings += (
                "Stored examples were preserved; generated sentences that did not "
                "fill an unoccupied polite/casual slot were discarded.",
            )
        if progress is not None:
            progress("Saving proposals")
        with contextlib.ExitStack() as output_locks:
            for path in sorted(
                {
                    config.normalized_file.absolute(),
                    fresh.request_manifest_path,
                    fresh.staging_path,
                },
                key=lambda item: str(item),
            ):
                output_locks.enter_context(exclusive_path_lock(path))
            _require_current_record(config, fresh)
            if fresh.staging_path.exists() or fresh.staging_path.is_symlink():
                raise AiEnrichmentError(
                    "The AI-enrichment staging target became occupied while the "
                    "provider answered; the exact reply was captured and nothing "
                    "was overwritten."
                )
            persisted: AiEnrichmentItemResult | None = None

            def persist() -> None:
                nonlocal persisted
                persisted = _persist_result(
                    plan,
                    fresh,
                    request_manifest=request_manifest,
                    manifest_revision=manifest_revision,
                    parsed=parsed,
                    outcome=outcome,
                    warnings=warnings,
                )

            journal.commit_result(
                fresh.operation_id,
                persist,
            )
            if persisted is None:  # pragma: no cover - OperationJournal contract
                raise operations.OperationError(
                    "AI-enrichment result commit did not persist its destination."
                )
            result = persisted
    except Exception as exc:  # noqa: BLE001 - post-dispatch state must be truthful
        try:
            classify_dispatch_failure(config, journal, fresh.operation_id, exc)
        except JankiError as journal_error:
            raise AiEnrichmentRunError(
                f"AI enrichment {fresh.operation_id} failed after dispatch: {exc} "
                f"Janki could not settle its journal entry: {journal_error}. This "
                "call may have been billed; inspect janki operations before retrying.",
                operation_id=fresh.operation_id,
                provider_dispatched=True,
            ) from exc
        raise AiEnrichmentRunError(
            f"AI enrichment {fresh.operation_id} failed after dispatch: {exc}",
            operation_id=fresh.operation_id,
            provider_dispatched=True,
        ) from exc
    return result


def run_ai_enrichment(
    config: ProjectConfig,
    expected: AiEnrichmentPlan,
    *,
    client: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: revision_provider.Spawn = subprocess.Popen,
    api_call: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AiEnrichmentRunResult:
    """Consume one exact batch authority, sequentially capturing and staging calls."""

    if not isinstance(expected, AiEnrichmentPlan):
        raise AiEnrichmentError("AI enrichment needs a typed application plan.")
    if expected.plan_fingerprint != _plan_identity(expected):
        raise AiEnrichmentError("The AI-enrichment batch plan no longer matches its fingerprint.")
    results: list[AiEnrichmentItemResult] = []
    for call in expected.calls:
        try:
            results.append(
                _run_call(
                    config,
                    expected,
                    call,
                    client=client,
                    provider_env=provider_env,
                    provider_runner=provider_runner,
                    provider_which=provider_which,
                    provider_spawn=provider_spawn,
                    api_call=api_call,
                    progress=progress,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve an already-durable prefix
            if not results:
                raise
            count = len(results)
            destinations = ", ".join(
                _relative(
                    expected.repository_root,
                    item.staging_path or item.request_manifest_path,
                    label="Completed AI-enrichment destination",
                )
                for item in results
            )
            durable = (
                "call reached its durable destination"
                if count == 1
                else "calls reached their durable destinations"
            )
            operation_id = getattr(exc, "operation_id", call.operation_id)
            dispatched = bool(getattr(exc, "provider_dispatched", False))
            raise AiEnrichmentRunError(
                f"{count} earlier enrichment {durable} at {destinations}; that "
                "completed work remains valid and must not "
                f"be repeated. The current call then failed: {exc}",
                operation_id=operation_id,
                provider_dispatched=dispatched,
                completed_results=results,
            ) from exc
    return AiEnrichmentRunResult(
        plan_fingerprint=expected.plan_fingerprint,
        results=tuple(results),
    )
