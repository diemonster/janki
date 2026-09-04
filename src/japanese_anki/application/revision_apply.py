"""Owner-authorized application of one staged conjugation-deck revision."""

from __future__ import annotations

import contextlib
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from japanese_anki import ai_schema, claude_client, ledger, operations, status
from japanese_anki.application import revision as revision_application
from japanese_anki.application import revision_provider
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import (
    YAML_LOADER,
    DataError,
    atomic_unlink_bound,
    atomic_write_bytes_bound,
    atomic_write_text_bound,
    exclusive_path_lock,
    prepare_bound_directory,
    read_bytes_bound,
)
from japanese_anki.models import ExampleSentence

__all__ = [
    "RevisionApplyError",
    "RevisionApplyPhase",
    "RevisionApplyPlan",
    "RevisionApplyProgress",
    "RevisionApplyResult",
    "execute_revision_apply",
    "execute_revision_apply_locked",
    "plan_revision_apply",
    "plan_revision_apply_archived_authority",
    "plan_revision_apply_recovery",
]

RevisionApplyPhase = Literal[
    "Preparing proposal",
    "Re-reading proposal",
    "Applying revision",
    "Archiving proposal",
]
RevisionApplyProgress = Callable[[RevisionApplyPhase], None]


class RevisionApplyError(JankiError):
    """A staged revision cannot be applied without widening its authority."""


@dataclass(frozen=True, slots=True)
class RevisionApplyPlan:
    repository_root: Path
    staging_path: Path
    live_sha256: str
    proposal_sha256: str
    state: str
    operation_id: str
    request_fingerprint: str
    provider: str
    billing_class: str
    auth_metadata: Mapping[str, Any]
    transport_metadata: Mapping[str, Any]
    cli_version: str | None
    request_bytes_sha256: str
    model: str
    deck_path: Path
    deck_relative_path: str
    deck_base_sha256: str
    selected_record_ids: tuple[str, ...]
    canonical_context_fingerprint: str
    current_form_note: str
    current_drill_examples: Mapping[str, tuple[ExampleSentence, ...]]
    form_note: str
    drill_examples: Mapping[str, tuple[ExampleSentence, ...]]
    intended_deck_text: str
    intended_deck_sha256: str
    archive_path: Path
    plan_fingerprint: str

    @property
    def billing_display(self) -> str:
        return revision_provider.billing_display(
            self.provider, self.billing_class, self.auth_metadata
        )


@dataclass(frozen=True, slots=True)
class RevisionApplyResult:
    operation_id: str
    deck_path: Path
    deck_sha256: str
    archive_path: Path
    recovered: bool


_TOP_PROPOSED = {
    "schema_version",
    "kind",
    "state",
    "operation_id",
    "target",
    "request",
    "canonical_context",
    "before",
    "staged_at",
    "proposal",
}
_TARGET = {"deck_path", "deck_sha256", "form", "selected_record_ids", "staging_path"}
_REQUEST = {
    "provider_plan",
    "owner_instruction",
    "style_guide",
    "task_template",
    "system_blocks",
    "user_turn",
    "plan_fingerprint",
    "canonical_context_fingerprint",
}
_ACCEPTANCE = {
    "authority",
    "accepted_at",
    "plan_fingerprint",
    "proposal_sha256",
    "intended_deck_sha256",
    "intended_deck_text",
}
_BEFORE_EXAMPLE = {
    "japanese",
    "furigana",
    "romaji",
    "english",
    "register",
    "audio",
    "spoken_japanese",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RevisionApplyError(f"Revision JSON repeats key {key!r}.")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise RevisionApplyError(f"Revision JSON contains non-finite number {value}.")


def _strict_json(wire: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            wire.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except RevisionApplyError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RevisionApplyError(f"Could not parse revision {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RevisionApplyError(f"Revision {path} must contain a JSON object.")
    return value


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RevisionApplyError(f"Revision has invalid {label} fields.")
    return value


def _contained_file(path: Path, root: Path, label: str) -> Path:
    target = path.absolute()
    try:
        target.relative_to(root.absolute())
    except ValueError as exc:
        raise RevisionApplyError(f"{label} escapes {root}: {target}") from exc
    return target


def _staging_target(config: ProjectConfig, staging_path: Path | str) -> Path:
    target = _contained_file(Path(staging_path), config.staging_dir, "Revision proposal")
    if target.parent != config.staging_dir.absolute() or target.suffix != ".json":
        raise RevisionApplyError(
            f"Revision proposal must be one JSON file directly under {config.staging_dir}."
        )
    return target


def _deck_target(config: ProjectConfig, relative: Any) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute():
        raise RevisionApplyError("Revision target deck_path must be repository-relative.")
    pure = PurePosixPath(relative)
    if ".." in pure.parts:
        raise RevisionApplyError("Revision target deck_path may not traverse directories.")
    target = _contained_file(config.root / Path(*pure.parts), config.deck_dir, "Revision deck")
    if target.parent != config.deck_dir.absolute():
        raise RevisionApplyError(f"Revision deck must be one direct configured deck file: {target}")
    if config.deck_dir.is_symlink() or target.is_symlink():
        raise RevisionApplyError(f"Revision deck path may not contain a symlink: {target}")
    try:
        configured = {path.absolute() for path in status.deck_files(config)}
    except (JankiError, OSError) as exc:
        raise RevisionApplyError(
            f"Could not resolve the configured deck set: {exc}"
        ) from exc
    if target not in configured:
        raise RevisionApplyError(
            f"Revision target is not one configured deck: {target}"
        )
    return target, pure.as_posix()


def _examples(
    proposal: Mapping[str, Any], selected: tuple[str, ...]
) -> tuple[str, dict[str, tuple[ExampleSentence, ...]]]:
    proposal = _exact(proposal, {"form_note", "drill_examples"}, "proposal")
    note = proposal["form_note"]
    raw = proposal["drill_examples"]
    if not isinstance(note, str) or not isinstance(raw, Mapping) or set(raw) != set(selected):
        raise RevisionApplyError("Revision proposal does not exactly match selected IDs.")
    parsed: dict[str, tuple[ExampleSentence, ...]] = {}
    for record_id in selected:
        rows = raw[record_id]
        if not isinstance(rows, list) or len(rows) != 2:
            raise RevisionApplyError(f"Revision {record_id!r} needs exactly two examples.")
        values: list[ExampleSentence] = []
        for row in rows:
            row = _exact(row, {"japanese", "furigana", "english", "register"}, "example")
            if any(not isinstance(value, str) for value in row.values()):
                raise RevisionApplyError("Revision example fields must be text.")
            if any(not str(row[field]).strip() for field in ("japanese", "furigana", "english")):
                raise RevisionApplyError(
                    "Revision examples need nonblank japanese, furigana, and english."
                )
            values.append(ExampleSentence.from_dict(dict(row)))
        counts = Counter(value.register for value in values)
        if counts != Counter({"polite": 1, "casual": 1}):
            raise RevisionApplyError(
                f"Revision {record_id!r} needs one polite and one casual example."
            )
        parsed[record_id] = tuple(values)
    return note, parsed


def _before_content(
    before: Any, selected: tuple[str, ...]
) -> tuple[str, dict[str, tuple[ExampleSentence, ...]]]:
    before = _exact(before, {"form_note", "drill_examples"}, "base content")
    note = before["form_note"]
    raw = before["drill_examples"]
    if not isinstance(note, str) or not isinstance(raw, Mapping) or set(raw) != set(selected):
        raise RevisionApplyError("Revision base content does not match selected IDs.")
    parsed: dict[str, tuple[ExampleSentence, ...]] = {}
    for record_id in selected:
        rows = raw[record_id]
        if not isinstance(rows, list) and not isinstance(rows, tuple):
            raise RevisionApplyError("Revision base examples must be a sequence.")
        values: list[ExampleSentence] = []
        for row in rows:
            row = _exact(row, _BEFORE_EXAMPLE, "base example")
            if any(not isinstance(value, str) for value in row.values()):
                raise RevisionApplyError("Revision base example fields must be text.")
            values.append(ExampleSentence.from_dict(dict(row)))
        counts = Counter(value.register for value in values)
        if len(values) != 2 or counts != Counter({"polite": 1, "casual": 1}):
            raise RevisionApplyError("Revision base needs one polite and one casual example.")
        parsed[record_id] = tuple(values)
    return note, parsed


def _validate_before_snapshot(
    content: pattern_cards.DrillDeckContent,
    selected: tuple[str, ...],
    note: str,
    examples: Mapping[str, tuple[ExampleSentence, ...]],
) -> None:
    current_selected = {record_id: content.drill_examples[record_id] for record_id in selected}
    if content.form_note != note or current_selected != examples:
        raise RevisionApplyError("Revision base values no longer match the bound deck.")


def _stored_provider_plan(
    request: Mapping[str, Any],
) -> revision_provider.RevisionProviderPlan:
    """Purely reproduce the exact paid/capped provider request from staging."""
    text_fields = (
        "owner_instruction",
        "style_guide",
        "task_template",
        "user_turn",
        "plan_fingerprint",
        "canonical_context_fingerprint",
    )
    if any(not isinstance(request.get(name), str) for name in text_fields):
        raise RevisionApplyError(
            "[revision-request-provenance] revision request identity fields must be text."
        )
    blocks = request.get("system_blocks")
    if not isinstance(blocks, list):
        raise RevisionApplyError(
            "[revision-request-provenance] revision system blocks must be a list."
        )
    expected_blocks = claude_client.system_blocks(
        request["style_guide"], request["task_template"]
    )
    if blocks != expected_blocks:
        raise RevisionApplyError(
            "[revision-request-provenance] revision system blocks do not reproduce."
        )
    provider_manifest = _exact(
        request.get("provider_plan"),
        set(revision_provider.PERSISTENT_MANIFEST_KEYS),
        "provider plan",
    )
    model = provider_manifest.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RevisionApplyError(
            "[revision-request-provenance] revision provider model must be nonblank text."
        )
    try:
        return revision_provider.provider_plan_from_manifest(
            provider_manifest,
            model=model,
            style_guide=request["style_guide"],
            task_template=request["task_template"],
            system_blocks=blocks,
            user_turn=request["user_turn"],
            schema=ai_schema.conjugation_deck_revision_schema(),
        )
    except revision_provider.RevisionProviderError as exc:
        raise RevisionApplyError(
            f"[revision-request-provenance] {exc}"
        ) from exc


def _verify_proposed_authority(
    config: ProjectConfig,
    plan: RevisionApplyPlan,
    manifest: Mapping[str, Any],
) -> None:
    """Reproduce the paid request and current context before owner acceptance."""
    target = _exact(manifest.get("target"), _TARGET, "target")
    request = _exact(manifest.get("request"), _REQUEST, "request")
    before = _exact(manifest.get("before"), {"form_note", "drill_examples"}, "base content")
    context = manifest.get("canonical_context")
    provider_plan = _stored_provider_plan(request)
    if _sha(_canonical(context).encode("utf-8")) != request[
        "canonical_context_fingerprint"
    ]:
        raise RevisionApplyError(
            "[revision-request-provenance] revision content fingerprints do not reproduce."
        )
    raw_examples = before["drill_examples"]
    assert isinstance(raw_examples, Mapping)  # established by _before_content
    assert isinstance(context, Mapping)  # established by _load_plan
    rebuilt_turn = revision_application._user_turn(
        deck_name=Path(plan.deck_relative_path).name,
        form=target["form"],
        form_note=before["form_note"],
        selected=plan.selected_record_ids,
        instruction=request["owner_instruction"],
        current={
            record_id: tuple(raw_examples[record_id])
            for record_id in plan.selected_record_ids
        },
        context={
            record_id: context[record_id]
            for record_id in plan.selected_record_ids
        },
    )
    rebuilt_plan = revision_application._plan_identity(
        deck_relative=plan.deck_relative_path,
        deck_sha256=plan.deck_base_sha256,
        selected=plan.selected_record_ids,
        owner_instruction=request["owner_instruction"],
        provider=provider_plan.provider,
        model=provider_plan.model,
        canonical_context_fingerprint=request["canonical_context_fingerprint"],
        request_fingerprint=provider_plan.request_fingerprint,
        staging_relative=plan.staging_path.relative_to(
            config.root.absolute()
        ).as_posix(),
        staging_revision=None,
    )
    if (
        rebuilt_turn != request["user_turn"]
        or rebuilt_plan != request["plan_fingerprint"]
        or provider_plan.request_fingerprint != plan.request_fingerprint
        or provider_plan.model != plan.model
        or provider_plan.provider != plan.provider
    ):
        raise RevisionApplyError(
            "[revision-request-provenance] revision request and plan fingerprints "
            "do not reproduce."
        )
    try:
        current_context = revision_application._canonical_context(
            config,
            plan.selected_record_ids,
            target["form"],
        )
    except (JankiError, OSError) as exc:
        raise RevisionApplyError(
            f"[revision-canonical-context-stale] could not reload the selected "
            f"canonical records: {exc}"
        ) from exc
    if current_context != context:
        raise RevisionApplyError(
            "[revision-canonical-context-stale] selected canonical record context "
            "changed after the paid revision request."
        )


def _plan_fingerprint(
    *,
    proposal_sha: str,
    operation_id: str,
    deck_relative: str,
    base_sha: str,
    selected: tuple[str, ...],
    context_fp: str,
    intended_sha: str,
    archive_relative: str,
) -> str:
    return _sha(
        _canonical(
            {
                "version": 1,
                "proposal_sha256": proposal_sha,
                "operation_id": operation_id,
                "deck_path": deck_relative,
                "deck_base_sha256": base_sha,
                "selected_record_ids": selected,
                "canonical_context_fingerprint": context_fp,
                "intended_deck_sha256": intended_sha,
                "archive_path": archive_relative,
            }
        ).encode("utf-8")
    )


def _read_optional(path: Path) -> tuple[bytes | None, str | None]:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None, None
    except (DataError, OSError) as exc:
        raise RevisionApplyError(f"Could not safely read {path}: {exc}") from exc
    return wire, _sha(wire)


def _load_plan(
    config: ProjectConfig,
    staging_path: Path,
    *,
    artifact_path: Path | None = None,
    validate_accepted_deck: bool = True,
) -> RevisionApplyPlan:
    read_path = staging_path if artifact_path is None else artifact_path
    wire, live_sha = _read_optional(read_path)
    if wire is None or live_sha is None:
        raise RevisionApplyError(f"Revision proposal no longer exists: {read_path}")
    manifest = _strict_json(wire, read_path)
    state = manifest.get("state")
    top = _TOP_PROPOSED if state == "proposed" else _TOP_PROPOSED | {"acceptance"}
    _exact(manifest, top, "top-level")
    if manifest.get("schema_version") != 2 or manifest.get("kind") != "conjugation_deck_revision":
        raise RevisionApplyError("Unsupported revision artifact.")
    if state not in {"proposed", "accepted"}:
        raise RevisionApplyError("Only a proposed or durably accepted revision can apply.")
    try:
        operation_id = str(uuid.UUID(str(manifest["operation_id"])))
    except (ValueError, AttributeError) as exc:
        raise RevisionApplyError("Revision operation_id must be a UUID.") from exc
    if manifest["operation_id"] != operation_id:
        raise RevisionApplyError("Revision operation_id must be a canonical UUID.")
    target = _exact(manifest["target"], _TARGET, "target")
    request = _exact(manifest["request"], _REQUEST, "request")
    provider_plan = _stored_provider_plan(request)
    deck_path, deck_relative = _deck_target(config, target["deck_path"])
    staging_relative = staging_path.relative_to(config.root.absolute()).as_posix()
    if target["staging_path"] != staging_relative:
        raise RevisionApplyError("Revision staging_path does not name this artifact.")
    base_sha = target["deck_sha256"]
    context_fp = request["canonical_context_fingerprint"]
    if not _is_sha(base_sha) or not _is_sha(context_fp):
        raise RevisionApplyError("Revision fingerprints must be SHA-256 text.")
    selected_raw = target["selected_record_ids"]
    if (
        not isinstance(selected_raw, list)
        or not selected_raw
        or any(not isinstance(value, str) or not value for value in selected_raw)
        or len(selected_raw) != len(set(selected_raw))
    ):
        raise RevisionApplyError("Revision selected_record_ids are malformed.")
    selected = tuple(selected_raw)
    context = manifest["canonical_context"]
    if not isinstance(context, Mapping) or set(context) != set(selected):
        raise RevisionApplyError("Revision canonical_context does not match selected IDs.")
    if _sha(_canonical(context).encode("utf-8")) != context_fp:
        raise RevisionApplyError("Revision canonical_context fingerprint is stale.")
    journal = operations.OperationJournal.load(config.operations_file)
    operation = journal.operations.get(operation_id)
    if operation is None:
        raise RevisionApplyError("Revision has no matching paid-operation journal entry.")
    if (
        operation.kind != "revise"
        or operation.state != "committed"
        or operation.cleanup is not None
        or operation.source_file != deck_relative
        or operation.source_sha256 != base_sha
        or operation.request_fp != provider_plan.request_fingerprint
        or operation.model != provider_plan.model
    ):
        raise RevisionApplyError(
            "Revision paid-operation journal identity does not match the proposal."
        )
    current_note, current_examples = _before_content(manifest["before"], selected)
    form_note, examples = _examples(manifest["proposal"], selected)
    archive = config.staging_dir / "done" / "revisions" / staging_path.name
    archive_relative = archive.relative_to(config.root.absolute()).as_posix()

    if state == "proposed":
        content = pattern_cards.read_drill_deck_content(deck_path)
        if _sha(content.revision.text.encode("utf-8")) != base_sha:
            raise RevisionApplyError("[revision-apply-stale] target deck changed.")
        if content.form != target["form"]:
            raise RevisionApplyError("Revision form no longer matches the deck.")
        _validate_before_snapshot(content, selected, current_note, current_examples)
        intended = pattern_cards.render_drill_deck_content(
            deck_path,
            expected=content.revision,
            form_note=form_note,
            drill_examples=examples,
        )
        intended_sha = _sha(intended.encode("utf-8"))
        proposal_sha = live_sha
        plan_fp = _plan_fingerprint(
            proposal_sha=proposal_sha,
            operation_id=operation_id,
            deck_relative=deck_relative,
            base_sha=base_sha,
            selected=selected,
            context_fp=context_fp,
            intended_sha=intended_sha,
            archive_relative=archive_relative,
        )
        archive_wire, _ = _read_optional(archive)
        if archive_wire is not None:
            raise RevisionApplyError(f"Revision archive already exists: {archive}")
    else:
        acceptance = _exact(manifest["acceptance"], _ACCEPTANCE, "acceptance")
        if acceptance["authority"] != "repository-owner":
            raise RevisionApplyError("Accepted revision lacks repository-owner authority.")
        if not isinstance(acceptance["accepted_at"], str):
            raise RevisionApplyError("Accepted revision timestamp is malformed.")
        try:
            datetime.fromisoformat(acceptance["accepted_at"])
        except ValueError as exc:
            raise RevisionApplyError("Accepted revision timestamp is malformed.") from exc
        intended = acceptance["intended_deck_text"]
        intended_sha = acceptance["intended_deck_sha256"]
        proposal_sha = acceptance["proposal_sha256"]
        plan_fp = acceptance["plan_fingerprint"]
        if not all(_is_sha(value) for value in (intended_sha, proposal_sha, plan_fp)):
            raise RevisionApplyError("Accepted revision fingerprints are malformed.")
        if not isinstance(intended, str) or _sha(intended.encode("utf-8")) != intended_sha:
            raise RevisionApplyError("Accepted revision intended deck is corrupt.")
        proposed_again = dict(manifest)
        proposed_again.pop("acceptance")
        proposed_again["state"] = "proposed"
        proposed_wire = (
            json.dumps(
                proposed_again,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if _sha(proposed_wire) != proposal_sha:
            raise RevisionApplyError("Accepted revision proposal content is corrupt.")
        calculated = _plan_fingerprint(
            proposal_sha=proposal_sha,
            operation_id=operation_id,
            deck_relative=deck_relative,
            base_sha=base_sha,
            selected=selected,
            context_fp=context_fp,
            intended_sha=intended_sha,
            archive_relative=archive_relative,
        )
        if calculated != plan_fp:
            raise RevisionApplyError("Accepted revision plan fingerprint is corrupt.")
        if validate_accepted_deck:
            current = pattern_cards.read_drill_deck_content(deck_path)
            current_sha = _sha(current.revision.text.encode("utf-8"))
            if current_sha not in {base_sha, intended_sha}:
                raise RevisionApplyError(
                    "Accepted revision deck diverged from both bound states."
                )
            if current_sha == base_sha:
                _validate_before_snapshot(
                    current, selected, current_note, current_examples
                )
        archive_wire, _ = _read_optional(archive)
        if archive_wire is not None and archive_wire != wire:
            raise RevisionApplyError("Revision archive differs from the accepted artifact.")
    return RevisionApplyPlan(
        repository_root=config.root.resolve(),
        staging_path=staging_path,
        live_sha256=live_sha,
        proposal_sha256=proposal_sha,
        state=state,
        operation_id=operation_id,
        deck_path=deck_path,
        request_fingerprint=provider_plan.request_fingerprint,
        provider=provider_plan.provider,
        billing_class=provider_plan.billing_class,
        auth_metadata=provider_plan.auth_metadata,
        transport_metadata=provider_plan.transport,
        cli_version=(
            str(provider_plan.transport["cli_version"])
            if "cli_version" in provider_plan.transport
            else None
        ),
        request_bytes_sha256=_sha(provider_plan.request_bytes),
        model=provider_plan.model,
        deck_relative_path=deck_relative,
        deck_base_sha256=base_sha,
        selected_record_ids=selected,
        canonical_context_fingerprint=context_fp,
        current_form_note=current_note,
        current_drill_examples=current_examples,
        form_note=form_note,
        drill_examples=examples,
        intended_deck_text=intended,
        intended_deck_sha256=intended_sha,
        archive_path=archive,
        plan_fingerprint=plan_fp,
    )


def plan_revision_apply(config: ProjectConfig, staging_path: Path | str) -> RevisionApplyPlan:
    """Plan one exact proposal apply without writing repository state."""
    return _load_plan(config, _staging_target(config, staging_path))


def plan_revision_apply_recovery(
    config: ProjectConfig,
    staging_path: Path | str,
    *,
    archive_path: Path | str,
    plan_fingerprint: str,
) -> RevisionApplyPlan:
    """Reconstruct one completed plan from its exact accepted archive binding."""
    logical_staging = _staging_target(config, staging_path)
    if not _is_sha(plan_fingerprint):
        raise RevisionApplyError("Revision recovery plan fingerprint is malformed.")
    live_wire, _ = _read_optional(logical_staging)
    if live_wire is not None:
        raise RevisionApplyError("Revision recovery requires the live proposal to be gone.")
    expected_archive = (config.staging_dir / "done" / "revisions" / logical_staging.name).absolute()
    bound_archive = _contained_file(
        Path(archive_path), config.staging_dir / "done" / "revisions", "Revision archive"
    )
    if bound_archive != expected_archive:
        raise RevisionApplyError("Revision recovery archive is not the logical proposal archive.")
    archive_wire, _ = _read_optional(bound_archive)
    if archive_wire is None:
        raise RevisionApplyError(f"Revision recovery archive no longer exists: {bound_archive}")
    recovered = _load_plan(
        config,
        logical_staging,
        artifact_path=bound_archive,
    )
    if (
        recovered.state != "accepted"
        or recovered.archive_path != bound_archive
        or recovered.plan_fingerprint != plan_fingerprint
    ):
        raise RevisionApplyError("Archived revision is not this accepted plan.")
    return recovered


def plan_revision_apply_archived_authority(
    config: ProjectConfig,
    staging_path: Path | str,
    *,
    archive_path: Path | str,
    plan_fingerprint: str,
) -> RevisionApplyPlan:
    """Reconstruct accepted authority after a caller-owned deck transaction.

    Unlike :func:`plan_revision_apply_recovery`, this does not assert that the
    canonical deck still equals the pre-audio revision bytes.  It is only for a
    higher-level transaction that separately validates every allowed post-apply
    deck evolution before proceeding.
    """

    logical_staging = _staging_target(config, staging_path)
    if not _is_sha(plan_fingerprint):
        raise RevisionApplyError("Archived revision plan fingerprint is malformed.")
    live_wire, _ = _read_optional(logical_staging)
    if live_wire is not None:
        raise RevisionApplyError(
            "Archived revision authority requires the live proposal to be gone."
        )
    expected_archive = (
        config.staging_dir / "done" / "revisions" / logical_staging.name
    ).absolute()
    bound_archive = _contained_file(
        Path(archive_path),
        config.staging_dir / "done" / "revisions",
        "Revision archive",
    )
    if bound_archive != expected_archive:
        raise RevisionApplyError(
            "Revision archive is not the logical proposal archive."
        )
    archive_wire, _ = _read_optional(bound_archive)
    if archive_wire is None:
        raise RevisionApplyError(
            f"Revision archive no longer exists: {bound_archive}"
        )
    recovered = _load_plan(
        config,
        logical_staging,
        artifact_path=bound_archive,
        validate_accepted_deck=False,
    )
    if (
        recovered.state != "accepted"
        or recovered.archive_path != bound_archive
        or recovered.plan_fingerprint != plan_fingerprint
    ):
        raise RevisionApplyError(
            "Archived revision is not this accepted authority."
        )
    return recovered


def _same_authority(expected: RevisionApplyPlan, fresh: RevisionApplyPlan) -> bool:
    return (
        expected.repository_root == fresh.repository_root
        and expected.staging_path == fresh.staging_path
        and expected.proposal_sha256 == fresh.proposal_sha256
        and expected.operation_id == fresh.operation_id
        and expected.request_fingerprint == fresh.request_fingerprint
        and expected.provider == fresh.provider
        and expected.billing_class == fresh.billing_class
        and expected.auth_metadata == fresh.auth_metadata
        and expected.transport_metadata == fresh.transport_metadata
        and expected.cli_version == fresh.cli_version
        and expected.request_bytes_sha256 == fresh.request_bytes_sha256
        and expected.model == fresh.model
        and expected.deck_path == fresh.deck_path
        and expected.deck_base_sha256 == fresh.deck_base_sha256
        and expected.selected_record_ids == fresh.selected_record_ids
        and expected.canonical_context_fingerprint == fresh.canonical_context_fingerprint
        and expected.current_form_note == fresh.current_form_note
        and expected.current_drill_examples == fresh.current_drill_examples
        and expected.intended_deck_sha256 == fresh.intended_deck_sha256
        and expected.archive_path == fresh.archive_path
        and expected.plan_fingerprint == fresh.plan_fingerprint
    )


def _refuse_pending_audio(config: ProjectConfig, plan: RevisionApplyPlan) -> None:
    content = pattern_cards.read_drill_deck_content(plan.deck_path)
    import yaml

    document = yaml.load(content.revision.text, Loader=YAML_LOADER)
    section = document["deck"]
    deck_id = section.get("deck_id")
    if isinstance(deck_id, bool) or not isinstance(deck_id, int):
        raise RevisionApplyError("Revision deck_id must be an integer.")
    owners = {
        pattern_cards.drill_audio_owner_id(deck_id, content.form, record_id)
        for record_id in content.record_ids
    }
    book = ledger.load(config.ledger_file)
    blocked = [
        key
        for key, entry in book.pending_audio.items()
        if isinstance(entry, Mapping) and entry.get("record_id") in owners
    ]
    if blocked:
        raise RevisionApplyError(
            "Revision deck has pending paid audio; finish that exact audio "
            "transaction before applying this proposal."
        )


def _accepted_wire(plan: RevisionApplyPlan, manifest: Mapping[str, Any]) -> str:
    value = dict(manifest)
    value["state"] = "accepted"
    value["acceptance"] = {
        "authority": "repository-owner",
        "accepted_at": datetime.now(UTC).isoformat(),
        "plan_fingerprint": plan.plan_fingerprint,
        "proposal_sha256": plan.proposal_sha256,
        "intended_deck_sha256": plan.intended_deck_sha256,
        "intended_deck_text": plan.intended_deck_text,
    }
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _emit_progress(
    progress: RevisionApplyProgress | None,
    phase: RevisionApplyPhase,
) -> None:
    """Progress is advisory and must never interrupt the durable transaction."""
    if progress is None:
        return
    try:
        progress(phase)
    except Exception:
        return


def execute_revision_apply(
    config: ProjectConfig,
    expected_plan: RevisionApplyPlan,
    *,
    progress: RevisionApplyProgress | None = None,
) -> RevisionApplyResult:
    """Consume an exact owner-confirmed plan and finish its recoverable CAS."""
    _emit_progress(progress, "Preparing proposal")
    if expected_plan.repository_root != config.root.resolve():
        raise RevisionApplyError("Revision apply plan belongs to another repository.")
    prepare_bound_directory(expected_plan.archive_path.parent)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        return execute_revision_apply_locked(config, expected_plan, progress=progress)


def execute_revision_apply_locked(
    config: ProjectConfig,
    expected_plan: RevisionApplyPlan,
    *,
    progress: RevisionApplyProgress | None = None,
) -> RevisionApplyResult:
    """Execute while the caller owns the repository audio-operation lock."""
    recovered = expected_plan.state == "accepted"
    with contextlib.ExitStack() as locks:
        for path in sorted(
            {
                expected_plan.deck_path,
                expected_plan.staging_path,
                expected_plan.archive_path,
                config.normalized_file,
            },
            key=lambda item: str(item.absolute()),
        ):
            locks.enter_context(exclusive_path_lock(path))
        _emit_progress(progress, "Re-reading proposal")
        live_wire, _ = _read_optional(expected_plan.staging_path)
        if live_wire is None:
            archive_wire, _ = _read_optional(expected_plan.archive_path)
            deck = pattern_cards.read_drill_deck_content(expected_plan.deck_path)
            journal = operations.OperationJournal.load(config.operations_file)
            operation = journal.operations.get(expected_plan.operation_id)
            if (
                archive_wire is None
                or operation is None
                or operation.kind != "revise"
                or operation.state != "committed"
                or operation.cleanup is not None
                or operation.source_file != expected_plan.deck_relative_path
                or operation.source_sha256 != expected_plan.deck_base_sha256
                or operation.request_fp != expected_plan.request_fingerprint
                or operation.model != expected_plan.model
                or _sha(deck.revision.text.encode("utf-8")) != expected_plan.intended_deck_sha256
            ):
                raise RevisionApplyError("Completed revision evidence is incomplete.")
            archived = _strict_json(archive_wire, expected_plan.archive_path)
            _exact(archived, _TOP_PROPOSED | {"acceptance"}, "archived top-level")
            archived_target = _exact(archived.get("target"), _TARGET, "archived target")
            archived_request = _exact(archived.get("request"), _REQUEST, "archived request")
            acceptance = _exact(archived.get("acceptance"), _ACCEPTANCE, "acceptance")
            proposed_again = dict(archived)
            proposed_again.pop("acceptance")
            proposed_again["state"] = "proposed"
            proposed_sha = _sha(
                (
                    json.dumps(
                        proposed_again,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            if (
                archived.get("state") != "accepted"
                or archived.get("operation_id") != expected_plan.operation_id
                or archived_target.get("deck_path") != expected_plan.deck_relative_path
                or archived_target.get("deck_sha256") != expected_plan.deck_base_sha256
                or tuple(archived_target.get("selected_record_ids") or ())
                != expected_plan.selected_record_ids
                or archived_target.get("staging_path")
                != expected_plan.staging_path.relative_to(config.root.absolute()).as_posix()
                or archived_request.get("canonical_context_fingerprint")
                != expected_plan.canonical_context_fingerprint
                or acceptance["authority"] != "repository-owner"
                or acceptance["plan_fingerprint"] != expected_plan.plan_fingerprint
                or acceptance["proposal_sha256"] != expected_plan.proposal_sha256
                or proposed_sha != expected_plan.proposal_sha256
                or acceptance["intended_deck_sha256"] != expected_plan.intended_deck_sha256
                or not isinstance(acceptance["intended_deck_text"], str)
                or _sha(acceptance["intended_deck_text"].encode("utf-8"))
                != expected_plan.intended_deck_sha256
            ):
                raise RevisionApplyError("Archived revision is not this confirmed plan.")
            return RevisionApplyResult(
                expected_plan.operation_id,
                expected_plan.deck_path,
                expected_plan.intended_deck_sha256,
                expected_plan.archive_path,
                True,
            )

        fresh = _load_plan(config, expected_plan.staging_path)
        if not _same_authority(expected_plan, fresh):
            raise RevisionApplyError("[revision-apply-plan-stale] reload the proposal.")
        _refuse_pending_audio(config, fresh)
        if fresh.state == "proposed":
            manifest = _strict_json(live_wire, fresh.staging_path)
            _verify_proposed_authority(config, fresh, manifest)
            accepted_text = _accepted_wire(fresh, manifest)
            atomic_write_text_bound(
                fresh.staging_path,
                accepted_text,
                expected_revision=fresh.live_sha256,
            )
            recovered = False
            fresh = _load_plan(config, fresh.staging_path)
        else:
            recovered = True

        _emit_progress(progress, "Applying revision")
        current = pattern_cards.read_drill_deck_content(fresh.deck_path)
        current_sha = _sha(current.revision.text.encode("utf-8"))
        if current_sha == fresh.deck_base_sha256:
            landed = pattern_cards.save_drill_deck_content(
                fresh.deck_path,
                expected=current.revision,
                form_note=fresh.form_note,
                drill_examples=fresh.drill_examples,
            )
            if _sha(landed.text.encode("utf-8")) != fresh.intended_deck_sha256:
                raise RevisionApplyError("Pattern writer produced a different deck.")
        elif current_sha != fresh.intended_deck_sha256:
            raise RevisionApplyError("Revision deck changed after acceptance.")

        _emit_progress(progress, "Archiving proposal")
        accepted_wire = read_bytes_bound(fresh.staging_path)
        archive_wire, _ = _read_optional(fresh.archive_path)
        if archive_wire is None:
            atomic_write_bytes_bound(fresh.archive_path, accepted_wire, expected_absent=True)
        elif archive_wire != accepted_wire:
            raise RevisionApplyError("Revision archive differs from accepted proposal.")
        accepted_sha = _sha(accepted_wire)
        atomic_unlink_bound(fresh.staging_path, expected_revision=accepted_sha)
        return RevisionApplyResult(
            fresh.operation_id,
            fresh.deck_path,
            fresh.intended_deck_sha256,
            fresh.archive_path,
            recovered,
        )
