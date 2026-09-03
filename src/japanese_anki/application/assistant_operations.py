"""Plan-bound Assistant access to the paid-operation journal.

This is a browser surface over :class:`~japanese_anki.operations.OperationJournal`,
not another recovery implementation.  Planning exposes only a safe projection of
one exact entry.  Execution passes the rendered operation back into the journal,
which compares it under the same lock used by the requested read or transition.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal

from japanese_anki import operations, prompts
from japanese_anki.application import assistant_agent, card_revision, revision
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError, read_bytes_bound

__all__ = [
    "AssistantOperationError",
    "OperationActionExecution",
    "OperationActionOption",
    "OperationActionPlan",
    "OperationChoice",
    "execute_operation_action",
    "list_operation_choices",
    "plan_operation_action",
]


OperationAction = Literal["recover", "show_reply", "end", "forget"]
RecoveryKind = Literal["assistant_agent", "card_revision", "conjugation_revision"]


class AssistantOperationError(JankiError):
    """An Assistant operation action is unavailable, stale, or unsafe."""


@dataclass(frozen=True, slots=True)
class OperationActionOption:
    """One exact local action the operation's current state permits."""

    action: OperationAction
    label: str
    accept_paid_output_loss: bool = False


@dataclass(frozen=True, slots=True)
class OperationChoice:
    """Safe browser-facing facts for one tracked paid operation."""

    operation_id: str
    kind: str
    state: str
    source_name: str
    model: str
    authorized_at: str
    blocks_spending: bool
    money_may_have_been_spent: bool
    has_captured_reply: bool
    has_response_spool: bool
    cleanup_pending: bool
    actions: tuple[OperationActionOption, ...]


@dataclass(frozen=True, slots=True)
class OperationActionPlan:
    """One exact operation and consequence rendered before owner authority."""

    repository_root: Path
    operations_path: Path
    operation: operations.Operation
    action: OperationAction
    force: bool
    projection_wire: str
    fingerprint: str
    recovery_kind: RecoveryKind | None = None
    manifest_path: Path | None = None
    result_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Operation action repository root must be canonical")
        if self.operations_path != _lexical(self.operations_path):
            raise ValueError("Operation action journal path must be lexical absolute")
        try:
            self.operations_path.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("Operation action journal must stay in its repository") from exc
        if self.action not in {"recover", "show_reply", "end", "forget"}:
            raise ValueError("Operation action kind is invalid")
        if not isinstance(self.force, bool):
            raise ValueError("Operation action force choice must be true or false")
        if not self.operation.operation_id:
            raise ValueError("Operation action must bind one operation")
        try:
            parsed = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Operation action projection must be JSON") from exc
        expected = {
            "schema_version": 1,
            "action": self.action,
            "force": self.force,
            "operation": _operation_projection(self.operation),
        }
        if self.action == "recover":
            if (
                self.recovery_kind is None
                or self.manifest_path is None
                or not self.result_paths
            ):
                raise ValueError("Operation recovery plan is incomplete")
            expected["recovery"] = _recovery_projection(
                self.operation,
                _RecoveryBinding(
                    kind=self.recovery_kind,
                    manifest_path=self.manifest_path,
                    manifest_sha256=_sha256(read_bytes_bound(self.manifest_path)),
                    result_paths=self.result_paths,
                ),
            )
        elif (
            self.recovery_kind is not None
            or self.manifest_path is not None
            or self.result_paths
        ):
            raise ValueError("Only a recover action may bind recovery output")
        if parsed != expected or _canonical_json(parsed) != self.projection_wire:
            raise ValueError("Operation action projection does not bind its plan")
        if _sha256(self.projection_wire.encode("utf-8")) != self.fingerprint:
            raise ValueError("Operation action fingerprint does not bind its projection")

    @property
    def projection(self) -> Mapping[str, Any]:
        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class OperationActionExecution:
    """Result of one journal-owned read or transition."""

    plan: OperationActionPlan
    operation: operations.Operation | None
    forgotten: int = 0
    reply: bytes | None = None
    reply_sha256: str | None = None
    private: bool = False
    recovery_kind: RecoveryKind | None = None
    result_names: tuple[str, ...] = ()
    result_sha256: tuple[str, ...] = ()
    assistant_answer: str | None = None
    assistant_action_intent_count: int = 0


@dataclass(frozen=True, slots=True)
class _RecoveryBinding:
    kind: RecoveryKind
    manifest_path: Path
    manifest_sha256: str
    result_paths: tuple[Path, ...]


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
        raise AssistantOperationError(
            f"The paid-operation action could not be fingerprinted: {exc}"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_name(value: str) -> str:
    name = PurePath(value.replace("\\", "/")).name if value else ""
    return name if name not in {".", ".."} else ""


def _requires_paid_output_loss(operation: operations.Operation) -> bool:
    return (
        operation.cleanup is None
        and operation.state != "committed"
        and (
            operation.artifact is not None
            or (
                operation.response_spool is not None
                and operation.response_spool.frame_count > 0
            )
        )
    )


def _action_options(
    operation: operations.Operation,
    *,
    recoverable: bool = False,
) -> tuple[OperationActionOption, ...]:
    values: list[OperationActionOption] = []
    if recoverable:
        values.append(
            OperationActionOption(
                action="recover",
                label="Recover captured result",
            )
        )
    if operation.artifact is not None or (
        operation.response_spool is not None
        and operation.response_spool.frame_count > 0
    ):
        values.append(
            OperationActionOption(
                action="show_reply",
                label="Show exact recovery reply",
            )
        )
    if operation.cleanup is None and operation.state in {
        "authorized",
        "dispatching",
        "running",
    }:
        values.append(
            OperationActionOption(
                action="end",
                label="End this operation",
            )
        )
    if operation.cleanup is not None or operation.state in (
        operations.TERMINAL_STATES | {"result_captured"}
    ):
        loses_paid_output = _requires_paid_output_loss(operation)
        values.append(
            OperationActionOption(
                action="forget",
                label=(
                    "Discard captured reply and forget"
                    if loses_paid_output
                    else "Forget this finished operation"
                ),
                accept_paid_output_loss=loses_paid_output,
            )
        )
    return tuple(values)


def list_operation_choices(config: ProjectConfig) -> tuple[OperationChoice, ...]:
    """List only blocking or recovery-bearing operations, oldest first."""

    try:
        journal = operations.OperationJournal.load(config.operations_file)
    except (operations.OperationError, OSError, ValueError) as exc:
        raise AssistantOperationError(str(exc)) from exc
    values: list[OperationChoice] = []
    for operation in sorted(
        journal.tracked(),
        key=lambda item: (
            not item.blocks_spending,
            item.authorized_at,
            item.operation_id,
        ),
    ):
        recoverable = False
        if operation.state == "result_captured" and operation.artifact is not None:
            try:
                _recovery_binding(config, operation)
            except AssistantOperationError:
                pass
            else:
                recoverable = True
        values.append(
            OperationChoice(
                operation_id=operation.operation_id,
                kind=operation.kind,
                state=operation.state,
                source_name=_safe_name(operation.source_file) or "unnamed source",
                model=operation.model,
                authorized_at=operation.authorized_at,
                blocks_spending=operation.blocks_spending,
                money_may_have_been_spent=operation.money_may_have_been_spent,
                has_captured_reply=operation.artifact is not None,
                has_response_spool=operation.response_spool is not None,
                cleanup_pending=operation.cleanup is not None,
                actions=_action_options(operation, recoverable=recoverable),
            )
        )
    return tuple(values)


def _operation_projection(operation: operations.Operation) -> dict[str, object]:
    exact_wire = _canonical_json(
        {
            "operation_id": operation.operation_id,
            "value": operation.to_dict(),
        }
    )
    artifact = operation.artifact
    spool = operation.response_spool
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "state": operation.state,
        "source_name": _safe_name(operation.source_file) or "unnamed source",
        "model": operation.model,
        "authorized_at": operation.authorized_at,
        "updated_at": operation.updated_at,
        "money_may_have_been_spent": operation.money_may_have_been_spent,
        "blocks_spending": operation.blocks_spending,
        "has_captured_reply": artifact is not None,
        "captured_reply_bytes": artifact.entry_state[2] if artifact is not None else None,
        "captured_reply_sha256": (
            artifact.content_sha256 if artifact is not None else None
        ),
        "has_response_spool": spool is not None,
        "committed_response_frames": spool.frame_count if spool is not None else 0,
        "committed_response_bytes": spool.committed_size if spool is not None else 0,
        "cleanup_pending": operation.cleanup is not None,
        "operation_revision": _sha256(exact_wire.encode("utf-8")),
    }


def _recovery_consequence(kind: RecoveryKind) -> str:
    if kind == "assistant_agent":
        return "commit the captured Assistant turn"
    if kind == "card_revision":
        return "stage the captured generic card revision for owner review"
    return "stage the captured rich conjugation revision for owner review"


def _recovery_projection(
    operation: operations.Operation,
    binding: _RecoveryBinding,
) -> dict[str, object]:
    artifact = operation.artifact
    if artifact is None:
        raise AssistantOperationError("Recovery requires one exact captured reply.")
    return {
        "kind": binding.kind,
        "manifest_name": binding.manifest_path.name,
        "manifest_sha256": binding.manifest_sha256,
        "captured_reply_sha256": artifact.content_sha256,
        "result_names": [path.name for path in binding.result_paths],
        "consequence": _recovery_consequence(binding.kind),
    }


def _manifest_binding(
    path: Path,
    *,
    operation_id: str,
    expected_kind: str,
    states: set[str],
    recovery_kind: RecoveryKind,
    result_paths: tuple[Path, ...],
) -> _RecoveryBinding | None:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise AssistantOperationError(
            f"Could not bind recovery manifest {path.name}: {exc}"
        ) from exc
    try:
        value = json.loads(wire.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AssistantOperationError(
            f"Recovery manifest {path.name} is not valid UTF-8 JSON."
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("kind") != expected_kind
        or value.get("state") not in states
        or value.get("operation_id") != operation_id
    ):
        raise AssistantOperationError(
            f"Recovery manifest {path.name} does not match paid operation "
            f"{operation_id}."
        )
    return _RecoveryBinding(
        kind=recovery_kind,
        manifest_path=path,
        manifest_sha256=_sha256(wire),
        result_paths=result_paths,
    )


def _recovery_binding(
    config: ProjectConfig,
    operation: operations.Operation,
) -> _RecoveryBinding:
    if operation.state != "result_captured" or operation.artifact is None:
        raise AssistantOperationError(
            f"Operation {operation.operation_id!r} has no captured result to recover."
        )
    if operation.kind == "assistant_agent":
        manifest = config.assistant_dir / f"{operation.operation_id}.json"
        binding = _manifest_binding(
            manifest,
            operation_id=operation.operation_id,
            expected_kind="assistant_agent",
            states={"request", "complete", "failed"},
            recovery_kind="assistant_agent",
            result_paths=(manifest,),
        )
        if binding is None:
            raise AssistantOperationError(
                f"Assistant operation {operation.operation_id!r} has no request manifest."
            )
        return binding
    if operation.kind != "revise":
        raise AssistantOperationError(
            f"Paid operation kind {operation.kind!r} has no Assistant recovery service."
        )

    generic_manifest = (
        config.staging_dir
        / f"card-revision-{operation.operation_id}.request.json"
    )
    generic_result = (
        config.staging_dir / f"card-revision-{operation.operation_id}.yaml"
    )
    generic = _manifest_binding(
        generic_manifest,
        operation_id=operation.operation_id,
        expected_kind="canonical_card_revision",
        states={"request", "result"},
        recovery_kind="card_revision",
        result_paths=(generic_manifest, generic_result),
    )
    rich_manifest = config.staging_dir / (
        f"revise-{prompts.fingerprint(operation.source_file)[:12]}-"
        f"{operation.request_fp[:16]}.json"
    )
    rich = _manifest_binding(
        rich_manifest,
        operation_id=operation.operation_id,
        expected_kind="conjugation_deck_revision",
        states={"request", "proposed"},
        recovery_kind="conjugation_revision",
        result_paths=(rich_manifest,),
    )
    found = [binding for binding in (generic, rich) if binding is not None]
    if len(found) != 1:
        raise AssistantOperationError(
            f"Revision operation {operation.operation_id!r} must have exactly one "
            "recognized request manifest."
        )
    return found[0]


def plan_operation_action(
    config: ProjectConfig,
    *,
    operation_id: str,
    action: OperationAction,
    accept_paid_output_loss: bool = False,
) -> OperationActionPlan:
    """Bind one currently available journal action without performing it."""

    if not isinstance(operation_id, str) or not operation_id.strip():
        raise AssistantOperationError("Choose one exact paid operation.")
    if action not in {"recover", "show_reply", "end", "forget"}:
        raise AssistantOperationError(f"Unknown paid-operation action {action!r}.")
    if not isinstance(accept_paid_output_loss, bool):
        raise AssistantOperationError(
            "Paid-output loss acceptance must be an explicit true or false choice."
        )
    try:
        journal = operations.OperationJournal.load(config.operations_file)
    except (operations.OperationError, OSError, ValueError) as exc:
        raise AssistantOperationError(str(exc)) from exc
    operation = journal.operations.get(operation_id)
    if operation is None:
        raise AssistantOperationError(
            f"Paid operation {operation_id!r} is no longer tracked."
        )
    recovery = _recovery_binding(config, operation) if action == "recover" else None
    available = {
        option.action: option
        for option in _action_options(operation, recoverable=recovery is not None)
    }
    option = available.get(action)
    if option is None:
        raise AssistantOperationError(
            f"Paid operation {operation_id!r} cannot currently {action.replace('_', ' ')}."
        )
    force = option.accept_paid_output_loss
    if force and not accept_paid_output_loss:
        raise AssistantOperationError(
            "Forgetting this operation would delete paid output that never reached "
            "its durable destination. The owner must explicitly accept losing it."
        )
    if accept_paid_output_loss and not force:
        raise AssistantOperationError(
            "This action does not currently require paid-output loss authority. "
            "Prepare its ordinary action instead."
        )
    projection = {
        "schema_version": 1,
        "action": action,
        "force": force,
        "operation": _operation_projection(operation),
    }
    if recovery is not None:
        projection["recovery"] = _recovery_projection(operation, recovery)
    projection_wire = _canonical_json(projection)
    root = config.root.resolve()
    return OperationActionPlan(
        repository_root=root,
        operations_path=_lexical(config.operations_file),
        operation=operation,
        action=action,
        force=force,
        projection_wire=projection_wire,
        fingerprint=_sha256(projection_wire.encode("utf-8")),
        recovery_kind=recovery.kind if recovery is not None else None,
        manifest_path=recovery.manifest_path if recovery is not None else None,
        result_paths=recovery.result_paths if recovery is not None else (),
    )


def execute_operation_action(
    config: ProjectConfig,
    plan: OperationActionPlan,
    *,
    progress: Callable[[str], None] | None = None,
) -> OperationActionExecution:
    """Execute only if the exact rendered operation still holds under lock."""

    if not isinstance(plan, OperationActionPlan):
        raise AssistantOperationError("The paid-operation plan is invalid.")
    if config.root.resolve() != plan.repository_root or _lexical(
        config.operations_file
    ) != plan.operations_path:
        raise AssistantOperationError(
            "The paid-operation journal changed after the action was rendered."
        )
    journal = operations.OperationJournal.load(config.operations_file)
    try:
        if plan.action == "recover":
            try:
                fresh = plan_operation_action(
                    config,
                    operation_id=plan.operation.operation_id,
                    action="recover",
                )
            except AssistantOperationError as exc:
                raise AssistantOperationError(
                    "The paid operation or recovery manifest changed after the "
                    "action was rendered."
                ) from exc
            if fresh.fingerprint != plan.fingerprint:
                raise AssistantOperationError(
                    "The paid operation or recovery manifest changed after the "
                    "action was rendered."
                )
            if plan.recovery_kind == "assistant_agent":
                recovered = assistant_agent.recover_agent(
                    config,
                    plan.operation.operation_id,
                    progress=progress,
                )
                returned_paths = (Path(recovered.manifest_path),)
                answer = recovered.answer
                intent_count = len(recovered.action_intents)
            elif plan.recovery_kind == "card_revision":
                recovered = card_revision.recover_card_revision(
                    config,
                    plan.operation.operation_id,
                    progress=progress,
                )
                returned_paths = (
                    Path(recovered.request_manifest_path),
                    Path(recovered.staging_path),
                )
                answer = None
                intent_count = 0
            elif plan.recovery_kind == "conjugation_revision":
                recovered = revision.recover_revision(
                    config,
                    plan.operation.operation_id,
                    progress=progress,
                )
                returned_paths = (Path(recovered.staging_path),)
                answer = None
                intent_count = 0
            else:
                raise AssistantOperationError("The paid-operation recovery kind is invalid.")
            if returned_paths != plan.result_paths:
                raise AssistantOperationError(
                    "The recovery service returned a different durable destination."
                )
            result_hashes = tuple(_sha256(read_bytes_bound(path)) for path in returned_paths)
            committed = operations.OperationJournal.load(
                config.operations_file
            ).operations.get(plan.operation.operation_id)
            if committed is None or committed.state != "committed":
                raise AssistantOperationError(
                    "The recovery service did not commit the exact paid operation."
                )
            return OperationActionExecution(
                plan=plan,
                operation=committed,
                recovery_kind=plan.recovery_kind,
                result_names=tuple(path.name for path in returned_paths),
                result_sha256=result_hashes,
                assistant_answer=answer,
                assistant_action_intent_count=intent_count,
            )
        if plan.action == "show_reply":
            reply = journal.read_inspectable_reply(
                plan.operation.operation_id,
                expected_operation=plan.operation,
            )
            return OperationActionExecution(
                plan=plan,
                operation=plan.operation,
                reply=reply,
                reply_sha256=_sha256(reply),
                private=True,
            )
        if plan.action == "end":
            ended = journal.end(
                plan.operation.operation_id,
                detail="ended by repository owner in Janki Assistant",
                expected_operation=plan.operation,
            )
            return OperationActionExecution(plan=plan, operation=ended)
        forgotten = journal.forget(
            [plan.operation.operation_id],
            force=plan.force,
            expected_operation=plan.operation,
        )
        if forgotten != 1:
            raise AssistantOperationError(
                "The exact paid operation was not forgotten; refresh its status."
            )
        return OperationActionExecution(
            plan=plan,
            operation=None,
            forgotten=forgotten,
        )
    except AssistantOperationError:
        raise
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantOperationError(str(exc)) from exc
