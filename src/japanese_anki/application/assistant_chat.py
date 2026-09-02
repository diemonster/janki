"""Journal one ordinary, non-mutating Assistant conversation turn.

The composer send is direct owner authority for exactly one conversational
call.  This pass receives only the selected deck scope, its locally resolved
revision capability, a bounded caller-supplied history, and the current
message; it never reads deck content and its provider runs without tools or
filesystem context.  The separate ``revise`` service remains the only
Assistant path that writes Japanese card proposals.

Every call publishes its exact request manifest before dispatch, captures the
provider's exact reply before decoding, and commits only after the decoded
answer and provenance are durable under ``data/assistant``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import ai_schema, claude_client, operations, prompts
from japanese_anki.application import revision_provider
from japanese_anki.application.extraction import classify_dispatch_failure
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    prepare_bound_directory,
    read_bytes_bound,
)

__all__ = [
    "ChatApplicationError",
    "ChatPlan",
    "ChatRunError",
    "ChatRunResult",
    "plan_chat",
    "recover_chat",
    "run_chat",
]


class ChatApplicationError(JankiError):
    """An ordinary Assistant turn could not be planned or completed safely."""


class ChatRunError(ChatApplicationError):
    """A chat turn stopped after its paid-operation identity was allocated."""

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
class ChatPlan:
    """The immutable provider request made by one composer send."""

    repository_root: Path
    deck_scope: str
    message: str
    history: tuple[tuple[str, str], ...]
    task_template_name: str
    task_template: str
    user_turn: str
    provider_plan: revision_provider.RevisionProviderPlan

    @property
    def provider(self) -> str:
        return self.provider_plan.provider

    @property
    def billing_class(self) -> str:
        return self.provider_plan.billing_class

    @property
    def billing_display(self) -> str:
        return self.provider_plan.billing_display

    @property
    def auth_metadata(self) -> Mapping[str, Any]:
        return self.provider_plan.auth_metadata

    @property
    def model(self) -> str:
        return self.provider_plan.model

    @property
    def transport(self) -> Mapping[str, Any]:
        return self.provider_plan.transport

    @property
    def request_fingerprint(self) -> str:
        return self.provider_plan.request_fingerprint


@dataclass(frozen=True, slots=True)
class ChatRunResult:
    """One conversational answer at its durable destination."""

    answer: str
    operation_id: str
    manifest_path: Path
    request_fingerprint: str


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json_value(item) for item in value]
    return value


def _report_progress(progress: Callable[[str], None] | None, label: str) -> None:
    if progress is not None:
        progress(label)


_HISTORY_LIMIT = 12
_HISTORY_BYTE_LIMIT = 24_000


def _validated_history(
    history: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(history, tuple):
        raise ChatApplicationError(
            "Assistant history must be an immutable tuple; nothing was sent."
        )
    if len(history) > _HISTORY_LIMIT:
        raise ChatApplicationError(
            f"Assistant history may contain at most {_HISTORY_LIMIT} messages; "
            "nothing was sent."
        )
    checked: list[tuple[str, str]] = []
    for item in history:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ChatApplicationError(
                "Each Assistant history item must be an immutable (role, text) "
                "pair; nothing was sent."
            )
        role, content = item
        if not isinstance(role, str) or role not in {"user", "assistant"}:
            raise ChatApplicationError(
                "Assistant history roles must be 'user' or 'assistant'; nothing "
                "was sent."
            )
        if not isinstance(content, str) or not content.strip():
            raise ChatApplicationError(
                "Assistant history messages must be nonblank text; nothing was sent."
            )
        checked.append((role, content))
    frozen = tuple(checked)
    history_bytes = _canonical_json(frozen).encode("utf-8")
    if len(history_bytes) > _HISTORY_BYTE_LIMIT:
        raise ChatApplicationError(
            f"Assistant history may use at most {_HISTORY_BYTE_LIMIT} UTF-8 "
            "bytes; nothing was sent."
        )
    return frozen


def _plan_chat(
    config: ProjectConfig,
    *,
    deck_scope: str,
    message: str,
    history: tuple[tuple[str, str], ...] = (),
    revision_supported: bool = True,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ChatPlan:
    if not isinstance(deck_scope, str) or not deck_scope.strip():
        raise ChatApplicationError(
            "The Assistant needs one selected deck scope; nothing was sent."
        )
    if not isinstance(message, str) or not message.strip():
        raise ChatApplicationError(
            "Write a nonblank Assistant message; nothing was sent."
        )
    if not isinstance(revision_supported, bool):
        raise ChatApplicationError(
            "Assistant revision capability must be true or false; nothing was sent."
        )
    checked_history = _validated_history(history)
    provider_name = str(config.assistant_provider).strip().lower()
    model = str(config.assistant_model).strip()
    if not model:
        raise ChatApplicationError(
            "Assistant model must be nonblank; nothing was sent."
        )

    task_template_name = (
        "assistant-chat" if revision_supported else "assistant-chat-only"
    )
    task_template = prompts.load(config.root, task_template_name)
    system_blocks = tuple(claude_client.system_blocks(task_template))
    user_turn = _user_turn(
        deck_scope=deck_scope,
        history=checked_history,
        message=message,
    )
    schema = ai_schema.assistant_chat_schema()
    provider_plan = revision_provider.plan_provider(
        provider_name,
        model=model,
        style_guide="",
        task_template=task_template,
        system_blocks=system_blocks,
        user_turn=user_turn,
        schema=schema,
        response_mode=(
            "plain-markdown"
            if provider_name == revision_provider.CLAUDE_CODE_PROVIDER
            else "structured"
        ),
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
    )
    return ChatPlan(
        repository_root=config.root.resolve(),
        deck_scope=deck_scope,
        message=message,
        history=checked_history,
        task_template_name=task_template_name,
        task_template=task_template,
        user_turn=user_turn,
        provider_plan=provider_plan,
    )


def plan_chat(
    config: ProjectConfig,
    *,
    deck_scope: str,
    message: str,
    history: tuple[tuple[str, str], ...] = (),
    revision_supported: bool = True,
) -> ChatPlan:
    """Plan one conversational call without granting authority or writing state."""
    return _plan_chat(
        config,
        deck_scope=deck_scope,
        message=message,
        history=history,
        revision_supported=revision_supported,
    )


def _fresh_plan(
    config: ProjectConfig,
    expected: ChatPlan,
    *,
    provider_env: Mapping[str, str] | None,
    provider_runner: Callable[..., Any],
    provider_which: Callable[..., str | None],
) -> ChatPlan:
    if expected.repository_root != config.root.resolve():
        raise ChatApplicationError(
            "This Assistant request belongs to a different repository; nothing "
            "was sent."
        )
    if expected.task_template_name not in {
        "assistant-chat",
        "assistant-chat-only",
    }:
        raise ChatApplicationError(
            "This Assistant request has no valid task template identity; nothing "
            "was sent."
        )
    fresh = _plan_chat(
        config,
        deck_scope=expected.deck_scope,
        message=expected.message,
        history=expected.history,
        revision_supported=expected.task_template_name == "assistant-chat",
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )
    if fresh.request_fingerprint != expected.request_fingerprint:
        raise ChatApplicationError(
            "The Assistant prompt, provider, model, authentication, or exact "
            "message changed after this request was planned; nothing was sent."
        )
    return fresh


def _request_manifest(plan: ChatPlan, *, operation_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "assistant_chat",
        "state": "request",
        "operation_id": operation_id,
        "deck_scope": plan.deck_scope,
        "request": {
            "history": [list(item) for item in plan.history],
            "message": plan.message,
            "task_template": plan.task_template,
            "task_template_sha256": prompts.fingerprint(plan.task_template),
            "system_blocks": _plain_json_value(plan.provider_plan.system_blocks),
            "user_turn": plan.user_turn,
            "provider_plan": plan.provider_plan.persistent_manifest(),
        },
    }


def _manifest_text(value: Mapping[str, Any]) -> str:
    return _canonical_json(value, pretty=True) + "\n"


def _complete_manifest(
    request_manifest: Mapping[str, Any],
    plan: ChatPlan,
    *,
    answer: str,
) -> str:
    value = dict(request_manifest)
    value["state"] = "complete"
    value["answer"] = answer
    value["provenance"] = _answer_provenance(plan)
    return _manifest_text(value)


def _answer_provenance(plan: ChatPlan) -> dict[str, str]:
    return {
        "provider": plan.provider,
        "billing_class": plan.billing_class,
        "model": plan.model,
        "request_fingerprint": plan.request_fingerprint,
    }


def _strict_json(raw: bytes, path: Path) -> Mapping[str, Any]:
    """Decode one Assistant manifest without duplicate keys or non-JSON values."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ChatApplicationError(
                    f"Assistant manifest {path} repeats key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ChatApplicationError(
            f"Assistant manifest {path} contains invalid JSON value {value!r}."
        )

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except ChatApplicationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ChatApplicationError(
            f"Could not parse Assistant manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ChatApplicationError(
            f"Assistant manifest {path} must contain one JSON object."
        )
    return value


def _require_exact_keys(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        found = sorted(str(key) for key in value) if isinstance(value, Mapping) else []
        raise ChatApplicationError(
            f"Assistant manifest has invalid {label} fields: {found}."
        )
    return value


def _history_from_manifest(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in value
    ):
        raise ChatApplicationError(
            "Assistant manifest has invalid conversation history."
        )
    pairs: list[tuple[str, str]] = []
    for item in value:
        role, content = item
        if not isinstance(role, str) or not isinstance(content, str):
            raise ChatApplicationError(
                "Assistant manifest has invalid conversation history."
            )
        pairs.append((role, content))
    return _validated_history(tuple(pairs))


def _user_turn(
    *,
    deck_scope: str,
    history: tuple[tuple[str, str], ...],
    message: str,
) -> str:
    return _canonical_json(
        {
            "deck_scope": deck_scope,
            "history": [
                {"role": role, "content": content} for role, content in history
            ],
            "user_message": message,
        }
    )


def _request_manifest_for_recovery(
    config: ProjectConfig,
    operation_id: str,
) -> tuple[
    operations.OperationJournal,
    Any,
    Path,
    Mapping[str, Any],
    str,
    ChatPlan,
    bytes,
]:
    """Prove one stored chat request and its exact captured provider reply."""
    journal = operations.OperationJournal.load(config.operations_file)
    held = journal.operations.get(operation_id)
    if held is None:
        raise ChatApplicationError(
            f"No journaled Assistant operation {operation_id!r} to recover."
        )
    if held.kind != "assistant_chat":
        raise ChatApplicationError(
            f"Operation {operation_id!r} is {held.kind!r}, not an Assistant chat."
        )
    if held.state != "result_captured" or held.artifact is None:
        raise ChatApplicationError(
            f"Assistant operation {operation_id!r} has no exact captured reply "
            f"ready for recovery; its state is {held.state!r}."
        )
    try:
        canonical_operation_id = str(uuid.UUID(operation_id))
    except (ValueError, AttributeError) as exc:
        raise ChatApplicationError(
            f"Assistant operation ID {operation_id!r} is invalid."
        ) from exc
    if canonical_operation_id != operation_id:
        raise ChatApplicationError(
            f"Assistant operation ID {operation_id!r} is not canonical."
        )

    manifest_path = config.assistant_dir / f"{operation_id}.json"
    try:
        manifest_wire = read_bytes_bound(manifest_path)
    except (FileNotFoundError, DataError, OSError) as exc:
        raise ChatApplicationError(
            f"Could not read the durable Assistant request manifest for "
            f"{operation_id}: {exc}"
        ) from exc
    manifest_revision = _sha256(manifest_wire)
    manifest = _strict_json(manifest_wire, manifest_path)
    manifest_state = manifest.get("state")
    top_level = {
        "schema_version",
        "kind",
        "state",
        "operation_id",
        "deck_scope",
        "request",
    }
    if manifest_state == "complete":
        top_level.update({"answer", "provenance"})
    _require_exact_keys(manifest, top_level, label="top-level")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "assistant_chat"
        or manifest_state not in {"request", "complete"}
        or manifest.get("operation_id") != operation_id
    ):
        raise ChatApplicationError(
            f"Assistant manifest {manifest_path} is not the recoverable result "
            f"for operation {operation_id}."
        )
    deck_scope = manifest.get("deck_scope")
    if not isinstance(deck_scope, str) or not deck_scope.strip():
        raise ChatApplicationError(
            "Assistant manifest has no valid selected deck scope."
        )
    if (
        held.source_file != deck_scope
        or held.source_sha256 != _sha256(deck_scope.encode("utf-8"))
    ):
        raise ChatApplicationError(
            "Assistant manifest does not match its journaled deck scope."
        )

    request = _require_exact_keys(
        manifest.get("request"),
        {
            "history",
            "message",
            "task_template",
            "task_template_sha256",
            "system_blocks",
            "user_turn",
            "provider_plan",
        },
        label="request",
    )
    scalar_fields = (
        "message",
        "task_template",
        "task_template_sha256",
        "user_turn",
    )
    if any(not isinstance(request.get(key), str) for key in scalar_fields):
        raise ChatApplicationError(
            "Assistant manifest contains a non-text request identity field."
        )
    message = request["message"]
    task_template = request["task_template"]
    if not message.strip() or not task_template:
        raise ChatApplicationError(
            "Assistant manifest contains an empty message or task template."
        )
    if request["task_template_sha256"] != prompts.fingerprint(task_template):
        raise ChatApplicationError(
            "Assistant manifest task template does not match its fingerprint."
        )
    history = _history_from_manifest(request["history"])
    rebuilt_turn = _user_turn(
        deck_scope=deck_scope,
        history=history,
        message=message,
    )
    if request["user_turn"] != rebuilt_turn:
        raise ChatApplicationError(
            "Assistant manifest's exact scope, history, message, and user turn "
            "do not agree."
        )
    expected_blocks = claude_client.system_blocks(task_template)
    if request["system_blocks"] != expected_blocks:
        raise ChatApplicationError(
            "Assistant manifest has inconsistent system prompt blocks."
        )
    provider_manifest = request["provider_plan"]
    if not isinstance(provider_manifest, Mapping):
        raise ChatApplicationError(
            "Assistant manifest has invalid provider metadata."
        )
    schema = ai_schema.assistant_chat_schema()
    try:
        provider_plan = revision_provider.provider_plan_from_manifest(
            provider_manifest,
            model=str(provider_manifest.get("model", "")),
            style_guide="",
            task_template=task_template,
            system_blocks=expected_blocks,
            user_turn=rebuilt_turn,
            schema=schema,
        )
    except revision_provider.RevisionProviderError as exc:
        raise ChatApplicationError(
            f"Assistant manifest has invalid provider provenance: {exc}"
        ) from exc
    if (
        provider_plan.model != held.model
        or provider_plan.request_fingerprint != held.request_fp
    ):
        raise ChatApplicationError(
            "Assistant provider metadata does not match its journaled operation."
        )
    plan = ChatPlan(
        repository_root=config.root.resolve(),
        deck_scope=deck_scope,
        message=message,
        history=history,
        task_template_name="captured-assistant-chat",
        task_template=task_template,
        user_turn=rebuilt_turn,
        provider_plan=provider_plan,
    )
    reply = journal.read_reply(operation_id)
    if _sha256(reply) != held.artifact.content_sha256:
        raise ChatApplicationError(
            f"Captured reply for Assistant operation {operation_id} does not "
            "match its journal receipt."
        )
    return (
        journal,
        held,
        manifest_path,
        manifest,
        manifest_revision,
        plan,
        reply,
    )


def _answer_from_result(result: Any, *, model: str, captured: bool) -> str:
    parsed = getattr(result, "parsed", None)
    if parsed is None:
        stop_reason = str(getattr(result, "stop_reason", "") or "unknown")
        refusal = getattr(result, "refusal", None)
        detail = ""
        if refusal is not None:
            category = str(getattr(refusal, "category", "") or "").strip()
            explanation = str(getattr(refusal, "explanation", "") or "").strip()
            detail = ": " + " — ".join(
                item for item in (category, explanation) if item
            )
        suffix = " Its exact reply was captured." if captured else ""
        raise ChatApplicationError(
            f"{model} returned no complete Assistant answer "
            f"({stop_reason}{detail}).{suffix}"
        )
    if isinstance(parsed, str):
        answer = parsed.strip()
    else:
        answer = str(getattr(parsed, "answer", "") or "").strip()
    if not answer:
        suffix = " Its exact reply was captured." if captured else ""
        raise ChatApplicationError(
            f"{model} returned an empty Assistant answer.{suffix}"
        )
    return answer


def recover_chat(
    config: ProjectConfig,
    operation_id: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> ChatRunResult:
    """Commit one exact captured Assistant reply without another provider call."""
    _report_progress(progress, "Preparing answer")
    journal, held, manifest_path, manifest, manifest_revision, plan, reply = (
        _request_manifest_for_recovery(config, operation_id)
    )
    _report_progress(progress, "Writing answer")
    try:
        result = revision_provider.provider_for(plan.provider).recover(
            plan.provider_plan,
            reply,
        )
    except revision_provider.RevisionProviderError as exc:
        raise ChatApplicationError(
            f"Captured Assistant reply for {operation_id} could not be decoded: {exc}"
        ) from exc
    answer = _answer_from_result(result, model=plan.model, captured=True)
    rendered = _complete_manifest(manifest, plan, answer=answer)
    if manifest["state"] == "complete" and (
        manifest.get("answer") != answer
        or manifest.get("provenance") != _answer_provenance(plan)
        or _sha256(rendered.encode("utf-8")) != manifest_revision
    ):
        raise ChatApplicationError(
            f"Completed Assistant manifest for {operation_id} does not match "
            "its exact captured answer and provenance."
        )
    _report_progress(progress, "Saving answer")
    with exclusive_path_lock(manifest_path):
        journal.commit_result(
            operation_id,
            lambda: atomic_write_text_bound(
                manifest_path,
                rendered,
                expected_revision=manifest_revision,
            ),
        )
    return ChatRunResult(
        answer=answer,
        operation_id=operation_id,
        manifest_path=manifest_path,
        request_fingerprint=held.request_fp,
    )


def run_chat(
    config: ProjectConfig,
    expected: ChatPlan,
    *,
    client: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    api_call: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ChatRunResult:
    """Dispatch the one chat call directly authorized by its composer send."""
    _report_progress(progress, "Preparing answer")
    fresh = _fresh_plan(
        config,
        expected,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )

    # Authentication and client construction can fail locally, so they happen
    # before the journal records authority that might otherwise need settling.
    provider = revision_provider.provider_for(fresh.provider)
    prepared_provider = provider.prepare(
        fresh.provider_plan,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
        client=client,
    )
    operation_id = str(uuid.uuid4())
    assistant_dir = config.assistant_dir
    manifest_path = assistant_dir / f"{operation_id}.json"
    prompt_path = prompts.path_for(config.root, fresh.task_template_name)

    with contextlib.ExitStack() as binding_locks:
        for path in sorted({manifest_path, prompt_path}, key=lambda item: str(item)):
            binding_locks.enter_context(exclusive_path_lock(path))
        fresh = _fresh_plan(
            config,
            expected,
            provider_env=provider_env,
            provider_runner=provider_runner,
            provider_which=provider_which,
        )
        operations.prepare_artifact_store(config.operations_file)
        prepare_bound_directory(assistant_dir)
        journal = operations.OperationJournal.load(config.operations_file)
        try:
            journal.authorize(
                operation_id,
                kind="assistant_chat",
                source_file=fresh.deck_scope,
                source_sha256=_sha256(fresh.deck_scope.encode("utf-8")),
                request_fp=fresh.request_fingerprint,
                model=fresh.model,
            )
        except Exception as exc:  # noqa: BLE001 - authority may have landed
            raise ChatRunError(
                f"Assistant authority could not be recorded safely: {exc} "
                "Inspect janki operations before retrying.",
                operation_id=operation_id,
                provider_dispatched=False,
            ) from exc

        request_manifest = _request_manifest(fresh, operation_id=operation_id)
        request_text = _manifest_text(request_manifest)
        request_revision = _sha256(request_text.encode("utf-8"))
        try:
            atomic_write_text_bound(
                manifest_path,
                request_text,
                expected_absent=True,
            )
        except Exception as exc:  # noqa: BLE001 - publication may have landed
            try:
                operations.OperationJournal.load(config.operations_file).advance(
                    operation_id,
                    "canceled_before_send",
                    detail="The durable Assistant request manifest could not be prepared.",
                )
            except JankiError as journal_error:
                raise ChatRunError(
                    f"Assistant turn {operation_id} was not sent, but its request "
                    f"manifest failed and its authority could not be retired: {exc}; "
                    f"{journal_error}. Inspect janki operations before retrying.",
                    operation_id=operation_id,
                    provider_dispatched=False,
                ) from exc
            raise ChatRunError(
                f"Assistant turn {operation_id} was canceled before send because "
                f"its exact request manifest could not be made durable: {exc}",
                operation_id=operation_id,
                provider_dispatched=False,
            ) from exc

    try:
        journal.advance(operation_id, "dispatching")
    except Exception as exc:  # noqa: BLE001 - transition may have landed
        try:
            current = operations.OperationJournal.load(config.operations_file)
            held = current.operations.get(operation_id)
            if held is not None and held.state == "authorized":
                current.advance(
                    operation_id,
                    "canceled_before_send",
                    detail="The Assistant dispatch boundary could not be recorded.",
                )
        except JankiError:
            pass
        raise ChatRunError(
            f"Assistant turn {operation_id} was not dispatched because its "
            f"dispatch boundary could not be recorded safely: {exc}. Inspect "
            "janki operations before retrying.",
            operation_id=operation_id,
            provider_dispatched=False,
        ) from exc

    try:
        _report_progress(progress, "Writing answer")

        def capture_reply(raw_reply: bytes) -> None:
            if not isinstance(raw_reply, bytes):
                raise operations.OperationError(
                    "Assistant provider capture must supply exact response bytes."
                )
            journal.capture_result(
                operation_id,
                lambda: operations.capture_artifact(
                    config.operations_file,
                    operation_id,
                    raw_reply,
                ),
            )

        result = provider.dispatch(
            prepared_provider,
            capture=capture_reply,
            runner=provider_runner,
            api_call=api_call,
        )
        captured = operations.OperationJournal.load(
            config.operations_file
        ).operations.get(operation_id)
        if captured is None or captured.state != "result_captured":
            raise operations.OperationError(
                f"Assistant turn {operation_id} returned data without durably "
                "capturing its exact provider reply."
            )
        answer = _answer_from_result(result, model=fresh.model, captured=True)
        rendered = _complete_manifest(request_manifest, fresh, answer=answer)
        _report_progress(progress, "Saving answer")
        with exclusive_path_lock(manifest_path):
            journal.commit_result(
                operation_id,
                lambda: atomic_write_text_bound(
                    manifest_path,
                    rendered,
                    expected_revision=request_revision,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - settle every post-dispatch failure
        try:
            classify_dispatch_failure(config, journal, operation_id, exc)
        except JankiError as journal_error:
            raise ChatRunError(
                f"Assistant turn {operation_id} failed after dispatch: {exc} "
                f"Janki could not settle its journal entry: {journal_error}. "
                "This call may have consumed allowance or been billed; do not "
                "retry until you inspect janki operations.",
                operation_id=operation_id,
                provider_dispatched=True,
            ) from exc
        raise ChatRunError(
            f"Assistant turn {operation_id} failed after dispatch: {exc}",
            operation_id=operation_id,
            provider_dispatched=True,
        ) from exc

    return ChatRunResult(
        answer=answer,
        operation_id=operation_id,
        manifest_path=manifest_path,
        request_fingerprint=fresh.request_fingerprint,
    )
