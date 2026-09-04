"""Journal one repository-aware Assistant turn and decode its typed intents.

The provider receives only the exact bounded context supplied by
``AssistantContextBroker``.  It has no Claude Code filesystem or shell tools.
Its structured answer may contain ordinary prose and closed application
intents. Those intents are untrusted data until an action broker resolves their
opaque identifiers and calls an existing Janki planner.

Sending a composer message is authority for this one Assistant inference call.
It is not authority to run an intent, write Japanese, generate audio, build a
package, or make an owner-only decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

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
from japanese_anki.models import VocabularyRecord

__all__ = [
    "AgentApplicationError",
    "AgentActionIntent",
    "AgentContext",
    "AgentPlan",
    "AgentRunError",
    "AgentRunResult",
    "plan_agent",
    "recover_agent",
    "run_agent",
]


class AgentApplicationError(JankiError):
    """A repository-aware Assistant turn could not complete safely."""


class AgentUnansweredError(AgentApplicationError):
    """The provider returned no Assistant answer at all.

    This is separate from every other refusal because it decides who settles
    the turn. A reply that carries no answer holds nothing an owner could act
    on, so its exact bytes and this message are the whole truth and janki
    records them itself. Every other failure leaves an answer somebody could
    still read — an intent naming a resource the turn never disclosed, or a
    `RevisionProviderError` for structured output that finished normally and
    only failed schema validation — and retiring one of those would spend the
    owner's decision for them.

    Both providers reach this the same way: an incomplete stop reason or an
    error envelope decodes to a result with no parsed answer, never to a
    provider error. A `RevisionProviderError` therefore always means a reply
    janki could not read, not a reply that said nothing.
    """


class AgentRunError(AgentApplicationError):
    """An Assistant turn stopped after allocating a paid-operation identity."""

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
class AgentActionIntent:
    """One closed planner request returned by the ordinary Assistant turn."""

    kind: str
    resource_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    instruction: str
    options_json: str = "{}"

    @property
    def options(self) -> Mapping[str, Any]:
        """The detached, canonical action-specific choices from the model."""

        value = _strict_json_text(self.options_json, label="Assistant action options")
        if not isinstance(value, Mapping):
            raise AgentApplicationError("Assistant action options must be one object.")
        return value


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Exact broker disclosure and the canonical cards it permits changing."""

    wire: str
    fingerprint: str
    resource_ids: tuple[str, ...]
    editable_records: tuple[VocabularyRecord, ...] = ()
    focus_resource_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentPlan:
    """One immutable repository-aware provider request."""

    repository_root: Path
    message: str
    history: tuple[tuple[str, str], ...]
    context: AgentContext
    task_template: str
    user_turn: str
    provider_plan: revision_provider.RevisionProviderPlan

    @property
    def provider(self) -> str:
        return self.provider_plan.provider

    @property
    def model(self) -> str:
        return self.provider_plan.model

    @property
    def billing_class(self) -> str:
        return self.provider_plan.billing_class

    @property
    def request_fingerprint(self) -> str:
        return self.provider_plan.request_fingerprint


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Durable Assistant prose and its optional untrusted action intent."""

    answer: str
    action_intents: tuple[AgentActionIntent, ...]
    operation_id: str
    manifest_path: Path
    request_fingerprint: str


_HISTORY_LIMIT = 12
_HISTORY_BYTE_LIMIT = 24_000
_SHA256_LENGTH = 64
_REQUEST_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "state",
        "operation_id",
        "context",
        "request",
    }
)
_COMPLETE_MANIFEST_KEYS = _REQUEST_MANIFEST_KEYS | {
    "answer",
    "action_intents",
    "provenance",
}
_FAILED_MANIFEST_KEYS = _REQUEST_MANIFEST_KEYS | {
    "failure",
    "provenance",
}
_FAILURE_KEYS = frozenset(
    {
        "message",
        "provider_reply_base64",
        "provider_reply_bytes",
        "provider_reply_sha256",
    }
)
_CONTEXT_MANIFEST_KEYS = frozenset(
    {
        "wire",
        "fingerprint",
        "resource_ids",
        "focus_resource_id",
        "editable_records",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "history",
        "message",
        "task_template",
        "task_template_sha256",
        "system_blocks",
        "user_turn",
        "provider_plan",
    }
)


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
        raise AgentApplicationError(f"Assistant data is not finite JSON: {exc}") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json_value(item) for item in value]
    return value


def _report_progress(progress: Callable[[str], None] | None, label: str) -> None:
    if progress is not None:
        progress(label)


def _validated_history(
    history: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(history, tuple) or len(history) > _HISTORY_LIMIT:
        raise AgentApplicationError(
            f"Assistant history must be an immutable tuple of at most "
            f"{_HISTORY_LIMIT} messages; nothing was sent."
        )
    checked: list[tuple[str, str]] = []
    for item in history:
        if not isinstance(item, tuple) or len(item) != 2:
            raise AgentApplicationError(
                "Each Assistant history item must be an immutable (role, text) "
                "pair; nothing was sent."
            )
        role, content = item
        if (
            not isinstance(role, str)
            or role not in {"user", "assistant"}
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise AgentApplicationError(
                "Assistant history needs nonblank user or assistant text; nothing "
                "was sent."
            )
        checked.append((role, content))
    frozen = tuple(checked)
    if len(_canonical_json(frozen).encode("utf-8")) > _HISTORY_BYTE_LIMIT:
        raise AgentApplicationError(
            f"Assistant history may use at most {_HISTORY_BYTE_LIMIT} UTF-8 bytes; "
            "nothing was sent."
        )
    return frozen


def _strict_json_text(value: str, *, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise AgentApplicationError(f"{label} repeats JSON key {key!r}.")
            result[key] = item
        return result

    def constant(item: str) -> Any:
        raise AgentApplicationError(f"{label} contains invalid JSON value {item!r}.")

    try:
        return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
    except AgentApplicationError:
        raise
    except json.JSONDecodeError as exc:
        raise AgentApplicationError(f"{label} is not valid JSON: {exc}") from exc


def _require_exact_keys(
    value: Any,
    keys: frozenset[str] | set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        found = sorted(str(key) for key in value) if isinstance(value, Mapping) else []
        raise AgentApplicationError(
            f"Captured Assistant manifest has invalid {label} fields: {found}."
        )
    return value


def _canonical_operation_id(operation_id: str) -> str:
    try:
        canonical = str(uuid.UUID(operation_id))
    except (ValueError, AttributeError) as exc:
        raise AgentApplicationError(
            f"Assistant operation ID {operation_id!r} is invalid."
        ) from exc
    if canonical != operation_id:
        raise AgentApplicationError(
            f"Assistant operation ID {operation_id!r} is not canonical."
        )
    return canonical


def _history_from_manifest(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in value
    ):
        raise AgentApplicationError(
            "Captured Assistant manifest has invalid conversation history."
        )
    pairs: list[tuple[str, str]] = []
    for item in value:
        role, content = item
        if not isinstance(role, str) or not isinstance(content, str):
            raise AgentApplicationError(
                "Captured Assistant manifest has invalid conversation history."
            )
        pairs.append((role, content))
    return _validated_history(tuple(pairs))


def _record_from_manifest(value: Any, *, index: int) -> VocabularyRecord:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AgentApplicationError(
            f"Captured Assistant editable record {index} must be one JSON object."
        )
    raw = dict(value)
    try:
        record = VocabularyRecord.from_dict(raw)
    except JankiError as exc:
        raise AgentApplicationError(
            f"Captured Assistant editable record {index} is invalid: {exc}"
        ) from exc
    if record.to_dict() != raw:
        raise AgentApplicationError(
            f"Captured Assistant editable record {index} does not round-trip exactly."
        )
    return record


def _detached_record(record: VocabularyRecord, *, label: str) -> VocabularyRecord:
    if not isinstance(record, VocabularyRecord):
        raise AgentApplicationError(f"{label} must be a VocabularyRecord.")
    raw = record.to_dict()
    try:
        detached = VocabularyRecord.from_dict(raw)
    except JankiError as exc:
        raise AgentApplicationError(f"{label} is invalid: {exc}") from exc
    if detached.to_dict() != raw:
        raise AgentApplicationError(f"{label} does not round-trip exactly.")
    return detached


def _validated_context(context: AgentContext) -> AgentContext:
    if not isinstance(context, AgentContext):
        raise AgentApplicationError("Assistant context must come from the typed broker.")
    encoded = context.wire.encode("utf-8") if isinstance(context.wire, str) else b""
    if not encoded or not _is_sha256(context.fingerprint):
        raise AgentApplicationError("Assistant context is empty or has no valid fingerprint.")
    if _sha256(encoded) != context.fingerprint:
        raise AgentApplicationError("Assistant context does not match its fingerprint.")
    parsed = _strict_json_text(context.wire, label="Assistant context")
    if not isinstance(parsed, Mapping):
        raise AgentApplicationError("Assistant context must contain one JSON object.")
    if (
        not isinstance(context.resource_ids, tuple)
        or any(not isinstance(item, str) or not item for item in context.resource_ids)
        or len(set(context.resource_ids)) != len(context.resource_ids)
    ):
        raise AgentApplicationError("Assistant context resource ids must be unique text.")
    if context.focus_resource_id is not None and (
        not isinstance(context.focus_resource_id, str)
        or not context.focus_resource_id.strip()
        or context.focus_resource_id not in context.resource_ids
    ):
        raise AgentApplicationError(
            "Assistant focus must be one of the exact disclosed resource ids."
        )
    records = tuple(
        _detached_record(record, label=f"Editable record {index + 1}")
        for index, record in enumerate(context.editable_records)
    )
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise AgentApplicationError("Assistant context repeats an editable record id.")
    return AgentContext(
        wire=context.wire,
        fingerprint=context.fingerprint,
        resource_ids=context.resource_ids,
        editable_records=records,
        focus_resource_id=context.focus_resource_id,
    )


def _user_turn(
    *,
    context: AgentContext,
    history: tuple[tuple[str, str], ...],
    message: str,
) -> str:
    return _canonical_json(
        {
            "repository_context": _strict_json_text(
                context.wire, label="Assistant context"
            ),
            "context_fingerprint": context.fingerprint,
            "resource_ids": list(context.resource_ids),
            "active_focus_resource_id": context.focus_resource_id,
            "history": [
                {"role": role, "content": content} for role, content in history
            ],
            "user_message": message,
        }
    )


def _plan_agent(
    config: ProjectConfig,
    *,
    context: AgentContext,
    message: str,
    history: tuple[tuple[str, str], ...] = (),
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> AgentPlan:
    if not isinstance(message, str) or not message.strip():
        raise AgentApplicationError("Write a nonblank Assistant message; nothing was sent.")
    checked_context = _validated_context(context)
    checked_history = _validated_history(history)
    provider_name = str(config.assistant_provider).strip().lower()
    model = str(config.assistant_model).strip()
    if not model:
        raise AgentApplicationError("Assistant model must be nonblank; nothing was sent.")
    # The depth is configured rather than model-resolved: one turn is answered
    # and read immediately, unlike a card-writing pass. Whether the model takes
    # the key at all is the provider's own `plan` refusal, which runs first.
    task_template = prompts.load(config.root, "assistant-agent")
    blocks = tuple(claude_client.system_blocks(task_template))
    user_turn = _user_turn(
        context=checked_context,
        history=checked_history,
        message=message.strip(),
    )
    provider_plan = revision_provider.plan_provider(
        provider_name,
        model=model,
        style_guide="",
        task_template=task_template,
        system_blocks=blocks,
        user_turn=user_turn,
        schema=ai_schema.assistant_agent_schema(),
        effort=config.assistant_effort,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
    )
    return AgentPlan(
        repository_root=config.root.resolve(),
        message=message.strip(),
        history=checked_history,
        context=checked_context,
        task_template=task_template,
        user_turn=user_turn,
        provider_plan=provider_plan,
    )


def plan_agent(
    config: ProjectConfig,
    *,
    context: AgentContext,
    message: str,
    history: tuple[tuple[str, str], ...] = (),
) -> AgentPlan:
    """Plan one repository-aware turn without writing or granting authority."""
    return _plan_agent(config, context=context, message=message, history=history)


def _decode_answer(
    result: Any,
    *,
    model: str,
    captured: bool,
    context: AgentContext,
) -> tuple[str, tuple[AgentActionIntent, ...]]:
    parsed = getattr(result, "parsed", None)
    if parsed is None:
        stop_reason = str(getattr(result, "stop_reason", "") or "unknown")
        suffix = " Its exact reply was captured." if captured else ""
        raise AgentUnansweredError(
            f"{model} returned no complete Assistant answer ({stop_reason}).{suffix}"
        )
    answer = str(getattr(parsed, "answer", "") or "").strip()
    if not answer:
        suffix = " Its exact reply was captured." if captured else ""
        raise AgentUnansweredError(f"{model} returned an empty Assistant answer.{suffix}")
    raw_intents = getattr(parsed, "action_intents", None)
    if not isinstance(raw_intents, list):
        raise AgentApplicationError("Assistant answer has no valid action_intents list.")
    allowed_resources = set(context.resource_ids)
    intents: list[AgentActionIntent] = []
    for raw_intent in raw_intents:
        kind = str(getattr(raw_intent, "kind", "") or "").strip()
        instruction = str(getattr(raw_intent, "instruction", "") or "").strip()
        resource_ids = tuple(getattr(raw_intent, "resource_ids", ()) or ())
        record_ids = tuple(getattr(raw_intent, "record_ids", ()) or ())
        raw_options = getattr(raw_intent, "options", None)
        if raw_options is None:
            options: Mapping[str, Any] = {}
        elif hasattr(raw_options, "model_dump"):
            dumped = raw_options.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            )
            if not isinstance(dumped, Mapping):
                raise AgentApplicationError("Assistant action options are invalid.")
            options = dumped
        elif isinstance(raw_options, Mapping):
            options = raw_options
        else:
            raise AgentApplicationError("Assistant action options are invalid.")
        options_json = _canonical_json(options)
        if (
            not kind
            or not instruction
            or any(not isinstance(item, str) or not item for item in resource_ids)
            or any(not isinstance(item, str) or not item for item in record_ids)
            or len(set(resource_ids)) != len(resource_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise AgentApplicationError(
                "Assistant action intents need complete, unique typed targets."
            )
        unknown_resources = sorted(set(resource_ids) - allowed_resources)
        destination_resource = options.get("destination_resource_id")
        if (
            isinstance(destination_resource, str)
            and destination_resource not in allowed_resources
        ):
            unknown_resources.append(destination_resource)
            unknown_resources.sort()
        if unknown_resources:
            raise AgentApplicationError(
                "Assistant requested targets absent from its exact disclosed "
                "context: resources " + ", ".join(unknown_resources)
            )
        intents.append(
            AgentActionIntent(
                kind=kind,
                resource_ids=resource_ids,
                record_ids=record_ids,
                instruction=instruction,
                options_json=options_json,
            )
        )
    return answer, tuple(intents)


def _intent_wire(intents: Sequence[AgentActionIntent]) -> list[dict[str, Any]]:
    return [
        {
            "kind": intent.kind,
            "resource_ids": list(intent.resource_ids),
            "record_ids": list(intent.record_ids),
            "instruction": intent.instruction,
            "options": dict(intent.options),
        }
        for intent in intents
    ]


def _request_manifest(plan: AgentPlan, *, operation_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "assistant_agent",
        "state": "request",
        "operation_id": operation_id,
        "context": {
            "wire": plan.context.wire,
            "fingerprint": plan.context.fingerprint,
            "resource_ids": list(plan.context.resource_ids),
            "focus_resource_id": plan.context.focus_resource_id,
            "editable_records": [
                record.to_dict() for record in plan.context.editable_records
            ],
        },
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


def _complete_manifest(
    request_manifest: Mapping[str, Any],
    plan: AgentPlan,
    *,
    answer: str,
    action_intents: tuple[AgentActionIntent, ...],
) -> str:
    value = dict(request_manifest)
    value["state"] = "complete"
    value["answer"] = answer
    value["action_intents"] = _intent_wire(action_intents)
    value["provenance"] = _provenance(plan)
    return _canonical_json(value, pretty=True) + "\n"


def _provenance(plan: AgentPlan) -> dict[str, str]:
    return {
        "provider": plan.provider,
        "billing_class": plan.billing_class,
        "model": plan.model,
        "request_fingerprint": plan.request_fingerprint,
    }


def _failed_manifest(
    request_manifest: Mapping[str, Any],
    plan: AgentPlan,
    *,
    message: str,
    provider_reply: bytes,
) -> str:
    value = dict(request_manifest)
    value["state"] = "failed"
    value["failure"] = {
        "message": message,
        "provider_reply_base64": base64.b64encode(provider_reply).decode("ascii"),
        "provider_reply_bytes": len(provider_reply),
        "provider_reply_sha256": _sha256(provider_reply),
    }
    value["provenance"] = _provenance(plan)
    return _canonical_json(value, pretty=True) + "\n"


def _failed_manifest_message(
    manifest: Mapping[str, Any],
    plan: AgentPlan,
    *,
    provider_reply: bytes,
) -> str:
    failure = _require_exact_keys(
        manifest.get("failure"),
        _FAILURE_KEYS,
        label="failure",
    )
    message = failure["message"]
    encoded = failure["provider_reply_base64"]
    byte_count = failure["provider_reply_bytes"]
    digest = failure["provider_reply_sha256"]
    if (
        not isinstance(message, str)
        or not message
        or not isinstance(encoded, str)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not _is_sha256(digest)
    ):
        raise AgentApplicationError(
            "Captured Assistant failure manifest has invalid field types."
        )
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise AgentApplicationError(
            "Captured Assistant failure reply is not canonical base64."
        ) from exc
    if (
        base64.b64encode(decoded).decode("ascii") != encoded
        or len(decoded) != byte_count
        or _sha256(decoded) != digest
        or decoded != provider_reply
    ):
        raise AgentApplicationError(
            "Captured Assistant failure reply differs from its exact provider bytes."
        )
    if manifest.get("provenance") != _provenance(plan):
        raise AgentApplicationError(
            "Captured Assistant failure provenance differs from its request."
        )
    return message


def _commit_captured_failure(
    config: ProjectConfig,
    plan: AgentPlan,
    *,
    operation_id: str,
    manifest_path: Path,
    request_manifest: Mapping[str, Any],
    manifest_revision: str,
    error: BaseException,
) -> bool:
    """Commit a captured ordinary-turn failure without discarding its reply.

    Ordinary Assistant output has no canonical-content authority. Once its exact
    provider bytes and the failure that prevented an answer are both durable in
    the Assistant manifest, leaving the operation live would protect no missing
    output; it would only wedge later conversation.
    """

    journal = operations.OperationJournal.load(config.operations_file)
    held = journal.operations.get(operation_id)
    if held is None or held.state != "result_captured" or held.artifact is None:
        return False
    reply = journal.read_reply(operation_id)
    rendered = _failed_manifest(
        request_manifest,
        plan,
        message=str(error),
        provider_reply=reply,
    )
    with exclusive_path_lock(manifest_path):
        journal.commit_result(
            operation_id,
            lambda: atomic_write_text_bound(
                manifest_path,
                rendered,
                expected_revision=manifest_revision,
            ),
        )
    return True


def _retire_unsent_turn(
    config: ProjectConfig,
    operation_id: str,
    *,
    detail: str,
    cause: BaseException,
) -> NoReturn:
    """Retire authority for a turn whose preparation failed before any send."""
    raise operations.cancel_before_send(
        config.operations_file,
        operation_id,
        error=AgentRunError,
        label="Assistant turn",
        detail=detail,
        cause=cause,
    ) from cause


def run_agent(
    config: ProjectConfig,
    expected: AgentPlan,
    *,
    client: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: revision_provider.Spawn = subprocess.Popen,
    api_call: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
    preview: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Dispatch one turn, capture it, and persist its untrusted closed intent.

    Sending the message authorizes exactly that one journaled turn over that
    message and its exact bounded context, so this dispatches the plan it was
    handed rather than replacing it with a newer one.  What the model receives
    is still bound to what was fingerprinted: the context is checked against
    its own fingerprint here, the plan's user turn embeds that exact wire, the
    provider's request bytes embed the user turn, and the provider refuses
    request bytes or a prompt channel that no longer match.  Preparation still
    refuses authentication or CLI drift before any authority exists.
    """
    _report_progress(progress, "Preparing answer")
    if expected.repository_root != config.root.resolve():
        raise AgentApplicationError(
            "This Assistant request belongs to a different repository; nothing was sent."
        )
    _validated_context(expected.context)
    # The fingerprint binds the context *wire* and nothing else, so the ids the
    # turn is authorized over — plus its focus, history and message — are bound
    # only where they already are: inside the user turn the provider is sent.
    # Rebuild it. Without this a plan whose `resource_ids` were widened after
    # planning dispatches a request that never mentions the extra resource,
    # then has an intent naming it accepted and committed into a manifest that
    # `recover_agent` refuses to read back.
    if (
        _user_turn(
            context=expected.context,
            history=expected.history,
            message=expected.message,
        )
        != expected.user_turn
        or expected.provider_plan.user_turn != expected.user_turn
    ):
        raise AgentApplicationError(
            "Assistant plan no longer matches its dispatched user turn; nothing was sent."
        )
    provider = revision_provider.provider_for(expected.provider)
    prepared_provider = provider.prepare(
        expected.provider_plan,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
        client=client,
    )
    operation_id = str(uuid.uuid4())
    manifest_path = config.assistant_dir / f"{operation_id}.json"

    with exclusive_path_lock(manifest_path):
        operations.prepare_artifact_store(config.operations_file)
        prepare_bound_directory(config.assistant_dir)
        journal = operations.OperationJournal.load(config.operations_file)
        source = expected.context.focus_resource_id or "janki-project"
        try:
            journal.authorize(
                operation_id,
                kind="assistant_agent",
                source_file=source,
                source_sha256=expected.context.fingerprint,
                request_fp=expected.request_fingerprint,
                model=expected.model,
            )
        except Exception as exc:  # noqa: BLE001 - authority may have landed
            raise AgentRunError(
                f"Assistant authority could not be recorded safely: {exc} Inspect "
                "janki operations before retrying.",
                operation_id=operation_id,
                provider_dispatched=False,
            ) from exc
        try:
            journal.begin_response_capture(operation_id)
        except Exception as exc:  # noqa: BLE001 - preparation may have landed
            _retire_unsent_turn(
                config,
                operation_id,
                detail="The Assistant response frame spool could not be prepared.",
                cause=exc,
            )
        request_manifest = _request_manifest(expected, operation_id=operation_id)
        request_text = _canonical_json(request_manifest, pretty=True) + "\n"
        request_revision = _sha256(request_text.encode("utf-8"))
        try:
            atomic_write_text_bound(manifest_path, request_text, expected_absent=True)
        except Exception as exc:  # noqa: BLE001 - publication may have landed
            _retire_unsent_turn(
                config,
                operation_id,
                detail="The durable Assistant request manifest could not be prepared.",
                cause=exc,
            )

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
        raise AgentRunError(
            f"Assistant turn {operation_id} was not dispatched because its dispatch "
            f"boundary could not be recorded safely: {exc}.",
            operation_id=operation_id,
            provider_dispatched=False,
        ) from exc

    def capture_reply(raw_reply: bytes) -> None:
        if not isinstance(raw_reply, bytes):
            raise operations.OperationError(
                "Assistant provider capture must supply exact response bytes."
            )
        journal.capture_result(
            operation_id,
            lambda: operations.capture_artifact(
                config.operations_file, operation_id, raw_reply
            ),
        )

    try:
        _report_progress(progress, "Writing answer")
        result = provider.dispatch(
            prepared_provider,
            capture=capture_reply,
            spawn=provider_spawn,
            api_call=api_call,
            frame=lambda payload: journal.append_response_frame(operation_id, payload),
            preview=preview,
        )
        captured = operations.OperationJournal.load(config.operations_file).operations.get(
            operation_id
        )
        if captured is None or captured.state != "result_captured":
            raise operations.OperationError(
                f"Assistant turn {operation_id} returned data without durably "
                "capturing its exact provider reply."
            )
        answer, intents = _decode_answer(
            result,
            model=expected.model,
            captured=True,
            context=expected.context,
        )
    except Exception as exc:  # noqa: BLE001 - settle every post-dispatch failure
        try:
            committed_failure = False
            if isinstance(exc, AgentUnansweredError):
                _report_progress(progress, "Saving answer")
                committed_failure = _commit_captured_failure(
                    config,
                    expected,
                    operation_id=operation_id,
                    manifest_path=manifest_path,
                    request_manifest=request_manifest,
                    manifest_revision=request_revision,
                    error=exc,
                )
            if not committed_failure:
                classify_dispatch_failure(config, journal, operation_id, exc)
        except (JankiError, OSError) as journal_error:
            raise AgentRunError(
                f"Assistant turn {operation_id} failed after dispatch: {exc} Janki "
                f"could not settle its journal entry: {journal_error}. This call may "
                "have consumed allowance or been billed; do not retry until you "
                "inspect janki operations.",
                operation_id=operation_id,
                provider_dispatched=True,
            ) from exc
        raise AgentRunError(
            f"Assistant turn {operation_id} failed after dispatch: {exc}",
            operation_id=operation_id,
            provider_dispatched=True,
        ) from exc

    try:
        rendered = _complete_manifest(
            request_manifest,
            expected,
            answer=answer,
            action_intents=intents,
        )
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
    except Exception as exc:  # noqa: BLE001 - preserve a captured valid answer
        try:
            classify_dispatch_failure(config, journal, operation_id, exc)
        except JankiError as journal_error:
            raise AgentRunError(
                f"Assistant turn {operation_id} failed after dispatch: {exc} Janki "
                f"could not settle its journal entry: {journal_error}. This call may "
                "have consumed allowance or been billed; do not retry until you "
                "inspect janki operations.",
                operation_id=operation_id,
                provider_dispatched=True,
            ) from exc
        raise AgentRunError(
            f"Assistant turn {operation_id} failed after dispatch: {exc}",
            operation_id=operation_id,
            provider_dispatched=True,
        ) from exc

    return AgentRunResult(
        answer=answer,
        action_intents=intents,
        operation_id=operation_id,
        manifest_path=manifest_path,
        request_fingerprint=expected.request_fingerprint,
    )


def recover_agent(
    config: ProjectConfig,
    operation_id: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Finish one captured Assistant answer without another provider call.

    Recovery intentionally rebuilds the exact provider plan and editable-card
    scope from the durable request manifest.  It never refreshes repository
    context: the captured model answered the snapshot recorded there.
    """
    _report_progress(progress, "Preparing answer")
    _canonical_operation_id(operation_id)
    journal = operations.OperationJournal.load(config.operations_file)
    held = journal.operations.get(operation_id)
    if held is None:
        raise AgentApplicationError(
            f"No journaled Assistant agent operation {operation_id!r} to recover."
        )
    if held.kind != "assistant_agent":
        raise AgentApplicationError(
            f"Operation {operation_id!r} is {held.kind!r}, not an Assistant agent turn."
        )
    if held.state != "result_captured" or held.artifact is None:
        raise AgentApplicationError(
            f"Assistant operation {operation_id!r} has no captured agent answer "
            f"ready for recovery; state is {held.state!r}."
        )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    try:
        manifest_bytes = read_bytes_bound(manifest_path)
        manifest = _strict_json_text(
            manifest_bytes.decode("utf-8", errors="strict"),
            label=f"Assistant manifest {manifest_path}",
        )
    except (DataError, OSError, UnicodeError) as exc:
        raise AgentApplicationError(
            f"Could not read Assistant manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise AgentApplicationError(
            f"Assistant manifest {manifest_path} must contain one JSON object."
        )
    manifest_state = manifest.get("state")
    manifest_keys = {
        "request": _REQUEST_MANIFEST_KEYS,
        "complete": _COMPLETE_MANIFEST_KEYS,
        "failed": _FAILED_MANIFEST_KEYS,
    }.get(manifest_state)
    if manifest_keys is None:
        raise AgentApplicationError(
            f"Assistant manifest {manifest_path} has an unknown state."
        )
    _require_exact_keys(
        manifest,
        manifest_keys,
        label="top-level",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "assistant_agent"
        or manifest_state not in {"request", "complete", "failed"}
        or manifest.get("operation_id") != operation_id
    ):
        raise AgentApplicationError(
            f"Assistant manifest {manifest_path} is not the captured agent request."
        )
    context_raw = _require_exact_keys(
        manifest.get("context"),
        _CONTEXT_MANIFEST_KEYS,
        label="context",
    )
    request = _require_exact_keys(
        manifest.get("request"),
        _REQUEST_KEYS,
        label="request",
    )
    try:
        wire = context_raw["wire"]
        fingerprint = context_raw["fingerprint"]
        resource_ids_raw = context_raw["resource_ids"]
        focus_resource_id = context_raw["focus_resource_id"]
        editable_records_raw = context_raw["editable_records"]
        if (
            not isinstance(wire, str)
            or not isinstance(fingerprint, str)
            or not isinstance(resource_ids_raw, list)
            or any(not isinstance(item, str) for item in resource_ids_raw)
            or (
                focus_resource_id is not None
                and not isinstance(focus_resource_id, str)
            )
            or not isinstance(editable_records_raw, list)
        ):
            raise AgentApplicationError(
                "Captured Assistant manifest has invalid context field types."
            )
        records = tuple(
            _record_from_manifest(item, index=index)
            for index, item in enumerate(editable_records_raw, start=1)
        )
        context = _validated_context(
            AgentContext(
                wire=wire,
                fingerprint=fingerprint,
                resource_ids=tuple(resource_ids_raw),
                editable_records=records,
                focus_resource_id=focus_resource_id,
            )
        )
        expected_source = context.focus_resource_id or "janki-project"
        if (
            held.source_file != expected_source
            or held.source_sha256 != context.fingerprint
        ):
            raise AgentApplicationError(
                "Captured Assistant context does not match its operation journal entry."
            )
        history = _history_from_manifest(request["history"])
        message = request["message"]
        task_template = request["task_template"]
        scalar_fields = (
            message,
            task_template,
            request["task_template_sha256"],
            request["user_turn"],
        )
        if any(not isinstance(item, str) for item in scalar_fields):
            raise AgentApplicationError(
                "Captured Assistant manifest has non-text request identity fields."
            )
        if not message.strip() or not task_template:
            raise AgentApplicationError(
                "Captured Assistant manifest has an empty message or task template."
            )
        if prompts.fingerprint(task_template) != request["task_template_sha256"]:
            raise AgentApplicationError("Captured Assistant prompt fingerprint differs.")
        user_turn = _user_turn(context=context, history=history, message=message)
        if user_turn != request["user_turn"]:
            raise AgentApplicationError("Captured Assistant user turn is inconsistent.")
        blocks = tuple(claude_client.system_blocks(task_template))
        if _plain_json_value(blocks) != request["system_blocks"]:
            raise AgentApplicationError("Captured Assistant system prompt is inconsistent.")
        provider_manifest = request["provider_plan"]
        if not isinstance(provider_manifest, Mapping):
            raise AgentApplicationError(
                "Captured Assistant manifest has invalid provider metadata."
            )
        provider_plan = revision_provider.provider_plan_from_manifest(
            provider_manifest,
            model=str(provider_manifest.get("model", "")),
            style_guide="",
            task_template=task_template,
            system_blocks=blocks,
            user_turn=user_turn,
            schema=ai_schema.assistant_agent_schema(),
        )
    except (KeyError, TypeError, JankiError) as exc:
        if isinstance(exc, AgentApplicationError):
            raise
        raise AgentApplicationError(
            f"Captured Assistant manifest has invalid request provenance: {exc}"
        ) from exc
    if (
        provider_plan.model != held.model
        or provider_plan.request_fingerprint != held.request_fp
    ):
        raise AgentApplicationError(
            "Captured Assistant manifest does not match its operation journal entry."
        )
    plan = AgentPlan(
        repository_root=config.root.resolve(),
        message=message,
        history=history,
        context=context,
        task_template=task_template,
        user_turn=user_turn,
        provider_plan=provider_plan,
    )
    reply = journal.read_reply(operation_id)
    if _sha256(reply) != held.artifact.content_sha256:
        raise AgentApplicationError("Captured Assistant reply differs from its receipt.")
    revision = _sha256(manifest_bytes)
    if manifest_state == "failed":
        failure_message = _failed_manifest_message(
            manifest,
            plan,
            provider_reply=reply,
        )
        _report_progress(progress, "Saving answer")

        def preserve_failed_manifest() -> None:
            if _sha256(read_bytes_bound(manifest_path)) != revision:
                raise AgentApplicationError(
                    "Captured Assistant failure manifest changed during recovery."
                )

        with exclusive_path_lock(manifest_path):
            journal.commit_result(operation_id, preserve_failed_manifest)
        return AgentRunResult(
            answer=f"The earlier Assistant turn failed: {failure_message}",
            action_intents=(),
            operation_id=operation_id,
            manifest_path=manifest_path,
            request_fingerprint=held.request_fp,
        )
    _report_progress(progress, "Writing answer")
    try:
        result = revision_provider.provider_for(plan.provider).recover(provider_plan, reply)
        answer, intents = _decode_answer(
            result,
            model=plan.model,
            captured=True,
            context=plan.context,
        )
    except JankiError as exc:
        if manifest_state != "request" or not isinstance(exc, AgentUnansweredError):
            raise AgentApplicationError(
                f"Captured Assistant reply for {operation_id} could not be decoded: "
                f"{exc}"
            ) from exc
        request_manifest = {key: manifest[key] for key in _REQUEST_MANIFEST_KEYS}
        request_manifest["state"] = "request"
        _report_progress(progress, "Saving answer")
        if not _commit_captured_failure(
            config,
            plan,
            operation_id=operation_id,
            manifest_path=manifest_path,
            request_manifest=request_manifest,
            manifest_revision=revision,
            error=exc,
        ):
            raise AgentApplicationError(
                f"Captured Assistant failure for {operation_id} changed during "
                "recovery."
            ) from exc
        return AgentRunResult(
            answer=f"The earlier Assistant turn failed: {exc}",
            action_intents=(),
            operation_id=operation_id,
            manifest_path=manifest_path,
            request_fingerprint=held.request_fp,
        )
    request_manifest = {key: manifest[key] for key in _REQUEST_MANIFEST_KEYS}
    request_manifest["state"] = "request"
    rendered = _complete_manifest(
        request_manifest,
        plan,
        answer=answer,
        action_intents=intents,
    )
    if manifest_state == "complete" and _sha256(rendered.encode("utf-8")) != _sha256(
        manifest_bytes
    ):
        raise AgentApplicationError(
            f"Completed Assistant manifest for {operation_id} does not match its "
            "exact captured answer, intents, and provenance."
        )
    _report_progress(progress, "Saving answer")
    with exclusive_path_lock(manifest_path):
        journal.commit_result(
            operation_id,
            lambda: atomic_write_text_bound(
                manifest_path,
                rendered,
                expected_revision=revision,
            ),
        )
    return AgentRunResult(
        answer=answer,
        action_intents=intents,
        operation_id=operation_id,
        manifest_path=manifest_path,
        request_fingerprint=held.request_fp,
    )
