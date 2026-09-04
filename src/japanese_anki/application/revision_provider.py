"""Provider transports for the existing-card ``revise`` pass.

The revision service owns authority, journaling, and staging.  This module owns
only the provider-specific request: planning its exact bytes, preparing its
authentication before authority exists, dispatching it, and decoding an exact
captured reply.  Both transports therefore share one byte-oriented capture
boundary even though Anthropic's API returns an SDK object and Claude Code
streams JSON frames on stdout.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from japanese_anki import claude_client, operations, prompts
from japanese_anki.credential_safety import redact_environment_credentials
from japanese_anki.errors import JankiError

__all__ = [
    "ANTHROPIC_API_PROVIDER",
    "CLAUDE_CODE_PROVIDER",
    "PERSISTENT_MANIFEST_KEYS",
    "PROVIDERS",
    "PreparedRevisionProvider",
    "RevisionProvider",
    "RevisionProviderError",
    "RevisionProviderPlan",
    "billing_display",
    "plan_provider",
    "provider_plan_from_manifest",
    "provider_for",
]

ANTHROPIC_API_PROVIDER = "anthropic-api"
CLAUDE_CODE_PROVIDER = "claude-code"

Runner = Callable[..., subprocess.CompletedProcess[Any]]
Spawn = Callable[..., subprocess.Popen[bytes]]
Which = Callable[..., str | None]
Capture = Callable[[bytes], None]

PERSISTENT_MANIFEST_KEYS = frozenset(
    {
        "provider",
        "billing_class",
        "auth",
        "model",
        "transport",
        "request_bytes_utf8",
        "request_fingerprint",
        "request_bytes_sha256",
        "response_schema_fingerprint",
    }
)


class RevisionProviderError(JankiError):
    """A revision provider could not be planned, prepared, or decoded safely."""


_SUPPORTED_REVISION_MODEL = "claude-opus-5"


def _require_supported_model(model: str) -> None:
    if model != _SUPPORTED_REVISION_MODEL:
        raise RevisionProviderError(
            "Revision model must be the repository's exact supported pinned model "
            f"{_SUPPORTED_REVISION_MODEL!r}."
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request_identity(
    *,
    provider: str,
    billing_class: str,
    auth: Mapping[str, Any],
    model: str,
    transport: Mapping[str, Any],
    request_bytes: bytes,
    response_schema_fingerprint: str,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "auth": _plain_value(auth),
                "billing_class": billing_class,
                "model": model,
                "provider": provider,
                "request_bytes_sha256": _sha256(request_bytes),
                "response_schema_fingerprint": response_schema_fingerprint,
                "transport": _plain_value(transport),
            }
        ).encode("utf-8")
    )


def billing_display(
    provider: str,
    billing_class: str,
    auth: Mapping[str, Any],
) -> str:
    """Presentation derived only from durable provider and authentication facts."""
    if provider == ANTHROPIC_API_PROVIDER:
        if (
            billing_class != "anthropic-platform-api"
            or auth.get("auth_method") != "environment-api-key"
        ):
            raise RevisionProviderError(
                "Anthropic API billing metadata is inconsistent."
            )
        return "Anthropic API billing"
    if provider == CLAUDE_CODE_PROVIDER:
        subscription = auth.get("subscription_type")
        if billing_class != "claude-subscription" or subscription not in {
            "pro",
            "max",
        }:
            raise RevisionProviderError(
                "Claude subscription billing metadata is inconsistent."
            )
        return f"Claude {str(subscription).title()} subscription via Claude Code"
    raise RevisionProviderError(f"Unknown revision provider {provider!r}.")


def _type_adapter(schema: Any) -> Any:
    try:
        import pydantic
    except ImportError as exc:  # pragma: no cover - installation boundary
        raise RevisionProviderError(
            "Revision providers need janki's AI support. Install it with: "
            "pip install -e '.[ai]'"
        ) from exc
    return pydantic.TypeAdapter(schema)


def _neutral_wire_schema(schema: Any) -> Mapping[str, Any]:
    """Provider-neutral strict JSON schema, without loading an API SDK."""
    value = _type_adapter(schema).json_schema()
    if not isinstance(value, Mapping):
        raise RevisionProviderError("Revision response schema must be one JSON object.")
    return value


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RevisionProviderError(
                    f"Captured {label} repeats JSON key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise RevisionProviderError(
            f"Captured {label} contains invalid JSON value {value!r}."
        )

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except RevisionProviderError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RevisionProviderError(f"Could not decode captured {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RevisionProviderError(f"Captured {label} must be one JSON object.")
    return value


def _frozen_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _frozen_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen_value(item) for item in value)
    return value


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-copy a plan mapping so no nested request identity can be mutated."""
    frozen = _frozen_value(value)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class RevisionProviderPlan:
    """Safe, exact provider identity rendered before owner confirmation."""

    provider: str
    billing_class: str
    auth_metadata: Mapping[str, Any]
    model: str
    transport: Mapping[str, Any]
    request_bytes: bytes = field(repr=False)
    request_fingerprint: str
    response_schema_fingerprint: str
    system_blocks: tuple[Mapping[str, Any], ...] = field(repr=False)
    user_turn: str = field(repr=False)
    schema: Any = field(repr=False, compare=False)

    @property
    def billing_display(self) -> str:
        return billing_display(
            self.provider,
            self.billing_class,
            self.auth_metadata,
        )

    def persistent_manifest(self) -> dict[str, Any]:
        """Serializable request provenance containing no credentials or host path."""
        return {
            "provider": self.provider,
            "billing_class": self.billing_class,
            "auth": _plain_value(self.auth_metadata),
            "model": self.model,
            "transport": _plain_value(self.transport),
            "request_bytes_utf8": self.request_bytes.decode("utf-8", errors="strict"),
            "request_fingerprint": self.request_fingerprint,
            "request_bytes_sha256": _sha256(self.request_bytes),
            "response_schema_fingerprint": self.response_schema_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PreparedRevisionProvider:
    """Ephemeral provider handle proved before the journal grants authority."""

    plan: RevisionProviderPlan
    resource: Any = field(default=None, repr=False, compare=False)
    executable: str | None = field(default=None, repr=False, compare=False)
    environment: Mapping[str, str] | None = field(default=None, repr=False, compare=False)


class RevisionProvider(Protocol):
    name: str

    def plan(
        self,
        *,
        model: str,
        style_guide: str,
        task_template: str,
        system_blocks: Sequence[Mapping[str, Any]],
        user_turn: str,
        schema: Any,
        effort: str | None,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
    ) -> RevisionProviderPlan: ...

    def prepare(
        self,
        plan: RevisionProviderPlan,
        *,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
        client: Any | None = None,
    ) -> PreparedRevisionProvider: ...

    def dispatch(
        self,
        prepared: PreparedRevisionProvider,
        *,
        capture: Capture,
        spawn: Spawn = subprocess.Popen,
        api_call: Callable[..., Any] | None = None,
        frame: Callable[[str], None] | None = None,
        preview: Callable[[str], None] | None = None,
    ) -> claude_client.CallResult: ...

    def recover(
        self, plan: RevisionProviderPlan, raw_reply: bytes
    ) -> claude_client.CallResult: ...


def _validate_plan(plan: RevisionProviderPlan, provider: str) -> None:
    if plan.provider != provider:
        raise RevisionProviderError(
            f"Provider {provider!r} cannot execute a {plan.provider!r} revision plan."
        )
    expected = _request_identity(
        provider=plan.provider,
        billing_class=plan.billing_class,
        auth=plan.auth_metadata,
        model=plan.model,
        transport=plan.transport,
        request_bytes=plan.request_bytes,
        response_schema_fingerprint=plan.response_schema_fingerprint,
    )
    if expected != plan.request_fingerprint:
        raise RevisionProviderError(
            "The revision provider request bytes no longer match their fingerprint."
        )


def _validate_api_channels(plan: RevisionProviderPlan) -> None:
    request = _strict_json(plan.request_bytes, label="Anthropic API request")
    output = request.get("output_config")
    output_format = output.get("format") if isinstance(output, Mapping) else None
    embedded_schema = (
        output_format.get("schema") if isinstance(output_format, Mapping) else None
    )
    expected_effort = plan.transport.get("effort")
    observed_effort = output.get("effort") if isinstance(output, Mapping) else None
    fixed = {
        "model": plan.model,
        "max_tokens": plan.transport.get("max_tokens"),
        "system": [_plain_value(block) for block in plan.system_blocks],
        "messages": [{"role": "user", "content": plan.user_turn}],
        "thinking": _plain_value(plan.transport.get("thinking")),
    }
    if (
        any(request.get(key) != value for key, value in fixed.items())
        or not isinstance(output_format, Mapping)
        or output_format.get("type") != "json_schema"
        or not isinstance(embedded_schema, Mapping)
        or observed_effort != expected_effort
        or prompts.schema_fingerprint(embedded_schema)
        != plan.response_schema_fingerprint
    ):
        raise RevisionProviderError(
            "The Anthropic API plan's prompt channels no longer match its exact "
            "request bytes."
        )


def _validate_cli_channels(plan: RevisionProviderPlan) -> Mapping[str, Any]:
    request = _strict_json(plan.request_bytes, label="Claude Code request")
    if set(request) != {
        "argv",
        "cli_version",
        "controlled_environment",
        "cwd",
        "stdin_utf8",
    }:
        raise RevisionProviderError("Claude Code plan has invalid request fields.")
    expected = {
        "argv": _plain_value(plan.transport.get("argv")),
        "cli_version": plan.transport.get("cli_version"),
        "controlled_environment": _plain_value(
            plan.transport.get("controlled_environment")
        ),
        "cwd": plan.transport.get("cwd"),
        "stdin_utf8": plan.user_turn,
    }
    if _canonical_json(request) != _canonical_json(expected):
        raise RevisionProviderError(
            "The Claude Code plan's command or prompt channels no longer match "
            "its exact request bytes."
        )
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RevisionProviderError("Claude Code plan has an invalid argv.")

    def option(name: str) -> str:
        positions = [index for index, value in enumerate(argv) if value == name]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise RevisionProviderError(
                f"Claude Code plan has an invalid {name} option."
            )
        return argv[positions[0] + 1]

    try:
        embedded_schema = json.loads(option("--json-schema"))
    except json.JSONDecodeError as exc:
        raise RevisionProviderError(
            "Claude Code plan has an invalid JSON schema."
        ) from exc
    invalid_response_contract = (
        not isinstance(embedded_schema, Mapping)
        or prompts.schema_fingerprint(embedded_schema)
        != plan.response_schema_fingerprint
    )
    # The depth is bound three ways: the command's option, the transport
    # identity the fingerprint covers, and the environment the child runs
    # under. They are compared against each other and against the CLI's own
    # vocabulary — never against today's configured level — so a recovered plan
    # is validated entirely from its stored bytes.
    stored_effort = plan.transport.get("effort")
    bound_effort = {
        option("--effort"),
        str(stored_effort),
        str(plan.transport["controlled_environment"]["CLAUDE_CODE_EFFORT_LEVEL"]),
    }
    # The whole command is bound, not a list of flags: it is rebuilt from the
    # plan's own model, prompt, schema and depth and compared as one value. A
    # stored plan that dropped --safe-mode, or asked for the single-envelope
    # output this module can no longer read, fails the same comparison as one
    # with the wrong model, and a flag added to _cli_argv is bound the moment
    # it exists.
    if (
        tuple(argv)
        != _cli_argv(
            model=plan.model,
            system_prompt=_combined_system_prompt(plan.system_blocks),
            schema_json=option("--json-schema"),
            effort=str(stored_effort),
        )
        or stored_effort not in claude_client.EFFORT_LEVELS
        or len(bound_effort) != 1
        or invalid_response_contract
    ):
        raise RevisionProviderError(
            "The Claude Code plan's bound model, prompts, or response contract "
            "do not match."
        )
    return request


def _api_result_from_reply(
    plan: RevisionProviderPlan, raw_reply: bytes
) -> claude_client.CallResult:
    payload = _strict_json(raw_reply, label="Anthropic API reply")
    stop_reason_raw = payload.get("stop_reason")
    stop_reason = str(stop_reason_raw) if stop_reason_raw is not None else None
    if stop_reason not in claude_client.COMPLETE_STOP_REASONS:
        refusal = None
        details = payload.get("stop_details")
        if stop_reason == "refusal" and isinstance(details, Mapping):
            refusal = claude_client.Refusal(
                category=str(details.get("category", "") or ""),
                explanation=str(details.get("explanation", "") or ""),
            )
        return claude_client.CallResult(None, stop_reason, refusal)
    content = payload.get("content")
    blocks = content if isinstance(content, list) else []
    answers = [
        block.get("text")
        for block in blocks
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if len(answers) != 1:
        raise RevisionProviderError(
            f"{plan.model} finished normally but its captured API reply must have "
            "exactly one text answer block."
        )
    answer = answers[0]
    try:
        parsed = _type_adapter(plan.schema).validate_json(answer)
    except Exception as exc:
        raise RevisionProviderError(
            f"{plan.model} finished normally but its captured answer did not match "
            f"the revision schema: {exc}"
        ) from exc
    return claude_client.CallResult(parsed, stop_reason, None)


class _AnthropicAPIRevisionProvider:
    name = ANTHROPIC_API_PROVIDER

    def plan(
        self,
        *,
        model: str,
        style_guide: str,
        task_template: str,
        system_blocks: Sequence[Mapping[str, Any]],
        user_turn: str,
        schema: Any,
        effort: str | None,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
    ) -> RevisionProviderPlan:
        del style_guide, task_template, env, runner, which
        _require_supported_model(model)
        wire_schema = claude_client.wire_schema(schema)
        blocks = tuple(_frozen_mapping(block) for block in system_blocks)
        # One owner for the API request shape: this is the same helper used by
        # claude_client.parse_call immediately before the SDK dispatches it.
        request = claude_client.request_body(
            model,
            tuple(_plain_value(block) for block in blocks),
            user_turn,
            schema,
            claude_client.DEFAULT_MAX_TOKENS,
            effort,
        )
        request_bytes = _canonical_json(request).encode("utf-8")
        transport = _frozen_mapping(
            {
                "kind": "anthropic-messages-api",
                "max_tokens": claude_client.DEFAULT_MAX_TOKENS,
                "effort": effort,
                "thinking": {"type": "adaptive"},
            }
        )
        billing_class = "anthropic-platform-api"
        auth = _frozen_mapping(
            {
                "auth_method": "environment-api-key",
                "api_key_source": claude_client.API_KEY_ENV,
            }
        )
        schema_fingerprint = prompts.schema_fingerprint(wire_schema)
        return RevisionProviderPlan(
            provider=self.name,
            billing_class=billing_class,
            auth_metadata=auth,
            model=model,
            transport=transport,
            request_bytes=request_bytes,
            request_fingerprint=_request_identity(
                provider=self.name,
                billing_class=billing_class,
                auth=auth,
                model=model,
                transport=transport,
                request_bytes=request_bytes,
                response_schema_fingerprint=schema_fingerprint,
            ),
            response_schema_fingerprint=schema_fingerprint,
            system_blocks=blocks,
            user_turn=user_turn,
            schema=schema,
        )

    def prepare(
        self,
        plan: RevisionProviderPlan,
        *,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
        client: Any | None = None,
    ) -> PreparedRevisionProvider:
        del runner, which
        _validate_plan(plan, self.name)
        _validate_api_channels(plan)
        resource = client if client is not None else claude_client.prepare_paid_client(env)
        return PreparedRevisionProvider(plan=plan, resource=resource)

    def dispatch(
        self,
        prepared: PreparedRevisionProvider,
        *,
        capture: Capture,
        spawn: Spawn = subprocess.Popen,
        api_call: Callable[..., Any] | None = None,
        frame: Callable[[str], None] | None = None,
        preview: Callable[[str], None] | None = None,
    ) -> claude_client.CallResult:
        # The SDK returns one object rather than a stream of frames, so this
        # transport has nothing to spool or preview.
        del spawn, frame, preview
        plan = prepared.plan
        _validate_plan(plan, self.name)
        _validate_api_channels(plan)
        call = claude_client.parse_call if api_call is None else api_call

        captured: bytes | None = None

        def capture_response(response: Any) -> None:
            nonlocal captured
            if captured is not None:
                raise RevisionProviderError("Anthropic API returned multiple replies.")
            raw = operations.serialize_response(response)
            capture(raw)
            captured = raw

        call(
            plan.model,
            tuple(_plain_value(block) for block in plan.system_blocks),
            plan.user_turn,
            plan.schema,
            client=prepared.resource,
            max_tokens=int(plan.transport["max_tokens"]),
            effort=plan.transport["effort"],
            capture=capture_response,
        )
        if captured is None:
            raise RevisionProviderError("Anthropic API returned no reply to capture.")
        return _api_result_from_reply(plan, captured)

    def recover(
        self, plan: RevisionProviderPlan, raw_reply: bytes
    ) -> claude_client.CallResult:
        _validate_plan(plan, self.name)
        _validate_api_channels(plan)
        return _api_result_from_reply(plan, raw_reply)


def _required_effort(effort: str | None) -> str:
    """The depth for one Claude Code call, which is never absent.

    The CLI takes ``--effort`` on every invocation, so there is no shape of
    this request that omits it — unlike the API, whose older models reject the
    key outright and must be sent none.
    """
    if effort is None:
        raise RevisionProviderError(
            "Claude Code revisions need one reasoning depth; the CLI takes "
            "--effort on every call."
        )
    return effort


def _controlled_claude_environment(effort: str) -> Mapping[str, str]:
    """The exact environment one planned Claude Code call runs under.

    Built per plan rather than held as a module constant because the depth is
    the caller's, and the plan stores these bytes: dispatch and recovery both
    read the stored mapping, so the environment a call ran under is the one its
    fingerprint covers.
    """
    return MappingProxyType(
        {
            "CLAUDE_CODE_SAFE_MODE": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_EFFORT_LEVEL": effort,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(claude_client.DEFAULT_MAX_TOKENS),
            "MAX_STRUCTURED_OUTPUT_RETRIES": "0",
            "CLAUDE_CODE_MAX_TURNS": "1",
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
            "CLAUDE_CODE_FORK_SUBAGENT": "0",
            "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
        }
    )


_HOST_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TMPDIR",
        "USER",
        "XDG_CONFIG_HOME",
    }
)


def _sanitized_claude_environment(
    env: Mapping[str, str] | None,
    controlled: Mapping[str, str],
) -> dict[str, str]:
    source = os.environ if env is None else env
    clean = {
        name: str(source[name]) for name in _HOST_ENV_ALLOWLIST if name in source
    }
    clean.update({str(name): str(value) for name, value in controlled.items()})
    return clean


def _as_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    return str(value).encode("utf-8")


def _resolve_claude(which: Which, env: Mapping[str, str]) -> str:
    try:
        executable = which("claude", path=env.get("PATH"))
    except TypeError:  # small injected test doubles commonly take one arg
        executable = which("claude")
    if not executable:
        raise RevisionProviderError(
            "The Claude Code CLI is not installed. Install it and sign in with "
            "Claude Pro or Max; no provider was contacted."
        )
    return str(executable)


def _run_probe(
    runner: Runner,
    command: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    label: str,
) -> bytes:
    try:
        completed = runner(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RevisionProviderError(
            "The Claude Code CLI disappeared before its subscription could be "
            "verified; no provider was contacted."
        ) from exc
    except OSError as exc:
        raise RevisionProviderError(
            f"Could not run the Claude Code {label} check: {exc}. No provider "
            "was contacted."
        ) from exc
    if completed.returncode != 0:
        raise RevisionProviderError(
            f"Claude Code {label} check failed (exit {completed.returncode}); "
            "no provider was contacted."
        )
    return _as_bytes(completed.stdout)


_VERSION = re.compile(rb"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?)(?![0-9])")


def _probe_claude(
    *,
    env: Mapping[str, str] | None,
    runner: Runner,
    which: Which,
    controlled: Mapping[str, str],
) -> tuple[str, dict[str, str], Mapping[str, Any], str]:
    clean = _sanitized_claude_environment(env, controlled)
    executable = _resolve_claude(which, clean)
    with tempfile.TemporaryDirectory(prefix="janki-claude-probe-") as temporary:
        version_output = _run_probe(
            runner,
            [executable, "--safe-mode", "--version"],
            cwd=temporary,
            env=clean,
            label="version",
        )
        auth_output = _run_probe(
            runner,
            [executable, "--safe-mode", "auth", "status", "--json"],
            cwd=temporary,
            env=clean,
            label="authentication",
        )
    matched = _VERSION.search(version_output)
    if matched is None:
        raise RevisionProviderError(
            "Claude Code returned an unrecognized version; no provider was contacted."
        )
    version = matched.group(1).decode("ascii")
    auth = _strict_json(auth_output, label="Claude Code authentication status")
    subscription = auth.get("subscriptionType")
    if (
        auth.get("loggedIn") is not True
        or auth.get("authMethod") != "claude.ai"
        or auth.get("apiProvider") != "firstParty"
        or subscription not in {"pro", "max"}
        or auth.get("apiKeySource") is not None
    ):
        raise RevisionProviderError(
            "Claude Code is not authenticated through a Claude Pro or Max "
            "subscription. Sign in with claude.ai and remove API, Bedrock, Vertex, "
            "or Foundry overrides; no model call was made."
        )
    safe_auth = _frozen_mapping(
        {
            "auth_method": "claude.ai",
            "api_provider": "firstParty",
            "subscription_type": str(subscription),
            "api_key_source": None,
        }
    )
    return executable, clean, safe_auth, version


def _combined_system_prompt(blocks: Sequence[Mapping[str, Any]]) -> str:
    texts: list[str] = []
    for block in blocks:
        text = block.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    if not texts:
        raise RevisionProviderError(
            "A Claude Code revision needs non-empty system instructions."
        )
    return "\n\n".join(texts)


def _cli_argv(
    *, model: str, system_prompt: str, schema_json: str, effort: str
) -> tuple[str, ...]:
    # ``--verbose`` is not optional decoration: with ``--print`` the CLI exits 1
    # on ``--output-format=stream-json`` without it.
    return (
        "claude",
        "-p",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--no-chrome",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--prompt-suggestions",
        "false",
        "--verbose",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--model",
        model,
        "--effort",
        effort,
        "--json-schema",
        schema_json,
        "--system-prompt",
        system_prompt,
    )


def _claude_code_plan(
    *,
    model: str,
    system_blocks: Sequence[Mapping[str, Any]],
    user_turn: str,
    schema: Any,
    effort: str,
    auth: Mapping[str, Any],
    version: str,
) -> RevisionProviderPlan:
    _require_supported_model(model)
    response_contract = _neutral_wire_schema(schema)
    schema_json = _canonical_json(response_contract)
    blocks = tuple(_frozen_mapping(block) for block in system_blocks)
    argv = _cli_argv(
        model=model,
        system_prompt=_combined_system_prompt(blocks),
        schema_json=schema_json,
        effort=effort,
    )
    controlled = _controlled_claude_environment(effort)
    transport = _frozen_mapping(
        {
            "kind": "claude-code-cli",
            "cli": "claude",
            "cli_version": version,
            "argv": argv,
            "cwd": "fresh-empty-temporary-directory",
            "stdin": "exact-user-turn-utf8",
            "effort": effort,
            "controlled_environment": controlled,
        }
    )
    request_bytes = _canonical_json(
        {
            "argv": argv,
            "cli_version": version,
            "controlled_environment": dict(controlled),
            "cwd": "fresh-empty-temporary-directory",
            "stdin_utf8": user_turn,
        }
    ).encode("utf-8")
    billing_class = "claude-subscription"
    schema_fingerprint = prompts.schema_fingerprint(_plain_value(response_contract))
    return RevisionProviderPlan(
        provider=CLAUDE_CODE_PROVIDER,
        billing_class=billing_class,
        auth_metadata=_frozen_mapping(auth),
        model=model,
        transport=transport,
        request_bytes=request_bytes,
        request_fingerprint=_request_identity(
            provider=CLAUDE_CODE_PROVIDER,
            billing_class=billing_class,
            auth=auth,
            model=model,
            transport=transport,
            request_bytes=request_bytes,
            response_schema_fingerprint=schema_fingerprint,
        ),
        response_schema_fingerprint=schema_fingerprint,
        system_blocks=blocks,
        user_turn=user_turn,
        schema=schema,
    )


def _cli_result_from_reply(
    plan: RevisionProviderPlan, raw_reply: bytes
) -> claude_client.CallResult:
    """Decode one captured stream, which settles in exactly one result frame.

    Every line is read, not just the last: a capture that is not wholly
    well-formed stream-json, or that carries no single result frame, is a reply
    janki cannot read rather than a reply that said nothing.
    """
    results: list[Mapping[str, Any]] = []
    for number, line in enumerate(raw_reply.split(b"\n"), start=1):
        if not line.strip():
            continue
        frame = _strict_json(line, label=f"Claude Code stream frame {number}")
        kind = frame.get("type")
        if not isinstance(kind, str) or not kind:
            raise RevisionProviderError(
                f"Captured Claude Code stream frame {number} has no frame type."
            )
        if kind == "result":
            results.append(frame)
    if len(results) != 1:
        raise RevisionProviderError(
            f"Captured Claude Code stream carries {len(results)} result frames; "
            "exactly one settles a call."
        )
    payload = results[0]
    if payload.get("is_error") is not False or payload.get("subtype") != "success":
        reason = str(payload.get("subtype") or "error")
        return claude_client.CallResult(None, reason, None)
    if "structured_output" not in payload:
        raise RevisionProviderError(
            "Claude Code finished normally but returned no structured output."
        )
    try:
        parsed = _type_adapter(plan.schema).validate_python(payload["structured_output"])
    except Exception as exc:
        raise RevisionProviderError(
            f"{plan.model} finished normally but its captured Claude Code answer "
            f"did not match the revision schema: {exc}"
        ) from exc
    return claude_client.CallResult(parsed, "end_turn", None)


class _FirstTextPreview:
    """Forward the first assistant text block as the CLI streams it.

    Only that block previews the answer.  Thinking is not the answer, the
    structured result arrives later as a tool call whose partial JSON is not
    displayable text, and a later message belongs to the schema round trip.
    The preview is best effort throughout: one decoding or callback failure
    ends it for the rest of the call and never touches the capture.
    """

    def __init__(self, forward: Callable[[str], None]) -> None:
        self._forward: Callable[[str], None] | None = forward
        self._index: Any = None

    def observe(self, line: bytes) -> None:
        if self._forward is None:
            return
        try:
            self._observe(line)
        except Exception:  # noqa: BLE001 - a preview never fails a paid call
            self._forward = None

    def _observe(self, line: bytes) -> None:
        if not line.strip():
            return
        payload = _strict_json(line, label="Claude Code preview frame")
        if payload.get("type") != "stream_event":
            return
        event = payload.get("event")
        if not isinstance(event, Mapping):
            return
        kind = event.get("type")
        if kind == "message_stop":
            # The first message ends the preview: everything after it is the
            # schema round trip, so nothing later can be forwarded.
            self._forward = None
            return
        if kind == "content_block_start":
            block = event.get("content_block")
            if (
                self._index is None
                and isinstance(block, Mapping)
                and block.get("type") == "text"
            ):
                self._index = event.get("index")
            return
        if kind != "content_block_delta" or self._index is None:
            return
        if event.get("index") != self._index:
            return
        delta = event.get("delta")
        if not isinstance(delta, Mapping) or delta.get("type") != "text_delta":
            return
        text = delta.get("text")
        if isinstance(text, str) and text:
            assert self._forward is not None
            self._forward(text)


def _send_claude_prompt(process: Any, prompt: bytes) -> None:
    """Write one exact user turn without blocking the frame reader.

    The close is the child's end-of-prompt, so it happens even when the write
    tore: a CLI still waiting on stdin would otherwise never reach EOF, and
    the pipe would outlive the turn holding a file descriptor open.
    """
    try:
        process.stdin.write(prompt)
    except (OSError, ValueError):
        # The child exited before reading its prompt; its exit code and
        # captured frames are what report that, not a torn write.
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()


def _stream_claude_reply(
    process: Any,
    prompt: bytes,
    *,
    frame: Callable[[str], None] | None,
    preview: Callable[[str], None] | None,
) -> tuple[int, list[bytes], tuple[int, UnicodeDecodeError] | None]:
    """Spool every exact byte, then read what the bytes say, then reap.

    The reply is the child's, paid for whatever it says, so a line joins the
    spool before anything looks at it.  The frame callback makes that line
    durable elsewhere and runs before the next read; a failure there means
    janki can no longer prove it holds what it is about to be told, so the
    call ends where it stands and dispatch captures nothing.  A line that is
    not UTF-8 is reported instead: the spool keeps growing to EOF so the exact
    reply is still captured, and the returned marker names the frame that
    cannot be read.

    However this ends, the child is killed on the way out and always waited
    for — before the writer is joined, since a writer blocked on a full stdin
    pipe is released by EPIPE only once the child is dead.
    """
    writer = threading.Thread(
        target=_send_claude_prompt, args=(process, prompt), daemon=True
    )
    writer.start()
    watcher = None if preview is None else _FirstTextPreview(preview)
    chunks: list[bytes] = []
    unreadable: tuple[int, UnicodeDecodeError] | None = None
    returncode: int
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            chunks.append(line)
            if unreadable is not None:
                # Nothing downstream can read this stream any more; only its
                # exact bytes still matter.
                continue
            try:
                text = line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                unreadable = (len(chunks), exc)
                continue
            if frame is not None:
                try:
                    frame(text)
                except Exception as exc:  # noqa: BLE001 - any failure ends the call
                    raise RevisionProviderError(
                        f"Claude Code frame capture failed after {len(chunks) - 1} "
                        f"durable frames: {exc}"
                    ) from exc
            if watcher is not None:
                watcher.observe(line)
    except BaseException:
        process.kill()
        raise
    finally:
        with contextlib.suppress(OSError):
            process.stdout.close()
        returncode = process.wait()
        writer.join()
    return returncode, chunks, unreadable


class _ClaudeCodeRevisionProvider:
    name = CLAUDE_CODE_PROVIDER

    def plan(
        self,
        *,
        model: str,
        style_guide: str,
        task_template: str,
        system_blocks: Sequence[Mapping[str, Any]],
        user_turn: str,
        schema: Any,
        effort: str | None,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
    ) -> RevisionProviderPlan:
        del style_guide, task_template
        _require_supported_model(model)
        # Before the probe, because the probe already runs under the depth the
        # planned call will use.
        depth = _required_effort(effort)
        _, _, auth, version = _probe_claude(
            env=env,
            runner=runner,
            which=which,
            controlled=_controlled_claude_environment(depth),
        )
        return _claude_code_plan(
            model=model,
            system_blocks=system_blocks,
            user_turn=user_turn,
            schema=schema,
            effort=depth,
            auth=auth,
            version=version,
        )

    def prepare(
        self,
        plan: RevisionProviderPlan,
        *,
        env: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
        which: Which = shutil.which,
        client: Any | None = None,
    ) -> PreparedRevisionProvider:
        del client
        _validate_plan(plan, self.name)
        _validate_cli_channels(plan)
        # The environment dispatch runs under is the bound one, read back from
        # the plan rather than rebuilt from today's module constants — so a
        # depth or limit that changed since planning cannot reach the child.
        executable, clean, auth, version = _probe_claude(
            env=env,
            runner=runner,
            which=which,
            controlled=plan.transport["controlled_environment"],
        )
        if (
            dict(auth) != dict(plan.auth_metadata)
            or version != plan.transport.get("cli_version")
        ):
            raise RevisionProviderError(
                "Claude Code's version or subscription authentication changed "
                "after this revision plan was rendered. Reload it; no model call "
                "was made."
            )
        return PreparedRevisionProvider(
            plan=plan,
            executable=executable,
            environment=MappingProxyType(clean),
        )

    def dispatch(
        self,
        prepared: PreparedRevisionProvider,
        *,
        capture: Capture,
        spawn: Spawn = subprocess.Popen,
        api_call: Callable[..., Any] | None = None,
        frame: Callable[[str], None] | None = None,
        preview: Callable[[str], None] | None = None,
    ) -> claude_client.CallResult:
        del api_call
        plan = prepared.plan
        _validate_plan(plan, self.name)
        request = _validate_cli_channels(plan)
        if prepared.executable is None or prepared.environment is None:
            raise RevisionProviderError("Claude Code was not prepared before dispatch.")
        stable_argv = request["argv"]
        if not isinstance(stable_argv, list) or not stable_argv or stable_argv[0] != "claude":
            raise RevisionProviderError("Claude Code plan has an invalid command identity.")
        command = [prepared.executable, *[str(value) for value in stable_argv[1:]]]
        prompt = str(request["stdin_utf8"]).encode("utf-8")
        with (
            tempfile.TemporaryDirectory(prefix="janki-claude-revise-") as temporary,
            tempfile.TemporaryFile(prefix="janki-claude-stderr-") as diagnostics,
        ):
            try:
                process = spawn(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=diagnostics,
                    cwd=temporary,
                    env=dict(prepared.environment),
                )
            except FileNotFoundError as exc:
                raise RevisionProviderError(
                    "The prepared Claude Code CLI disappeared before dispatch."
                ) from exc
            except OSError as exc:
                raise RevisionProviderError(
                    f"Could not start the prepared Claude Code CLI: {exc}"
                ) from exc
            returncode, chunks, unreadable = _stream_claude_reply(
                process, prompt, frame=frame, preview=preview
            )
            diagnostics.seek(0)
            stderr_bytes = diagnostics.read()
        raw_reply = b"".join(chunks)
        if raw_reply:
            # The exact paid/capped reply is durable before exit classification,
            # outer JSON decoding, or Pydantic validation.
            capture(raw_reply)
        if unreadable is not None:
            number, error = unreadable
            raise RevisionProviderError(
                f"Claude Code stream frame {number} is not UTF-8: {error}"
            ) from error
        if returncode != 0:
            diagnostic = redact_environment_credentials(
                stderr_bytes.decode("utf-8", errors="replace")
            ).strip()
            suffix = f": {diagnostic[-1000:]}" if diagnostic else ""
            raise RevisionProviderError(
                f"Claude Code {plan.model} failed (exit {returncode}){suffix}"
            )
        if not raw_reply:
            raise RevisionProviderError(
                f"Claude Code {plan.model} returned no JSON reply to capture."
            )
        return self.recover(plan, raw_reply)

    def recover(
        self, plan: RevisionProviderPlan, raw_reply: bytes
    ) -> claude_client.CallResult:
        _validate_plan(plan, self.name)
        _validate_cli_channels(plan)
        return _cli_result_from_reply(plan, raw_reply)


PROVIDERS: Mapping[str, RevisionProvider] = MappingProxyType(
    {
        ANTHROPIC_API_PROVIDER: _AnthropicAPIRevisionProvider(),
        CLAUDE_CODE_PROVIDER: _ClaudeCodeRevisionProvider(),
    }
)


def provider_for(name: str) -> RevisionProvider:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROVIDERS))
        raise RevisionProviderError(
            f"Unknown revision provider {name!r}; choose one of: {choices}."
        ) from exc


def plan_provider(name: str, **kwargs: Any) -> RevisionProviderPlan:
    """Plan one named provider through the shared registry."""
    return provider_for(name).plan(**kwargs)


def provider_plan_from_manifest(
    manifest: Mapping[str, Any],
    *,
    model: str,
    style_guide: str,
    task_template: str,
    system_blocks: Sequence[Mapping[str, Any]],
    user_turn: str,
    schema: Any,
) -> RevisionProviderPlan:
    """Purely reconstruct and verify stored provider provenance.

    Recovery and proposal application use this path after a call has already
    consumed API money or subscription allowance.  It performs no executable
    lookup, authentication probe, client construction, or network access.
    """
    if set(manifest) != PERSISTENT_MANIFEST_KEYS:
        found = sorted(str(key) for key in manifest)
        raise RevisionProviderError(
            f"Revision provider manifest has invalid fields: {found}."
        )
    provider_name = manifest.get("provider")
    if not isinstance(provider_name, str):
        raise RevisionProviderError("Revision provider manifest has no text provider.")
    if manifest.get("model") != model:
        raise RevisionProviderError(
            "Revision provider manifest does not match its stored model."
        )
    del style_guide, task_template
    _require_supported_model(model)
    auth = manifest.get("auth")
    transport = manifest.get("transport")
    if not isinstance(auth, Mapping) or not isinstance(transport, Mapping):
        raise RevisionProviderError("Revision provider manifest has invalid metadata.")
    raw_text = manifest.get("request_bytes_utf8")
    if not isinstance(raw_text, str):
        raise RevisionProviderError("Revision provider manifest has no request bytes.")
    request_bytes = raw_text.encode("utf-8")
    request = _strict_json(request_bytes, label="stored provider request")
    if _canonical_json(request).encode("utf-8") != request_bytes:
        raise RevisionProviderError("Stored provider request bytes are not canonical JSON.")
    if manifest.get("request_bytes_sha256") != _sha256(request_bytes):
        raise RevisionProviderError("Stored provider request bytes do not match their hash.")
    if provider_name == CLAUDE_CODE_PROVIDER:
        if not isinstance(auth, Mapping) or set(auth) != {
            "auth_method",
            "api_provider",
            "subscription_type",
            "api_key_source",
        }:
            raise RevisionProviderError(
                "Revision provider manifest has invalid Claude subscription metadata."
            )
        if (
            auth.get("auth_method") != "claude.ai"
            or auth.get("api_provider") != "firstParty"
            or auth.get("subscription_type") not in {"pro", "max"}
            or auth.get("api_key_source") is not None
        ):
            raise RevisionProviderError(
                "Revision provider manifest is not bound to Claude Pro or Max."
            )
        version = transport.get("cli_version")
        if (
            not isinstance(version, str)
            or _VERSION.fullmatch(version.encode("ascii", "ignore")) is None
        ):
            raise RevisionProviderError(
                "Revision provider manifest has an invalid Claude Code version."
            )
    elif provider_name == ANTHROPIC_API_PROVIDER:
        if dict(auth) != {
            "auth_method": "environment-api-key",
            "api_key_source": claude_client.API_KEY_ENV,
        }:
            raise RevisionProviderError(
                "Revision provider manifest has invalid API authentication metadata."
            )
    else:
        provider_for(str(provider_name))
        raise AssertionError("unreachable")
    billing_class = manifest.get("billing_class")
    fingerprint = manifest.get("request_fingerprint")
    schema_fingerprint = manifest.get("response_schema_fingerprint")
    identity_values = (billing_class, fingerprint, schema_fingerprint)
    if not all(isinstance(value, str) for value in identity_values):
        raise RevisionProviderError("Revision provider manifest has invalid identity metadata.")
    candidate = RevisionProviderPlan(
        provider=provider_name,
        billing_class=billing_class,
        auth_metadata=_frozen_mapping(auth),
        model=model,
        transport=_frozen_mapping(transport),
        request_bytes=request_bytes,
        request_fingerprint=fingerprint,
        response_schema_fingerprint=schema_fingerprint,
        system_blocks=tuple(_frozen_mapping(block) for block in system_blocks),
        user_turn=user_turn,
        schema=schema,
    )
    _validate_plan(candidate, provider_name)
    billing_display(provider_name, billing_class, auth)
    if provider_name == CLAUDE_CODE_PROVIDER:
        _validate_cli_channels(candidate)
    else:
        _validate_api_channels(candidate)
    return candidate
