from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from japanese_anki import claude_client
from japanese_anki.application import revision_provider


class Answer(BaseModel):
    examples: list[str]


BLOCKS = (
    {"type": "text", "text": "Style\n"},
    {"type": "text", "text": "Task\n"},
)
USER_TURN = "Use 食べられる。"
MODEL = "claude-opus-5"


def _auth(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
        "apiKeySource": None,
        # The CLI may return account details; neither is safe provenance.
        "email": "owner@example.test",
        "organizationId": "org-secret",
    }
    value.update(changes)
    return value


class FakeClaudeRunner:
    def __init__(
        self,
        *,
        auth: Mapping[str, Any] | None = None,
        version: bytes = b"2.1.246 (Claude Code)\n",
        reply: bytes | None = None,
        returncode: int = 0,
    ) -> None:
        self.auth = dict(_auth() if auth is None else auth)
        self.version = version
        self.reply = reply
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(command), kwargs))
        cwd = Path(kwargs["cwd"])
        assert cwd.is_dir()
        assert list(cwd.iterdir()) == []
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, self.version, b"")
        if "auth" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(self.auth).encode("utf-8"),
                b"",
            )
        return subprocess.CompletedProcess(
            command,
            self.returncode,
            self.reply or b"",
            b"local diagnostic",
        )


def _which(name: str, **kwargs: Any) -> str:
    assert name == "claude"
    del kwargs
    return "/private/test/bin/claude"


def _cli_plan(
    runner: FakeClaudeRunner | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[revision_provider.RevisionProviderPlan, FakeClaudeRunner]:
    used = runner or FakeClaudeRunner()
    plan = revision_provider.plan_provider(
        "claude-code",
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
        env=env or {"PATH": "/bin", "BENIGN": "kept"},
        runner=used,
        which=_which,
    )
    return plan, used


def test_registry_has_only_the_two_revision_transports() -> None:
    assert set(revision_provider.PROVIDERS) == {"anthropic-api", "claude-code"}
    with pytest.raises(revision_provider.RevisionProviderError, match="Unknown"):
        revision_provider.provider_for("anthropic")


def test_claude_plan_binds_safe_subscription_request_without_host_identity() -> None:
    plan, runner = _cli_plan()

    manifest = plan.persistent_manifest()
    rendered = json.dumps(manifest, ensure_ascii=False)
    assert manifest["billing_class"] == "claude-subscription"
    assert "billing_display" not in manifest
    assert "Max subscription" in plan.billing_display
    assert manifest["auth"] == {
        "auth_method": "claude.ai",
        "api_provider": "firstParty",
        "subscription_type": "max",
        "api_key_source": None,
    }
    assert manifest["transport"]["cli_version"] == "2.1.246"
    assert "/private/test/bin/claude" not in rendered
    assert "owner@example.test" not in rendered
    assert "org-secret" not in rendered
    assert runner.calls[0][0] == [
        "/private/test/bin/claude",
        "--safe-mode",
        "--version",
    ]
    assert runner.calls[1][0] == [
        "/private/test/bin/claude",
        "--safe-mode",
        "auth",
        "status",
        "--json",
    ]


def test_claude_environment_scrubs_diversions_and_sets_exact_limits() -> None:
    dangerous = {
        "PATH": "/bin",
        "BENIGN": "kept",
        "ANTHROPIC_API_KEY": "api-secret",
        "ANTHROPIC_AUTH_TOKEN": "bearer-secret",
        "ANTHROPIC_BASE_URL": "https://gateway.invalid",
        "ANTHROPIC_MODEL": "opus",
        "ANTHROPIC_FEDERATION_TOKEN": "federated",
        "CLAUDE_CODE_OAUTH_TOKEN": "different-account",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_SESSION_TOKEN": "cloud-secret",
        "MAX_THINKING_TOKENS": "1",
        "MAX_STRUCTURED_OUTPUT_RETRIES": "9",
        "FALLBACK_FOR_ALL_PRIMARY_MODELS": "1",
        "OPENAI_API_KEY": "unrelated-secret",
    }
    _, runner = _cli_plan(env=dangerous)

    child = runner.calls[0][1]["env"]
    assert child["PATH"] == "/bin"
    assert "BENIGN" not in child
    for name in dangerous.keys() - {"PATH", "MAX_STRUCTURED_OUTPUT_RETRIES"}:
        assert name not in child
    assert child["CLAUDE_CODE_EFFORT_LEVEL"] == "xhigh"
    assert child["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    assert child["MAX_STRUCTURED_OUTPUT_RETRIES"] == "0"
    assert child["CLAUDE_CODE_MAX_TURNS"] == "1"
    assert child["CLAUDE_CODE_MAX_RETRIES"] == "0"
    assert child["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"] == "1"
    assert child["CLAUDE_CODE_FORK_SUBAGENT"] == "0"
    assert child["CLAUDE_CODE_DISABLE_WORKFLOWS"] == "1"


@pytest.mark.parametrize(
    "name",
    [
        "ENABLE_PROMPT_CACHING_1H",
        "FORCE_PROMPT_CACHING_5M",
        "DISABLE_PROMPT_CACHING_OPUS",
        "DISABLE_INTERLEAVED_THINKING",
        "CLAUDECODE",
    ],
)
def test_claude_environment_never_inherits_named_behavior_overrides(name: str) -> None:
    _, runner = _cli_plan(env={"PATH": "/bin", name: "host-value"})
    assert name not in runner.calls[0][1]["env"]


def test_claude_environment_preserves_the_login_user_for_subscription_auth() -> None:
    _, runner = _cli_plan(env={"PATH": "/bin", "USER": "subscription-owner"})
    assert runner.calls[0][1]["env"]["USER"] == "subscription-owner"


def test_claude_planning_does_not_load_the_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        claude_client,
        "load_anthropic",
        lambda: pytest.fail("Claude Code schema planning must be provider-neutral"),
    )
    plan, _ = _cli_plan()
    assert plan.model == MODEL


@pytest.mark.parametrize(
    ("changes", "remove"),
    [
        ({"loggedIn": False}, None),
        ({"authMethod": "apiKey"}, None),
        ({"apiProvider": "bedrock"}, None),
        ({"subscriptionType": None}, None),
        ({"subscriptionType": "team"}, None),
        ({"apiKeySource": "ANTHROPIC_API_KEY"}, None),
    ],
)
def test_claude_plan_refuses_every_non_subscription_auth_shape(
    changes: Mapping[str, Any], remove: str | None
) -> None:
    auth = _auth(**changes)
    if remove is not None:
        del auth[remove]
    runner = FakeClaudeRunner(auth=auth)

    with pytest.raises(
        revision_provider.RevisionProviderError, match="Pro or Max"
    ):
        _cli_plan(runner)


def test_claude_plan_accepts_omitted_api_key_source_from_subscription_status() -> None:
    auth = _auth()
    del auth["apiKeySource"]

    plan, _ = _cli_plan(FakeClaudeRunner(auth=auth))

    assert plan.auth_metadata["api_key_source"] is None


@pytest.mark.parametrize("model", ["claude-opus", "claude-foo"])
def test_claude_plan_requires_the_exact_supported_model(model: str) -> None:
    with pytest.raises(revision_provider.RevisionProviderError, match="exact supported"):
        revision_provider.plan_provider(
            "claude-code",
            model=model,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
            runner=lambda *_args, **_kwargs: pytest.fail("must fail before probing"),
            which=_which,
        )


def test_claude_command_has_exact_isolation_and_exact_prompt_channels() -> None:
    reply = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"examples": ["食べられます。"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    plan, runner = _cli_plan(FakeClaudeRunner(reply=reply))
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(
        plan,
        env={"PATH": "/bin", "BENIGN": "kept"},
        runner=runner,
        which=_which,
    )
    captured: list[bytes] = []

    result = provider.dispatch(prepared, capture=captured.append, runner=runner)

    assert result.parsed == Answer(examples=["食べられます。"])
    assert captured == [reply]
    command, kwargs = runner.calls[-1]
    assert command[0] == "/private/test/bin/claude"
    assert "--bare" not in command
    for flag in (
        "-p",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--no-chrome",
        "--tools",
        "--permission-mode",
        "--prompt-suggestions",
        "--output-format",
        "--model",
        "--effort",
        "--json-schema",
        "--system-prompt",
    ):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--prompt-suggestions") + 1] == "false"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--model") + 1] == MODEL
    assert command[command.index("--effort") + 1] == "xhigh"
    assert command[command.index("--system-prompt") + 1] == "Style\n\n\nTask\n"
    assert json.loads(command[command.index("--json-schema") + 1])["type"] == "object"
    assert kwargs["input"] == USER_TURN.encode("utf-8")
    assert "shell" not in kwargs


def test_claude_captures_nonzero_stdout_before_reporting_failure() -> None:
    raw = b'{"type":"result","subtype":"error","is_error":true}'
    plan, runner = _cli_plan(FakeClaudeRunner(reply=raw, returncode=3))
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(plan, runner=runner, which=_which)
    events: list[bytes] = []

    with pytest.raises(revision_provider.RevisionProviderError, match="exit 3"):
        provider.dispatch(prepared, capture=events.append, runner=runner)

    assert events == [raw]


def test_claude_captures_before_outer_json_and_schema_validation() -> None:
    for raw in (
        b"not-json",
        b'{"type":"result","subtype":"success","is_error":false,'
        b'"structured_output":{"examples":"wrong"}}',
    ):
        plan, runner = _cli_plan(FakeClaudeRunner(reply=raw))
        provider = revision_provider.provider_for("claude-code")
        prepared = provider.prepare(plan, runner=runner, which=_which)
        events: list[bytes] = []

        with pytest.raises(revision_provider.RevisionProviderError):
            provider.dispatch(prepared, capture=events.append, runner=runner)

        assert events == [raw]


def test_claude_success_requires_an_explicit_false_error_flag() -> None:
    plan, _ = _cli_plan()
    raw = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "structured_output": {"examples": ["食べられます。"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")

    result = revision_provider.provider_for("claude-code").recover(plan, raw)

    assert result.parsed is None


def test_claude_plan_is_deeply_immutable_and_purely_recoverable() -> None:
    plan, _ = _cli_plan()
    with pytest.raises(TypeError):
        plan.transport["argv"][0] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.transport["controlled_environment"]["CLAUDE_CODE_MAX_TURNS"] = "9"  # type: ignore[index]

    recovered = revision_provider.provider_plan_from_manifest(
        plan.persistent_manifest(),
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )

    assert recovered.request_bytes == plan.request_bytes
    assert recovered.request_fingerprint == plan.request_fingerprint


def test_claude_prepare_refuses_auth_or_version_drift_before_dispatch() -> None:
    plan, _ = _cli_plan()
    provider = revision_provider.provider_for("claude-code")

    with pytest.raises(revision_provider.RevisionProviderError, match="changed"):
        provider.prepare(
            plan,
            runner=FakeClaudeRunner(version=b"2.1.247\n"),
            which=_which,
        )
    with pytest.raises(revision_provider.RevisionProviderError, match="changed"):
        provider.prepare(
            plan,
            runner=FakeClaudeRunner(auth=_auth(subscriptionType="pro")),
            which=_which,
        )


def test_replaced_prompt_field_cannot_escape_the_bound_request_bytes() -> None:
    plan, runner = _cli_plan()
    changed = replace(plan, user_turn="A different unconfirmed instruction")
    provider = revision_provider.provider_for("claude-code")

    with pytest.raises(
        revision_provider.RevisionProviderError, match="prompt channels"
    ):
        provider.prepare(changed, runner=runner, which=_which)


def test_billing_display_copy_is_not_durable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _ = _cli_plan()
    manifest = plan.persistent_manifest()
    monkeypatch.setattr(revision_provider, "billing_display", lambda *_args: "New copy")
    assert plan.billing_display == "New copy"
    assert plan.persistent_manifest() == manifest


def test_each_claude_transport_identity_change_changes_the_fingerprint() -> None:
    max_plan, _ = _cli_plan(FakeClaudeRunner(auth=_auth(subscriptionType="max")))
    pro_plan, _ = _cli_plan(FakeClaudeRunner(auth=_auth(subscriptionType="pro")))
    version_plan, _ = _cli_plan(FakeClaudeRunner(version=b"2.1.247\n"))
    turn_plan = revision_provider.plan_provider(
        "claude-code",
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn="Different",
        schema=Answer,
        runner=FakeClaudeRunner(),
        which=_which,
    )

    assert len(
        {
            max_plan.request_fingerprint,
            pro_plan.request_fingerprint,
            version_plan.request_fingerprint,
            turn_plan.request_fingerprint,
        }
    ) == 4


class FakeAPIResponse:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def model_dump_json(self) -> str:
        return self.raw.decode("utf-8")


def test_api_adapter_keeps_existing_call_shape_and_byte_capture() -> None:
    provider = revision_provider.provider_for("anthropic-api")
    plan = provider.plan(
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )
    prepared = provider.prepare(plan, client=object())
    raw = json.dumps(
        {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"examples": ["食べられます。"]}),
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    captured: list[bytes] = []

    def call(*args: Any, **kwargs: Any) -> claude_client.CallResult:
        assert args[:4] == (
            MODEL,
            tuple(dict(block) for block in BLOCKS),
            USER_TURN,
            Answer,
        )
        kwargs["capture"](FakeAPIResponse(raw))
        return claude_client.CallResult(
            Answer(examples=["different live decoder result"]), "end_turn", None
        )

    result = provider.dispatch(
        prepared,
        capture=captured.append,
        api_call=call,
    )

    assert result.parsed == Answer(examples=["食べられます。"])
    assert captured == [raw]
    assert provider.recover(plan, raw).parsed == result.parsed


def test_api_dispatch_and_recovery_both_refuse_multiple_text_answers() -> None:
    provider = revision_provider.provider_for("anthropic-api")
    plan = provider.plan(
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )
    prepared = provider.prepare(plan, client=object())
    raw = json.dumps(
        {
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": '{"examples":["one"]}'},
                {"type": "text", "text": '{"examples":["two"]}'},
            ],
        }
    ).encode()
    captured: list[bytes] = []

    def call(*_args: Any, **kwargs: Any) -> None:
        kwargs["capture"](FakeAPIResponse(raw))

    with pytest.raises(revision_provider.RevisionProviderError, match="exactly one"):
        provider.dispatch(prepared, capture=captured.append, api_call=call)
    with pytest.raises(revision_provider.RevisionProviderError, match="exactly one"):
        provider.recover(plan, raw)
    assert captured == [raw]


def test_api_dispatch_refuses_two_capture_callbacks() -> None:
    provider = revision_provider.provider_for("anthropic-api")
    plan = provider.plan(
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )
    prepared = provider.prepare(plan, client=object())
    raw = json.dumps({"stop_reason": "max_tokens", "content": []}).encode()
    captured: list[bytes] = []

    def call(*_args: Any, **kwargs: Any) -> None:
        kwargs["capture"](FakeAPIResponse(raw))
        kwargs["capture"](FakeAPIResponse(raw))

    with pytest.raises(revision_provider.RevisionProviderError, match="multiple replies"):
        provider.dispatch(prepared, capture=captured.append, api_call=call)
    assert captured == [raw]


def test_stored_manifest_tampering_refuses_without_a_probe() -> None:
    plan, _ = _cli_plan()
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    manifest["transport"]["cli_version"] = "2.1.999"

    with pytest.raises(
        revision_provider.RevisionProviderError, match="fingerprint"
    ):
        revision_provider.provider_plan_from_manifest(
            manifest,
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
        )


def test_stored_request_bytes_are_independently_hash_bound() -> None:
    plan, _ = _cli_plan()
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    manifest["request_bytes_utf8"] = manifest["request_bytes_utf8"].replace(
        USER_TURN, "Changed"
    )
    with pytest.raises(revision_provider.RevisionProviderError, match="hash"):
        revision_provider.provider_plan_from_manifest(
            manifest,
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
        )


@pytest.mark.parametrize("provider_name", ["anthropic-api", "claude-code"])
def test_reconstruction_does_not_regenerate_todays_schema(
    provider_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if provider_name == "claude-code":
        plan, _ = _cli_plan()
    else:
        plan = revision_provider.plan_provider(
            provider_name,
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
        )
    monkeypatch.setattr(
        revision_provider,
        "_neutral_wire_schema",
        lambda *_args: pytest.fail("recovery regenerated today's schema"),
    )
    monkeypatch.setattr(
        claude_client,
        "wire_schema",
        lambda *_args: pytest.fail("recovery regenerated the SDK schema"),
    )
    monkeypatch.setattr(
        claude_client,
        "request_body",
        lambda *_args, **_kwargs: pytest.fail("recovery regenerated today's defaults"),
    )
    recovered = revision_provider.provider_plan_from_manifest(
        plan.persistent_manifest(),
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )
    assert recovered.request_bytes == plan.request_bytes


def test_api_reconstruction_uses_the_stored_thinking_contract() -> None:
    plan = revision_provider.plan_provider(
        "anthropic-api",
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    request = json.loads(manifest["request_bytes_utf8"])
    request["thinking"] = {"type": "disabled"}
    manifest["transport"]["thinking"] = {"type": "disabled"}
    request_bytes = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["request_bytes_utf8"] = request_bytes.decode("utf-8")
    manifest["request_bytes_sha256"] = revision_provider._sha256(request_bytes)
    manifest["request_fingerprint"] = revision_provider._request_identity(
        provider=manifest["provider"],
        billing_class=manifest["billing_class"],
        auth=manifest["auth"],
        model=manifest["model"],
        transport=manifest["transport"],
        request_bytes=request_bytes,
        response_schema_fingerprint=manifest["response_schema_fingerprint"],
    )

    recovered = revision_provider.provider_plan_from_manifest(
        manifest,
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=Answer,
    )

    assert recovered.transport["thinking"] == {"type": "disabled"}
