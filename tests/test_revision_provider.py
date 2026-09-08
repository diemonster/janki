from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import subprocess
import sys
import threading
import time
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


class StreamedAnswer(BaseModel):
    """The response contract the captured stream fixture was really planned with."""

    answer: str


BLOCKS = (
    {"type": "text", "text": "Style\n"},
    {"type": "text", "text": "Task\n"},
)
USER_TURN = "Use 食べられる。"
MODEL = "claude-opus-5"
#: The exact prepared-input block shape input preparation hands this transport.
#: The bytes are deliberately not a valid PDF or PNG: this module transports
#: whatever preparation already proved, and re-deriving that here would test the
#: fixture instead of the wire.
PDF_DATA = base64.b64encode(b"%PDF-1.7 GENKI page \xe3\x81\x82").decode("ascii")
PNG_DATA = base64.b64encode(b"\x89PNG\r\n\x1a\nscan").decode("ascii")
JPEG_DATA = base64.b64encode(b"\xff\xd8\xff\xe0photo").decode("ascii")


def _document_block(data: str = PDF_DATA) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": data,
        },
    }


def _image_block(media_type: str, data: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _input_blocks() -> list[dict[str, Any]]:
    return [
        _document_block(),
        _image_block("image/png", PNG_DATA),
        _image_block("image/jpeg", JPEG_DATA),
    ]


def _expected_frame(
    blocks: list[dict[str, Any]], user_turn: str = USER_TURN
) -> bytes:
    """One canonical stream-json user message, exactly one newline long."""
    return (
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [*blocks, {"type": "text", "text": user_turn}],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
#: One real ``claude -p --output-format stream-json`` capture, redacted. Frame
#: order, event shapes, and the gap between the streamed prose and the final
#: structured answer are all the CLI's, not this suite's guess at them.
STREAM_FIXTURE = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"


def _stream_lines() -> list[bytes]:
    return STREAM_FIXTURE.read_bytes().splitlines(keepends=True)


def _without_result_line() -> list[bytes]:
    return [
        line
        for line in _stream_lines()
        if json.loads(line).get("type") != "result"
    ]


def _result_line() -> bytes:
    [line] = [
        line for line in _stream_lines() if json.loads(line).get("type") == "result"
    ]
    return line


def _fixture_answer() -> str:
    answer = json.loads(_result_line())["structured_output"]["answer"]
    assert isinstance(answer, str)
    return answer


def _stream_with_torn_frame(number: int) -> bytes:
    """The captured stream with one frame ending in a truncated rune.

    A child killed mid-write, or a CLI that wrote a rune in two syscalls, ends
    a line no decoder can read — which does not make the bytes any less paid
    for.
    """
    lines = _stream_lines()
    lines[number - 1] = lines[number - 1].rstrip(b"\n") + b"\xe3\x81\n"  # torn "あ"
    return b"".join(lines)


def _fixture_preview() -> tuple[str, ...]:
    """Each prose chunk the CLI streamed before it called for structured output."""
    chunks = []
    for line in _stream_lines():
        payload = json.loads(line)
        if payload.get("type") != "stream_event":
            continue
        event = payload["event"]
        if event["type"] != "content_block_delta":
            continue
        if event["delta"]["type"] == "text_delta":
            chunks.append(event["delta"]["text"])
    assert chunks
    return tuple(chunks)


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


class FakeStdin(io.BytesIO):
    """A pipe that keeps its exact bytes after the writer closes it.

    ``blocking`` is a full pipe whose child is not reading: the write only
    completes once that child is gone, which is what releases a real writer
    thread. A write still waiting when the test gives up records that, so an
    unreaped child is a failure rather than a slow pass.
    """

    def __init__(self, *, blocking: bool = False) -> None:
        super().__init__()
        self.written = b""
        self.timed_out = False
        self._blocking = blocking
        self._child_gone = threading.Event()

    def child_exited(self) -> None:
        self._child_gone.set()

    def write(self, payload: Any) -> int:
        if self._blocking and not self._child_gone.wait(3):
            self.timed_out = True
        return super().write(payload)

    def close(self) -> None:
        if not self.closed:
            self.written = self.getvalue()
        super().close()


class FakeStdout(io.BytesIO):
    """The child's stdout, optionally torn by a read error mid-stream."""

    def __init__(self, payload: bytes, *, error_on_line: int | None = None) -> None:
        super().__init__(payload)
        self._error_on_line = error_on_line
        self._lines = 0

    def readline(self, *args: Any) -> bytes:
        self._lines += 1
        if self._lines == self._error_on_line:
            raise OSError(errno.EIO, "Input/output error")
        return super().readline(*args)


class FakePopen:
    """The exact child surface the streaming Claude Code transport drives."""

    def __init__(
        self,
        stdout: bytes,
        *,
        returncode: int,
        stderr: Any,
        read_error: int | None = None,
        blocking_stdin: bool = False,
    ) -> None:
        self.stdin = FakeStdin(blocking=blocking_stdin)
        self.stdout = FakeStdout(stdout, error_on_line=read_error)
        self.returncode = returncode
        self.killed = False
        self.waits = 0
        stderr.write(b"local diagnostic")

    def wait(self) -> int:
        self.waits += 1
        self.stdin.child_exited()
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class FakeClaudeRunner:
    def __init__(
        self,
        *,
        auth: Mapping[str, Any] | None = None,
        version: bytes = b"2.1.246 (Claude Code)\n",
        reply: bytes | None = None,
        returncode: int = 0,
        read_error: int | None = None,
        blocking_stdin: bool = False,
    ) -> None:
        self.auth = dict(_auth() if auth is None else auth)
        self.version = version
        self.reply = reply
        self.returncode = returncode
        self.read_error = read_error
        self.blocking_stdin = blocking_stdin
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.spawned: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakePopen] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(command), kwargs))
        cwd = Path(kwargs["cwd"])
        assert cwd.is_dir()
        assert list(cwd.iterdir()) == []
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, self.version, b"")
        assert "auth" in command, command
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(self.auth).encode("utf-8"),
            b"",
        )

    def spawn(self, command: list[str], **kwargs: Any) -> FakePopen:
        self.spawned.append((list(command), kwargs))
        cwd = Path(kwargs["cwd"])
        assert cwd.is_dir()
        assert list(cwd.iterdir()) == []
        process = FakePopen(
            self.reply or b"",
            returncode=self.returncode,
            stderr=kwargs["stderr"],
            read_error=self.read_error,
            blocking_stdin=self.blocking_stdin,
        )
        self.processes.append(process)
        return process


def _which(name: str, **kwargs: Any) -> str:
    assert name == "claude"
    del kwargs
    return "/private/test/bin/claude"


def _cli_plan(
    runner: FakeClaudeRunner | None = None,
    *,
    env: Mapping[str, str] | None = None,
    effort: str = "medium",
    schema: Any = Answer,
    input_blocks: Any = None,
) -> tuple[revision_provider.RevisionProviderPlan, FakeClaudeRunner]:
    used = runner or FakeClaudeRunner()
    attachments = {} if input_blocks is None else {"input_blocks": input_blocks}
    plan = revision_provider.plan_provider(
        "claude-code",
        model=MODEL,
        style_guide="Style\n",
        task_template="Task\n",
        system_blocks=BLOCKS,
        user_turn=USER_TURN,
        schema=schema,
        effort=effort,
        env=env or {"PATH": "/bin", "BENIGN": "kept"},
        runner=used,
        which=_which,
        **attachments,
    )
    return plan, used


def _prepared_stream(
    reply: bytes,
    *,
    returncode: int = 0,
    read_error: int | None = None,
    blocking_stdin: bool = False,
) -> tuple[
    revision_provider.RevisionProvider,
    revision_provider.PreparedRevisionProvider,
    FakeClaudeRunner,
]:
    plan, runner = _cli_plan(
        FakeClaudeRunner(
            reply=reply,
            returncode=returncode,
            read_error=read_error,
            blocking_stdin=blocking_stdin,
        ),
        schema=StreamedAnswer,
    )
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(plan, runner=runner, which=_which)
    return provider, prepared, runner


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
    assert child["CLAUDE_CODE_EFFORT_LEVEL"] == "medium"
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
        # Readable JSON whose subscription field is not text: a refusal, not a
        # TypeError from testing an unhashable value against a frozenset.
        ({"subscriptionType": ["max"]}, None),
        ({"subscriptionType": {"plan": "max"}}, None),
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
            effort="medium",
            runner=lambda *_args, **_kwargs: pytest.fail("must fail before probing"),
            which=_which,
        )


def test_claude_command_has_exact_isolation_and_exact_prompt_channels() -> None:
    reply = (
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {"examples": ["食べられます。"]},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    plan, runner = _cli_plan(FakeClaudeRunner(reply=reply))
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(
        plan,
        env={"PATH": "/bin", "BENIGN": "kept"},
        runner=runner,
        which=_which,
    )
    captured: list[bytes] = []

    result = provider.dispatch(prepared, capture=captured.append, spawn=runner.spawn)

    assert result.parsed == Answer(examples=["食べられます。"])
    assert captured == [reply]
    command, kwargs = runner.spawned[-1]
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
        "--verbose",
        "--output-format",
        "--include-partial-messages",
        "--model",
        "--effort",
        "--json-schema",
        "--system-prompt",
    ):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--prompt-suggestions") + 1] == "false"
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert command[command.index("--model") + 1] == MODEL
    assert command[command.index("--effort") + 1] == "medium"
    assert command[command.index("--system-prompt") + 1] == "Style\n\n\nTask\n"
    assert json.loads(command[command.index("--json-schema") + 1])["type"] == "object"
    # The prompt is a pipe the transport writes and closes, never an argument
    # and never a keyword the caller could substitute.
    assert "input" not in kwargs
    assert runner.processes[-1].stdin.written == USER_TURN.encode("utf-8")
    assert "shell" not in kwargs


def test_claude_plan_binds_effort_into_argv_environment_and_transport() -> None:
    """One depth, three bound channels — the caller's, not a module default.

    The Assistant's turn is the one pass that configures this, so a level that
    reached only the command line would leave the environment and the stored
    transport describing a call that never happened.
    """
    plan, runner = _cli_plan(effort="high")

    argv = list(plan.transport["argv"])
    assert argv[argv.index("--effort") + 1] == "high"
    assert plan.transport["effort"] == "high"
    assert plan.transport["controlled_environment"]["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    # The probe already runs under the depth the call will use.
    assert runner.calls[0][1]["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    request = json.loads(plan.persistent_manifest()["request_bytes_utf8"])
    assert request["argv"][request["argv"].index("--effort") + 1] == "high"
    assert request["controlled_environment"]["CLAUDE_CODE_EFFORT_LEVEL"] == "high"


def test_claude_transport_effort_must_agree_with_argv_on_reconstruction() -> None:
    """A stored depth that disagrees with the command is refused, not preferred.

    The transport is what the fingerprint covers and what a receipt displays;
    the argv is what actually ran. A manifest whose two disagree describes a
    call nobody made, so it cannot be reconstructed at all.
    """
    plan, _ = _cli_plan(effort="medium")
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    manifest["transport"]["effort"] = "max"
    manifest["request_fingerprint"] = revision_provider._request_identity(
        provider=manifest["provider"],
        billing_class=manifest["billing_class"],
        auth=manifest["auth"],
        model=manifest["model"],
        transport=manifest["transport"],
        request_bytes=manifest["request_bytes_utf8"].encode("utf-8"),
        response_schema_fingerprint=manifest["response_schema_fingerprint"],
    )

    with pytest.raises(
        revision_provider.RevisionProviderError, match="bound model, prompts"
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


def test_claude_command_is_bound_whole_not_flag_by_flag() -> None:
    """A stored command missing a flag the CLI needs is refused on reconstruction.

    Binding a command by checking the options janki happens to name leaves every
    other flag free to drift: a manifest whose argv lost --safe-mode described a
    call with a wider permission surface than the one that was fingerprinted,
    and it validated. The command is rebuilt from the plan's own fields and
    compared as one value, so a flag is bound the moment _cli_argv emits it.
    """
    plan, _ = _cli_plan(effort="medium")
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    argv = [item for item in manifest["transport"]["argv"] if item != "--safe-mode"]
    assert len(argv) == len(manifest["transport"]["argv"]) - 1
    manifest["transport"]["argv"] = argv
    request = json.loads(manifest["request_bytes_utf8"])
    request["argv"] = argv
    manifest["request_bytes_utf8"] = revision_provider._canonical_json(request)
    manifest["request_bytes_sha256"] = hashlib.sha256(
        manifest["request_bytes_utf8"].encode("utf-8")
    ).hexdigest()
    manifest["request_fingerprint"] = revision_provider._request_identity(
        provider=manifest["provider"],
        billing_class=manifest["billing_class"],
        auth=manifest["auth"],
        model=manifest["model"],
        transport=manifest["transport"],
        request_bytes=manifest["request_bytes_utf8"].encode("utf-8"),
        response_schema_fingerprint=manifest["response_schema_fingerprint"],
    )

    with pytest.raises(
        revision_provider.RevisionProviderError, match="bound model, prompts"
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


def test_claude_prepare_binds_the_dispatch_environment_to_the_plan() -> None:
    """The child runs under the plan's environment, not today's constants.

    Preparation re-probes the CLI, and the environment that probe builds is the
    one dispatch reuses. Rebuilding it from the module would let a configured
    depth changed since planning reach a call whose fingerprint says otherwise.
    """
    reply = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"examples": ["食べられます。"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    plan, runner = _cli_plan(FakeClaudeRunner(reply=reply), effort="low")
    provider = revision_provider.provider_for("claude-code")

    prepared = provider.prepare(plan, env={"PATH": "/bin"}, runner=runner, which=_which)
    provider.dispatch(prepared, capture=lambda _raw: None, spawn=runner.spawn)

    assert prepared.environment is not None
    assert prepared.environment["CLAUDE_CODE_EFFORT_LEVEL"] == "low"
    assert runner.spawned[-1][1]["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "low"


def test_claude_dispatch_captures_exact_stdout_and_decodes_the_result_line() -> None:
    """The artifact is the whole stream; the answer is its one result frame.

    Capturing only the frame janki reads would leave the paid call's own record
    of what it did — its init, its usage, its turn count — unrecoverable.
    """
    raw = STREAM_FIXTURE.read_bytes()
    provider, prepared, runner = _prepared_stream(raw)
    captured: list[bytes] = []

    result = provider.dispatch(prepared, capture=captured.append, spawn=runner.spawn)

    assert captured == [raw]
    assert result.parsed == StreamedAnswer(answer=_fixture_answer())
    assert result.stop_reason == "end_turn"


def test_claude_reply_needs_exactly_one_result_line() -> None:
    """No result frame and two result frames are both unreadable, not answers.

    Silently preferring one of several would settle a call on a frame nobody
    proved was the last, and a capture with none finished nothing at all.
    """
    provider = revision_provider.provider_for("claude-code")
    plan, _ = _cli_plan(schema=StreamedAnswer)

    for reply in (
        b"".join(_without_result_line()),
        STREAM_FIXTURE.read_bytes() + _result_line(),
    ):
        with pytest.raises(
            revision_provider.RevisionProviderError, match="result frames"
        ):
            provider.recover(plan, reply)


def test_claude_error_result_line_decodes_to_no_answer() -> None:
    """A failed result frame is a turn that said nothing, not a torn capture."""
    failed = json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
        }
    ).encode("utf-8")
    reply = b"".join([*_without_result_line(), failed, b"\n"])
    provider, prepared, runner = _prepared_stream(reply)
    captured: list[bytes] = []

    result = provider.dispatch(prepared, capture=captured.append, spawn=runner.spawn)

    assert captured == [reply]
    assert result.parsed is None
    assert result.stop_reason == "error_during_execution"


def test_claude_recover_decodes_a_captured_stream_reply_identically() -> None:
    """Recovery reads the same whole capture dispatch did, not its first frame."""
    raw = STREAM_FIXTURE.read_bytes()
    provider, prepared, runner = _prepared_stream(raw)

    dispatched = provider.dispatch(
        prepared, capture=lambda _raw: None, spawn=runner.spawn
    )
    recovered = provider.recover(prepared.plan, raw)

    assert recovered.parsed == dispatched.parsed
    assert recovered.stop_reason == dispatched.stop_reason


def test_claude_stream_frames_concatenate_to_the_captured_artifact() -> None:
    """Every spooled frame keeps its exact bytes, newline included.

    The spool exists so an interrupted call can be read back; a frame stripped
    of its boundary would rebuild an artifact the receipt no longer matches.
    """
    raw = STREAM_FIXTURE.read_bytes()
    provider, prepared, runner = _prepared_stream(raw)
    frames: list[str] = []
    captured: list[bytes] = []

    provider.dispatch(
        prepared,
        capture=captured.append,
        spawn=runner.spawn,
        frame=frames.append,
    )

    assert len(frames) == len(_stream_lines())
    assert "".join(frames).encode("utf-8") == captured[0]


def test_claude_frame_failure_kills_the_process_and_captures_nothing() -> None:
    """A frame janki could not make durable ends the call where it stands.

    Reading on would consume a reply whose earlier bytes are already lost, and
    capturing the remainder would publish an artifact missing them.
    """
    raw = STREAM_FIXTURE.read_bytes()
    provider, prepared, runner = _prepared_stream(raw)
    frames: list[str] = []
    captured: list[bytes] = []

    def spool(payload: str) -> None:
        frames.append(payload)
        if len(frames) == 3:
            raise OSError("the frame spool is full")

    with pytest.raises(
        revision_provider.RevisionProviderError, match="frame capture failed"
    ):
        provider.dispatch(
            prepared,
            capture=captured.append,
            spawn=runner.spawn,
            frame=spool,
        )

    assert captured == []
    assert runner.processes[-1].killed


def test_claude_undecodable_stream_line_is_captured_then_refused() -> None:
    """One unreadable byte never costs the reply that carried it.

    The child was paid for the whole stream whatever the bytes turned out to
    be, so reading runs to EOF, the exact artifact is captured, and only then
    is the call refused — naming the frame nobody can read.
    """
    corrupted = _stream_with_torn_frame(16)
    provider, prepared, runner = _prepared_stream(corrupted)
    frames: list[str] = []
    captured: list[bytes] = []

    with pytest.raises(
        revision_provider.RevisionProviderError, match="frame 16 is not UTF-8"
    ):
        provider.dispatch(
            prepared,
            capture=captured.append,
            spawn=runner.spawn,
            frame=frames.append,
        )

    assert captured == [corrupted]
    assert frames == [line.decode("utf-8") for line in _stream_lines()[:15]]
    assert not runner.processes[-1].killed


def test_claude_refused_stream_spool_is_a_prefix_of_the_artifact() -> None:
    """The spool stops where reading did, so it is a prefix and not a hole.

    Spooling on past the unreadable frame would leave a durable record that
    rebuilds to a stream missing the very bytes that refused it, and every
    later frame would then be attributed to a reply nobody could read.
    """
    corrupted = _stream_with_torn_frame(16)
    provider, prepared, runner = _prepared_stream(corrupted)
    frames: list[str] = []
    captured: list[bytes] = []

    with pytest.raises(revision_provider.RevisionProviderError, match="not UTF-8"):
        provider.dispatch(
            prepared,
            capture=captured.append,
            spawn=runner.spawn,
            frame=frames.append,
        )

    spooled = "".join(frames).encode("utf-8")
    assert captured[0].startswith(spooled)
    assert len(captured[0]) > len(spooled)


def test_claude_read_failure_kills_and_reaps_the_child() -> None:
    """A torn read leaves no child behind in a directory about to be deleted.

    Dispatch reads inside a temporary cwd it removes on the way out, so a
    child still running there writes into a directory that is gone. It is
    killed and waited for exactly once, and only then is its prompt writer
    joined: a writer blocked on a full stdin pipe is released by the child's
    death, not before it. Nothing is captured either — a read that tore
    mid-stream holds bytes janki cannot claim are the whole reply.
    """
    provider, prepared, runner = _prepared_stream(
        STREAM_FIXTURE.read_bytes(), read_error=3, blocking_stdin=True
    )
    captured: list[bytes] = []

    with pytest.raises(OSError, match="Input/output error"):
        provider.dispatch(prepared, capture=captured.append, spawn=runner.spawn)

    process = runner.processes[-1]
    assert process.killed
    assert process.waits == 1
    assert not process.stdin.timed_out
    assert process.stdin.closed
    assert captured == []


def test_claude_prompt_write_failure_still_closes_stdin() -> None:
    """A torn prompt write still ends the prompt.

    The close is the CLI's end of input. Skipping it because the write failed
    leaves a child waiting on a stdin that never reaches EOF, and holds the
    pipe open past the turn that opened it.
    """

    class BrokenPipe(FakeStdin):
        def write(self, payload: Any) -> int:
            raise OSError(errno.EPIPE, "Broken pipe")

    process = FakePopen(b"", returncode=0, stderr=io.BytesIO())
    process.stdin = BrokenPipe()

    revision_provider._send_claude_prompt(process, b"one exact user turn")

    assert process.stdin.closed


#: A child that writes its whole reply from a thread, waits a bounded moment for
#: that write to land, and then force-exits zero whatever is left unwritten.
#: That is the shape of the installed CLI's own shutdown: it writes stdout
#: without awaiting backpressure, drains for a capped interval, and exits with
#: the requested code even when the drain timed out. Nothing here is a model,
#: a network call, or a Node dependency — it is stdlib Python over a real pipe.
BOUNDED_DRAIN_CHILD = """
import os, sys, threading

blob = open(sys.argv[1], "rb").read()


def pump():
    view = memoryview(blob)
    while view:
        view = view[os.write(1, view[:65536]):]


writer = threading.Thread(target=pump, daemon=True)
writer.start()
writer.join(float(sys.argv[2]))
os._exit(0)
"""


def _synthetic_ndjson(frames: int) -> bytes:
    """Synthetic ASCII NDJSON far larger than any pipe buffer, plus a result.

    The prose frames are the shape the preview decoder reads, so this proves
    frame order and preview order on the same bytes it proves the byte count on.
    """
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            }
        ),
        *(
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": f"chunk-{number} "},
                    },
                }
            )
            for number in range(2)
        ),
        *(
            json.dumps({"type": "filler", "index": number, "text": "A" * 96})
            for number in range(frames)
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {"answer": "done"},
            }
        ),
    ]
    return "".join(line + "\n" for line in lines).encode("ascii")


def test_claude_stream_drains_a_real_pipe_while_a_frame_is_made_durable(
    tmp_path: Path,
) -> None:
    """A slow durable frame must not cost the reply bytes it was paid for.

    The CLI writes its stream without waiting for anyone to read it and drains
    only briefly before force-exiting zero, so a reader that stops reading
    while it makes one frame durable loses everything past the pipe buffer —
    and the loss arrives as a clean exit with a truncated capture, which is
    indistinguishable from a short reply. Stdout is therefore drained
    independently of what the frame callback is doing; here the very first
    callback deliberately blocks until the child is gone, and every byte the
    child wrote still arrives, in order, terminal frame included.
    """
    payload = _synthetic_ndjson(24000)
    assert len(payload) > 2 * 1024 * 1024
    reply = tmp_path / "reply.ndjson"
    reply.write_bytes(payload)
    process = subprocess.Popen(  # noqa: S603 - stdlib child, no shell, no network
        [sys.executable, "-c", BOUNDED_DRAIN_CHILD, str(reply), "1.0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    order: list[tuple[str, str]] = []
    frames: list[str] = []
    exited: list[bool] = []

    def durable(payload_text: str) -> None:
        if not frames:
            # The blocked fsync this exists to survive: nothing is read while
            # it runs. The failsafe only bounds a hang; the child's own drain
            # budget is what ends the wait.
            deadline = time.monotonic() + 30.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            exited.append(process.poll() is not None)
        frames.append(payload_text)
        order.append(("frame", payload_text))

    threads = threading.active_count()
    returncode, chunks, unreadable = revision_provider._stream_claude_reply(
        process,
        b"{}\n",
        frame=durable,
        preview=lambda text: order.append(("preview", text)),
    )

    assert exited == [True]
    assert unreadable is None
    assert returncode == 0
    assert b"".join(chunks) == payload
    assert chunks == payload.splitlines(keepends=True)
    assert "".join(frames).encode("utf-8") == payload
    assert json.loads(frames[-1])["type"] == "result"
    # Every preview is the frame that immediately preceded it: nothing is read
    # out of the queue before its own durable callback returned.
    for index, (kind, _text) in enumerate(order):
        if kind == "preview":
            assert order[index - 1][0] == "frame"
    assert [text for kind, text in order if kind == "preview"] == [
        "chunk-0 ",
        "chunk-1 ",
    ]
    assert process.stdout.closed
    assert process.stdin.closed
    assert threading.active_count() == threads


def test_claude_preview_receives_only_first_text_block_deltas() -> None:
    """The live preview is the streamed prose, and nothing else on the wire.

    Thinking is not an answer, the structured answer arrives later as partial
    tool-call JSON that is not displayable text, and the fixture proves the
    prose is only a near-paraphrase of what the schema finally returns. The
    schema round trip is a second message, and prose it happens to stream is
    the model talking to the tool call, not to the reader — so this splices
    exactly that into the real capture, which never contained any.
    """
    later_prose = b"".join(
        json.dumps(payload).encode("utf-8") + b"\n"
        for payload in (
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "LATER"},
                },
            },
            {
                "type": "stream_event",
                "event": {"type": "content_block_stop", "index": 1},
            },
        )
    )
    lines = _stream_lines()
    # After the last block the CLI really closed, which is the tool call the
    # second message was opened for.
    cut = 1 + max(
        index
        for index, line in enumerate(lines)
        if json.loads(line).get("event", {}).get("type") == "content_block_stop"
    )
    raw = b"".join([*lines[:cut], later_prose, *lines[cut:]])
    provider, prepared, runner = _prepared_stream(raw)
    deltas: list[str] = []

    provider.dispatch(
        prepared,
        capture=lambda _raw: None,
        spawn=runner.spawn,
        preview=deltas.append,
    )

    assert "LATER" not in "".join(deltas)
    assert tuple(deltas) == _fixture_preview()
    assert "".join(deltas) != _fixture_answer()


def test_claude_preview_keeps_the_first_text_block_when_a_second_starts() -> None:
    """The preview is one block's prose, so a later block cannot join it.

    A message may open more than one text block; only the first is the answer
    being previewed, and appending a second would show two passes as one.
    """
    second_block = b"".join(
        json.dumps(payload).encode("utf-8") + b"\n"
        for payload in (
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "text", "text": ""},
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": "A second pass."},
                },
            },
        )
    )
    lines = _stream_lines()
    cut = next(
        index
        for index, line in enumerate(lines)
        if json.loads(line).get("event", {}).get("type") == "message_delta"
    )
    raw = b"".join([*lines[:cut], second_block, *lines[cut:]])
    provider, prepared, runner = _prepared_stream(raw)
    deltas: list[str] = []

    provider.dispatch(
        prepared,
        capture=lambda _raw: None,
        spawn=runner.spawn,
        preview=deltas.append,
    )

    assert tuple(deltas) == _fixture_preview()


def test_claude_preview_survives_a_blank_stdout_line() -> None:
    """A blank line is not a frame, and not the end of the preview either.

    The decoder skips blank lines, so a preview that treated one as a fatal
    error would go quiet for the rest of a turn the CLI is still streaming —
    and say nothing about why.
    """
    lines = _stream_lines()
    first_delta = next(
        index
        for index, line in enumerate(lines)
        if json.loads(line).get("event", {}).get("delta", {}).get("type")
        == "text_delta"
    )
    raw = b"".join([*lines[:first_delta], b"\n", *lines[first_delta:]])
    provider, prepared, runner = _prepared_stream(raw)
    deltas: list[str] = []

    provider.dispatch(
        prepared,
        capture=lambda _raw: None,
        spawn=runner.spawn,
        preview=deltas.append,
    )

    assert tuple(deltas) == _fixture_preview()


def test_claude_captures_nonzero_stdout_before_reporting_failure() -> None:
    raw = b'{"type":"result","subtype":"error","is_error":true}'
    plan, runner = _cli_plan(FakeClaudeRunner(reply=raw, returncode=3))
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(plan, runner=runner, which=_which)
    events: list[bytes] = []

    with pytest.raises(revision_provider.RevisionProviderError, match="exit 3"):
        provider.dispatch(prepared, capture=events.append, spawn=runner.spawn)

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
            provider.dispatch(prepared, capture=events.append, spawn=runner.spawn)

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
        effort="medium",
        runner=FakeClaudeRunner(),
        which=_which,
    )

    effort_plan, _ = _cli_plan(effort="high")

    assert len(
        {
            max_plan.request_fingerprint,
            pro_plan.request_fingerprint,
            version_plan.request_fingerprint,
            turn_plan.request_fingerprint,
            effort_plan.request_fingerprint,
        }
    ) == 5


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
        effort="xhigh",
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
        effort="xhigh",
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
        effort="xhigh",
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
            effort="xhigh",
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
        effort="xhigh",
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


SUCCESS_REPLY = (
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"examples": ["食べられます。"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    + b"\n"
)


def _attachment_dispatch(
    blocks: list[dict[str, Any]],
    *,
    reply: bytes = SUCCESS_REPLY,
) -> tuple[
    revision_provider.RevisionProviderPlan,
    FakeClaudeRunner,
    revision_provider.RevisionProvider,
    revision_provider.PreparedRevisionProvider,
]:
    plan, runner = _cli_plan(FakeClaudeRunner(reply=reply), input_blocks=blocks)
    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(plan, runner=runner, which=_which)
    return plan, runner, provider, prepared


def test_claude_sends_prepared_input_blocks_as_one_stream_json_stdin_frame() -> None:
    """The attachment turn is one canonical frame, blocks first and text last.

    The CLI reads a document or image only from a stream-json user message, so
    the exact base64 preparation produced has to reach stdin unaltered and in
    order — anything else silently sends a text-only turn the owner did not
    plan, at full subscription cost.
    """
    blocks = _input_blocks()
    _, runner, provider, prepared = _attachment_dispatch(blocks)
    captured: list[bytes] = []

    result = provider.dispatch(prepared, capture=captured.append, spawn=runner.spawn)

    assert result.parsed == Answer(examples=["食べられます。"])
    assert captured == [SUCCESS_REPLY]
    command, kwargs = runner.spawned[-1]
    written = runner.processes[-1].stdin.written
    assert written == _expected_frame(blocks)
    assert written.count(b"\n") == 1 and written.endswith(b"\n")
    assert runner.processes[-1].stdin.closed
    # The exact bytes preparation produced, never re-encoded on the way through.
    for data in (PDF_DATA, PNG_DATA, JPEG_DATA):
        assert data.encode("ascii") in written
    frame = json.loads(written)
    assert frame["type"] == "user"
    assert frame["message"]["role"] == "user"
    assert frame["message"]["content"] == [*blocks, {"type": "text", "text": USER_TURN}]
    assert command[command.index("--input-format") + 1] == "stream-json"
    assert command[command.index("--output-format") + 1] == "stream-json"
    # ``-p`` is already the print flag; a second one would be a duplicate.
    assert command.count("-p") == 1
    assert "--print" not in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "--safe-mode" in command
    assert "input" not in kwargs and "shell" not in kwargs
    # The child still runs in a fresh empty directory: FakeClaudeRunner.spawn
    # asserts that, and the plan says so.
    assert kwargs["cwd"] != ""


def test_claude_text_only_turn_keeps_its_exact_plain_stdin_and_command() -> None:
    """No attachment means no new flag and no framing — the same bytes as before."""
    plain, _ = _cli_plan()
    empty, runner = _cli_plan(FakeClaudeRunner(reply=SUCCESS_REPLY), input_blocks=[])

    assert empty.request_bytes == plain.request_bytes
    assert empty.request_fingerprint == plain.request_fingerprint
    assert empty.input_blocks == ()
    assert plain.input_blocks == ()
    assert "--input-format" not in tuple(plain.transport["argv"])
    assert json.loads(plain.request_bytes)["stdin_utf8"] == USER_TURN

    provider = revision_provider.provider_for("claude-code")
    prepared = provider.prepare(empty, runner=runner, which=_which)
    provider.dispatch(prepared, capture=lambda _raw: None, spawn=runner.spawn)

    assert runner.processes[-1].stdin.written == USER_TURN.encode("utf-8")
    assert "--input-format" not in runner.spawned[-1][0]


def test_input_blocks_are_frozen_and_bound_into_the_request_fingerprint() -> None:
    """A caller that keeps its blocks cannot edit a plan already rendered.

    The plan is what the owner confirms and what the fingerprint covers, so the
    source bytes it will send must stop being the caller's to change — and two
    different documents must never look like the same paid request.
    """
    blocks = _input_blocks()
    plan, _ = _cli_plan(input_blocks=blocks)
    before = plan.request_bytes

    blocks[0]["source"]["data"] = base64.b64encode(b"swapped").decode("ascii")
    blocks.append(_image_block("image/png", PNG_DATA))

    assert plan.request_bytes == before
    assert json.loads(plan.request_bytes)["stdin_utf8"] == _expected_frame(
        _input_blocks()
    ).decode("utf-8")
    assert len(plan.input_blocks) == 3
    assert plan.input_blocks[0]["source"]["data"] == PDF_DATA
    with pytest.raises(TypeError):
        plan.input_blocks[0]["source"]["data"] = "other"  # type: ignore[index]

    other, _ = _cli_plan(
        input_blocks=[_document_block(base64.b64encode(b"other pages").decode("ascii"))]
    )
    text_only, _ = _cli_plan()
    assert (
        len(
            {
                plan.request_fingerprint,
                other.request_fingerprint,
                text_only.request_fingerprint,
            }
        )
        == 3
    )


def test_stored_attachment_manifest_only_reconstructs_with_its_own_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery replays the stored request or refuses, and never probes to do it.

    The manifest keeps its existing field set because the request bytes already
    carry the frame; the blocks are supplied back in. Blocks that are missing or
    different rebuild a different stdin, which is a different call than the one
    that was paid for.
    """
    blocks = _input_blocks()
    plan, _ = _cli_plan(input_blocks=blocks)
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    assert set(manifest) == set(revision_provider.PERSISTENT_MANIFEST_KEYS)
    monkeypatch.setattr(
        revision_provider,
        "_probe_claude",
        lambda **_kwargs: pytest.fail("recovery probed the CLI"),
    )
    monkeypatch.setattr(
        revision_provider,
        "_resolve_claude",
        lambda *_args: pytest.fail("recovery looked for an executable"),
    )
    monkeypatch.setattr(
        claude_client,
        "prepare_paid_client",
        lambda *_args, **_kwargs: pytest.fail("recovery built an API client"),
    )

    def recover(**changes: Any) -> revision_provider.RevisionProviderPlan:
        return revision_provider.provider_plan_from_manifest(
            manifest,
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=changes.pop("user_turn", USER_TURN),
            schema=Answer,
            **changes,
        )

    recovered = recover(input_blocks=blocks)
    assert recovered.request_bytes == plan.request_bytes
    assert recovered.request_fingerprint == plan.request_fingerprint
    assert recovered.input_blocks == plan.input_blocks

    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        recover()
    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        recover(input_blocks=blocks[:2])
    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        recover(
            input_blocks=[
                _document_block(base64.b64encode(b"other pages").decode("ascii")),
                *blocks[1:],
            ]
        )
    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        recover(input_blocks=blocks, user_turn="A different unconfirmed instruction")


def test_replaced_plan_input_blocks_cannot_escape_the_bound_request_bytes() -> None:
    """A plan whose blocks were swapped after rendering is refused before spawn."""
    blocks = _input_blocks()
    plan, runner = _cli_plan(input_blocks=blocks)
    provider = revision_provider.provider_for("claude-code")

    forged = replace(
        plan,
        input_blocks=(_document_block(base64.b64encode(b"forged").decode("ascii")),),
    )
    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        provider.prepare(forged, runner=runner, which=_which)

    stripped = replace(plan, input_blocks=())
    with pytest.raises(revision_provider.RevisionProviderError, match="prompt channels"):
        provider.prepare(stripped, runner=runner, which=_which)
    assert runner.spawned == []


def test_attachment_call_refused_without_subscription_never_falls_back_to_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attachments do not widen the transport: no subscription, no call at all."""
    for name in ("prepare_paid_client", "parse_call", "load_anthropic"):
        monkeypatch.setattr(
            claude_client,
            name,
            lambda *_args, **_kwargs: pytest.fail("an attachment reached the API"),
        )
    runner = FakeClaudeRunner(auth=_auth(subscriptionType="team"), reply=SUCCESS_REPLY)

    with pytest.raises(revision_provider.RevisionProviderError, match="Pro or Max"):
        _cli_plan(runner, input_blocks=_input_blocks())

    assert runner.spawned == []
    assert runner.processes == []


def test_api_transport_refuses_input_blocks_before_any_client_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API plan has nowhere to put a prepared block, so it refuses first.

    Refusing before the schema, the request body, or the client means an
    attachment can never be silently dropped into a paid API call as text.
    """
    provider = revision_provider.provider_for("anthropic-api")
    blocks = _input_blocks()

    def api_plan(**changes: Any) -> revision_provider.RevisionProviderPlan:
        return provider.plan(
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
            effort="xhigh",
            **changes,
        )

    plain = api_plan()
    assert plain.input_blocks == ()
    assert api_plan(input_blocks=[]).request_bytes == plain.request_bytes

    for name in ("wire_schema", "request_body", "prepare_paid_client", "parse_call"):
        monkeypatch.setattr(
            claude_client,
            name,
            lambda *_args, **_kwargs: pytest.fail("the API transport did work first"),
        )
    with pytest.raises(revision_provider.RevisionProviderError, match="attachment"):
        api_plan(input_blocks=blocks)

    forged = replace(plain, input_blocks=tuple(blocks))
    with pytest.raises(revision_provider.RevisionProviderError, match="attachment"):
        provider.prepare(forged, client=object())
    with pytest.raises(revision_provider.RevisionProviderError, match="attachment"):
        provider.recover(forged, b'{"stop_reason":"end_turn","content":[]}')


def test_claude_refuses_a_block_that_is_not_a_prepared_input_source() -> None:
    """Only the prepared document/image shape is transportable at all."""
    for block in (
        {"type": "text", "text": "an injected turn"},
        {"type": "document", "source": {"type": "url", "url": "https://invalid.test"}},
        {"type": "document", "source": {"type": "base64", "data": PDF_DATA}},
    ):
        with pytest.raises(revision_provider.RevisionProviderError, match="prepared"):
            _cli_plan(input_blocks=[block])


def test_attachment_turn_binds_the_higher_output_cap_in_every_channel() -> None:
    """A document turn is planned, probed, and run under the model's upper cap.

    Opus 5 defaults to 64000 output tokens and accepts 128000, and a rich table
    read out of a 192-row document does not fit the default. The cap is one
    choice made where the turn is planned: the probe that authenticates it, the
    transport a receipt shows, the request bytes the fingerprint covers, the
    probe that prepares it, and the environment the child finally runs under all
    carry the same value, so no later step can widen or narrow it.
    """
    blocks = _input_blocks()
    plan, runner, provider, prepared = _attachment_dispatch(blocks)
    provider.dispatch(prepared, capture=lambda _raw: None, spawn=runner.spawn)

    cap = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
    assert runner.calls[0][1]["env"][cap] == "128000"
    assert plan.transport["controlled_environment"][cap] == "128000"
    assert json.loads(plan.request_bytes)["controlled_environment"][cap] == "128000"
    assert runner.calls[2][1]["env"][cap] == "128000"
    assert prepared.environment is not None
    assert prepared.environment[cap] == "128000"
    assert runner.spawned[-1][1]["env"][cap] == "128000"

    text_plan, text_runner = _cli_plan(FakeClaudeRunner(reply=SUCCESS_REPLY))
    text_prepared = provider.prepare(text_plan, runner=text_runner, which=_which)
    provider.dispatch(text_prepared, capture=lambda _raw: None, spawn=text_runner.spawn)

    assert text_runner.calls[0][1]["env"][cap] == "64000"
    assert text_plan.transport["controlled_environment"][cap] == "64000"
    assert json.loads(text_plan.request_bytes)["controlled_environment"][cap] == "64000"
    assert text_runner.spawned[-1][1]["env"][cap] == "64000"
    assert plan.request_fingerprint != text_plan.request_fingerprint


def test_stored_output_cap_must_agree_with_the_turn_that_was_planned() -> None:
    """A manifest whose cap does not match its own turn describes no real call.

    The cap follows from the turn, not from an environment somebody supplies
    later, so a re-signed manifest that keeps the attachment frame but stores
    the text cap is refused rather than run under it.
    """
    blocks = _input_blocks()
    plan, _ = _cli_plan(input_blocks=blocks)
    manifest = json.loads(json.dumps(plan.persistent_manifest()))
    cap = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
    assert manifest["transport"]["controlled_environment"][cap] == "128000"
    manifest["transport"]["controlled_environment"][cap] = "64000"
    request = json.loads(manifest["request_bytes_utf8"])
    request["controlled_environment"][cap] = "64000"
    manifest["request_bytes_utf8"] = revision_provider._canonical_json(request)
    manifest["request_bytes_sha256"] = hashlib.sha256(
        manifest["request_bytes_utf8"].encode("utf-8")
    ).hexdigest()
    manifest["request_fingerprint"] = revision_provider._request_identity(
        provider=manifest["provider"],
        billing_class=manifest["billing_class"],
        auth=manifest["auth"],
        model=manifest["model"],
        transport=manifest["transport"],
        request_bytes=manifest["request_bytes_utf8"].encode("utf-8"),
        response_schema_fingerprint=manifest["response_schema_fingerprint"],
    )

    with pytest.raises(
        revision_provider.RevisionProviderError, match="bound model, prompts"
    ):
        revision_provider.provider_plan_from_manifest(
            manifest,
            model=MODEL,
            style_guide="Style\n",
            task_template="Task\n",
            system_blocks=BLOCKS,
            user_turn=USER_TURN,
            schema=Answer,
            input_blocks=blocks,
        )


def test_attachment_reply_is_captured_before_it_is_read(
    tmp_path: Path,
) -> None:
    """An attachment turn is paid for whatever it replies, so bytes come first.

    The stream is spooled frame by frame and captured whole before any JSON or
    schema decoding refuses it — the same lifecycle a text turn already has.
    """
    del tmp_path
    unreadable = b'{"type":"result","subtype":"success"\n'
    mismatched = (
        b'{"type":"result","subtype":"success","is_error":false,'
        b'"structured_output":{"examples":"wrong"}}\n'
    )
    for reply in (unreadable, mismatched):
        blocks = _input_blocks()
        _, runner, provider, prepared = _attachment_dispatch(blocks, reply=reply)
        captured: list[bytes] = []
        frames: list[str] = []

        with pytest.raises(revision_provider.RevisionProviderError):
            provider.dispatch(
                prepared,
                capture=captured.append,
                spawn=runner.spawn,
                frame=frames.append,
            )

        assert captured == [reply]
        assert "".join(frames).encode("utf-8") == reply
        assert runner.processes[-1].stdin.written == _expected_frame(blocks)
