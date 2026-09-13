"""Subscription-only providers for development, planning, and code review.

A provider owns CLI dialect and authentication evidence. The shared launcher
owns selection, process context and dispatch. Adding a provider means adding
an implementation and registering it; it never means adding a fallback route.
Runtime content providers and their paid-operation journal are separate.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from japanese_anki import subscription_auth

Role = Literal["implementation", "review", "planning"]
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ROLES = ("implementation", "review", "planning")


class LaunchRefused(Exception):
    """Authentication or the selected execution context could not be proven."""


@dataclass(frozen=True)
class ModelRequest:
    model: str
    effort: str
    role: Role
    json_output: bool = False

    def validate(self) -> None:
        # This rejects generic aliases and native option injection. Availability
        # of an explicitly named model is the CLI's to establish, never a reason
        # for this launcher to choose a replacement model.
        if not re.fullmatch(r"[a-zA-Z0-9]+-[a-zA-Z0-9._:-]+(?:\[1m\])?", self.model):
            raise ValueError("--model must name an explicit model id, not an alias or CLI option")
        if self.effort not in EFFORTS or self.role not in ROLES:
            raise ValueError("unknown reasoning effort or development role")


class LLMProvider(Protocol):
    """The complete provider seam; implementations do not dispatch themselves."""

    name: str
    executable: str

    def environment(self, host: Mapping[str, str]) -> dict[str, str]: ...
    def authentication_arguments(self, cwd: Path) -> tuple[str, ...]: ...
    def verify_auth(self, completed: subprocess.CompletedProcess[bytes]) -> None: ...
    def run_arguments(self, request: ModelRequest, cwd: Path) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class LaunchPlan:
    provider: str
    executable: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path


class ClaudeProvider:
    name = "claude"
    executable = "claude"
    _base = ("--safe-mode", "--setting-sources", "")

    def environment(self, host: Mapping[str, str]) -> dict[str, str]:
        return subscription_auth.sanitized_environment(host)

    def authentication_arguments(self, cwd: Path) -> tuple[str, ...]:
        return (*self._base, *subscription_auth.AUTH_STATUS_ARGUMENTS)

    def verify_auth(self, completed: subprocess.CompletedProcess[bytes]) -> None:
        try:
            payload = subscription_auth.parse_auth_status(completed.stdout)
        except subscription_auth.SubscriptionAuthError as exc:
            raise LaunchRefused("Claude returned unreadable authentication status") from exc
        reason = subscription_auth.subscription_auth_reason(payload)
        if reason:
            raise LaunchRefused(f"Claude Pro/Max login required: {reason}")

    def run_arguments(self, request: ModelRequest, cwd: Path) -> tuple[str, ...]:
        tools = ("Read", "Glob", "Grep")
        if request.role == "implementation":
            tools = ("Read", "Glob", "Grep", "Write", "Edit", "Bash")
        output = ("--output-format", "stream-json", "--verbose") if request.json_output else ()
        return (
            *self._base, "-p", "--model", request.model, "--effort", request.effort,
            "--permission-mode", "dontAsk", "--tools", ",".join(tools),
            "--allowedTools", *tools, *output,
        )


class CodexProvider:
    name = "codex"
    executable = "codex"
    _routing = (
        "-c", 'forced_login_method="chatgpt"',
        "-c", 'model_provider="openai"',
        # login status cannot ignore user config like exec can. Choose the
        # same credential store explicitly in both processes.
        "-c", 'cli_auth_credentials_store="auto"',
    )

    def environment(self, host: Mapping[str, str]) -> dict[str, str]:
        # Same host allowlist, without Claude-specific controlled values. In
        # particular CODEX_API_KEY, OPENAI_API_KEY and CODEX_HOME cannot sneak
        # a different bill or a different credential store into this process.
        return subscription_auth.sanitized_environment(host, {})

    def authentication_arguments(self, cwd: Path) -> tuple[str, ...]:
        return (*self._routing, "login", "status")

    def verify_auth(self, completed: subprocess.CompletedProcess[bytes]) -> None:
        # This CLI publishes login status as a terse stderr message, not JSON.
        # Accept one exact positive shape; unfamiliar/new diagnostics refuse.
        raw = completed.stdout + completed.stderr
        if raw.strip() != b"Logged in using ChatGPT":
            raise LaunchRefused("Codex must report an active ChatGPT login")

    def run_arguments(self, request: ModelRequest, cwd: Path) -> tuple[str, ...]:
        sandbox = "workspace-write" if request.role == "implementation" else "read-only"
        # Command-line configuration outranks project/user defaults. Built-in
        # openai is reserved by Codex, so a custom provider cannot shadow it.
        # Skip project config/hooks as well as user config for this invocation.
        trust = tuple(
            argument
            for path in (cwd, *cwd.parents)
            for argument in ("-c", f'projects.{_toml_string(str(path))}.trust_level="untrusted"')
        )
        disabled = tuple(
            argument
            for name in ("hooks", "plugins", "apps", "multi_agent", "browser_use",
                         "computer_use", "image_generation", "skill_search")
            for argument in ("--disable", name)
        )
        return (
            *self._routing, "-a", "never", "exec", "--ignore-user-config",
            "--sandbox", sandbox, "--color", "never", "--model", request.model,
            "-c", f'model_reasoning_effort="{request.effort}"',
            "-c", 'web_search="disabled"', *trust, *disabled,
            *(("--json",) if request.json_output else ()), "-",
        )


def _toml_string(value: str) -> str:
    # JSON strings are valid TOML basic strings for these filesystem paths.
    import json
    return json.dumps(value, ensure_ascii=False)


PROVIDERS: Mapping[str, LLMProvider] = MappingProxyType({
    provider.name: provider for provider in (ClaudeProvider(), CodexProvider())
})


def prepare_launch(
    provider: LLMProvider,
    request: ModelRequest | None,
    *,
    host: Mapping[str, str],
    cwd: Path,
    runner=subprocess.run,
) -> LaunchPlan:
    """Resolve, authenticate and freeze exactly one provider; make no model call."""
    if request is not None:
        request.validate()
    directory = cwd.resolve(strict=True)
    environment = provider.environment(host)
    executable = shutil.which(provider.executable, path=environment.get("PATH", os.defpath))
    if executable is None:
        raise LaunchRefused(f"{provider.executable} is not installed on the controlled PATH")
    executable = os.path.abspath(executable)
    try:
        completed = runner(
            [executable, *provider.authentication_arguments(directory)],
            cwd=str(directory), env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchRefused(f"{provider.name} authentication check could not complete") from exc
    if completed.returncode != 0:
        raise LaunchRefused(f"{provider.name} authentication check exited {completed.returncode}")
    provider.verify_auth(completed)
    arguments = provider.run_arguments(request, directory) if request is not None else ()
    return LaunchPlan(
        provider.name, executable, (executable, *arguments),
        MappingProxyType(environment), directory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--provider", required=True, choices=tuple(PROVIDERS))
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=EFFORTS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args(argv)
    request = None
    if not args.check:
        if args.role is None or args.model is None or args.effort is None:
            parser.error("--role, --model and --effort are required for a model launch")
        request = ModelRequest(args.model, args.effort, args.role, args.json)
    try:
        if request is not None:
            request.validate()
        # Open the task before probing; errors cannot spend a call. Keep it on
        # stdin, not in shell text or the process's public argument list.
        prompt = args.prompt_file.open("rb") if args.prompt_file and not args.check else None
        plan = prepare_launch(PROVIDERS[args.provider], request, host=os.environ, cwd=Path.cwd())
        if args.check:
            print(f"{plan.provider}: subscription login verified ({plan.executable}).")
            return 0
        if prompt is not None:
            os.dup2(prompt.fileno(), sys.stdin.fileno())
            prompt.close()
        # Exec preserves stdin, native status, and watchdog/signal handling.
        os.execve(plan.executable, plan.argv, dict(plan.environment))
    except ValueError as exc:
        parser.error(str(exc))
    except (LaunchRefused, OSError) as exc:
        print(f"llm: refusing {args.provider}: {exc}.", file=sys.stderr)
        print("No model call was made. No API fallback is permitted.", file=sys.stderr)
        return 78
    return 0
