#!/usr/bin/env python3
"""Start the Claude Code CLI on the owner's Pro/Max subscription, or refuse.

This is the mandatory entry point for every development, planning and review
model launch in this repository — `scripts/janki-review.sh` goes through it,
and so does a human or agent starting Claude to work on janki. It exists
because a valid Max login and an inherited `ANTHROPIC_API_KEY` look identical
from the outside: the CLI starts, answers, and exits 0 either way, and the only
visible difference is a Console invoice that arrives later.

What it does:

- builds the child's environment by allowlist rather than by unsetting names,
  so an unsafe value in the developer's shell is *ignored* — nothing here
  deletes a credential, edits a shell profile, or changes a setting the owner
  owns;
- resolves an absolute `claude` executable from that allowlisted PATH;
- runs the free `auth status --json` probe under exactly the environment,
  working directory and enforced arguments the launch itself will use, because
  a probe run under different conditions describes a different process;
- refuses on anything short of a claude.ai first-party Pro or Max login, and
  refuses without a fallback: there is no path from here to API billing;
- otherwise `exec`s the CLI, so the caller's argv, stdin, exit status and
  signals — including `janki-review.sh`'s watchdog kill — reach it unchanged.

What it does not do: it does not narrow the CLI for coding. The one-turn,
no-tools, single-schema limits janki's *runtime* revision transport imposes
belong to that transport, and pinning them here would make the launcher
useless for the work it is meant to launch.

This wrapper protects this repository's entry points. It cannot protect a
`claude` you type yourself in another terminal; that is what `--check` and the
instructions in AGENTS.md are for.

Usage:

    scripts/claude-subscription.py --check      # verify the login, free, then stop
    scripts/claude-subscription.py [claude arguments...]
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

PROGRAM = "claude-subscription"

EXIT_USAGE = 2
# EX_CONFIG: the launch context is wrong. Nothing was dispatched and nothing
# was billed.
EXIT_REFUSED = 78

# Always sent, to the probe and to the launch alike. `--setting-sources ''`
# means no settings file, project or user, can add a provider, a model, or an
# API key helper behind the login this launcher just verified.
ENFORCED_ARGUMENTS = ("--safe-mode", "--setting-sources", "")

# Arguments a caller may not pass, because each one makes the verified probe
# describe a process other than the one that would run. There is no safe value
# for any of them here, so they are refused rather than overridden.
REJECTED_ARGUMENTS = {
    "--settings": "would load settings this launcher deliberately disabled",
    "--setting-sources": "is pinned to '' so no settings file can re-add a provider",
    "--bare": "changes which configuration and session context the CLI starts in",
    "--add-dir": "widens the launch beyond the directory the probe ran in",
    "--cwd": "moves the launch away from the directory the probe ran in",
    "-C": "moves the launch away from the directory the probe ran in",
    "--chdir": "moves the launch away from the directory the probe ran in",
    "--no-safe-mode": "undoes an environment guarantee the probe was made under",
    # The CLI can also move, resume or create the session somewhere else
    # entirely. A probe run here describes this directory and this login; a
    # session that lands in another worktree, on another machine, or in a
    # detached process is not the one that was verified.
    "--worktree": "runs the session in another worktree, not the one the probe ran in",
    "-w": "runs the session in another worktree, not the one the probe ran in",
    "--teleport": "moves the session to another machine, outside the verified context",
    "--cloud": "runs the session remotely, where this verified local login is not what answers",
    "--environment": "selects a remote environment other than the one the probe verified",
    "--bg": "detaches the session from the process this probe verified",
    "--background": "detaches the session from the process this probe verified",
}

_ROOT = Path(__file__).resolve().parents[1]
_AUTH_MODULE = _ROOT / "src" / "japanese_anki" / "subscription_auth.py"


def refuse(reason: str) -> NoReturn:
    print(f"{PROGRAM}: refusing to launch Claude Code.", file=sys.stderr)
    print(f"  {reason}", file=sys.stderr)
    print("  No model call was made and nothing was billed.", file=sys.stderr)
    print(
        "  This launcher is the only supported way to start Claude for janki"
        " development, planning or review, and it never falls back to API billing.",
        file=sys.stderr,
    )
    print(
        "  Sign in with `claude /login` (Pro or Max), then re-check with:"
        f" scripts/{PROGRAM}.py --check",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_REFUSED)


def usage_error(reason: str) -> NoReturn:
    print(f"{PROGRAM}: {reason}", file=sys.stderr)
    print(f"  usage: scripts/{PROGRAM}.py [--check] [claude arguments...]", file=sys.stderr)
    raise SystemExit(EXIT_USAGE)


def load_subscription_auth() -> Any:
    """Load the shared auth/env facts by path, not by import.

    A fresh clone has no venv and no installed `japanese_anki`, and this
    launcher is exactly what someone runs there.
    """
    if not _AUTH_MODULE.is_file():
        refuse(f"the shared subscription rules are missing: {_AUTH_MODULE}")
    spec = importlib.util.spec_from_file_location("janki_subscription_auth", _AUTH_MODULE)
    if spec is None or spec.loader is None:
        refuse(f"the shared subscription rules could not be loaded: {_AUTH_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen_arguments(arguments: list[str]) -> None:
    for argument in arguments:
        name = argument.split("=", 1)[0]
        why = REJECTED_ARGUMENTS.get(name)
        if why is not None:
            usage_error(f"{name} {why}, so this launch is refused.")


def probe(subscription_auth: Any, base: list[str], *, env: dict[str, str], cwd: str) -> dict:
    """The free subscription check, run as the exact process about to launch."""
    command = [*base, *subscription_auth.AUTH_STATUS_ARGUMENTS]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        refuse(f"the Claude Code subscription check could not run: {exc}")
    if completed.returncode != 0:
        # Its stderr is not repeated: `auth status` names the account, and a
        # refusal that echoed it would put an email into every review report.
        refuse(
            "`claude auth status --json` exited "
            f"{completed.returncode}, so the login could not be verified."
        )
    try:
        payload = subscription_auth.parse_auth_status(completed.stdout)
    except subscription_auth.SubscriptionAuthError as exc:
        refuse(f"the Claude Code authentication status was unreadable: {exc}")
    reason = subscription_auth.subscription_auth_reason(payload)
    if reason is not None:
        refuse(f"this is not a Claude Pro or Max subscription login: {reason}.")
    return payload


def report(payload: dict, executable: str) -> None:
    """The safe summary: only the fields that were checked, never the account."""
    print(f"{PROGRAM}: subscription login verified.")
    print("  auth method:     claude.ai")
    print("  api provider:    firstParty")
    print(f"  subscription:    {payload['subscriptionType']}")
    print("  api key source:  none")
    print(f"  executable:      {executable}")
    print("  enforced:        " + " ".join(repr(part) for part in ENFORCED_ARGUMENTS))


def main(argv: list[str]) -> int:
    subscription_auth = load_subscription_auth()

    arguments = list(argv)
    check_only = "--check" in arguments
    if check_only:
        if arguments != ["--check"]:
            usage_error("--check runs the free probe alone and takes no other arguments.")
        arguments = []
    screen_arguments(arguments)

    env = subscription_auth.sanitized_environment(
        os.environ, subscription_auth.BASE_CONTROLLED_ENVIRONMENT
    )
    found = shutil.which("claude", path=env.get("PATH") or os.defpath)
    if not found:
        refuse(
            "the Claude Code CLI is not on PATH. Install it and sign in with"
            " Claude Pro or Max."
        )
    executable = os.path.abspath(found)
    cwd = os.getcwd()
    base = [executable, *ENFORCED_ARGUMENTS]

    payload = probe(subscription_auth, base, env=env, cwd=cwd)
    if check_only:
        report(payload, executable)
        return 0

    try:
        # exec, not a child: the caller's stdin, exit status and signals are
        # the CLI's own from here, and janki-review.sh's watchdog kills the pid
        # it started rather than an orphaned wrapper.
        os.execve(executable, [*base, *arguments], env)
    except OSError as exc:
        refuse(f"the verified Claude Code CLI could not be started: {exc}")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:  # 128 + SIGINT, the shell's own convention
        raise SystemExit(130) from None
