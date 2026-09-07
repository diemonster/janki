"""The mandatory development/review launcher: subscription login or nothing.

Every assertion here is made against a *fake* `claude` on a controlled PATH.
Nothing in this module may reach a provider: a test that contacted one would be
the exact mistake the launcher exists to prevent, so the fake records each
invocation's argv, environment, cwd, pid and stdin onto a log whose path is
baked into its source. It cannot read that path out of the environment, because
the environment is precisely what the launcher scrubs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "claude-subscription.py"

# The launcher's contract, written out here rather than imported from it: a
# test that reads the allowlist out of the module under test would pass against
# any allowlist at all.
ALLOWED_HOST_ENVIRONMENT = frozenset(
    {"HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "TMPDIR", "USER", "XDG_CONFIG_HOME"}
)
ENFORCED_ENVIRONMENT = {
    "CLAUDE_CODE_SAFE_MODE": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
ENFORCED_ARGUMENTS = ["--safe-mode", "--setting-sources", ""]
AUTH_STATUS = ["auth", "status", "--json"]

# macOS adds these to a child regardless of the environment handed to exec, and
# a Python child coerces its own locale. Neither selects a provider, an account,
# or a settings file, and neither is something the launcher passed on.
PLATFORM_INJECTED = frozenset({"__CF_USER_TEXT_ENCODING", "__PYVENV_LAUNCHER__"})

EXIT_USAGE = 2
EXIT_REFUSED = 78

# What the CLI reports for a healthy Max login. The last two keys are the ones
# that must never reach a terminal, a review report, or a commit.
SUBSCRIPTION = {
    "loggedIn": True,
    "authMethod": "claude.ai",
    "apiProvider": "firstParty",
    "subscriptionType": "max",
    "apiKeySource": None,
    "email": "owner@example.test",
    "organizationId": "org-secret-0451",
}
ACCOUNT_DETAILS = ("owner@example.test", "org-secret-0451")

# An inherited environment that would silently bill the Console: the exact
# shape that caused this launcher to exist.
HOSTILE = {
    "ANTHROPIC_API_KEY": "sk-ant-test-DEADBEEF",
    "ANTHROPIC_AUTH_TOKEN": "oat-test-DEADBEEF",
    "ANTHROPIC_BASE_URL": "https://gateway.invalid",
    "ANTHROPIC_MODEL": "claude-not-ours",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLAUDE_CONFIG_DIR": "/tmp/somebody-elses-config",
    "AWS_PROFILE": "console-billing",
}

FAKE_CLAUDE = """#!{python}
import json, os, sys

argv = sys.argv[1:]
with open({log!r}, "a", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            {{
                "argv": argv,
                "env": dict(os.environ),
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "stdin": sys.stdin.read(),
            }}
        )
        + "\\n"
    )

if argv[-3:] == ["auth", "status", "--json"]:
    sys.stdout.write({status!r})
    sys.stderr.write("status diagnostics for {account!s}\\n")
    raise SystemExit({status_exit})

sys.stdout.write("dispatched\\n")
raise SystemExit({exit_code})
"""


@dataclass(frozen=True)
class Fixture:
    launcher: Path
    log: Path
    cwd: Path
    env: dict[str, str]


def fake_claude(
    path: Path,
    *,
    log: Path,
    status: str,
    status_exit: int = 0,
    exit_code: int = 0,
) -> Path:
    path.write_text(
        FAKE_CLAUDE.format(
            python=sys.executable,
            log=str(log),
            status=status,
            status_exit=status_exit,
            exit_code=exit_code,
            account=SUBSCRIPTION["email"],
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def build(
    tmp_path: Path,
    *,
    status: object = SUBSCRIPTION,
    status_exit: int = 0,
    exit_code: int = 0,
    install_claude: bool = True,
) -> Fixture:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "claude-calls.jsonl"
    if install_claude:
        fake_claude(
            bin_dir / "claude",
            log=log,
            status=status if isinstance(status, str) else json.dumps(status),
            status_exit=status_exit,
            exit_code=exit_code,
        )
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    return Fixture(
        launcher=LAUNCHER,
        log=log,
        cwd=work,
        env={
            "HOME": str(home),
            "PATH": str(bin_dir),
            "LANG": "en_US.UTF-8",
            "USER": "tester",
            **HOSTILE,
        },
    )


def run_launcher(fixture: Fixture, *arguments: str, input: str = ""):
    return subprocess.run(
        [sys.executable, str(fixture.launcher), *arguments],
        cwd=str(fixture.cwd),
        env=fixture.env,
        input=input,
        text=True,
        capture_output=True,
        check=False,
    )


def records(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def kind(call: dict) -> str:
    return "probe" if call["argv"][3:] == AUTH_STATUS else "dispatch"


def output(result) -> str:
    return result.stdout + result.stderr


def test_hostile_environment_is_stripped_from_both_the_probe_and_the_dispatch(
    tmp_path: Path,
) -> None:
    fixture = build(tmp_path)

    result = run_launcher(fixture, "-p", "review this", "--model", "claude-opus-5")

    assert result.returncode == 0, result.stderr
    invocations = records(fixture.log)
    assert [kind(call) for call in invocations] == ["probe", "dispatch"]
    for call in invocations:
        environment = call["env"]
        for name in HOSTILE:
            assert name not in environment, f"{name} reached the {kind(call)}"
        allowed = ALLOWED_HOST_ENVIRONMENT | set(ENFORCED_ENVIRONMENT) | PLATFORM_INJECTED
        assert set(environment) <= allowed
        for name, value in ENFORCED_ENVIRONMENT.items():
            assert environment[name] == value
        assert environment["PATH"] == fixture.env["PATH"]
        assert environment["HOME"] == fixture.env["HOME"]
        # The probe has to describe the process the dispatch will be: same
        # environment, same working directory, same enforced settings sources.
        assert Path(call["cwd"]).resolve() == fixture.cwd.resolve()
        assert call["argv"][:3] == ENFORCED_ARGUMENTS


def test_an_unsafe_inherited_credential_is_ignored_rather_than_refused(tmp_path: Path) -> None:
    """The launcher scrubs what it passes on; it never scolds, unsets, or edits."""
    fixture = build(tmp_path)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == 0, result.stderr
    assert [kind(call) for call in records(fixture.log)] == ["probe", "dispatch"]
    assert "ANTHROPIC_API_KEY" not in output(result)


@pytest.mark.parametrize(
    ("changes", "removed"),
    [
        ({"loggedIn": False}, None),
        ({"loggedIn": "yes"}, None),
        ({}, "loggedIn"),
        ({"authMethod": "apiKey"}, None),
        ({"apiProvider": "bedrock"}, None),
        ({"subscriptionType": None}, None),
        ({"subscriptionType": "team"}, None),
        ({"apiKeySource": "ANTHROPIC_API_KEY"}, None),
        ({"apiKeySource": "/Users/owner/.anthropic/key"}, None),
        # Readable JSON whose subscription field is not text. Refusing is the
        # promised answer; testing an unhashable value for membership in a set
        # is a TypeError, which is a crash rather than a refusal.
        ({"subscriptionType": ["max"]}, None),
        ({"subscriptionType": {"plan": "max"}}, None),
    ],
)
def test_a_non_subscription_login_never_reaches_a_dispatch(
    tmp_path: Path, changes: dict, removed: str | None
) -> None:
    status = dict(SUBSCRIPTION, **changes)
    if removed is not None:
        del status[removed]
    fixture = build(tmp_path, status=status)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == EXIT_REFUSED
    assert [kind(call) for call in records(fixture.log)] == ["probe"]
    for secret in (*ACCOUNT_DETAILS, HOSTILE["ANTHROPIC_API_KEY"]):
        assert secret not in output(result)


def test_an_omitted_api_key_source_is_still_a_subscription(tmp_path: Path) -> None:
    status = dict(SUBSCRIPTION)
    del status["apiKeySource"]
    fixture = build(tmp_path, status=status)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == 0, result.stderr
    assert [kind(call) for call in records(fixture.log)] == ["probe", "dispatch"]


@pytest.mark.parametrize(
    "body",
    [
        "",
        "not json at all\n",
        "[]\n",
        '"claude.ai"\n',
        '{"loggedIn": true, "loggedIn": false}\n',
        '{"loggedIn": NaN}\n',
        '{"loggedIn": true} trailing\n',
    ],
)
def test_a_malformed_auth_status_never_reaches_a_dispatch(tmp_path: Path, body: str) -> None:
    fixture = build(tmp_path, status=body)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == EXIT_REFUSED
    assert [kind(call) for call in records(fixture.log)] == ["probe"]
    assert "claude-subscription" in result.stderr


def test_an_auth_probe_that_fails_never_reaches_a_dispatch(tmp_path: Path) -> None:
    fixture = build(tmp_path, status_exit=3)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == EXIT_REFUSED
    assert [kind(call) for call in records(fixture.log)] == ["probe"]
    # The CLI's own diagnostics may name the account; the refusal must not.
    for secret in ACCOUNT_DETAILS:
        assert secret not in output(result)


@pytest.mark.parametrize(
    "argument",
    [
        "--settings",
        "--settings=/tmp/other.json",
        "--setting-sources",
        "--setting-sources=user,project",
        "--bare",
        "--add-dir",
        "--cwd",
        "-C",
        "--chdir",
        "--no-safe-mode",
        # Each of these moves, resumes, or creates the session somewhere other
        # than the directory and login this probe just verified.
        "--worktree",
        "--worktree=review",
        "-w",
        "--teleport",
        "--cloud",
        "--environment",
        "--environment=some-remote",
        "--bg",
        "--background",
    ],
)
def test_a_settings_or_cwd_override_is_refused_before_any_claude_runs(
    tmp_path: Path, argument: str
) -> None:
    fixture = build(tmp_path)

    result = run_launcher(fixture, "-p", "review this", argument, "value")

    assert result.returncode == EXIT_USAGE
    assert records(fixture.log) == [], "nothing may run once the launch context is in doubt"
    assert argument.split("=", 1)[0] in result.stderr


def test_ordinary_coding_arguments_stdin_and_exit_status_are_preserved(tmp_path: Path) -> None:
    fixture = build(tmp_path, exit_code=7)
    arguments = [
        "-p",
        "Review the changes in the git range `a..b`.",
        "--model",
        "claude-opus-5",
        "--effort",
        "max",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read",
        "Glob",
        "Bash(git *)",
    ]

    result = run_launcher(fixture, *arguments, input="prompt on stdin\n")

    assert result.returncode == 7, "the caller's exit status is the CLI's own"
    dispatch = [call for call in records(fixture.log) if kind(call) == "dispatch"]
    assert len(dispatch) == 1
    assert dispatch[0]["argv"] == ENFORCED_ARGUMENTS + arguments
    assert dispatch[0]["stdin"] == "prompt on stdin\n", "the probe must not eat the prompt"


def test_the_dispatch_replaces_the_launcher_so_a_watchdog_can_kill_it(tmp_path: Path) -> None:
    """`scripts/janki-review.sh` kills the pid it started; exec makes that the CLI."""
    fixture = build(tmp_path)

    process = subprocess.Popen(
        [sys.executable, str(fixture.launcher), "-p", "review this"],
        cwd=str(fixture.cwd),
        env=fixture.env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.communicate(timeout=60)

    probe, dispatch = records(fixture.log)
    assert dispatch["pid"] == process.pid
    assert probe["pid"] != process.pid


def test_check_runs_only_the_free_probe_and_prints_no_account_details(tmp_path: Path) -> None:
    fixture = build(tmp_path)

    result = run_launcher(fixture, "--check")

    assert result.returncode == 0, result.stderr
    assert [kind(call) for call in records(fixture.log)] == ["probe"]
    assert "claude.ai" in result.stdout
    assert "firstParty" in result.stdout
    assert "max" in result.stdout
    for secret in (*ACCOUNT_DETAILS, HOSTILE["ANTHROPIC_API_KEY"]):
        assert secret not in output(result)


def test_check_refuses_to_share_the_command_line_with_a_prompt(tmp_path: Path) -> None:
    fixture = build(tmp_path)

    result = run_launcher(fixture, "--check", "-p", "review this")

    assert result.returncode == EXIT_USAGE
    assert records(fixture.log) == []


def test_a_missing_claude_refuses_instead_of_falling_back(tmp_path: Path) -> None:
    fixture = build(tmp_path, install_claude=False)

    result = run_launcher(fixture, "-p", "review this")

    assert result.returncode == EXIT_REFUSED
    assert records(fixture.log) == []
    assert "claude-subscription" in result.stderr
