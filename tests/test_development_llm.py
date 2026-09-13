"""Development providers are tested with local recording CLIs, never a model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/llm.py"
HOSTILE = {
    "ANTHROPIC_API_KEY": "ant-secret-test",
    "OPENAI_API_KEY": "openai-secret-test",
    "CODEX_API_KEY": "codex-secret-test",
    "ANTHROPIC_BASE_URL": "https://bad.invalid",
    "OPENAI_BASE_URL": "https://bad.invalid",
    "CODEX_HOME": "/tmp/wrong-login",
    "CLAUDE_CONFIG_DIR": "/tmp/wrong-claude",
    "PYTHONPATH": "/tmp/wrong-code",
    "UNRECOGNIZED_PROVIDER_TOKEN": "future-secret-test",
}
CLAUDE_AUTH = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "subscriptionType": "max", "apiKeySource": None,
})


def cli(tmp_path, provider, *, auth=None, auth_exit=0, exit_code=0):
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    log = tmp_path / f"{provider}.jsonl"
    path = directory / provider
    if auth is None:
        auth = CLAUDE_AUTH if provider == "claude" else "Logged in using ChatGPT\n"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "probe = args[-3:] == ['auth','status','--json'] or "
        "args[-2:] == ['login','status']\n"
        f"with open({str(log)!r}, 'a') as handle:\n"
        "    handle.write(json.dumps({'args':args,'env':dict(os.environ),"
        "'cwd':os.getcwd(),'stdin':'' if probe else sys.stdin.read()})+'\\n')\n"
        "if probe:\n"
        f"    (sys.stdout if {provider!r} == 'claude' else sys.stderr).write({auth!r})\n"
        f"    raise SystemExit({auth_exit})\n"
        "print('VERDICT: CLEAN')\n"
        f"raise SystemExit({exit_code})\n"
    )
    path.chmod(0o755)
    env = dict(os.environ, **HOSTILE, PATH=f"{directory}{os.pathsep}{os.defpath}")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    return log, env


def run(tmp_path, provider="codex", *, env, extra=(), prompt="Review exact patch.\n"):
    return subprocess.run(
        [sys.executable, str(LAUNCHER), "--provider", provider,
         "--role", "review", "--model",
         "gpt-6-astra" if provider == "codex" else "claude-opus-5",
         "--effort", "max", *extra],
        input=prompt, text=True, capture_output=True, cwd=tmp_path, env=env,
    )


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_subscription_probe_and_dispatch_share_sanitized_context(tmp_path, provider):
    log, env = cli(tmp_path, provider)
    result = run(tmp_path, provider, env=env)
    assert result.returncode == 0, result.stderr
    probe, dispatch = calls(log)
    assert probe["cwd"] == dispatch["cwd"] == str(tmp_path.resolve())
    assert probe["env"] == dispatch["env"]
    assert not (set(HOSTILE) & set(dispatch["env"]))
    assert dispatch["stdin"] == "Review exact patch.\n"
    assert "--model" in dispatch["args"]
    if provider == "codex":
        assert '--ignore-user-config' in dispatch['args']
        for item in (probe, dispatch):
            assert 'forced_login_method="chatgpt"' in item['args']
            assert 'model_provider="openai"' in item['args']
        assert dispatch['args'][dispatch['args'].index('--sandbox') + 1] == 'read-only'
    else:
        for item in (probe, dispatch):
            assert item["args"][:3] == ["--safe-mode", "--setting-sources", ""]
        assert "Edit" not in dispatch['args']


def test_codex_probe_and_dispatch_pin_the_same_credential_store(tmp_path):
    # login status reads user config, while exec --ignore-user-config does not.
    # Pin both so a user-config keyring selection cannot make the probe inspect
    # different credentials from the process that actually calls the model.
    log, env = cli(tmp_path, "codex")
    config = Path(env["HOME"]) / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('cli_auth_credentials_store = "keyring"\n')

    result = run(tmp_path, env=env)

    assert result.returncode == 0, result.stderr
    probe, dispatch = calls(log)
    for call in (probe, dispatch):
        args = call["args"]
        overrides = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
        stores = [item for item in overrides if item.startswith("cli_auth_credentials_store=")]
        assert stores == ['cli_auth_credentials_store="auto"']


@pytest.mark.parametrize("provider,auth", [
    ("codex", "Logged in using an API key: secret-do-not-print"),
    ("codex", "Logged in using ChatGPT\nunknown-extra-status"),
    ("claude", '{"loggedIn":true,"authMethod":"api_key","email":"private-email"}'),
    ("claude", '{"loggedIn":true,"loggedIn":false}'),
])
def test_unverified_auth_refuses_without_dispatch_or_exposing_reply(tmp_path, provider, auth):
    log, env = cli(tmp_path, provider, auth=auth)
    result = run(tmp_path, provider, env=env)
    assert result.returncode == 78
    assert len(calls(log)) == 1
    assert "secret-do-not-print" not in result.stdout + result.stderr
    assert "private-email" not in result.stdout + result.stderr
    assert "No model call" in result.stderr


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_free_check_only_probes_and_actual_cli_exit_is_preserved(tmp_path, provider):
    log, env = cli(tmp_path, provider, exit_code=19)
    checked = run(tmp_path, provider, env=env, extra=("--check",))
    assert checked.returncode == 0, checked.stderr
    assert len(calls(log)) == 1
    assert "verified" in checked.stdout
    dispatched = run(tmp_path, provider, env=env)
    assert dispatched.returncode == 19
    assert len(calls(log)) == 3


@pytest.mark.parametrize("extra", [
    ("--config", 'forced_login_method="api"'),
    ("--settings", "bad.json"), ("--provider", "unknown"),
    ("--model", "latest"), ("--role", "unrestricted"),
])
def test_native_override_and_unknown_provider_are_rejected_before_probe(tmp_path, extra):
    log, env = cli(tmp_path, "codex")
    result = run(tmp_path, env=env, extra=extra)
    assert result.returncode == 2
    assert calls(log) == []


def test_a_model_launch_without_a_role_is_rejected_before_probe(tmp_path):
    log, env = cli(tmp_path, "codex")
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--provider", "codex",
         "--model", "gpt-6-astra", "--effort", "max"],
        input="Implement this change.\n", text=True, capture_output=True,
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert calls(log) == []
    assert "--role" in result.stderr


def test_prompt_file_is_forwarded_exactly_and_native_json_mode_is_explicit(tmp_path):
    log, env = cli(tmp_path, "codex")
    prompt = tmp_path / "task.txt"
    prompt.write_bytes(b"literal $() and `text`\nsecond line\n")
    result = run(tmp_path, env=env, extra=("--prompt-file", str(prompt), "--json"))
    assert result.returncode == 0, result.stderr
    assert calls(log)[-1]["stdin"].encode() == prompt.read_bytes()
    assert "--json" in calls(log)[-1]["args"]


def test_missing_selected_provider_does_not_try_the_other_provider(tmp_path):
    log, env = cli(tmp_path, "claude")
    result = run(tmp_path, "codex", env=env)
    assert result.returncode == 78
    assert calls(log) == []
    assert "codex is not installed" in result.stderr


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_failed_auth_probe_cannot_dispatch_even_with_a_positive_reply(tmp_path, provider):
    log, env = cli(tmp_path, provider, auth_exit=7)
    result = run(tmp_path, provider, env=env)
    assert result.returncode == 78
    assert len(calls(log)) == 1


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_implementation_permissions_and_model_effort_are_explicit(tmp_path, provider):
    log, env = cli(tmp_path, provider)
    result = run(tmp_path, provider, env=env, extra=("--role", "implementation"))
    assert result.returncode == 0, result.stderr
    args = calls(log)[-1]["args"]
    expected = "gpt-6-astra" if provider == "codex" else "claude-opus-5"
    assert args[args.index("--model") + 1] == expected
    if provider == "codex":
        assert args[args.index("--sandbox") + 1] == "workspace-write"
        assert 'model_reasoning_effort="max"' in args
        assert "--ignore-user-config" in args
        assert "hooks" in args
        assert "multi_agent" in args
    else:
        assert "Write" in args and "Edit" in args
        assert args[args.index("--effort") + 1] == "max"


def test_auth_timeout_refuses_without_echoing_captured_secrets(tmp_path):
    from japanese_anki.development_llm import CodexProvider, LaunchRefused, prepare_launch

    _log, env = cli(tmp_path, "codex")

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(command, 30, output=b"private-account-secret")

    with pytest.raises(LaunchRefused, match="could not complete") as refusal:
        prepare_launch(CodexProvider(), None, host=env, cwd=tmp_path, runner=timeout)
    assert "private-account-secret" not in str(refusal.value)


def test_claude_review_exposes_only_reading_tools(tmp_path):
    log, env = cli(tmp_path, "claude")
    result = run(tmp_path, "claude", env=env)
    assert result.returncode == 0, result.stderr
    args = calls(log)[-1]["args"]
    assert args[args.index("--tools") + 1] == "Read,Glob,Grep"
    assert args[args.index("--allowedTools") + 1:] == ["Read", "Glob", "Grep"]


def test_fresh_clone_launcher_loads_its_own_source_without_an_install(tmp_path):
    checkout = tmp_path / "checkout"
    for rel in ("scripts/llm.py", "src/japanese_anki/__init__.py",
                "src/japanese_anki/development_llm.py", "src/japanese_anki/subscription_auth.py"):
        target = checkout / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    log, env = cli(tmp_path, "codex")
    result = subprocess.run(
        [sys.executable, str(checkout / "scripts/llm.py"), "--provider", "codex", "--check"],
        env=env, cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(calls(log)) == 1


def test_third_provider_implements_interface_without_branching_the_launcher(tmp_path):
    from japanese_anki.development_llm import ModelRequest, prepare_launch

    class ThirdProvider:
        name = "third"
        executable = "third-cli"

        def environment(self, host):
            return {"PATH": str(tmp_path), "HOME": host["HOME"]}

        def authentication_arguments(self, cwd):
            return ("auth-check",)

        def verify_auth(self, completed):
            assert completed.stdout == b"subscription"

        def run_arguments(self, request, cwd):
            return ("run", request.model, request.effort)

    executable = tmp_path / "third-cli"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    requests = []

    def runner(command, **kwargs):
        requests.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"subscription", b"")

    plan = prepare_launch(
        ThirdProvider(), ModelRequest("third-model-1", "max", "review"),
        host={"HOME": str(tmp_path)}, cwd=tmp_path, runner=runner,
    )
    assert plan.argv == (str(executable), "run", "third-model-1", "max")
    assert len(requests) == 1
