"""Bootstrap's Git LFS setup, exercised against fake git/make/python."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Every fake records its argv to one shared log, so ordering across commands is
# observable. Extras add just enough behaviour to keep bootstrap.sh moving.
FAKE = """#!{exe}
import json, os, sys
with open(os.environ["BOOTSTRAP_LOG"], "a") as fh:
    fh.write(json.dumps({{"cmd": os.path.basename(sys.argv[0]), "argv": sys.argv[1:]}}) + "\\n")
{extra}
"""
EXTRAS = {
    # Only the version-check heredoc feeds us stdin; reading it anywhere else
    # would block on the caller's terminal.
    "python": 'if sys.argv[1:2] == ["-"]:\n    sys.stdin.read()',
    "git": (
        'if sys.argv[1:3] == ["lfs", "pull"]:\n'
        '    sys.exit(int(os.environ.get("LFS_PULL_STATUS", "0")))'
    ),
    "git-lfs": "",
    "make": "",
    "install-review-hooks.sh": "",
}


def _fake(path: Path, name: str) -> None:
    path.write_text(FAKE.format(exe=sys.executable, extra=EXTRAS[name]))
    path.chmod(0o755)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A disposable repo holding a copy of bootstrap.sh and nothing real."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "bootstrap.sh", tmp_path / "scripts" / "bootstrap.sh")
    _fake(tmp_path / "scripts" / "install-review-hooks.sh", "install-review-hooks.sh")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "activate").write_text("")  # sourced, never used
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name in ("python", "git", "git-lfs", "make"):
        _fake(fakebin / name, name)
    return tmp_path


def _run(tree: Path, **overrides: str):
    log = tree / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{tree / 'fakebin'}:/usr/bin:/bin",
        "PYTHON_BIN": str(tree / "fakebin" / "python"),
        "BOOTSTRAP_LOG": str(log),
        **overrides,
    }
    proc = subprocess.run(
        ["bash", str(tree / "scripts" / "bootstrap.sh")],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, [(call["cmd"], call["argv"]) for call in calls]


def _index(calls, cmd, argv) -> int:
    for i, call in enumerate(calls):
        if call == (cmd, argv):
            return i
    raise AssertionError(f"{cmd} {argv} never ran; calls were {calls}")


def test_installs_local_filters_and_hydrates_every_object_before_gates(tree):
    proc, calls = _run(tree)

    assert proc.returncode == 0, proc.stderr
    filters = _index(calls, "git", ["lfs", "install", "--local", "--skip-repo"])
    installer = _index(calls, "install-review-hooks.sh", [])
    # Empty include and exclude: every tracked object, not just the audio tree.
    hydrate = _index(calls, "git", ["lfs", "pull", "--include=", "--exclude="])
    gates = _index(calls, "make", ["gates"])
    assert filters < installer, (
        f"the installer's hooks need the local filters in place; calls were {calls}"
    )
    assert installer < hydrate, (
        f"hydration must run through the installed hooks; calls were {calls}"
    )
    assert hydrate < gates, (
        f"gates must read hydrated objects, not pointer files; calls were {calls}"
    )


def test_failed_hydration_aborts_before_gates(tree):
    proc, calls = _run(tree, LFS_PULL_STATUS="1")

    assert proc.returncode != 0, "a failed lfs pull must fail bootstrap"
    assert ("make", ["gates"]) not in calls, f"gates ran on pointer files; calls were {calls}"
