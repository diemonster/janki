"""The suite must not spend the owner's Claude subscription allowance.

Nothing here starts a process, and nothing here needs the machine to have
Claude installed. The guard's idea of "the installed CLI" and the real `Popen`
constructor are both replaced with synthetic stand-ins, so a blocked call is
proved by the sentinel never running and an allowed call by it running exactly
once. Watching a real CLI decline would be the bug this file exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import conftest


class Sentinel:
    """Stands in for the real `Popen.__init__`, and records nothing else."""

    def __init__(self) -> None:
        self.spawned: list[Any] = []

    def __call__(self, _self: Any, args: Any = (), *_rest: Any, **_kwargs: Any) -> None:
        self.spawned.append(args)


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A synthetic Claude installation, shaped like a real one.

    Real installs are version-named and reached through a symlink whose target
    is not necessarily called `claude` at all, so the guard cannot work from
    the name it was handed.
    """
    binary = tmp_path / "install" / "versions" / "2.1.246" / "cli"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher = tmp_path / "install" / "bin" / "claude"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(binary)

    monkeypatch.setattr(
        conftest, "INSTALLED_CLAUDE", frozenset({launcher.resolve(), binary.resolve()})
    )
    monkeypatch.setenv("PATH", str(launcher.parent))
    return {"binary": binary, "launcher": launcher}


@pytest.fixture
def sentinel(monkeypatch: pytest.MonkeyPatch) -> Sentinel:
    stand_in = Sentinel()
    monkeypatch.setattr(conftest, "_REAL_POPEN_INIT", stand_in)
    return stand_in


def test_a_bare_name_is_resolved_through_PATH_before_it_is_judged(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    """`claude` with no directory in it is the ordinary way the CLI is spawned.

    Nothing about the string says where it goes; only PATH does.
    """
    with pytest.raises(AssertionError):
        subprocess.Popen(["claude", "-p", "hello"])

    assert sentinel.spawned == []


def test_the_installed_symlink_is_refused(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    with pytest.raises(AssertionError):
        subprocess.Popen([str(install["launcher"]), "-p", "hello"])

    assert sentinel.spawned == []


def test_the_resolved_version_named_binary_is_refused(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    """Its name is `cli`. Only comparing against the real target catches it."""
    with pytest.raises(AssertionError):
        subprocess.Popen([str(install["binary"]), "-p", "hello"])

    assert sentinel.spawned == []


def test_an_alias_under_the_temp_root_is_still_the_installed_cli(
    install: dict[str, Path], sentinel: Sentinel, tmp_path: Path
) -> None:
    """Living where a fixture's files live is not what makes a program a fake."""
    alias = tmp_path / "shim" / "claude"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(install["launcher"])

    with pytest.raises(AssertionError):
        subprocess.Popen([str(alias), "-p", "hello"])

    assert sentinel.spawned == []


def test_the_executable_keyword_is_judged_too(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    with pytest.raises(AssertionError):
        subprocess.Popen(["anything"], executable=str(install["launcher"]))

    assert sentinel.spawned == []


def test_a_fixtures_own_fake_claude_still_runs(
    install: dict[str, Path], sentinel: Sentinel, tmp_path: Path
) -> None:
    """The launcher, auth-probe and provider suites all build one of these."""
    fake = tmp_path / "fake" / "claude"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")

    subprocess.Popen([str(fake), "--version"])

    assert sentinel.spawned == [[str(fake), "--version"]]


def test_a_bare_name_resolving_to_a_fake_on_PATH_still_runs(
    sentinel: Sentinel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fakebin" / "claude"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(conftest, "INSTALLED_CLAUDE", frozenset())
    monkeypatch.setenv("PATH", str(fake.parent))

    subprocess.Popen(["claude", "--version"])

    assert sentinel.spawned == [["claude", "--version"]]


def test_unrelated_subprocesses_are_left_alone(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    subprocess.Popen([chrome, "--headless"])
    subprocess.Popen(["/usr/bin/git", "status"])

    assert sentinel.spawned == [[chrome, "--headless"], ["/usr/bin/git", "status"]]


def test_the_guard_holds_the_seam_a_function_default_captured(
    install: dict[str, Path], sentinel: Sentinel
) -> None:
    """`revision_provider.dispatch` binds `spawn=subprocess.Popen` at def time.

    Rebinding the module attribute would leave that default pointing at the
    real class, so the guard patches the constructor the class itself uses.
    """
    captured_at_def_time = subprocess.Popen

    with pytest.raises(AssertionError):
        captured_at_def_time([str(install["launcher"]), "-p", "hello"])

    assert sentinel.spawned == []


def test_there_is_no_opt_out(install: dict[str, Path]) -> None:
    """No test in this suite is authorized to spend the owner's allowance.

    A marker that lifted the guard would be the only thing standing between a
    typo and a billed content call, so there is not one.
    """
    assert "allow_claude_cli" not in Path(conftest.__file__).read_text(
        encoding="utf-8"
    )
