"""The `voicevox` target, exercised rather than read.

It exists for one reason: `janki audio` voices a hundred clips and a run that
discovers the engine is down on the forty-first has written forty files and
half a ledger. So the target's whole job is to fail *before* that, fast, with a
reason.

It has now been broken three commits running, each time by a fix to the
previous break, and each break was the same shape — a docker step whose failure
does not stop the recipe, so the run falls through to sixty seconds of polling
and blames a container that was never started. That kept happening because
nothing here executed it: `tests/test_build_only_new.py` reads the `gates:`
recipe as text, which is enough for "does the gate pass --output" and cannot
see control flow at all.

These run the real recipe against a stub `docker` and a stub `curl` on PATH,
so what is asserted is what the shell does. `sleep` is stubbed to nothing, so
a test that would poll for a minute takes milliseconds — which also means a
regression to "polls the full loop" shows up as the wrong *message*, not as a
slow test that someone eventually deletes.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: A stub `docker` whose behaviour each test picks with environment variables,
#: so one script covers every branch and the tests read as a table.
DOCKER_STUB = """#!/bin/sh
case "$1" in
  info)
    exit ${STUB_INFO_RC:-0} ;;
  ps)
    case "$*" in
      *-aq*) [ -n "$STUB_PS_AQ_RC" ] && exit "$STUB_PS_AQ_RC"
             [ -n "$STUB_EXISTS" ] && echo abc123
             exit 0 ;;
      *)     [ -n "$STUB_PS_Q_RC" ] && exit "$STUB_PS_Q_RC"
             [ -n "$STUB_RUNNING" ] && echo abc123
             exit 0 ;;
    esac ;;
  start)   exit ${STUB_START_RC:-0} ;;
  restart) echo "STUB-FORBIDDEN: docker restart" >&2; exit 0 ;;
  run)     exit ${STUB_RUN_RC:-0} ;;
  rm)      echo "STUB-FORBIDDEN: docker rm" >&2; exit 0 ;;
  *)       exit 0 ;;
esac
"""

CURL_STUB = """#!/bin/sh
exit ${STUB_CURL_RC:-7}
"""

COUNTING_CURL_STUB = """#!/bin/sh
echo x >> "$STUB_COUNT_FILE"
exit ${STUB_CURL_RC:-7}
"""


@dataclass
class Run:
    code: int
    out: str
    err: str

    @property
    def text(self) -> str:
        return self.out + self.err


def _bin(tmp_path: Path, **scripts: str) -> Path:
    """A PATH directory holding the stubs, plus what make itself needs."""
    where = tmp_path / "bin"
    where.mkdir(exist_ok=True)
    for name, body in scripts.items():
        script = where / name
        script.write_text(body, encoding="utf-8")
        script.chmod(0o755)
    # Everything the recipe and make invoke, symlinked rather than copied so
    # the stubs are the only thing this PATH changes.
    for tool in ("make", "sh", "echo", "printf", "grep", "sed", "sort", "cat", "expr"):
        found = _which(tool)
        if found and not (where / tool).exists():
            (where / tool).symlink_to(found)
    return where


def _which(tool: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _make(target: str, path: Path, **env: str) -> Run:
    """Run one make target with `path` as the *whole* PATH."""
    result = subprocess.run(
        ["make", "-C", str(PROJECT_ROOT), target],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(path), **env},
        check=False,
    )
    return Run(result.returncode, result.stdout, result.stderr)


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    # `sleep` does nothing: the loop's timing is not what is under test, and a
    # test that really waited 60 seconds would be deleted rather than fixed.
    return _bin(
        tmp_path,
        docker=DOCKER_STUB,
        curl=CURL_STUB,
        sleep="#!/bin/sh\nexit 0\n",
    )


def test_a_dead_daemon_is_named_rather_than_waited_out(stubs: Path) -> None:
    """`docker info` failing means every later docker call fails too.

    The original bug: `docker ps -aq` also fails, prints nothing, and an
    unchecked read of that takes the *create* branch — announcing a container
    it never ran, then polling for a minute.
    """
    run = _make("voicevox", stubs, STUB_INFO_RC="1")

    assert run.code != 0
    assert "daemon is not running" in run.text
    assert "waiting for the engine" not in run.text, "it never got that far"


def test_a_docker_ps_that_fails_before_the_loop_stops_the_run(stubs: Path) -> None:
    """`docker info` succeeding does not make `docker ps` succeed.

    A switched context, a changed DOCKER_HOST, an API-version mismatch. An
    unchecked `ps` here reads as "no such container" and creates a second one.
    """
    run = _make("voicevox", stubs, STUB_PS_AQ_RC="1")

    assert run.code != 0
    assert "'docker ps' failed" in run.text
    assert "running voicevox" not in run.text, "it must not claim to have started one"


def test_a_docker_ps_that_fails_inside_the_loop_stops_the_run(stubs: Path) -> None:
    """The regression that shipped, and the reason this file exists.

    `if ok="$(docker ps ...)" && [ -z "$ok" ]` consults the exit status and
    short-circuits false on failure, so the branch never fires and the loop
    runs its full sixty polls before blaming a container. It was checked and
    not acted on, which reads identical in a diff.
    """
    run = _make("voicevox", stubs, STUB_EXISTS="1", STUB_PS_Q_RC="1")

    assert run.code != 0
    assert "'docker ps' stopped answering" in run.text
    assert "60 tries" not in run.text, "it must not poll the loop out"


def test_a_container_that_exits_while_starting_is_noticed(stubs: Path) -> None:
    """A dead container answers `docker ps -q` with nothing, and the loop is
    the only place that can see it happen."""
    run = _make("voicevox", stubs, STUB_EXISTS="1")

    assert run.code != 0
    assert "exited while starting up" in run.text
    assert "60 tries" not in run.text


def test_a_container_that_never_answers_times_out_saying_so(stubs: Path) -> None:
    """The one case where polling the loop out *is* the right answer."""
    run = _make("voicevox", stubs, STUB_EXISTS="1", STUB_RUNNING="1")

    assert run.code != 0
    assert "60 tries and the engine never answered" in run.text


def test_the_loop_polls_exactly_sixty_times(tmp_path: Path) -> None:
    """Sixty, not fifty-nine and not sixty-one.

    The count was `seq 1 60` and is now a shell counter, which is the kind of
    rewrite that quietly loses an iteration. One curl happens before the loop,
    so the total is 61.
    """
    counter = tmp_path / "curls"
    stubs = _bin(
        tmp_path,
        docker=DOCKER_STUB,
        curl=COUNTING_CURL_STUB,
        sleep="#!/bin/sh\nexit 0\n",
    )

    _make(
        "voicevox",
        stubs,
        STUB_EXISTS="1",
        STUB_RUNNING="1",
        STUB_COUNT_FILE=str(counter),
    )

    assert counter.read_text().count("x") == 61, "1 pre-check + 60 in the loop"


def test_the_engine_already_answering_starts_nothing(stubs: Path) -> None:
    """Adopting a running engine — the VOICEVOX app, or a container someone
    else started — is the common case and must not touch docker at all."""
    run = _make("voicevox", stubs, STUB_CURL_RC="0")

    assert run.code == 0
    assert "already answering" in run.text
    assert "STUB-FORBIDDEN" not in run.text


def test_a_running_container_is_started_not_restarted(stubs: Path) -> None:
    """`docker start` is a no-op on a running container; `docker restart` tears
    down the speaker model VOICEVOX loads on boot.

    This branch is reached whenever curl failed, and "curl failed" includes an
    engine that is up and still loading — so `restart` fixes the rare wedged
    case by breaking the common slow one. The stub screams if it is called.
    """
    run = _make("voicevox", stubs, STUB_EXISTS="1", STUB_RUNNING="1")

    assert "STUB-FORBIDDEN: docker restart" not in run.text


def test_a_failed_create_does_not_delete_a_container(stubs: Path) -> None:
    """`docker run` fails with a name conflict precisely when the container
    exists — so a cleanup `docker rm -f` on that path destroys the engine
    another `make audio` is using. The stub screams if it is called."""
    run = _make("voicevox", stubs, STUB_RUN_RC="125")

    assert run.code != 0
    assert "STUB-FORBIDDEN: docker rm" not in run.text
    assert "could not start the container" in run.text


def test_a_missing_curl_is_named_rather_than_polled_out(tmp_path: Path) -> None:
    """Without curl the recipe cannot tell whether the engine is up, and every
    probe fails — which is sixty polls and a message blaming docker."""
    stubs = _bin(tmp_path, docker=DOCKER_STUB, sleep="#!/bin/sh\nexit 0\n")

    run = _make("voicevox", stubs)

    assert run.code != 0
    assert "curl is not installed" in run.text
    assert "60 tries" not in run.text


def test_help_lists_every_target_and_expands_the_container_name(stubs: Path) -> None:
    """`help` is the default goal, so it is the only discovery surface.

    It reads the Makefile as text, which is why the container name needs
    expanding by hand — and why a target that grows a `##` line but no entry
    here would be invisible.
    """
    run = _make("help", stubs)

    assert run.code == 0
    assert "$(VOICEVOX_CONTAINER)" not in run.out, "the name must be expanded"
    assert "janki-voicevox" in run.out

    lines = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    targets = {
        line.split(":", 1)[0]
        for line in lines
        # A rule, not a variable: `NAME := value` and `NAME ?= value` both have
        # a colon-ish prefix, and `$(ROOT)` would otherwise read as a target.
        if line
        and not line.startswith(("\t", " ", "#", ".", "\n"))
        and ":" in line.split("=", 1)[0]
        and not line.split(":", 1)[0].strip().endswith(("?", ":"))
        and " " not in line.split(":", 1)[0]
    }
    missing = sorted(name for name in targets if name not in run.out)
    assert not missing, f"help does not mention: {missing}"
