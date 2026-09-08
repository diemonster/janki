"""Suite-wide guards.

Three rules, all about the machine rather than the code: **no test touches the
developer's real Anki collection**, **no test opens a billed API client**, and
**no test runs the Claude CLI installed on this machine.**
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import claude_client, collection, prompts, status


@pytest.fixture(scope="session")
def _empty_anki_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One directory for the whole run.

    `mktemp` in a function-scoped fixture made a new numbered directory per
    test — 1500+ of them — and pytest parses every existing suffix on each
    call, so the guard against wasted I/O was itself quadratic in the number of
    tests. The directory is empty and never written to.
    """
    return tmp_path_factory.mktemp("no-anki")


@pytest.fixture(autouse=True)
def _no_real_anki_collection(
    _empty_anki_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point Anki discovery at an empty directory for every test.

    `janki status` looks for a collection when `janki.toml` names none, and
    nearly every status test writes a `janki.toml` with no `[anki]` section. On
    a machine with exactly one Anki profile — the ordinary case — that made the
    suite copy the real `collection.anki2` and read it: measured at **32 reads,
    219 MB copied, for one test file**. Worse than slow, it made unrelated
    tests' stderr depend on how many profiles the machine happens to have: this
    one has two, so discovery returned "several profiles, name one" and nothing
    was read at all, and the leak stayed invisible here while costing anyone
    with a single profile a fifth of a gigabyte per file.

    An empty directory rather than `None`, so `find_profiles` runs its real code
    path and returns `{}` honestly. Tests that mean to exercise discovery pass
    their own root to `find_profiles`, or monkeypatch `status.find_profiles`,
    and are unaffected by this.
    """
    # Both namespaces: `status` imports the name into its own globals, so
    # patching only `collection` left `resolve_collection`'s "not found under
    # …" message reading the developer's real Anki path — machine-dependent
    # output inside the suite.
    monkeypatch.setattr(collection, "default_anki_root", lambda: _empty_anki_root)
    monkeypatch.setattr(status, "default_anki_root", lambda: _empty_anki_root)
    return _empty_anki_root


@pytest.fixture(autouse=True)
def _no_billed_client(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Refuse to build a real Anthropic client.

    `build_client` is the only path that opens a billed connection, so a test
    that fakes `parse_call` or injects a client never reaches this. What it
    catches is the gap that opens when a *default* moves: the enrichment
    provider default changed from codex to anthropic, and a helper that
    patched only `codex_client.parse_call` silently began routing twelve tests
    at the live API. That does not fail — it bills, and it blocks.

    A convention cannot catch that, because the test that breaks is one nobody
    edited. This can: the next default change turns twelve silent live calls
    into twelve loud errors naming the fixture that needs updating.

    Tests that legitimately construct a client opt out with
    ``@pytest.mark.allow_build_client``.
    """
    if request.node.get_closest_marker("allow_build_client"):
        return

    class RefuseBilledTransport:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(
                "This test reached the Anthropic transport through "
                f"client.{name}. Patch the parse_call your code path actually "
                "uses — both providers if a config default decides it — or "
                "mark the test @pytest.mark.allow_build_client if it means to."
            )

    def inert_client(*_args: object, **_kwargs: object) -> object:
        return RefuseBilledTransport()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-never-send")
    monkeypatch.setattr(claude_client, "build_client", inert_client)
    # Keep the public preflight body in every journaled call path. Its bound
    # test-only key reaches only the inert transport above, so a forgotten
    # parse_call fake fails loudly without constructing or contacting a
    # provider.


REPO_ROOT = Path(__file__).resolve().parents[1]


# --- no live Claude CLI -------------------------------------------------------
#
# The API guard above catches a billed *API* client. It cannot see the other
# paid path: the subscription transport spends the owner's Claude Pro/Max
# allowance by spawning the `claude` binary installed on this machine, and it
# reaches that binary through `subprocess`, not through `claude_client`.
#
# That path is now a *default*. `[ai] extract_provider` defaults to
# `claude-code`, so an extraction fixture that fakes `claude_client.parse_call`
# and says nothing about the provider no longer fakes the path its code takes —
# it dispatches a real content call. This is the same shape of failure the API
# guard exists for, one transport along, and the same argument applies: the test
# that breaks when a default moves is one nobody edited, so a convention cannot
# catch it and this can.

#: The real constructor, captured once at import so a guarded test cannot lose
#: it. Patching `subprocess.Popen.__init__` rather than `subprocess.Popen` is
#: deliberate: `application/revision_provider.py` binds `spawn=subprocess.Popen`
#: as a *function default*, which is evaluated at def time, so rebinding the
#: module attribute leaves that dispatch seam pointing at the real class.
_REAL_POPEN_INIT = subprocess.Popen.__init__


def _installed_claude() -> frozenset[Path]:
    """This machine's real `claude`, by every path that reaches it.

    Both the name on PATH and what it resolves to: an install is version-named
    and reached through a symlink whose target may be called something else
    entirely, so neither path alone identifies it.
    """
    found = shutil.which("claude")
    if not found:
        return frozenset()
    launcher = Path(found)
    with contextlib.suppress(OSError):  # resolve() is total on this platform
        return frozenset({launcher, launcher.resolve()})
    return frozenset({launcher})


#: Resolved once, at import. Tests that exercise the guard replace it with a
#: synthetic installation so the suite behaves the same on a machine that has
#: the CLI and one that does not.
INSTALLED_CLAUDE = _installed_claude()


def _programs(args: Any, executable: Any) -> Iterator[str]:
    """Every string this `Popen` call could name a program with."""
    if executable is not None:
        yield os.fsdecode(executable)
    if isinstance(args, str | bytes | os.PathLike):
        text = os.fsdecode(args)
        words = text.split()
        yield words[0] if words else text
        return
    try:
        first = next(iter(args))
    except (TypeError, StopIteration):
        return
    yield os.fsdecode(first)


def names_the_installed_claude_cli(
    args: Any, executable: Any = None, env: Any = None
) -> str | None:
    """The program this call would run, when that program is the real CLI.

    The question is not what the program is *called* — a bare `claude` says
    nothing about where it goes, an install's real binary is often not called
    `claude` at all, and a fixture's fake is called exactly that. It is whether
    the file this call would execute is the file `shutil.which` found. So
    resolve the call the way the OS will: a bare name through PATH, then
    symlinks, and compare against the installation.
    """
    path = None
    if env is not None:
        try:
            path = env.get("PATH")
        except AttributeError:
            path = None
    if path is None:
        path = os.environ.get("PATH")
    for program in _programs(args, executable):
        if os.path.dirname(program):
            target: str | None = program
        else:
            target = shutil.which(program, path=path)
            if target is None:
                # Unresolvable here, but PATH at exec time is not this PATH.
                # A bare `claude` that proves to be nothing is still refused;
                # anything else would fail with FileNotFoundError regardless.
                if Path(program).name == "claude":
                    return program
                continue
        candidates = {Path(target)}
        with contextlib.suppress(OSError):  # resolve() is total on this platform
            candidates.add(Path(target).resolve())
        if candidates & INSTALLED_CLAUDE:
            return program
    return None


@pytest.fixture(autouse=True)
def _no_live_claude_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse to spawn the Claude CLI this machine has installed.

    A test that injects a fake runner or a fake `spawn` never reaches here, and
    neither does one whose fixture builds its own `claude`. What this catches
    is a real dispatch — including the refusal paths, which must be exercised
    against a fake rather than by starting the CLI to watch it decline.

    There is no opt-out marker. No test in this suite is authorized to spend
    the owner's subscription allowance, so an escape hatch here would be the
    only thing between a typo and a billed content call.
    """

    def guarded_init(self: Any, args: Any = (), *rest: Any, **kwargs: Any) -> None:
        program = names_the_installed_claude_cli(
            args, kwargs.get("executable"), kwargs.get("env")
        )
        if program is not None:
            raise AssertionError(
                f"This test spawned the installed Claude CLI ({program}). That "
                "spends the owner's subscription allowance on a real content "
                "call. The subscription transport is the *default* now, so a "
                "fixture that fakes only claude_client.parse_call no longer "
                "covers the path its code takes: set [ai] extract_provider "
                "(or revise_provider) to 'anthropic-api' in the fixture's "
                "janki.toml if this test means to exercise the API path, or "
                "inject the fake transport the subscription path takes — a "
                "runner= (FakeClaudeRunner) or spawn= — if it means to "
                "exercise that one."
            )
        _REAL_POPEN_INIT(self, args, *rest, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)


def seed_prompts(root: Path) -> Path:
    """Copy this checkout's real `prompts/` into a temp project.

    The repository's own files, not stubs. A temp project that invented its
    own prompt text would let a CLI test pass while the shipped template said
    something else entirely — and these files are the deliverable now, so the
    suite should exercise them. Cheap enough to copy the small Markdown set per
    project.
    """
    target = Path(root) / prompts.DIRECTORY
    target.mkdir(parents=True, exist_ok=True)
    for source in (REPO_ROOT / prompts.DIRECTORY).glob("*.md"):
        shutil.copy2(source, target / source.name)
    return target


def seed_promotion_deck(root: Path) -> Path:
    """Give an isolated promotion fixture exactly one word-deck owner."""
    deck_dir = Path(root) / "data" / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    path = deck_dir / "all.yaml"
    path.write_text(
        "deck:\n"
        "  name: Promotion fixture\n"
        '  source: "../../vocabulary.json"\n',
        encoding="utf-8",
    )
    return path
