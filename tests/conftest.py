"""Suite-wide guards.

Two rules, both about the machine rather than the code: **no test touches the
developer's real Anki collection**, and **no test opens a billed API client.**
"""

from __future__ import annotations

import shutil
from pathlib import Path

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

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "This test reached claude_client.build_client(), which opens a "
            "billed connection. Patch the parse_call your code path actually "
            "uses — both providers if a config default decides it — or mark "
            "the test @pytest.mark.allow_build_client if it means to."
        )

    monkeypatch.setattr(claude_client, "build_client", refuse)


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_prompts(root: Path) -> Path:
    """Copy this checkout's real `prompts/` into a temp project.

    The repository's own files, not stubs. A temp project that invented its
    own prompt text would let a CLI test pass while the shipped template said
    something else entirely — and these files are the deliverable now, so the
    suite should exercise them. Cheap enough to do per project: seven small
    Markdown files.
    """
    target = Path(root) / prompts.DIRECTORY
    target.mkdir(parents=True, exist_ok=True)
    for source in (REPO_ROOT / prompts.DIRECTORY).glob("*.md"):
        shutil.copy2(source, target / source.name)
    return target
