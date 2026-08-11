"""Suite-wide guards.

One rule so far, and it is about the machine rather than the code: **no test
touches the developer's real Anki collection.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from japanese_anki import collection, status


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
