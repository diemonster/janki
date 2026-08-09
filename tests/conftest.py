"""Suite-wide guards.

One rule so far, and it is about the machine rather than the code: **no test
touches the developer's real Anki collection.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from japanese_anki import collection


@pytest.fixture(autouse=True)
def _no_real_anki_collection(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
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
    empty = tmp_path_factory.mktemp("no-anki")
    monkeypatch.setattr(collection, "default_anki_root", lambda: empty)
    return empty
