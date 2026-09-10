"""§6.5's lock order, at the four existing revision-finish entrypoints.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §6.5 requires every mutation entry
that can reach promotion to take the staging mutation coordination guard
**before any existing janki lock**, keep its own internal order after it, and
it ends "no code path may invert this ordering".
`application/card_revision_finish` spells that order four separate times — one
`with` header per public entrypoint — so it can be inverted four separate
ways, and each entrypoint needs its own case.

Both cases below are runtime interleavings of the real entrypoints against the
real `study_curation.curation_guard` and the real `.janki-audio-operation`
path lock. Neither reads production source text, and neither treats a timeout
as evidence: each waits, bounded, for a **positive** event that only the
correct order can produce.

Holding the guard and watching a finish block on it is not sufficient evidence
by itself — an inverted entry blocks there too, having already taken
`.janki-audio-operation` on its way in. What separates the two orders is
whether that lock is still free while the finish waits. So:

* `test_a_finish_entry_waits_for_the_guard_with_the_audio_lock_still_free`
  holds the guard, then *acquires* `.janki-audio-operation` from a third
  thread at the one moment an inverted entry would provably already hold it.
* `test_a_finish_entry_already_holds_the_guard_when_it_waits_for_audio` holds
  `.janki-audio-operation` instead and requires the finish to be inside the
  real guard while it waits for it.

The guard is held through the same production context manager a curation holds
before its own staging locks (§6.3 step 1). For the duration of each case that
context manager is wrapped so the test can tell *when* a finish reaches it;
the wrapper delegates to the real guard, so every acquisition, every block and
every release is production's. The wrapper's "reaching" event is published
**before** the real acquisition, which is what makes the probe exact: at that
instant the finish already holds every lock its own `with` header takes ahead
of the guard, and holds nothing it takes after.

The entrypoints run against a scratch project holding no proposal and no
finish receipt, so each refuses inside its guarded region. That is deliberate:
the ordering under test is decided by the `with` header alone, and the refusal
arriving only *after* the guard is released is what proves the body was
reached through it. No provider is resolved (both are passed in), nothing is
written, and no network, model or `data/` path is touched.
"""

from __future__ import annotations

import contextlib
import threading
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from japanese_anki.application import card_revision_finish, study_curation
from japanese_anki.application.card_revision_finish import CardRevisionFinishError
from japanese_anki.config import ProjectConfig
from japanese_anki.io import exclusive_path_lock

#: Every wait in this module is bounded. A correct run never spends it: each
#: awaited event is published by another thread that is already runnable. A
#: run against an inverted entry spends it once and fails.
BOUND = 15.0

AUDIO_OPERATION_LOCK = ".janki-audio-operation"


def _project(tmp_path: Path) -> ProjectConfig:
    """A scratch project with a staging directory and nothing staged in it."""
    (tmp_path / "janki.toml").write_text(
        '[paths]\nstaging_dir = "staging"\n', encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    return config


# --- the four public entrypoints, each with its own `with` header --------------


def _plan_card(config: ProjectConfig) -> Any:
    return card_revision_finish.plan_card_revision_finish(
        config,
        "revision:absent",
        "Clarify this card's usage note.",
        record_ids=("vocab-absent",),
        # Supplied, so provider resolution — which happens before the locks —
        # cannot be what the entry does or does not reach.
        word_provider=object(),
        sentence_provider=object(),
    )


def _plan_ai_enrichment(config: ProjectConfig) -> Any:
    return card_revision_finish.plan_ai_enrichment_finish(
        config,
        "enrichment:absent",
        "Complete the empty slots on these cards.",
        record_ids=("vocab-absent",),
        word_provider=object(),
        sentence_provider=object(),
    )


def _execute_card(config: ProjectConfig) -> Any:
    # `execute_card_revision_finish` reads only these attributes before its
    # `with` header (the repository check) and re-plans from `resource_id`
    # inside it, so a stand-in carrying the real repository root reaches the
    # locks and then refuses on the absent proposal. A full
    # `CardRevisionFinishPlan` would need a real audio and package plan and
    # would prove nothing more about the header under test.
    expected = types.SimpleNamespace(
        repository_root=config.root.resolve(),
        proposal_kind="card_revision",
        resource_id="revision:absent",
        instruction="Clarify this card's usage note.",
        review=None,
        record_ids=("vocab-absent",),
        authority={},
        fingerprint="0" * 64,
    )
    return card_revision_finish.execute_card_revision_finish(
        config,
        expected,
        word_provider=object(),
        sentence_provider=object(),
    )


def _resume_card(config: ProjectConfig) -> Any:
    return card_revision_finish.resume_card_revision_finish(config, "0" * 64)


ENTRIES: tuple[tuple[str, Callable[[ProjectConfig], Any]], ...] = (
    ("plan_card_revision_finish", _plan_card),
    ("plan_ai_enrichment_finish", _plan_ai_enrichment),
    ("execute_card_revision_finish", _execute_card),
    ("resume_card_revision_finish", _resume_card),
)

_ENTRY_CASES = pytest.mark.parametrize(
    "entry",
    [entry for _name, entry in ENTRIES],
    ids=[name for name, _entry in ENTRIES],
)


# --- watching the real guard without replacing it -----------------------------


class _WatchedGuard:
    """The production guard, plus two events about one thread's progress.

    ``reached`` is published before the real acquisition and ``held`` after
    it, so between them the finish is provably inside `curation_guard` and
    provably past nothing that follows it.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._real = study_curation.curation_guard
        self.reached = threading.Event()
        self.held = threading.Event()
        self.entries: list[str] = []
        monkeypatch.setattr(study_curation, "curation_guard", self._guard)

    @contextlib.contextmanager
    def _guard(self, config: ProjectConfig) -> Iterator[None]:
        self.entries.append("reached")
        self.reached.set()
        with self._real(config):
            self.entries.append("held")
            self.held.set()
            yield

    @contextlib.contextmanager
    def held_elsewhere(self, config: ProjectConfig) -> Iterator[None]:
        """Hold the guard the way a curation does, unwatched."""
        with self._real(config):
            yield


class _Runner:
    """One finish entry on its own thread, with its outcome kept for the end."""

    def __init__(
        self, entry: Callable[[ProjectConfig], Any], config: ProjectConfig
    ) -> None:
        self._entry = entry
        self._config = config
        self.done = threading.Event()
        self.outcome: list[Any] = []
        self.thread = threading.Thread(target=self._run, name="finish-entry")

    def _run(self) -> None:
        try:
            self.outcome.append(self._entry(self._config))
        except BaseException as exc:  # reported from the main thread
            self.outcome.append(exc)
        finally:
            self.done.set()

    def start(self) -> None:
        self.thread.start()

    def refusal(self) -> BaseException:
        """The entry's own domain refusal, once it has been allowed to run."""
        assert self.done.wait(BOUND), "the finish entry never returned"
        self.thread.join(BOUND)
        assert not self.thread.is_alive()
        assert len(self.outcome) == 1
        result = self.outcome[0]
        assert isinstance(result, CardRevisionFinishError), result
        return result


def _join(thread: threading.Thread) -> None:
    thread.join(BOUND)
    assert not thread.is_alive(), f"{thread.name} never finished"


@_ENTRY_CASES
def test_a_finish_entry_waits_for_the_guard_with_the_audio_lock_still_free(
    entry: Callable[[ProjectConfig], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§6.5: the guard comes first, so a blocked finish holds no other lock.

    A curation holds the coordination guard. The finish entry reaches the
    guard and stops there — and at that moment `.janki-audio-operation` is
    still free, which a third thread proves by taking it. Only after the
    curation lets the guard go does the entry get past it and refuse on the
    proposal it was given.

    Mutant: swap the two context managers in this entry's `with` header
    (`.janki-audio-operation` first, `study_curation.curation_guard` second).
    The entry then already holds that lock when it reaches the guard, the
    third thread cannot take it, and this case fails. Blocking alone does not
    catch that mutant, which is why the case is written around the probe.
    """
    config = _project(tmp_path)
    guard = _WatchedGuard(monkeypatch)
    finish = _Runner(entry, config)

    probe_held = threading.Event()
    probe_release = threading.Event()

    def take_audio_operation_lock() -> None:
        with exclusive_path_lock(config.root / AUDIO_OPERATION_LOCK):
            probe_held.set()
            probe_release.wait(BOUND)

    probe = threading.Thread(target=take_audio_operation_lock, name="audio-probe")
    try:
        with guard.held_elsewhere(config):
            finish.start()
            assert guard.reached.wait(BOUND), (
                "the finish entry never reached the coordination guard, so this "
                "case proved nothing about the order it takes locks in"
            )
            assert not guard.held.is_set(), "the guard is exclusive"

            # The finish is inside `curation_guard` and past nothing after it.
            # Whatever its header takes *before* the guard, it holds now.
            probe.start()
            assert probe_held.wait(BOUND), (
                "the finish entry already holds .janki-audio-operation while it "
                "waits for the coordination guard: its lock order is inverted"
            )
            assert not finish.done.is_set(), (
                "the finish entry ran to completion without the guard"
            )
            assert not guard.held.is_set()

            probe_release.set()
            _join(probe)
            assert not finish.done.is_set()

        # The guard is free; only now may the entry proceed.
        refusal = finish.refusal()
    finally:
        probe_release.set()
        probe.join(BOUND)
        finish.thread.join(BOUND)

    assert guard.entries == ["reached", "held"]
    assert "absent" in str(refusal) or "no longer exists" in str(refusal)


@_ENTRY_CASES
def test_a_finish_entry_already_holds_the_guard_when_it_waits_for_audio(
    entry: Callable[[ProjectConfig], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§6.5: the same order seen from the inner lock's side.

    With `.janki-audio-operation` held by another audio operation, the entry
    must get *past* the guard — holding it — before it waits for that lock, so
    a curation arriving meanwhile is the one that waits. Releasing the audio
    lock then lets the entry through to its own refusal.

    Mutant: swap the two context managers in this entry's `with` header. The
    entry then blocks on `.janki-audio-operation` without ever holding the
    guard, `held` is never published, and this case fails.
    """
    config = _project(tmp_path)
    guard = _WatchedGuard(monkeypatch)
    finish = _Runner(entry, config)

    try:
        with exclusive_path_lock(config.root / AUDIO_OPERATION_LOCK):
            finish.start()
            assert guard.held.wait(BOUND), (
                "the finish entry never took the coordination guard while it "
                "waited for .janki-audio-operation: its lock order is inverted"
            )
            assert not finish.done.is_set(), (
                "the finish entry ran to completion without .janki-audio-operation"
            )

        refusal = finish.refusal()
    finally:
        finish.thread.join(BOUND)

    assert guard.entries == ["reached", "held"]
    assert "absent" in str(refusal) or "no longer exists" in str(refusal)
