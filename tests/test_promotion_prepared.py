"""The prepared promotion writer: freeze, apply, recover.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §7.7. One transaction is planned
completely before a byte of it is written, and a process that dies between two
of its writes finishes from that plan alone — same archive, same receipt, same
dates, on the following day as on the day it started.

Everything checked here is an artifact fact: which bytes are at which digest,
under whose configuration, proven by which receipt. Nothing reads Japanese and
nothing judges a model answer. No network: the dictionary witness is the
suite's existing fake transport, and most cases skip the reading check outright
because the reading verdict is not what they are about.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from test_promote import (
    ONE_GOOD,
    _pattern_only_meta,
    accounted_extract,
    ai_staging_meta,
    client_for,
    hanasu_jpdb,
    parsed_candidate,
    project,
    record,
    staging_file,
)

from japanese_anki import patterns, promote, staging
from japanese_anki.application import promotion as promotion_application
from japanese_anki.application import study_curation, study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import exclusive_path_lock, load_records, records_revision
from japanese_anki.staging import read_staging, write_staging

BOUND = 15.0

FROZEN = date(2026, 9, 12)
NEXT_DAY = date(2026, 9, 13)

HELD_ONLY = """\
source_file: lesson.pdf
records:
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''
    source: {type: extract, imported_from: lesson.pdf}
"""

LANDS_AND_HOLDS = """\
source_file: lesson.pdf
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
    meanings: [to speak]
    source:
      type: extract
      imported_from: lesson.pdf
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''
    source: {type: extract, imported_from: lesson.pdf}
"""

LEGACY_EMPTY = """\
source_file: lesson.pdf
records: []
"""


# --- fixtures -----------------------------------------------------------------


def _config(tmp_path: Path, records: list[Any] | None = None) -> ProjectConfig:
    return ProjectConfig.load(project(tmp_path, records or []))


def _decide(
    config: ProjectConfig, path: Path, **kwargs: Any
) -> promotion_application.PromotionDecision:
    kwargs.setdefault("skip_reading_check", True)
    return promotion_application.decide_promotion(config, path, **kwargs)


def _prepared(
    config: ProjectConfig,
    path: Path,
    *,
    now: date = FROZEN,
    part_name: str = "part-01",
    **kwargs: Any,
) -> promotion_application.PreparedSourcePromotion:
    decision = _decide(config, path, **kwargs)
    return promotion_application.prepare_source_promotion(
        config, decision, part_name=part_name, now=now
    )


def _roundtrip(
    prepared: promotion_application.PreparedSourcePromotion,
) -> promotion_application.PreparedSourcePromotion:
    """Exactly what a study job persists and reads back."""
    return promotion_application.PreparedSourcePromotion.from_dict(
        json.loads(json.dumps(prepared.to_dict(), ensure_ascii=False))
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _land_component(
    prepared: promotion_application.PreparedSourcePromotion, role: str
) -> None:
    """Write exactly one frozen component, the way the writer would have."""
    component = prepared.component(role)
    assert component is not None and component.writes, role
    target = Path(component.path)
    if component.removes:
        target.unlink()
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(component.after_text or "", encoding="utf-8")


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _guarded(config: ProjectConfig, prepared, **kwargs):
    """Apply the way a finish coordinator does: guard outermost."""
    with study_curation.curation_guard(config):
        return promotion_application.apply_prepared_source_promotion(
            config, prepared, **kwargs
        )


# --- preparation writes nothing and freezes everything ------------------------


def test_prepare_publishes_nothing_and_binds_the_whole_component_vector(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    before = _files(config.root)

    prepared = _prepared(config, path)

    assert _files(config.root) == before
    assert prepared.projected_state == "lands"
    assert prepared.landed_ids == ("word:話す:はなす",)
    assert prepared.held_ids == ()
    assert [item.role for item in prepared.components] == [
        "canonical",
        "ledger",
        "archive",
        "live_staging",
    ]
    canonical = prepared.component("canonical")
    ledger_component = prepared.component("ledger")
    archive = prepared.component("archive")
    live = prepared.component("live_staging")
    assert canonical is not None and ledger_component is not None
    assert archive is not None and live is not None
    # Every payload is complete text, not a delta: a resume cannot recompute
    # bytes from a store that has moved.
    assert canonical.after_text and "word:話す:はなす" in canonical.after_text
    assert ledger_component.after_text and FROZEN.isoformat() in ledger_component.after_text
    assert archive.after_text and prepared.receipt_id in archive.after_text
    # No row is held, so the live review is removed rather than rewritten.
    assert live.expected_after is None and live.after_text is None
    assert live.removes
    assert prepared.ledger_dates == {
        "added_at": FROZEN.isoformat(),
        "seen_at": FROZEN.isoformat(),
        "enriched_at": FROZEN.isoformat(),
    }
    assert _roundtrip(prepared) == prepared
    assert _roundtrip(prepared).fingerprint == prepared.fingerprint


def test_prepare_binds_a_held_remainder_as_a_rewrite_not_a_removal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, LANDS_AND_HOLDS)

    prepared = _prepared(config, path)

    live = prepared.component("live_staging")
    assert live is not None
    assert prepared.landed_ids == ("word:話す:はなす",)
    assert prepared.held_ids == ("word:食べ物:",)
    assert not live.removes
    assert live.after_text is not None
    held, _meta = staging.read_staging_text(live.after_text, source=str(path))
    assert [item.id for item in held] == ["word:食べ物:"]
    assert promote.HOLD_MISSING_READING in staging.annotations(held[0])["hold_reason"]


@pytest.mark.parametrize(
    ("state", "roles"),
    [
        ("nothing", []),
        ("pattern_only", ["archive", "live_staging"]),
        ("nothing_lands", ["canonical", "ledger", "archive", "live_staging"]),
        ("lands", ["canonical", "ledger", "archive", "live_staging"]),
    ],
)
def test_each_promotable_state_binds_its_own_component_vector(
    tmp_path: Path, state: str, roles: list[str]
) -> None:
    config, path = _fixture_for(tmp_path, state)
    before = _files(config.root)

    prepared = _prepared(config, path)

    assert prepared.projected_state == state
    assert [item.role for item in prepared.components] == roles
    assert _files(config.root) == before
    # The two states that write no canonical byte still bind the file, so an
    # external collection edit is not silently accepted.
    if state == "nothing_lands":
        canonical = prepared.component("canonical")
        assert canonical is not None and not canonical.writes


def _fixture_for(tmp_path: Path, state: str) -> tuple[ProjectConfig, Path]:
    """One real staging file per promotable decision state."""
    config = _config(tmp_path)
    if state == "nothing":
        return config, staging_file(config.root, LEGACY_EMPTY)
    if state == "nothing_lands":
        return config, staging_file(config.root, HELD_ONLY)
    if state == "lands":
        return config, staging_file(config.root, ONE_GOOD)
    if state == "pattern_only":
        proposed = patterns.PatternSet(
            source="lesson.pdf",
            kind="lesson",
            title="Te-form",
            patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
        )
        meta = _pattern_only_meta(proposed)
        path = config.staging_dir / "lesson.pdf.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_staging(path, [], meta)
        staged = patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"])
        patterns.save_store(
            config.patterns_file, {proposed.source: replace(staged, reviewed=True)}
        )
        return config, path
    raise AssertionError(state)


# --- a projected decision never reaches the writer ----------------------------


@pytest.mark.parametrize(
    "state", ["nothing", "pattern_only", "nothing_lands", "lands"]
)
def test_execute_refuses_a_projected_decision_in_every_early_branch(
    tmp_path: Path, state: str
) -> None:
    """The refusal is at function entry, ahead of every branch that writes.

    `nothing`, `pattern_only` and `archive_retry` each archive or delete a live
    file before they reach the landing path, so a refusal placed after them
    would already have consumed the review.
    """
    config, path = _fixture_for(tmp_path, state)
    decision = _decide(config, path)
    projected = replace(decision, projected=True)
    before = _files(config.root)

    with pytest.raises(
        promote.PromoteError, match="promotion-projection-not-executable"
    ):
        promotion_application.execute_promotion(config, projected)

    assert _files(config.root) == before


def test_any_single_injection_marks_a_decision_projected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    wire, records, meta = promotion_application.record_review_snapshot(path)

    for injection in (
        {"_record_snapshot": (wire, records, meta)},
        {"_pattern_store_snapshot": {}},
    ):
        decision = _decide(config, path, **injection)
        assert decision.projected is True
        with pytest.raises(
            promote.PromoteError, match="promotion-projection-not-executable"
        ):
            promotion_application.execute_promotion(config, decision)


# --- the ordinary path still behaves exactly as it did ------------------------


def test_the_ordinary_writer_lands_through_the_prepared_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)

    result = promotion_application.execute_promotion(config, _decide(config, path))

    assert result.state == "landed"
    assert result.promoted_ids == ("word:話す:はなす",)
    assert not path.exists()
    archived, archived_meta = read_staging(config.staging_dir / "done" / path.name)
    (batch,) = promotion_application.promotion_batches(
        archived_meta, archived=archived, archive_file=path.name
    )
    assert result.receipt_id == batch.receipt_id
    assert [item.id for item in load_records(config.normalized_file)] == [
        "word:話す:はなす"
    ]


# --- resuming a part whose own writes already began ---------------------------


CRASH_SEAMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("after_canonical", ("canonical",)),
    ("after_ledger", ("canonical", "ledger")),
    ("after_archive", ("canonical", "ledger", "archive")),
)


@pytest.mark.parametrize(
    ("seam", "landed_roles"), CRASH_SEAMS, ids=[name for name, _ in CRASH_SEAMS]
)
@pytest.mark.parametrize("holds", [False, True], ids=["live-removed", "live-held"])
def test_a_resumed_apply_finishes_its_own_half_written_vector(
    tmp_path: Path, seam: str, landed_roles: tuple[str, ...], holds: bool
) -> None:
    """The vector is classified before anything is decided again.

    Re-deciding first cannot work: the canonical file already holds the row,
    and on the last seam the live review is gone — so a fresh decision either
    refuses or describes a different transaction, and in both cases the part's
    own completed writes are read as somebody else's.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, LANDS_AND_HOLDS if holds else ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    for role in landed_roles:
        _land_component(prepared, role)

    result = _guarded(config, prepared)

    assert result.state == "landed"
    assert result.promoted_ids == prepared.landed_ids
    assert result.receipt_id == prepared.receipt_id
    for component in prepared.components:
        assert _digest(Path(component.path)) == component.expected_after
    assert [item.id for item in load_records(config.normalized_file)] == [
        "word:話す:はなす"
    ]
    if holds:
        held, _meta = read_staging(path)
        assert [item.id for item in held] == ["word:食べ物:"]
    else:
        assert not path.exists()


def test_a_resumed_apply_reports_a_part_whose_every_write_already_landed(
    tmp_path: Path,
) -> None:
    """Completion is proven by the bound digests, never by a missing file."""
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    for role in ("canonical", "ledger", "archive", "live_staging"):
        _land_component(prepared, role)
    after = _files(config.root)

    result = _guarded(config, prepared)

    assert result.state == "landed"
    assert result.promoted_ids == prepared.landed_ids
    assert result.receipt_id == prepared.receipt_id
    assert _files(config.root) == after


def test_a_resumed_apply_re_decides_an_unstarted_part_under_the_locks(
    tmp_path: Path,
) -> None:
    """An intent is a plan; the repository still has to agree with it."""
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))

    result = _guarded(config, prepared)

    assert result.state == "landed"
    assert result.receipt_id == prepared.receipt_id
    ledger_wire = json.loads(config.ledger_file.read_text(encoding="utf-8"))
    entry = ledger_wire["records"]["word:話す:はなす"]
    assert entry["added_at"] == FROZEN.isoformat()
    assert [source["seen_at"] for source in entry["sources"]] == [FROZEN.isoformat()]


def test_a_resumed_apply_refuses_an_unstarted_part_the_repository_now_decides_differently(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    # The owner edited the review after the intent was recorded: the same file
    # now proposes a different row.
    records, meta = read_staging(path)
    write_staging(
        path,
        [replace(records[0], id="word:聞く:きく", expression="聞く", reading="きく")],
        meta,
        force=True,
    )
    before = _files(config.root)

    with pytest.raises(promote.PromoteError, match="promotion-intent-stale"):
        _guarded(config, prepared)

    assert _files(config.root) == before


def test_a_resumed_apply_replays_frozen_dates_on_the_following_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.7's date boundary: a resume the next day is not a third state.

    `ledger._iso_date(None)` reads the clock, so a ledger payload recomputed at
    apply time is different bytes at midnight — and the part would find its own
    bound component at neither digest and refuse.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path, now=FROZEN))

    class _Tomorrow(date):
        @classmethod
        def today(cls) -> date:
            return NEXT_DAY

    monkeypatch.setattr(promotion_application, "date", _Tomorrow)

    result = _guarded(config, prepared)

    assert result.state == "landed"
    ledger_wire = config.ledger_file.read_text(encoding="utf-8")
    assert FROZEN.isoformat() in ledger_wire
    assert NEXT_DAY.isoformat() not in ledger_wire
    ledger_component = prepared.component("ledger")
    assert ledger_component is not None
    assert _digest(config.ledger_file) == ledger_component.expected_after


# --- recovery from the persisted intent alone ---------------------------------


@pytest.mark.parametrize(
    ("seam", "landed_roles"), CRASH_SEAMS, ids=[name for name, _ in CRASH_SEAMS]
)
def test_recovery_finishes_each_seam_with_the_frozen_payloads(
    tmp_path: Path, seam: str, landed_roles: tuple[str, ...]
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, LANDS_AND_HOLDS)
    prepared = _roundtrip(_prepared(config, path))
    for role in landed_roles:
        _land_component(prepared, role)

    recovery = promotion_application.recover_promotion_intent(config, prepared)

    assert recovery.state == "landed"
    assert recovery.part_name == "part-01"
    assert recovery.already_complete == tuple(sorted(landed_roles))
    assert recovery.receipt_id == prepared.receipt_id
    assert recovery.landed_ids == prepared.landed_ids
    for component in prepared.components:
        assert _digest(Path(component.path)) == component.expected_after
    # Idempotent: the second pass finishes nothing and claims nothing new.
    again = promotion_application.recover_promotion_intent(config, prepared)
    assert again.finished == ()
    assert set(again.already_complete) == {
        item.role for item in prepared.components
    }


def test_recovery_refuses_a_component_at_a_third_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    _land_component(prepared, "canonical")
    config.ledger_file.parent.mkdir(parents=True, exist_ok=True)
    config.ledger_file.write_text('{"version": 1, "records": {}}\n', encoding="utf-8")
    before = _files(config.root)

    with pytest.raises(promote.PromoteError) as caught:
        promotion_application.recover_promotion_intent(config, prepared)

    message = str(caught.value)
    assert "promotion-intent-stale" in message
    assert "ledger" in message
    ledger_component = prepared.component("ledger")
    assert ledger_component is not None
    assert str(ledger_component.expected_after) in message
    assert _files(config.root) == before


def test_recovery_refuses_a_component_repointed_at_another_path(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _prepared(config, path)
    wire = prepared.to_dict()
    elsewhere = config.root / "elsewhere.json"
    for component in wire["components"]:
        if component["role"] == "canonical":
            component["path"] = str(elsewhere)
    altered = promotion_application.PreparedSourcePromotion.from_dict(wire)
    before = _files(config.root)

    with pytest.raises(promote.PromoteError, match="promotion-intent-stale"):
        promotion_application.recover_promotion_intent(config, altered)

    assert not elsewhere.exists()
    assert _files(config.root) == before


def test_recovery_reproves_deck_ownership_before_a_pending_canonical_write(
    tmp_path: Path,
) -> None:
    """An intent is not authority to land a card no deck owns.

    The canonical write is the one that puts a row in front of a study deck.
    If it is still pending, recovery re-proves exactly one configured owner
    under the deck lock — the same question `commit_canonical_state` asks —
    rather than replaying frozen bytes over a repository whose deck rules have
    since changed.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    (config.deck_dir / "all.yaml").unlink()
    before = _files(config.root)

    with pytest.raises(JankiError, match="deck"):
        promotion_application.recover_promotion_intent(config, prepared)

    assert _files(config.root) == before


def test_recovery_refuses_while_a_curation_barrier_is_open(tmp_path: Path) -> None:
    """The pending-curation barrier is rechecked under the guard, not assumed.

    A barrier can be published between the preparation and the resume. A
    skipped barrier is a lifted one, so recovery asks again rather than
    trusting that planning already did.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    _land_component(prepared, "canonical")
    # A job document janki cannot read is a barrier it cannot prove is closed.
    jobs = study_job.study_jobs_dir(config)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{uuid.uuid4()}.json").write_text("not a job document", encoding="utf-8")
    before = _files(config.root)

    with pytest.raises(promote.PromoteError, match="cannot be read"):
        promotion_application.recover_promotion_intent(config, prepared)

    assert _files(config.root) == before


def test_apply_refuses_while_a_curation_barrier_is_open(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    jobs = study_job.study_jobs_dir(config)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{uuid.uuid4()}.json").write_text("not a job document", encoding="utf-8")
    before = _files(config.root)

    with pytest.raises(promote.PromoteError, match="cannot be read"):
        _guarded(config, prepared)

    assert _files(config.root) == before


# --- §7.7's unstarted / started distinction, on the recovery entry -------------


def _pattern_intent_whose_review_moved(
    root: Path,
) -> tuple[ProjectConfig, Path, promotion_application.PreparedSourcePromotion]:
    """A pattern-only intent prepared while its store entry was reviewed.

    The mark then moves back to `false`, which is the one owner judgement this
    state's whole archive rests on. Every component of the intent is still at
    its bound before-state, so the part is unstarted.
    """
    config, path = _fixture_for(root, "pattern_only")
    prepared = _roundtrip(_prepared(config, path))
    store = patterns.load_store(config.patterns_file)
    patterns.save_store(
        config.patterns_file,
        {key: replace(entry, reviewed=False) for key, entry in store.items()},
    )
    return config, path, prepared


def test_an_unstarted_recovery_and_an_apply_refuse_the_same_moved_pattern_review(
    tmp_path: Path,
) -> None:
    """An all-pending part is re-decided by **either** public entry.

    §7.7 sends an unstarted part back through a fresh decision precisely so
    that a check the ordinary promote applies cannot be skipped by resuming
    instead of applying. Recovery used to replay the frozen payloads
    unconditionally: it archived a pattern set nobody had reviewed and deleted
    the live review, where the ordinary apply refused and left both alone.
    """
    refusals: dict[str, str] = {}
    for mode in ("apply", "recovery"):
        root = tmp_path / mode
        root.mkdir()
        config, path, prepared = _pattern_intent_whose_review_moved(root)
        before = _files(config.root)

        with pytest.raises(promote.PromoteError) as caught:
            if mode == "apply":
                _guarded(config, prepared)
            else:
                promotion_application.recover_promotion_intent(config, prepared)

        refusals[mode] = str(caught.value)
        assert path.exists(), "the live review survives a refusal"
        assert _files(config.root) == before

    assert "patterns-unreviewed" in refusals["apply"]
    assert refusals["recovery"] == refusals["apply"]


def test_recovery_resumes_a_started_pattern_intent_without_deciding_it_again(
    tmp_path: Path,
) -> None:
    """The other half of the distinction, over the same moved review.

    The archive already landed, so this part's own writes have begun and a
    fresh decision would read them as somebody else's — it refuses over this
    very repository, as the paired case above proves. Cleanup therefore
    finishes from the frozen payloads instead, and removes the live review.
    """
    config, path, prepared = _pattern_intent_whose_review_moved(tmp_path)
    _land_component(prepared, "archive")
    archive = prepared.component("archive")
    assert archive is not None

    recovery = promotion_application.recover_promotion_intent(config, prepared)

    assert recovery.state == "pattern_only"
    assert recovery.already_complete == ("archive",)
    assert recovery.finished == ("live_staging",)
    assert not path.exists()
    assert Path(archive.path).read_text(encoding="utf-8") == archive.after_text


def test_an_unstarted_recovery_needs_the_recorded_dictionary_witness(
    tmp_path: Path,
) -> None:
    """A consulted part is re-decided against the recorded book, or not at all.

    Re-deciding without the witness would silently ask the offline question
    instead of the one whose answer was approved; fetching one here would let a
    dictionary refresh change what the owner accepted. Both are refusals.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(
        _prepared(
            config, path, client=client_for(hanasu_jpdb()), skip_reading_check=False
        )
    )
    assert prepared.reading_check == "consulted"
    before = _files(config.root)

    with pytest.raises(
        promote.PromoteError, match="promotion-intent-witness-required"
    ):
        promotion_application.recover_promotion_intent(config, prepared)

    assert _files(config.root) == before

    recovery = promotion_application.recover_promotion_intent(
        config, prepared, witness=client_for(hanasu_jpdb())
    )

    assert recovery.state == "landed"
    assert recovery.finished == ("canonical", "ledger", "archive", "live_staging")


# --- the shared curation guard ------------------------------------------------


class _WatchedGuard:
    """The production guard, plus two events about one thread's progress."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._real = study_curation.staging_curation_guard
        self.reached = threading.Event()
        self.held = threading.Event()
        self.entries: list[str] = []
        monkeypatch.setattr(study_curation, "staging_curation_guard", self._guard)

    @contextlib.contextmanager
    def _guard(self, staging_dir: Path):
        self.entries.append("reached")
        self.reached.set()
        with self._real(staging_dir):
            self.entries.append("held")
            self.held.set()
            yield

    @contextlib.contextmanager
    def held_elsewhere(self, staging_dir: Path):
        with self._real(staging_dir):
            yield


def test_recovery_waits_for_the_guard_with_its_component_locks_still_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: the coordination guard is outermost on the recovery entry too.

    Mutant: drop `curation_guard` from `recover_promotion_intent`, or take it
    inside the component locks. In the first case the recovery finishes while a
    curation holds the guard; in the second the probe below cannot take the
    canonical lock. Blocking alone separates neither.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    _land_component(prepared, "canonical")
    guard = _WatchedGuard(monkeypatch)

    outcome: list[object] = []
    done = threading.Event()

    def run() -> None:
        try:
            outcome.append(
                promotion_application.recover_promotion_intent(config, prepared)
            )
        except BaseException as exc:  # reported from the main thread
            outcome.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=run, name="promotion-recovery")
    probe_held = threading.Event()
    probe_release = threading.Event()

    def take_canonical_lock() -> None:
        with exclusive_path_lock(config.normalized_file):
            probe_held.set()
            probe_release.wait(BOUND)

    probe = threading.Thread(target=take_canonical_lock, name="canonical-probe")
    try:
        with guard.held_elsewhere(config.staging_dir):
            worker.start()
            assert guard.reached.wait(BOUND), (
                "promotion recovery never took the coordination guard, so a "
                "curation and a recovery can write staged bytes at once"
            )
            assert not guard.held.is_set(), "the guard is exclusive"
            probe.start()
            assert probe_held.wait(BOUND), (
                "the recovery already holds a component lock while it waits for "
                "the coordination guard: its lock order is inverted"
            )
            assert not done.is_set(), "the recovery completed without the guard"
            probe_release.set()
            probe.join(BOUND)
            assert not done.is_set()

        assert done.wait(BOUND), "the recovery never returned"
    finally:
        probe_release.set()
        if probe.ident is not None:
            probe.join(BOUND)
        if worker.ident is not None:
            worker.join(BOUND)

    assert guard.entries == ["reached", "held"]
    assert len(outcome) == 1 and not isinstance(outcome[0], BaseException), outcome
    assert getattr(outcome[0], "state", None) == "landed"


def test_recovery_under_guard_does_not_take_the_nonreentrant_guard_again(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    _land_component(prepared, "canonical")

    outcome: list[object] = []
    done = threading.Event()

    def run() -> None:
        try:
            with study_curation.curation_guard(config):
                outcome.append(
                    promotion_application.recover_promotion_intent_under_guard(
                        config, prepared
                    )
                )
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=run, name="guarded-recovery")
    worker.start()
    try:
        assert done.wait(BOUND), (
            "a recovery inside the coordination guard took it a second time and "
            "deadlocked on the non-reentrant lock"
        )
    finally:
        worker.join(BOUND)

    assert len(outcome) == 1 and not isinstance(outcome[0], BaseException), outcome
    assert getattr(outcome[0], "state", None) == "landed"


def test_a_curation_and_a_promotion_contend_for_one_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary one-command promote waits behind a curation, not beside it."""
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    guard = _WatchedGuard(monkeypatch)

    outcome: list[object] = []
    done = threading.Event()

    def run() -> None:
        try:
            with study_curation.curation_guard(config):
                outcome.append(
                    promotion_application.apply_prepared_source_promotion(
                        config,
                        promotion_application.prepare_source_promotion(
                            config, _decide(config, path), now=FROZEN
                        ),
                    )
                )
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=run, name="promotion")
    try:
        with guard.held_elsewhere(config.staging_dir):
            worker.start()
            assert guard.reached.wait(BOUND), "the promotion never reached the guard"
            assert not done.is_set(), "the promotion ran while a curation held it"
        assert done.wait(BOUND), "the promotion never returned"
    finally:
        worker.join(BOUND)

    assert len(outcome) == 1 and not isinstance(outcome[0], BaseException), outcome
    assert getattr(outcome[0], "state", None) == "landed"


# --- the fold -----------------------------------------------------------------


def test_the_fold_chains_canonical_state_across_a_part_that_lands_nothing(
    tmp_path: Path,
) -> None:
    """§7.6: the carry advances only for a part whose state is `lands`.

    The middle part holds every row. Its projection is bound to the canonical
    state the first part left, it writes no canonical byte, and the third part
    projects against that same state rather than against an empty `merged`.
    """
    config = _config(tmp_path)
    first = staging_file(config.root, ONE_GOOD, name="p01.yaml")
    middle = staging_file(config.root, HELD_ONLY, name="p02.yaml")
    last = staging_file(
        config.root,
        ONE_GOOD.replace("話す", "聞く").replace("はなす", "きく"),
        name="p03.yaml",
    )
    parts = [
        promotion_application.SourcePromotionPart(
            part_name=name,
            staging_path=path,
            expected_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for name, path in (("p03", last), ("p01", first), ("p02", middle))
    ]
    before = _files(config.root)

    fold = promotion_application.fold_source_extraction_promotions(
        config,
        parts,
        collection=[],
        collection_revision=records_revision(config.normalized_file),
        pattern_store={},
        witness=client_for(hanasu_jpdb()),
    )

    assert _files(config.root) == before
    assert [part.part_name for part in fold.parts] == ["p01", "p02", "p03"]
    assert [part.state for part in fold.parts] == ["lands", "nothing_lands", "lands"]
    digests = fold.canonical_digests
    assert len(digests) == 4
    # The held part writes no canonical byte, so its after-state is exactly its
    # before-state and the empty `merged` digest never enters the chain.
    assert digests[0] != digests[1]
    assert digests[1] == digests[2] != digests[3]
    assert fold.parts[1].expected_before == fold.parts[1].expected_after == digests[1]
    assert fold.parts[1].landed_ids == ()
    assert fold.parts[1].held_ids == ("word:食べ物:",)
    assert [item.id for item in fold.records_after] == [
        "word:聞く:きく",
        "word:話す:はなす",
    ]
    assert all(part.decision is not None for part in fold.parts)
    assert all(part.decision.projected for part in fold.parts)  # type: ignore[union-attr]


def test_the_fold_reads_an_absent_collection_as_absence(tmp_path: Path) -> None:
    """`None` is a real canonical state, not an empty collection.

    A fresh repository has no collection file at all, and every part that
    writes none carries that absence forward — a digest of `[]` would claim a
    file exists that does not, and the first landing's compare-and-swap binds
    against absence.
    """
    config = _config(tmp_path)
    config.normalized_file.unlink()
    parts = [
        promotion_application.SourcePromotionPart(
            part_name=name,
            staging_path=path,
            expected_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for name, path in (
            ("p01", staging_file(config.root, HELD_ONLY, name="p01.yaml")),
            ("p02", staging_file(config.root, HELD_ONLY, name="p02.yaml")),
        )
    ]

    fold = promotion_application.fold_source_extraction_promotions(
        config,
        parts,
        collection=[],
        collection_revision=records_revision(config.normalized_file),
        pattern_store={},
        witness=client_for(hanasu_jpdb()),
    )

    assert [part.state for part in fold.parts] == ["nothing_lands", "nothing_lands"]
    assert fold.canonical_digests == (None, None, None)
    assert all(part.expected_before is None for part in fold.parts)
    assert all(part.expected_after is None for part in fold.parts)
    assert fold.records_after == ()
    assert not config.normalized_file.exists()


def test_the_fold_refuses_a_part_whose_bytes_moved(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD, name="p01.yaml")
    part = promotion_application.SourcePromotionPart(
        part_name="p01",
        staging_path=path,
        expected_revision="0" * 64,
    )

    with pytest.raises(promote.PromoteError, match="source-review-stale"):
        promotion_application.fold_source_extraction_promotions(
            config,
            [part],
            collection=[],
            collection_revision=records_revision(config.normalized_file),
            pattern_store={},
            witness=client_for(hanasu_jpdb()),
        )


def test_the_fold_applies_the_review_writers_own_example_flags(
    tmp_path: Path,
) -> None:
    """The projection flags what `ReviewPanel.submit` flags, not what the two
    neighbouring projections filter."""
    config = _config(tmp_path)
    records, meta = accounted_extract(
        tmp_path, parsed_candidate(expression="話す", reading="はなす")
    )
    path = config.staging_dir / "lesson.pdf.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(path, records, meta)

    decision = promotion_application.project_source_extraction_review_promotion(
        config,
        path,
        expected_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
        review_record_ids=(records[0].id,),
        review_patterns=False,
        pattern_store={},
        coverage_approval=None,
        collection=[],
        collection_revision=records_revision(config.normalized_file),
        witness=client_for(hanasu_jpdb()),
    )

    assert decision.projected is True
    flagged = {item.id: item for item in decision.records}
    from japanese_anki.models import EXAMPLE_AUTHORITY_KEY

    assert EXAMPLE_AUTHORITY_KEY in flagged[records[0].id].source.raw_fields
    assert path.read_bytes() == path.read_bytes()


# --- the two ledger-incomplete splits -----------------------------------------


def test_a_failed_ledger_save_leaves_an_intent_recovery_can_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`landed_ledger_incomplete` is preserved, and it is recoverable.

    The ordinary non-AI split archives and prunes and reports the ledger gap;
    the durable intent is what turns that report into something a resume can
    act on without re-deciding a transaction whose live review is already gone.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    decision = _decide(config, path)
    prepared = _roundtrip(
        promotion_application.prepare_source_promotion(
            config, decision, part_name="part-01", now=FROZEN
        )
    )
    monkeypatch.setattr(
        promotion_application.ledger.Ledger,
        "_save_locked",
        lambda self: (_ for _ in ()).throw(
            promotion_application.ledger.LedgerError("disk full")
        ),
    )

    result = promotion_application.execute_promotion(config, decision)

    assert result.state == "landed_ledger_incomplete"
    assert not path.exists()
    ledger_component = prepared.component("ledger")
    assert ledger_component is not None
    assert _digest(config.ledger_file) == ledger_component.expected_before

    monkeypatch.undo()
    recovery = promotion_application.recover_promotion_intent(config, prepared)

    assert recovery.finished == ("ledger",)
    assert set(recovery.already_complete) == {"canonical", "archive", "live_staging"}
    assert _digest(config.ledger_file) == ledger_component.expected_after
    assert recovery.receipt_id == prepared.receipt_id


def test_an_ai_ledger_failure_keeps_the_live_review_and_recovers_from_its_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deliberate AI split: canonical landed, nothing else did.

    `execute_promotion` raises before archive and prune on purpose, because the
    reviewed proposal is the only recoverable model attribution. The intent
    finishes all three remaining writes with the frozen payloads.
    """
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    config = _config(tmp_path, [original])
    path = config.staging_dir / "ai.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = ai_staging_meta(
        [original], {original.id: {"meanings": (original.meanings, proposal.meanings)}}
    )
    write_staging(path, [proposal], meta)
    decision = _decide(config, path)
    prepared = _roundtrip(
        promotion_application.prepare_source_promotion(
            config, decision, part_name="part-01", now=FROZEN
        )
    )
    monkeypatch.setattr(
        promotion_application.ledger.Ledger,
        "_save_locked",
        lambda self: (_ for _ in ()).throw(
            promotion_application.ledger.LedgerError("disk full")
        ),
    )

    result = promotion_application.execute_promotion(config, decision)

    assert result.state == "landed_ai_ledger_incomplete"
    assert path.exists(), "the only recoverable AI attribution stays live"
    assert not (config.staging_dir / "done" / "ai.yaml").exists()

    monkeypatch.undo()
    recovery = promotion_application.recover_promotion_intent(config, prepared)

    assert recovery.already_complete == ("canonical",)
    assert recovery.finished == ("ledger", "archive", "live_staging")
    for component in prepared.components:
        assert _digest(Path(component.path)) == component.expected_after
    assert not path.exists()
    ledger_wire = config.ledger_file.read_text(encoding="utf-8")
    assert FROZEN.isoformat() in ledger_wire
    assert "claude-opus-5" in ledger_wire


# --- one stale later component refuses the whole vector -----------------------


def test_a_resumed_apply_refuses_a_stale_later_component_before_any_write(
    tmp_path: Path,
) -> None:
    """The archive is measured before the canonical write, not after it."""
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    archive = prepared.component("archive")
    assert archive is not None and archive.expected_before is None
    Path(archive.path).parent.mkdir(parents=True, exist_ok=True)
    Path(archive.path).write_text("source_file: unrelated\nrecords: []\n", encoding="utf-8")
    before = _files(config.root)

    with pytest.raises(promote.PromoteError) as caught:
        _guarded(config, prepared)

    message = str(caught.value)
    assert "promotion-intent-stale" in message and "archive" in message
    assert str(archive.expected_after) in message
    assert _files(config.root) == before


# --- archive retry ------------------------------------------------------------


def _archive_retry_fixture(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    """A live review whose every row is already in this run's own archive."""
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    wire = path.read_bytes()
    result = promotion_application.execute_promotion(config, _decide(config, path))
    assert result.state == "landed"
    path.write_bytes(wire)
    return config, path


def test_an_archive_retry_binds_its_frozen_receipt_and_writes_no_canonical_byte(
    tmp_path: Path,
) -> None:
    config, path = _archive_retry_fixture(tmp_path)
    canonical_before = _digest(config.normalized_file)

    prepared = _prepared(config, path)

    assert prepared.projected_state == "archive_retry"
    assert prepared.archive_retry_ids == ("word:話す:はなす",)
    assert prepared.landed_ids == ()
    assert [item.role for item in prepared.components] == ["archive", "live_staging"]
    archive = prepared.component("archive")
    assert archive is not None and not archive.writes
    assert prepared.receipt_id is not None

    result = _guarded(config, _roundtrip(prepared))

    assert result.state == "archive_retry"
    assert result.receipt_id == prepared.receipt_id
    assert not path.exists()
    assert _digest(config.normalized_file) == canonical_before


def test_execute_refuses_a_projected_archive_retry(tmp_path: Path) -> None:
    config, path = _archive_retry_fixture(tmp_path)
    decision = _decide(config, path)
    assert decision.state == "archive_retry"
    before = _files(config.root)

    with pytest.raises(
        promote.PromoteError, match="promotion-projection-not-executable"
    ):
        promotion_application.execute_promotion(config, replace(decision, projected=True))

    assert _files(config.root) == before


def test_a_resumed_archive_retry_reports_a_vector_that_already_landed(
    tmp_path: Path,
) -> None:
    config, path = _archive_retry_fixture(tmp_path)
    prepared = _roundtrip(_prepared(config, path))
    _land_component(prepared, "live_staging")
    after = _files(config.root)

    result = _guarded(config, prepared)

    assert result.state == "archive_retry"
    assert result.receipt_id == prepared.receipt_id
    assert _files(config.root) == after


def test_a_resumed_pattern_only_review_finishes_from_its_intent(
    tmp_path: Path,
) -> None:
    config, path = _fixture_for(tmp_path, "pattern_only")
    prepared = _roundtrip(_prepared(config, path))
    assert [item.role for item in prepared.components] == ["archive", "live_staging"]
    _land_component(prepared, "archive")

    result = _guarded(config, prepared)

    assert result.state == "pattern_only"
    assert not path.exists()
    archived, archived_meta = read_staging(config.staging_dir / "done" / path.name)
    assert archived == []
    assert archived_meta["reviewed_pattern_set"]["reviewed"] is True


def test_a_repeated_all_held_promotion_carries_no_live_payload(
    tmp_path: Path,
) -> None:
    """A second pass over a review nobody can promote writes nothing at all.

    The first pass records each row's hold reason. The second renders the same
    document, so its live component is bound and unwritten — and carries **no**
    payload, because a durable job document should not embed a whole staging
    file per part per attempt for a file that changes nothing.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, HELD_ONLY)
    first = _guarded(config, _roundtrip(_prepared(config, path)))
    assert first.state == "nothing_lands"
    settled = path.read_bytes()

    prepared = _prepared(config, path)

    assert prepared.projected_state == "nothing_lands"
    assert prepared.held_ids == ("word:食べ物:",)
    live = prepared.component("live_staging")
    assert live is not None
    assert not live.writes
    assert live.expected_before == live.expected_after
    assert live.after_text is None
    assert json.dumps(prepared.to_dict()).count("食べ物") < 5

    result = _guarded(config, _roundtrip(prepared))

    assert result.state == "nothing_lands"
    assert path.read_bytes() == settled


# --- §7.5 lock order, held through the proof and the effects ------------------


def _contender(
    target: Path, failures: list[BaseException]
) -> tuple[threading.Thread, threading.Event, threading.Event]:
    """A real second writer waiting for one exact janki lock.

    It takes the lock janki itself takes — not a stand-in — so "held" is proved
    by the thread being unable to enter, and "released" by the same thread
    entering afterwards. A thread that never ran would satisfy the first
    assertion for the wrong reason, so it reports when it started as well.
    """
    started = threading.Event()
    acquired = threading.Event()

    def run() -> None:
        try:
            started.set()
            with exclusive_path_lock(target):
                acquired.set()
        except BaseException as exc:  # pragma: no cover - reported by the test
            failures.append(exc)

    return (
        threading.Thread(target=run, name=f"contender-{target.name}", daemon=True),
        started,
        acquired,
    )


def test_the_intent_locks_take_the_ordinary_writer_order(tmp_path: Path) -> None:
    """Live review, archive, deck directory, canonical, ledger — §7.5's order.

    The ordinary writer acquires exactly these, in exactly this order, through
    `_finish_record_review` → `commit_canonical_state` → the canonical and
    ledger saves. Sorting the bound paths by name instead reverses it on an
    ordinary layout, so a recovery holding canonical and waiting for the live
    review could meet a promote holding the live review and waiting for
    canonical.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)

    prepared = _prepared(config, path)

    assert promotion_application._intent_lock_paths(config, prepared) == [
        Path(os.path.realpath(path)),
        Path(os.path.realpath(config.staging_dir / "done" / path.name)),
        Path(os.path.realpath(config.deck_dir)),
        Path(os.path.realpath(config.normalized_file)),
        Path(os.path.realpath(config.ledger_file)),
    ]
    # The two the writer takes itself are not taken twice: `exclusive_path_lock`
    # is not re-entrant.
    assert promotion_application._writer_lock_paths(config, prepared) == [
        Path(os.path.realpath(config.deck_dir)),
        Path(os.path.realpath(config.normalized_file)),
        Path(os.path.realpath(config.ledger_file)),
    ]


def test_a_pattern_only_intent_locks_only_the_files_it_writes(tmp_path: Path) -> None:
    """No deck proof, no canonical write, so neither lock is taken."""
    config, path = _fixture_for(tmp_path, "pattern_only")

    prepared = _prepared(config, path)

    assert promotion_application._intent_lock_paths(config, prepared) == [
        Path(os.path.realpath(path)),
        Path(os.path.realpath(config.staging_dir / "done" / path.name)),
    ]
    assert promotion_application._writer_lock_paths(config, prepared) == []


def test_recovery_holds_the_deck_lock_through_its_canonical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof and the write it authorizes happen under one deck lock.

    `_reprove_landing_authority_under_locks` asks whether exactly one
    configured study deck owns every landing row. Releasing the deck lock
    between that answer and the canonical replay lets a cooperating deck writer
    change the rules in between, so what reaches the collection is authorized
    by a repository that no longer exists.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    failures: list[BaseException] = []
    thread, started, acquired = _contender(config.deck_dir, failures)
    original = promotion_application._write_component
    at_seam: list[bool] = []

    def probe(component: staging.PreparedComponent) -> None:
        if component.role == "canonical":
            thread.start()
            assert started.wait(BOUND), "the contending deck writer never ran"
            at_seam.append(acquired.wait(0.25))
        original(component)

    monkeypatch.setattr(promotion_application, "_write_component", probe)
    try:
        recovery = promotion_application.recover_promotion_intent(config, prepared)
    finally:
        if thread.is_alive() or thread.ident is not None:
            thread.join(BOUND)

    assert not failures, failures
    assert at_seam == [False], "a deck writer entered before the canonical replay"
    assert acquired.wait(BOUND), "the contending deck writer never got the lock"
    assert not thread.is_alive()
    assert recovery.state == "landed"


def test_the_ordinary_promote_holds_its_writer_locks_across_the_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole vector is measured, and written, under §7.5's locks.

    `precheck` joins the deck directory, canonical and ledger before it
    measures anything and they stay held until the apply returns, so nothing
    can move a bound file between the measurement and the write. Proved from
    inside `commit_canonical_state`, which runs after the precheck and before
    the canonical write.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    decision = _decide(config, path)
    prepared = promotion_application.prepare_source_promotion(
        config, decision, part_name="part-01", now=FROZEN
    )
    failures: list[BaseException] = []
    contenders = {
        name: _contender(target, failures)
        for name, target in (
            ("deck", config.deck_dir),
            ("canonical", config.normalized_file),
            ("ledger", config.ledger_file),
        )
    }
    original = promotion_application.require_exact_deck_ownership
    at_seam: list[dict[str, bool]] = []

    def probe(*args: Any, **kwargs: Any) -> Any:
        for thread, started, _acquired in contenders.values():
            thread.start()
            assert started.wait(BOUND)
        at_seam.append(
            {
                name: acquired.wait(0.25)
                for name, (_thread, _started, acquired) in contenders.items()
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(promotion_application, "require_exact_deck_ownership", probe)
    try:
        result = _guarded(config, prepared, decision=decision)
    finally:
        for thread, _started, _acquired in contenders.values():
            if thread.ident is not None:
                thread.join(BOUND)

    assert not failures, failures
    assert at_seam == [{"deck": False, "canonical": False, "ledger": False}]
    for name, (thread, _started, acquired) in contenders.items():
        assert acquired.wait(BOUND), f"the {name} contender never got its lock"
        assert not thread.is_alive()
    assert result.state == "landed"


# --- §7.7 deck-input bindings, revalidated where the writer revalidates them ---


def _edit_deck(config: ProjectConfig) -> None:
    """Change the configured deck bytes without changing what it declares."""
    deck = next(config.deck_dir.glob("*.yaml"))
    deck.write_text(
        deck.read_text(encoding="utf-8") + "\n# the owner edited this deck\n",
        encoding="utf-8",
    )


def test_a_landing_intent_binds_the_deck_inputs_its_decision_was_taken_against(
    tmp_path: Path,
) -> None:
    """The pre-landing collection contribution, not the merged one.

    `_require_deck_inputs` asks what the collection contributed *before* this
    promotion, and by recovery time the canonical file may already hold the
    landing. Binding the merged ids instead would make the check refuse its own
    successful landing.
    """
    config = _config(tmp_path, [record(id="word:聞く:きく", expression="聞く", reading="きく")])
    path = staging_file(config.root, ONE_GOOD)

    prepared = _prepared(config, path)

    assert prepared.deck_revision == promotion_application._deck_configuration_revision(
        config
    )
    assert prepared.existing_ids == ("word:聞く:きく",)
    assert prepared.stored_ids == ("word:聞く:きく",)
    assert prepared.unreadable_decks == ()
    assert prepared.landed_ids == ("word:話す:はなす",)
    assert prepared.landed_ids[0] not in prepared.existing_ids
    assert _roundtrip(prepared) == prepared


def test_recovery_and_the_ordinary_writer_refuse_the_same_moved_deck_inputs(
    tmp_path: Path,
) -> None:
    """Recovery is not a way around a check the ordinary promote applies.

    Both are given the same repository: an authorized decision, and a deck file
    edited afterwards. Neither publishes anything. The ordinary writer refuses
    at the canonical commit seam, where it proves the deck inputs it decided
    against; recovery meets the same repository one step earlier, because §7.7
    sends this unstarted part back through a fresh decision and the deck
    binding is one of the things that comparison covers. The seam is still
    asked — see the resumed-race case below, which is the window no comparison
    can see.
    """
    refusals: dict[str, str] = {}
    for mode in ("ordinary", "recovery"):
        root = tmp_path / mode
        root.mkdir()
        config = _config(root)
        path = staging_file(config.root, ONE_GOOD)
        decision = _decide(config, path)
        prepared = _roundtrip(
            promotion_application.prepare_source_promotion(
                config, decision, part_name="part-01", now=FROZEN
            )
        )
        _edit_deck(config)
        before = _files(config.root)

        with pytest.raises(promote.PromoteError) as caught:
            if mode == "ordinary":
                promotion_application.execute_promotion(config, decision)
            else:
                promotion_application.recover_promotion_intent(config, prepared)

        refusals[mode] = str(caught.value)
        assert _files(config.root) == before

    assert "promotion-input-stale" in refusals["ordinary"]
    assert "promotion-intent-stale" in refusals["recovery"]
    assert "deck configuration" in refusals["recovery"]


def test_recovery_reasks_the_deck_inputs_after_it_has_decided_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window the fresh comparison cannot see, closed under the deck lock.

    A re-decision reads the decks before the component locks are taken, because
    `decide_promotion` reads the same paths and `exclusive_path_lock` is not
    re-entrant. A deck edited in that window agrees with everything the fresh
    plan compared, so the only thing left to refuse it is
    `_require_deck_inputs`, asked at the canonical seam under the deck lock
    this recovery holds through the write it authorizes.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    original = promotion_application._redecide_for_intent

    def decide_then_move_the_decks(*args: Any, **kwargs: Any) -> Any:
        decision = original(*args, **kwargs)
        _edit_deck(config)
        return decision

    monkeypatch.setattr(
        promotion_application, "_redecide_for_intent", decide_then_move_the_decks
    )
    canonical_before = _digest(config.normalized_file)
    ledger_before = _digest(config.ledger_file)

    with pytest.raises(promote.PromoteError, match="promotion-input-stale"):
        promotion_application.recover_promotion_intent(config, prepared)

    assert _digest(config.normalized_file) == canonical_before
    assert _digest(config.ledger_file) == ledger_before
    assert path.exists(), "the live review survives a refusal"
    assert not (config.staging_dir / "done" / path.name).exists()


def test_a_resumed_unstarted_part_refuses_a_changed_deck_configuration(
    tmp_path: Path,
) -> None:
    """§7.7: an unstarted intent re-decides, and the comparison covers the decks.

    No component digest can catch this — a deck file is not one of this
    intent's paths — so the deck-input binding is compared directly.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))
    _edit_deck(config)
    before = _files(config.root)

    with pytest.raises(promote.PromoteError) as caught:
        _guarded(config, prepared)

    message = str(caught.value)
    assert "promotion-intent-stale" in message and "deck configuration" in message
    assert _files(config.root) == before


def test_recovery_finishes_a_landing_whose_decks_never_moved(tmp_path: Path) -> None:
    """The deck-input re-ask must not refuse an untouched repository."""
    config = _config(tmp_path, [record(id="word:聞く:きく", expression="聞く", reading="きく")])
    path = staging_file(config.root, ONE_GOOD)
    prepared = _roundtrip(_prepared(config, path))

    recovery = promotion_application.recover_promotion_intent(config, prepared)

    assert recovery.state == "landed"
    assert recovery.finished == ("canonical", "ledger", "archive", "live_staging")
    assert sorted(item.id for item in load_records(config.normalized_file)) == [
        "word:聞く:きく",
        "word:話す:はなす",
    ]


# --- §7.7 a resumed result is the ordinary writer's result --------------------


def _nothing_lands_with_an_archive_retry(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    """One row already in this run's archive, one row the witness holds back."""
    config = _config(tmp_path)
    path = staging_file(config.root, LANDS_AND_HOLDS)
    wire = path.read_bytes()
    assert (
        promotion_application.execute_promotion(config, _decide(config, path)).state
        == "landed"
    )
    path.write_bytes(wire)
    return config, path


def test_a_resumed_nothing_lands_with_a_retry_reports_its_finished_intent(
    tmp_path: Path,
) -> None:
    """§7.13's crash-after-archive shape, resumed through `apply`.

    The held remainder was written and the process died before the coordinator
    recorded anything. The part is complete, and reporting it must not depend
    on a return value nobody saved — nor raise out of the result invariant for
    a transaction that really did archive a retry.
    """
    config, path = _nothing_lands_with_an_archive_retry(tmp_path)
    prepared = _roundtrip(_prepared(config, path))
    assert prepared.projected_state == "nothing_lands"
    assert prepared.archive_retry_ids == ("word:話す:はなす",)
    assert prepared.held_ids == ("word:食べ物:",)
    _land_component(prepared, "live_staging")
    after = _files(config.root)

    result = _guarded(config, prepared)

    assert result.state == "nothing_lands"
    assert result.archive_path == (config.staging_dir / "done" / path.name).resolve()
    assert [item.id for item in result.retry_records] == ["word:話す:はなす"]
    assert [item.id for item in result.held] == ["word:食べ物:"]
    assert result.removed == 1
    assert result.receipt_id == prepared.receipt_id
    assert [dict(row)["id"] for row in prepared.retry_rows] == ["word:話す:はなす"]
    assert _files(config.root) == after


def test_a_resumed_empty_live_archive_retry_reports_what_the_writer_reports(
    tmp_path: Path,
) -> None:
    """An empty live review beside a full archive prunes nothing.

    Its already-archived ids are every row in the archive, so counting *those*
    as removals reports a prune that never happened and denies the empty-live
    case the CLI prints differently.
    """
    config, path = _archive_retry_fixture(tmp_path)
    meta = dict(read_staging(path)[1])
    write_staging(path, [], meta, force=True)
    decision = _decide(config, path)
    assert decision.state == "archive_retry"
    assert decision.already_archived == ("word:話す:はなす",)
    prepared = _roundtrip(
        promotion_application.prepare_source_promotion(
            config, decision, part_name="part-01", now=FROZEN
        )
    )
    _land_component(prepared, "live_staging")

    resumed = _guarded(config, prepared)

    assert resumed.state == "archive_retry"
    assert resumed.empty_live_retry is True
    assert resumed.removed == 0
    assert resumed.retry_records == ()
    assert resumed.receipt_id == prepared.receipt_id
    # Nothing was pruned, so the intent froze no retry rows either.
    assert prepared.retry_rows == ()


def test_a_resumed_landing_reports_its_held_remainder_and_pruned_retries(
    tmp_path: Path,
) -> None:
    """`removed` is the writer's own count: the rows that leave the live file."""
    config = _config(tmp_path)
    path = staging_file(config.root, LANDS_AND_HOLDS)
    prepared = _roundtrip(_prepared(config, path))
    for role in ("canonical", "ledger", "archive"):
        _land_component(prepared, role)

    result = _guarded(config, prepared)

    assert result.state == "landed"
    assert [item.id for item in result.promoted] == ["word:話す:はなす"]
    assert [item.id for item in result.held] == ["word:食べ物:"]
    assert promote.HOLD_MISSING_READING in staging.annotations(result.held[0])[
        "hold_reason"
    ]
    assert result.removed == 1
    assert result.retry_records == ()


# --- one disposition definition, projected and prepared -----------------------


def test_an_archive_retry_intent_discloses_exactly_what_the_fold_projects(
    tmp_path: Path,
) -> None:
    """§7.7's equality is an identity, not two derivations kept in step.

    A row already in this run's own archive is disclosed under `archive_retry`.
    Reporting it as `excluded` as well names a row that already landed as one
    still needing an owner's exclusion or deferral decision, and makes the
    coordinator's projected-versus-intent comparison refuse a legitimate
    resume.
    """
    config, path = _archive_retry_fixture(tmp_path)
    part = promotion_application.SourcePromotionPart(
        part_name="part-01",
        staging_path=path,
        expected_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    fold = promotion_application.fold_source_extraction_promotions(
        config,
        [part],
        collection=list(load_records(config.normalized_file)),
        collection_revision=records_revision(config.normalized_file),
        pattern_store={},
        witness=client_for(hanasu_jpdb()),
    )
    prepared = _prepared(config, path)

    projected = fold.parts[0]
    assert projected.state == "archive_retry" == prepared.projected_state
    assert projected.archive_retry_ids == ("word:話す:はなす",)
    assert prepared.archive_retry_ids == projected.archive_retry_ids
    assert projected.excluded_ids == ()
    assert prepared.excluded_ids == projected.excluded_ids
    assert prepared.landed_ids == projected.landed_ids == ()
    assert prepared.held_ids == projected.held_ids == ()


def test_a_landing_intent_without_its_deck_binding_is_refused(tmp_path: Path) -> None:
    """An intent that carries no deck inputs cannot be finished as a landing.

    The binding is what makes `_require_deck_inputs` answerable at recovery, so
    an intent missing it is unreadable rather than exempt — otherwise dropping
    the field would be a way to skip the check.
    """
    config = _config(tmp_path)
    path = staging_file(config.root, ONE_GOOD)
    prepared = _prepared(config, path)
    wire = prepared.to_dict()
    wire["deck_revision"] = ""
    stripped = promotion_application.PreparedSourcePromotion.from_dict(wire)
    before = _files(config.root)

    with pytest.raises(promote.PromoteError, match="promotion-intent-invalid"):
        promotion_application.recover_promotion_intent(config, stripped)

    assert _files(config.root) == before
