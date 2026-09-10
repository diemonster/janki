"""Many prepared corpus parts, sent as one bounded, one-consent batch.

Every provider here is faked (IMPLEMENTATION_PLAN rule 6): none of these tests
is about what Claude answers. They are about the parts of a batch that are the
project's own — that every child is revalidated before *any* reservation, that
the journal rather than a manifest is what permits a send, that one child's
bad answer does not cost its siblings their reserved work, and that a retry is
an explicit discard decision with fresh operation identities.

The sources are ASCII stand-ins. What matters about them is their bytes and
their names, because those are what the request identity is computed over.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import claude_client, operations, patterns, staging
from japanese_anki.application import extraction, extraction_batch
from japanese_anki.application.extraction_batch import (
    ExtractionBatchPlan,
    SourcePartLineage,
    dispatch_extraction_batch,
    extraction_batch_status,
    list_extraction_batches,
    plan_extraction_batch,
    plan_extraction_batch_retry,
    resume_extraction_batch,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

SCENARIO = "table_exhaustive"


def _answer() -> dict[str, Any]:
    """One fixture answer in the shape the extraction schema validates."""
    return json.loads((RESPONSES / f"{SCENARIO}.json").read_text(encoding="utf-8"))


def _stream(answer: Any) -> bytes:
    """A Claude Code stream whose final result carries that structured answer."""
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any Anthropic API use a loud failure rather than a silent bill."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _vanishing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """The subscription transport refuses after the batch was reserved.

    The refusal a logged-out login or a removed CLI produces, at the one point
    a batch can meet it that a single call cannot: after its children hold
    authority. Nothing may fall back to the metered API, and no child may be
    sent.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the subscription CLI is not available")

    monkeypatch.setattr(extraction_batch, "prepare_extraction_transport", refuse)


def _project(tmp_path: Path) -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _source(config: ProjectConfig, name: str, body: bytes = b"page one") -> Path:
    """One corpus source, already durable, as every extraction surface requires."""
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _runner() -> FakeClaudeRunner:
    return FakeClaudeRunner(reply=_stream(_answer()))


def _plan(
    config: ProjectConfig,
    sources: list[Path],
    runner: FakeClaudeRunner,
    **kwargs: Any,
) -> ExtractionBatchPlan:
    return plan_extraction_batch(
        config,
        sources,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        **kwargs,
    )


def _dispatch(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    runner: FakeClaudeRunner,
    *,
    spawn: Any = None,
    progress: Any = None,
) -> Any:
    return dispatch_extraction_batch(
        config,
        plan,
        progress=progress,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn if spawn is None else spawn,
    )


def _states(config: ProjectConfig) -> dict[str, str]:
    journal = operations.OperationJournal.load(config.operations_file)
    return {key: value.state for key, value in journal.operations.items()}


def test_a_batch_plan_binds_every_child_without_touching_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning names what would be sent. It reserves nothing and writes nothing."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]

    plan = _plan(config, sources, _runner())

    assert [child.index for child in plan.children] == [1, 2]
    assert [child.source for child in plan.children] == sources
    assert plan.provider == "claude-code"
    assert plan.model == "claude-opus-5"
    assert plan.concurrency_limit == 2
    assert plan.retry_of == ""
    assert plan.discards == ()
    # Nothing is authorized and no durable manifest exists yet: a plan is a
    # value a surface renders, not a receipt.
    assert not config.operations_file.exists()
    assert not (config.operations_file.parent / "extraction_batches").exists()
    # The manifest bytes are the plan's own exact serialization.
    assert plan.manifest_sha256 == __import__("hashlib").sha256(
        plan.manifest_bytes
    ).hexdigest()
    assert ExtractionBatchPlan.from_dict(json.loads(plan.manifest_bytes)) == plan


def test_equal_source_hashes_are_legal_and_a_repeated_source_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two pages may hold identical bytes; the same request must not ride twice."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    same = b"identical page bytes"
    first = _source(config, "page-01.pdf", same)
    second = _source(config, "page-02.pdf", same)

    plan = _plan(config, [first, second], _runner())

    assert plan.children[0].source_sha256 == plan.children[1].source_sha256
    assert (
        plan.children[0].request_fingerprint != plan.children[1].request_fingerprint
    )
    assert plan.children[0].staging_path != plan.children[1].staging_path

    with pytest.raises(JankiError) as caught:
        _plan(config, [first, first], _runner())
    assert "twice" in str(caught.value)
    assert not config.operations_file.exists()


def test_lineage_records_the_parent_without_replacing_the_child_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part's own path and hash stay its own; the parent is documentary."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "part-1.pdf", b"first"), _source(config, "part-2.pdf", b"second")]
    lineage = (
        SourcePartLineage("book.pdf", "a" * 64, "pages 1-2"),
        SourcePartLineage("book.pdf", "a" * 64, "pages 3-4"),
    )

    plan = _plan(config, sources, _runner(), lineage=lineage)

    for child, source, part in zip(plan.children, sources, lineage, strict=True):
        assert child.lineage == part
        assert child.source == source
        assert child.source_sha256 != part.parent_sha256
    assert ExtractionBatchPlan.from_dict(json.loads(plan.manifest_bytes)) == plan


def test_drift_in_any_child_reserves_nothing_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One moved source cancels the whole batch before a single call."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    first = _source(config, "one.pdf", b"one")
    second = _source(config, "two.pdf", b"two")
    runner = _runner()
    plan = _plan(config, [first, second], runner)

    # The second page is re-scanned between the consent page and the click.
    second.write_bytes(b"two, corrected")

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a provider was dispatched after drift")

    with pytest.raises(JankiError) as caught:
        _dispatch(config, plan, runner, spawn=never)

    assert "changed" in str(caught.value)
    assert not config.operations_file.exists()
    assert runner.spawned == []


def test_the_whole_batch_runs_under_its_own_bounded_worker_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four children, two slots: the limit is real, and every child commits."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [
        _source(config, f"page-{index}.pdf", f"page {index}".encode("ascii"))
        for index in range(1, 5)
    ]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=2)

    lock = threading.Lock()
    live = 0
    peak = 0

    def spawn(command: list[str], **kwargs: Any) -> Any:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            time.sleep(0.05)
            return runner.spawn(command, **kwargs)
        finally:
            with lock:
                live -= 1

    seen: list[Any] = []
    outcome = _dispatch(config, plan, runner, spawn=spawn, progress=seen.append)

    assert peak == 2
    assert outcome.committed_count == 4
    assert outcome.failed_count == 0
    assert outcome.pending_count == 0
    assert sorted(child.index for child in outcome.children) == [1, 2, 3, 4]
    assert all(child.state == "committed" for child in outcome.children)
    assert all(child.staging_path.exists() for child in outcome.children)
    assert all(child.bookkeeping_complete for child in outcome.children)
    # Every child keeps its own staging document; nothing is merged on write.
    assert len({child.staging_path for child in outcome.children}) == 4
    assert {event.index for event in seen} == {1, 2, 3, 4}
    assert all(event.operation_id for event in seen)


def test_one_childs_failure_leaves_its_siblings_and_their_work_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unusable answer is that child's problem, not the batch's."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)
    broken = FakeClaudeRunner(reply=b"not a stream at all\n")
    calls = {"count": 0}

    def spawn(command: list[str], **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            return broken.spawn(command, **kwargs)
        return runner.spawn(command, **kwargs)

    outcome = _dispatch(config, plan, runner, spawn=spawn)

    first, second = outcome.children
    assert first.state != "committed"
    assert first.error
    assert second.state == "committed"
    assert second.staging_path.exists()
    assert second.records > 0
    assert outcome.committed_count == 1


def test_a_captured_answer_is_recovered_on_resume_without_a_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paid bytes are the answer. Resume reads them; it never sends again."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    real_write = extraction.write_staging_under_lock

    def refuse_second(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.startswith("two"):
            raise staging.StagingError("the disk went away")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_second)
    first_pass = _dispatch(config, plan, runner)
    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)

    stalled = first_pass.children[1]
    assert stalled.state == "result_captured"
    assert not stalled.staging_path.exists()

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resume dispatched a provider for a captured answer")

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=never,
    )

    assert resumed.committed_count == 2
    assert resumed.children[1].state == "committed"
    assert resumed.children[1].staging_path.exists()


def test_resume_never_reclaims_committed_in_flight_or_unknown_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a child still holding unused authority may be dispatched again."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [
        _source(config, f"page-{index}.pdf", f"page {index}".encode("ascii"))
        for index in range(1, 4)
    ]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=2)

    _vanishing_cli(monkeypatch)
    # Reserve the whole batch, then refuse every transport: the reservation is
    # real, nothing was sent, and every child is left holding its authority.
    paused = _dispatch(config, plan, runner)
    assert paused.pending_count == 3
    assert runner.spawned == []
    monkeypatch.undo()
    _no_api(monkeypatch)

    journal = operations.OperationJournal.load(config.operations_file)
    for position in (0, 1):
        journal.claim_batch_dispatch(
            plan.children[position].operation_id,
            batch_id=plan.batch_id,
            request_fp=plan.children[position].request_fingerprint,
            manifest_sha256=plan.manifest_sha256,
        )
    journal.advance(plan.children[1].operation_id, "outcome_unknown")

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resume dispatched an in-flight or settled child")

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=never,
    )

    states = {child.index: child.state for child in resumed.children}
    assert states[1] == "dispatching"
    assert states[2] == "outcome_unknown"
    # The third child still holds unused authority, but the unknown outcome
    # occupies a slot, so its work waits for a person rather than spinning.
    assert states[3] == "authorized"
    assert resumed.unknown_count == 1
    assert resumed.pending_count == 2


def test_a_selected_retry_takes_fresh_identities_and_discloses_its_discard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same request may ride again, but never on the old authority."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)
    broken = FakeClaudeRunner(reply=b"not a stream at all\n")
    calls = {"count": 0}

    def spawn(command: list[str], **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            return broken.spawn(command, **kwargs)
        return runner.spawn(command, **kwargs)

    first_pass = _dispatch(config, plan, runner, spawn=spawn)
    assert first_pass.children[0].state != "committed"

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    assert retry.retry_of == plan.batch_id
    assert [child.index for child in retry.children] == [1]
    assert retry.children[0].operation_id != plan.children[0].operation_id
    # The request itself is unchanged: retrying is about authority, not content.
    assert (
        retry.children[0].request_fingerprint
        == plan.children[0].request_fingerprint
    )
    # The discards are the journal's own exact snapshots, not a summary of
    # them: they are both what a person is asked to throw away and the guard
    # `forget` compares under its lock.
    assert [discard.operation_id for discard in retry.discards] == [
        plan.children[0].operation_id
    ]
    assert all(
        isinstance(discard, operations.Operation) for discard in retry.discards
    )

    # Retrying a committed child is never automatic.
    with pytest.raises(JankiError) as caught:
        plan_extraction_batch_retry(
            config,
            plan.batch_id,
            [2],
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )
    assert "committed" in str(caught.value)

    outcome = _dispatch(config, retry, runner)
    assert outcome.committed_count == 1
    states = _states(config)
    assert plan.children[0].operation_id not in states
    assert states[retry.children[0].operation_id] == "committed"
    assert states[plan.children[1].operation_id] == "committed"


def test_a_discard_that_moved_stops_the_retry_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot in the plan is what is being discarded, or nothing is."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    _vanishing_cli(monkeypatch)
    _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance(plan.children[0].operation_id, "failed_before_send")

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    # Somebody settles the same entry by hand between the plan and the click.
    operations.OperationJournal.load(config.operations_file).forget(
        [plan.children[0].operation_id]
    )

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a retry dispatched after its discard had moved")

    with pytest.raises(JankiError):
        _dispatch(config, retry, runner, spawn=never)

    states = _states(config)
    assert retry.children[0].operation_id not in states


def test_a_committed_child_with_a_failed_pattern_write_is_not_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging succeeded. The bookkeeping owes a person, not a second call."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise patterns.PatternError("the pattern store is unwritable")

    monkeypatch.setattr(patterns, "save_store_under_lock", refuse)
    outcome = _dispatch(config, plan, runner)

    child = outcome.children[0]
    assert child.state == "committed"
    assert child.staging_path.exists()
    assert not child.bookkeeping_complete
    assert child.error

    with pytest.raises(JankiError) as caught:
        plan_extraction_batch_retry(
            config,
            plan.batch_id,
            [1],
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )
    assert "committed" in str(caught.value)
    assert child.staging_path.exists()


def test_status_surfaces_a_manifest_that_was_never_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt written before the reservation is a state, not a permission."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power during the reservation")

    monkeypatch.setattr(
        operations.OperationJournal, "authorize_batch", die, raising=False
    )
    with pytest.raises(JankiError):
        _dispatch(config, plan, runner)

    manifest = (
        config.operations_file.parent / "extraction_batches" / f"{plan.batch_id}.json"
    )
    assert manifest.exists()
    assert not config.operations_file.exists()

    status = extraction_batch_status(config, plan.batch_id)
    assert status.batch_id == plan.batch_id
    assert status.pending_count == 1
    assert status.committed_count == 0
    assert [batch.batch_id for batch in list_extraction_batches(config)] == [
        plan.batch_id
    ]
    # Named, not empty: "no reservation exists" is a state a surface renders,
    # and it is a different fact from "reserved and not yet sent".
    assert status.children[0].state == "unreserved"
    assert status.children[0].records is None
    # The owner's dispatch was confirmed and validated before the interruption,
    # so this one *is* resumable — unlike a manifest alone.
    assert status.resume_available
    assert status.resume_refusal == ""


def test_the_consent_fingerprint_covers_the_whole_confirmed_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fingerprint for one confirmed action, not for its request text.

    A surface binds a confirmation to this value, so everything the owner is
    agreeing to has to be inside it: how many calls run at once, which reviews
    would be destroyed, which entries would be retired, and which fresh
    authorities would be written. A retry sends the identical request and is a
    different decision.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan = _plan(config, sources, _runner(), concurrency_limit=2)

    # Exactly serializable: the same action reads back with the same identity.
    restored = ExtractionBatchPlan.from_dict(json.loads(plan.manifest_bytes))
    assert restored == plan
    assert restored.fingerprint == plan.fingerprint

    assert replace(plan, concurrency_limit=1).fingerprint != plan.fingerprint
    assert replace(plan, retry_of="x" * 8).fingerprint != plan.fingerprint
    reidentified = replace(
        plan,
        children=(
            replace(plan.children[0], operation_id="00000000-0000-4000-8000-000000000000"),
            plan.children[1],
        ),
    )
    assert reidentified.fingerprint != plan.fingerprint
    rereviewed = replace(
        plan,
        children=(
            replace(
                plan.children[0],
                expectation=replace(
                    plan.children[0].expectation,
                    replacement_revision=extraction.ExtractionRevision(
                        staging_sha256="b" * 64,
                        pattern_entry_sha256=None,
                        pattern_reviewed=None,
                        pattern_has_patterns=None,
                    ),
                    replacement_confirmed=True,
                ),
            ),
            plan.children[1],
        ),
    )
    assert rereviewed.fingerprint != plan.fingerprint


def test_a_retry_is_a_different_confirmation_from_the_call_it_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same request, fresh authority, and therefore a different consent."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    _vanishing_cli(monkeypatch)
    _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    operations.OperationJournal.load(config.operations_file).advance(
        plan.children[0].operation_id, "failed_before_send"
    )

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    assert (
        retry.children[0].request_fingerprint == plan.children[0].request_fingerprint
    )
    assert retry.fingerprint != plan.fingerprint
    assert retry.discards[0] == operations.OperationJournal.load(
        config.operations_file
    ).operations[plan.children[0].operation_id]
    # The snapshot survives the manifest exactly, because it is what `forget`
    # is guarded against.
    restored = ExtractionBatchPlan.from_dict(json.loads(retry.manifest_bytes))
    assert restored.discards == retry.discards


def test_a_retired_member_is_never_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A membership with no entry left says what happened, and refuses more."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner)
    _dispatch(config, plan, runner)

    operations.OperationJournal.load(config.operations_file).forget(
        [plan.children[0].operation_id]
    )

    status = extraction_batch_status(config, plan.batch_id)
    assert status.children[0].state == "retired"
    assert status.children[0].records is None

    with pytest.raises(JankiError) as caught:
        plan_extraction_batch_retry(
            config,
            plan.batch_id,
            [1],
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )
    assert "retired" in str(caught.value)
    assert plan.children[0].staging_path.exists()


def test_captured_recovery_needs_neither_prompts_nor_a_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid answer is recoverable from what was saved beside it. Full stop.

    The request that produced it is in the capture, so recovery must not ask
    today's prompt files what was sent or probe a login that could refuse. A
    person who logged out, or edited a prompt, still owns the answer they paid
    for.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    real_write = extraction.write_staging_under_lock

    def refuse_second(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.startswith("two"):
            raise staging.StagingError("the disk went away")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_second)
    first_pass = _dispatch(config, plan, runner)
    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)
    assert first_pass.children[1].state == "result_captured"

    # The prompts are gone and the CLI would refuse. Neither is needed.
    shutil.rmtree(config.root / "prompts")
    already_sent = len(runner.spawned)

    def never_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("captured recovery probed the provider")

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=never_probe,
        provider_which=never_probe,
        provider_spawn=never_probe,
    )

    assert resumed.children[1].state == "committed"
    assert resumed.children[1].staging_path.exists()
    assert resumed.committed_count == 2
    # Recovery is not a send: the paid call it finishes was the first pass's.
    assert len(runner.spawned) == already_sent


def test_a_spawn_failure_never_returns_a_claimed_child_to_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authority is one-use. A call that may have gone stays gone."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the CLI died with the request already written")

    outcome = _dispatch(config, plan, runner, spawn=die)

    assert outcome.children[0].state == "outcome_unknown"
    assert outcome.unknown_count == 1
    assert _states(config)[plan.children[0].operation_id] == "outcome_unknown"

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resume redispatched an unknown outcome")

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=never,
    )
    assert resumed.children[0].state == "outcome_unknown"
    assert not resumed.resume_available
    assert resumed.resume_refusal


def test_a_request_manifest_alone_never_authorizes_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record of what would be sent is not a record that anyone said send it."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power writing the receipt")

    monkeypatch.setattr(extraction_batch, "_write_execution_receipt", die)
    with pytest.raises(JankiError):
        _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    store = config.operations_file.parent / "extraction_batches"
    assert (store / f"{plan.batch_id}.json").exists()
    assert not (store / f"{plan.batch_id}.execution.json").exists()

    status = extraction_batch_status(config, plan.batch_id)
    assert not status.resume_available
    assert status.resume_refusal

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an unconfirmed manifest authorized a send")

    with pytest.raises(JankiError):
        resume_extraction_batch(
            config,
            plan.batch_id,
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
            provider_spawn=never,
        )

    assert not config.operations_file.exists()
    assert runner.spawned == []


def test_an_ended_unknown_outcome_may_be_retried_as_a_fresh_exact_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner ended a live call; the request may be made again, freshly.

    `end` is what turns a call nobody can account for into a terminal record,
    and nothing can end it twice. What a retry owes that record is disclosure:
    the exact snapshot, including that money may already have gone, thrown
    away by the same confirmation that authorizes the new call.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the CLI died with the request already written")

    _dispatch(config, plan, runner, spawn=die)
    journal = operations.OperationJournal.load(config.operations_file)
    ended = journal.operations[plan.children[0].operation_id]
    assert ended.state == "outcome_unknown"
    assert ended.money_may_have_been_spent
    before = _states(config)

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    # Planning is a value, not a decision: nothing moved.
    assert _states(config) == before
    assert retry.discards == (ended,)
    assert retry.discards[0].state == "outcome_unknown"
    assert retry.discards[0].money_may_have_been_spent
    assert retry.children[0].operation_id != plan.children[0].operation_id
    assert (
        retry.children[0].request_fingerprint == plan.children[0].request_fingerprint
    )

    outcome = _dispatch(config, retry, runner)
    assert outcome.committed_count == 1
    after = _states(config)
    assert plan.children[0].operation_id not in after
    assert after[retry.children[0].operation_id] == "committed"


def test_an_interrupted_confirmed_retry_finishes_its_decisions_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between two retirements, and before the reservation, recovers.

    There is no atomic retire-and-reserve, so the confirmed intent is durable
    before either happens. Resuming finishes exactly the decisions that
    confirmation named: the retirement that already landed is recognized
    rather than forgotten twice, the one that did not is completed, the
    unrelated success is not touched, and each fresh child is sent once.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [
        _source(config, "one.pdf", b"one"),
        _source(config, "two.pdf", b"two"),
        _source(config, "three.pdf", b"three"),
    ]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    _vanishing_cli(monkeypatch)
    _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance(plan.children[0].operation_id, "failed_before_send")
    journal.advance(plan.children[1].operation_id, "failed_before_send")

    # The third source is unrelated work that succeeded, and must stay so.
    resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    succeeded = operations.OperationJournal.load(config.operations_file).operations[
        plan.children[2].operation_id
    ]
    assert succeeded.state == "committed"
    assert plan.children[2].staging_path.exists()

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1, 2],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    assert len(retry.discards) == 2

    real_forget = operations.OperationJournal.forget
    forgotten: list[str] = []

    def forget_once(self: Any, operation_ids: Any, **kwargs: Any) -> int:
        wanted = list(operation_ids)
        if forgotten:
            raise JankiError("the machine lost power between two retirements")
        forgotten.extend(wanted)
        return real_forget(self, wanted, **kwargs)

    monkeypatch.setattr(operations.OperationJournal, "forget", forget_once)
    sent_before = len(runner.spawned)
    with pytest.raises(JankiError):
        _dispatch(config, retry, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    interrupted = _states(config)
    assert forgotten == [retry.discards[0].operation_id]
    assert retry.discards[0].operation_id not in interrupted
    assert retry.discards[1].operation_id in interrupted
    assert all(
        child.operation_id not in interrupted for child in retry.children
    )
    assert len(runner.spawned) == sent_before

    resumed = resume_extraction_batch(
        config,
        retry.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert resumed.committed_count == 2
    assert len(runner.spawned) == sent_before + 2
    finished = operations.OperationJournal.load(config.operations_file)
    for child in retry.children:
        assert finished.operations[child.operation_id].state == "committed"
    for discard in retry.discards:
        assert discard.operation_id not in finished.operations
    # The unrelated success is exactly as it was.
    assert finished.operations[plan.children[2].operation_id] == succeeded


def test_two_resumes_of_one_execution_reserve_and_send_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal is the one-use gate, so the loser reports rather than sends."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power before the reservation")

    monkeypatch.setattr(extraction_batch, "_reserve_batch", die)
    with pytest.raises(JankiError):
        _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    def resume() -> Any:
        return resume_extraction_batch(
            config,
            plan.batch_id,
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
            provider_spawn=runner.spawn,
        )

    first = resume()
    assert first.committed_count == 1
    assert len(runner.spawned) == 1

    # The second caller finds the work done and says so, without claiming
    # anything: one authority, consumed once.
    second = resume()
    assert second.committed_count == 1
    assert not second.resume_available
    assert second.resume_refusal
    assert len(runner.spawned) == 1


def test_the_same_dispatch_is_idempotent_and_divergent_bytes_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One batch id names one exact confirmed action, or nothing at all."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power before the reservation")

    monkeypatch.setattr(extraction_batch, "_reserve_batch", die)
    with pytest.raises(JankiError):
        _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    # The same exact action, dispatched again, is the same action.
    outcome = _dispatch(config, plan, runner)
    assert outcome.committed_count == 1

    manifest = config.operations_file.parent / "extraction_batches" / (
        f"{plan.batch_id}.json"
    )
    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["concurrency_limit"] = 7
    manifest.write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(JankiError) as caught:
        _dispatch(config, plan, runner)
    assert "changed" in str(caught.value) or "differ" in str(caught.value)


def test_an_exact_redispatch_after_partial_cleanup_finishes_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same confirmed action, sent again, is the same action.

    A crash between one retirement and the reservation leaves a batch whose
    first discard is already gone. Handing that exact plan back to dispatch
    must finish it, not refuse it: the snapshot it cannot find any more is one
    this very confirmation retired.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=2)

    _vanishing_cli(monkeypatch)
    _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance(plan.children[0].operation_id, "failed_before_send")
    journal.advance(plan.children[1].operation_id, "failed_before_send")

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1, 2],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def die(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power before the reservation")

    monkeypatch.setattr(extraction_batch, "_reserve_batch", die)
    with pytest.raises(JankiError):
        _dispatch(config, retry, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    assert all(
        discard.operation_id
        not in operations.OperationJournal.load(config.operations_file).operations
        for discard in retry.discards
    )

    outcome = _dispatch(config, retry, runner)

    assert outcome.committed_count == 2
    assert len(runner.spawned) == 2
    finished = operations.OperationJournal.load(config.operations_file)
    for child in retry.children:
        assert finished.operations[child.operation_id].state == "committed"
    assert len(finished.batches[retry.batch_id].child_operation_ids) == 2


def test_an_interruption_inside_a_retirement_resumes_that_same_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable cleanup decision is finished, never replaced by a new one.

    `forget` writes its exact deletion intent before it deletes anything, so an
    interruption leaves an entry that has moved on from the snapshot the plan
    holds — a newer `updated_at`, a recorded cleanup, possibly an exact spool
    extension the journal legitimately adopted. Resuming has to continue *that*
    decision under the journal's own compare-and-swap, not force a fresh one.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the CLI died with the request already written")

    _dispatch(config, plan, runner, spawn=die)
    original = operations.OperationJournal.load(config.operations_file).operations[
        plan.children[0].operation_id
    ]
    assert original.state == "outcome_unknown"

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def stop_cleanup(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power mid-deletion")

    monkeypatch.setattr(operations, "_retire_artifact", stop_cleanup)
    with pytest.raises(JankiError):
        _dispatch(config, retry, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    interrupted = operations.OperationJournal.load(config.operations_file).operations[
        original.operation_id
    ]
    assert interrupted.cleanup is not None
    assert interrupted != original
    assert interrupted.request_fp == original.request_fp

    outcome = _dispatch(config, retry, runner)

    assert outcome.committed_count == 1
    finished = operations.OperationJournal.load(config.operations_file)
    assert original.operation_id not in finished.operations
    assert finished.operations[retry.children[0].operation_id].state == "committed"


def test_a_cleanup_bearing_entry_that_is_not_the_confirmed_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a retirement is only ever the retirement that was confirmed."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def die(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the CLI died with the request already written")

    _dispatch(config, plan, runner, spawn=die)
    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def stop_cleanup(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the machine lost power mid-deletion")

    monkeypatch.setattr(operations, "_retire_artifact", stop_cleanup)
    with pytest.raises(JankiError):
        _dispatch(config, retry, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)

    # Something rewrote that row's request identity while cleanup was pending.
    # It is no longer the entry anybody agreed to throw away.
    wire = json.loads(config.operations_file.read_text(encoding="utf-8"))
    wire["operations"][plan.children[0].operation_id]["request_fp"] = "f" * 64
    config.operations_file.write_text(
        json.dumps(wire, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(JankiError) as caught:
        _dispatch(config, retry, runner)
    assert "changed" in str(caught.value) or "different" in str(caught.value)
    assert runner.spawned == []
    still = operations.OperationJournal.load(config.operations_file)
    assert plan.children[0].operation_id in still.operations
    assert retry.children[0].operation_id not in still.operations


def test_a_saved_answer_is_recovered_while_a_sibling_cannot_be_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logged-out machine cannot cost somebody an answer they already bought.

    One child's reply is on disk; the other has never been sent and its login
    now refuses. Recovering the first needs no provider and no prompt, so it
    happens regardless. The second stays exactly as it was — unspent, with a
    truthful refusal — and nothing is sent.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    real_write = extraction.write_staging_under_lock
    real_prepare = extraction_batch.prepare_extraction_transport

    def refuse_first_write(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.startswith("one"):
            raise staging.StagingError("temporary output failure")
        return real_write(path, *args, **kwargs)

    def refuse_second_transport(call_plan: Any, target: Any, **kwargs: Any) -> Any:
        if target.name == "two.pdf":
            raise JankiError("temporary login refusal")
        return real_prepare(call_plan, target, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_first_write)
    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", refuse_second_transport
    )
    first_pass = _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    assert [child.state for child in first_pass.children] == [
        "result_captured",
        "authorized",
    ]

    shutil.rmtree(config.root / "prompts")
    already_sent = len(runner.spawned)

    def no_provider(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a saved answer was recovered through the provider")

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=no_provider,
        provider_which=no_provider,
        provider_spawn=no_provider,
    )

    assert resumed.children[0].state == "committed"
    assert resumed.children[0].staging_path.exists()
    # The unsent sibling is untouched and says why it did not go.
    assert resumed.children[1].state == "authorized"
    assert resumed.children[1].error
    assert len(runner.spawned) == already_sent


def test_a_tampered_manifest_is_refused_before_any_recovery_or_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal authenticates the manifest; a redirected output never runs."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    def refuse_write(*args: Any, **kwargs: Any) -> Any:
        raise staging.StagingError("temporary output failure")

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_write)
    first_pass = _dispatch(config, plan, runner)
    monkeypatch.undo()
    _no_api(monkeypatch)
    assert first_pass.children[0].state == "result_captured"

    manifest = extraction_batch.batch_manifest_path(config, plan.batch_id)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    redirected = config.staging_dir / "unconfirmed-redirect.yaml"
    raw["children"][0]["staging_path"] = str(redirected)
    manifest.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    def no_provider(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a tampered manifest reached the provider")

    with pytest.raises(JankiError):
        resume_extraction_batch(
            config,
            plan.batch_id,
            provider_env={},
            provider_runner=no_provider,
            provider_which=no_provider,
            provider_spawn=no_provider,
        )
    # Even reading a status must not present tampered bindings as this batch's.
    with pytest.raises(JankiError):
        extraction_batch_status(config, plan.batch_id)

    assert not redirected.exists()
    assert not plan.children[0].staging_path.exists()
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[plan.children[0].operation_id]
        .state
        == "result_captured"
    )
    assert list_extraction_batches(config) == ()

    # The confirmed-execution receipt is a file too, so deleting it must not
    # turn a tampered manifest into a trusted one: the reservation itself
    # records which manifest was authorized.
    extraction_batch.execution_receipt_path(config, plan.batch_id).unlink()
    with pytest.raises(JankiError) as caught:
        extraction_batch_status(config, plan.batch_id)
    assert "authorized" in str(caught.value)
    with pytest.raises(JankiError):
        resume_extraction_batch(
            config,
            plan.batch_id,
            provider_env={},
            provider_runner=no_provider,
            provider_which=no_provider,
            provider_spawn=no_provider,
        )
    assert not redirected.exists()

    # Membership and the worker limit are bound by the same reservation.
    for field, value in (("concurrency_limit", 3), ("children", raw["children"])):
        tampered = json.loads(json.dumps(raw))
        if field == "children":
            tampered["children"][0]["operation_id"] = (
                "00000000-0000-4000-8000-000000000000"
            )
        else:
            tampered[field] = value
        manifest.write_text(
            json.dumps(tampered, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(JankiError):
            extraction_batch_status(config, plan.batch_id)
    assert runner.spawned


@pytest.mark.parametrize("limit", [True, 1.0, 0, -1, 5, 64])
def test_the_worker_limit_is_a_small_whole_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: Any
) -> None:
    """A batch buys a little parallelism, not an unbounded fan-out."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()

    with pytest.raises(JankiError):
        _plan(config, sources, runner, concurrency_limit=limit)

    # Refused before anything probes the subscription.
    assert runner.calls == []
    assert not config.operations_file.exists()


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_every_worker_limit_inside_the_cap_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """The boundaries themselves are ordinary, including both ends."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]

    plan = _plan(config, sources, _runner(), concurrency_limit=limit)

    assert plan.concurrency_limit == limit


# --- finding the child one operation id belongs to -------------------------------
#
# Recovery keyed by an operation id has to find the expectations that operation
# was bound to, and those live only in a manifest. What these pin is that the
# lookup is by exact reserved id: never the newest manifest, never a guess, and
# never a silent skip past a manifest that could not be read.


def test_a_reserved_operation_id_finds_its_own_batch_and_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact id, out of several batches, with no ordering involved.

    Mutant: return the first child of the newest manifest.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    runner = _runner()
    first = _plan(config, [_source(config, "one.pdf", b"one")], runner)
    _dispatch(config, first, runner)
    second_sources = [
        _source(config, "two.pdf", b"two"),
        _source(config, "three.pdf", b"three"),
    ]
    second = _plan(config, second_sources, runner)
    _dispatch(config, second, runner)

    wanted = second.children[1]
    found_plan, found_child = extraction_batch.find_capture_child(
        config, wanted.operation_id
    )

    assert found_plan.batch_id == second.batch_id
    assert found_child == wanted
    assert found_child.source.name == "three.pdf"


def test_an_operation_no_batch_reserved_refuses_rather_than_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-batch or unknown id has no expectations here, and says so.

    Mutant: fall back to the only manifest present when nothing matches.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    runner = _runner()
    plan = _plan(config, [_source(config, "one.pdf", b"one")], runner)
    _dispatch(config, plan, runner)

    with pytest.raises(JankiError) as caught:
        extraction_batch.find_capture_child(config, "not-a-reserved-operation")

    assert "reserved operation" in str(caught.value)


def test_two_manifests_naming_one_operation_refuse_instead_of_choosing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One id, two claims: janki writes nothing rather than pick a winner.

    Mutant: return the first match instead of refusing an ambiguous one.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    runner = _runner()
    plan = _plan(config, [_source(config, "one.pdf", b"one")], runner)
    _dispatch(config, plan, runner)

    directory = config.operations_file.parent / "extraction_batches"
    original = json.loads((directory / f"{plan.batch_id}.json").read_text("utf-8"))
    duplicate_id = "11111111-2222-4333-8444-555555555555"
    original["batch_id"] = duplicate_id
    (directory / f"{duplicate_id}.json").write_text(
        json.dumps(original), encoding="utf-8"
    )

    with pytest.raises(JankiError) as caught:
        extraction_batch.find_capture_child(
            config, plan.children[0].operation_id
        )

    message = str(caught.value)
    assert "more than one batch manifest" in message
    assert plan.batch_id in message and duplicate_id in message


def test_an_unreadable_manifest_is_named_rather_than_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that cannot be read may be the one that holds the answer.

    Mutant: swallow an unreadable manifest and report "no batch has it".
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    runner = _runner()
    plan = _plan(config, [_source(config, "one.pdf", b"one")], runner)
    _dispatch(config, plan, runner)

    directory = config.operations_file.parent / "extraction_batches"
    broken = "99999999-8888-4777-8666-555555555555"
    (directory / f"{broken}.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(JankiError) as caught:
        extraction_batch.find_capture_child(config, "0" * 36)

    assert broken in str(caught.value)


def test_ordinary_recovery_writes_no_capture_recovery_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staging serializer's new seam is off unless a caller supplies it.

    A child recovered the ordinary way has exactly one place its answer came
    from, so it says nothing about envelopes: the ``capture_recovery`` key
    exists to record a *choice*, and there was none here.

    Mutant: default ``complete_extraction``'s ``capture_recovery`` to ``{}``
    so an ordinary recovery writes an empty block.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, concurrency_limit=1)

    real_write = extraction.write_staging_under_lock
    calls = {"count": 0}

    def refuse_first(path: Path, *args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise staging.StagingError("the disk went away")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_first)
    first_pass = _dispatch(config, plan, runner)
    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)
    assert first_pass.children[0].state == "result_captured"

    resumed = resume_extraction_batch(
        config,
        plan.batch_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert resumed.children[0].state == "committed"
    _records, meta = staging.read_staging(plan.children[0].staging_path)
    assert "capture_recovery" not in meta


# --- the optional study-job backlink ------------------------------------------
#
# Contracts §2.4 makes `job_id` optional and serializes it only when it is set,
# so a batch nobody planned for a job keeps exactly the wire, manifest hash and
# consent fingerprint it had before study jobs existed. That shape is current
# and fully supported: there is no legacy reader and nothing is backfilled.


JOB = "11111111-1111-4111-8111-111111111111"


def test_a_jobless_batch_keeps_its_exact_pre_job_wire_and_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional backlink adds nothing at all to a batch without one."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan = _plan(config, sources, _runner())

    wire = json.loads(plan.manifest_bytes)
    assert set(wire) == {
        "version",
        "batch_id",
        "concurrency_limit",
        "provider",
        "model",
        "scope_id",
        "children",
    }
    assert "job_id" not in wire
    assert plan.job_id == ""
    # The exact bytes, recomputed from the key set above rather than trusted.
    assert plan.manifest_bytes == (
        json.dumps(wire, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert plan.manifest_sha256 == __import__("hashlib").sha256(
        plan.manifest_bytes
    ).hexdigest()
    receipt = json.loads(
        extraction_batch._execution_receipt_bytes(plan).decode("utf-8")
    )
    assert set(receipt) == {
        "version",
        "batch_id",
        "manifest_sha256",
        "fingerprint",
        "child_operation_ids",
        "discards",
    }
    assert "job_id" not in receipt


def test_a_job_batch_carries_its_job_id_through_wire_and_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set, it rides in the manifest, the receipt and back out of `from_dict`."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    jobless = _plan(config, sources, _runner())
    owned = replace(jobless, job_id=JOB)

    wire = json.loads(owned.manifest_bytes)
    assert wire["job_id"] == JOB
    # Different bytes, and so a different manifest hash and a different
    # consent fingerprint. Correct: a batch bound to a job is a different
    # thing to agree to than a bare one.
    assert owned.manifest_sha256 != jobless.manifest_sha256
    assert owned.fingerprint != jobless.fingerprint
    assert {key: value for key, value in wire.items() if key != "job_id"} == json.loads(
        jobless.manifest_bytes
    )
    # The round trip is what keeps `_require_authentic_reservation` able to
    # recompute the hash the journal authorized.
    assert ExtractionBatchPlan.from_dict(wire).job_id == JOB
    assert ExtractionBatchPlan.from_dict(wire) == owned
    receipt = json.loads(
        extraction_batch._execution_receipt_bytes(owned).decode("utf-8")
    )
    assert receipt["job_id"] == JOB


def test_planning_a_job_batch_binds_the_job_and_dispatch_still_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reserved lifecycle works with the backlink present."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = _runner()
    plan = _plan(config, sources, runner, job_id=JOB)

    assert plan.job_id == JOB
    _dispatch(config, plan, runner)

    stored = json.loads(
        extraction_batch.batch_manifest_path(config, plan.batch_id).read_text(
            encoding="utf-8"
        )
    )
    assert stored["job_id"] == JOB
    # Reading it back reproduces the authorized hash, so status and resume do
    # not refuse a batch for carrying the backlink they asked for.
    assert extraction_batch_status(config, plan.batch_id).children[0].state == (
        "committed"
    )


def test_a_retry_inherits_its_job_and_refuses_a_different_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry belongs to the job its batch did; it is never re-parented."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    sources = [_source(config, "one.pdf", b"one")]
    runner = FakeClaudeRunner(reply=b'{"type":"result","is_error":true}\n')
    plan = _plan(config, sources, runner, job_id=JOB)
    _dispatch(config, plan, runner)

    retry = plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=_runner(),
        provider_which=_which,
    )
    assert retry.job_id == JOB
    assert json.loads(retry.manifest_bytes)["job_id"] == JOB
    assert retry.batch_id != plan.batch_id

    with pytest.raises(JankiError) as error:
        plan_extraction_batch_retry(
            config,
            plan.batch_id,
            [1],
            job_id="22222222-2222-4222-8222-222222222222",
            provider_env={},
            provider_runner=_runner(),
            provider_which=_which,
        )
    assert JOB in str(error.value)
