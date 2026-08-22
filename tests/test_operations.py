"""The journal for paid model calls that have not finished.

`WORKBENCH_PLAN.md` W3, and the `docs/DESIGN.md` sentence W1.0 added. These
tests are almost all about one question — *what do we know about the money?* —
because that is the only question the journal exists to answer, and every wrong
answer costs someone real credits or real work.

Nothing here calls a provider. The journal never does either: it records what a
caller is about to do and what came back, and refuses transitions that would
let one authorization become two charges.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from japanese_anki.operations import (
    LIVE_STATES,
    TERMINAL_STATES,
    Operation,
    OperationError,
    OperationJournal,
)


def _journal(tmp_path: Path) -> OperationJournal:
    return OperationJournal.load(tmp_path / "operations.json")


def _authorize(journal: OperationJournal, operation_id: str = "op-1") -> Operation:
    return journal.authorize(
        operation_id,
        kind="extract",
        source_file="lesson-8.pdf",
        source_sha256="a" * 64,
        request_fp="f" * 64,
        model="claude-opus-5",
    )


# --- what the states mean about money ---------------------------------------


def test_authorizing_spends_nothing(tmp_path: Path) -> None:
    """The point of journaling *before* dispatch: at this moment the decision
    exists on disk and the money does not."""
    operation = _authorize(_journal(tmp_path))

    assert operation.state == "authorized"
    assert operation.money_may_have_been_spent is False


def test_dispatching_already_counts_as_maybe_spent(tmp_path: Path) -> None:
    """The interval this whole module exists for. A process that dies here may
    have got the request out the door a microsecond earlier, and a journal
    claiming otherwise would be guessing about someone's credits."""
    journal = _journal(tmp_path)
    _authorize(journal)

    operation = journal.advance("op-1", "dispatching")

    assert operation.money_may_have_been_spent is True


def test_a_call_that_never_left_is_not_counted_as_spent(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)

    canceled = journal.advance("op-1", "canceled_before_send", detail="user said no")

    assert canceled.money_may_have_been_spent is False
    assert canceled.state in TERMINAL_STATES


# --- authority is one-use ---------------------------------------------------


def test_an_operation_id_cannot_be_authorized_twice(tmp_path: Path) -> None:
    """An authorization is a person agreeing to spend money once. Reusing the
    identity would let a replayed request ride on a decision already made."""
    journal = _journal(tmp_path)
    _authorize(journal)

    with pytest.raises(OperationError, match="one-use"):
        _authorize(journal)


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES))
def test_no_terminal_state_can_be_reused(tmp_path: Path, terminal: str) -> None:
    """Including `committed`: a finished operation's ID is spent, not free."""
    journal = _journal(tmp_path)
    _authorize(journal)
    route = {
        "committed": ("dispatching", "running", "result_captured", "committed"),
        "outcome_unknown": ("dispatching", "outcome_unknown"),
        "failed_before_send": ("failed_before_send",),
        "canceled_before_send": ("canceled_before_send",),
        "expired": ("expired",),
    }[terminal]
    for step in route:
        journal.advance(
            "op-1",
            step,
            artifact=".pending/op-1.json" if step == "result_captured" else "",
        )

    with pytest.raises(OperationError, match="one-use"):
        _authorize(journal)


# --- the transition nobody may make -----------------------------------------


@pytest.mark.parametrize("target", ["dispatching", "running", "result_captured"])
def test_an_unknown_outcome_never_returns_to_a_live_state(
    tmp_path: Path, target: str
) -> None:
    """The rule that stops one authorization becoming two charges. A provider
    call whose outcome nobody knows must not be quietly retried; a fresh charge
    needs fresh authority, which means a new operation a person agreed to."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "outcome_unknown", detail="the socket died")

    with pytest.raises(OperationError, match="cannot move"):
        journal.advance("op-1", target)


def test_committing_requires_a_captured_answer(tmp_path: Path) -> None:
    """Committing means "the exact answer became staging". With no artifact
    there is nothing that could have, so the claim would be empty."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    journal.advance("op-1", "result_captured")  # captured, artifact not recorded

    with pytest.raises(OperationError, match="without a captured"):
        journal.advance("op-1", "committed")

    journal.advance("op-1", "committed", artifact=".pending/op-1.json")
    assert journal.operations["op-1"].state == "committed"


def test_skipping_dispatch_is_refused(tmp_path: Path) -> None:
    """A result cannot exist for a call that was never sent."""
    journal = _journal(tmp_path)
    _authorize(journal)

    with pytest.raises(OperationError, match="cannot move"):
        journal.advance("op-1", "result_captured")


# --- durability -------------------------------------------------------------


def test_every_transition_is_on_disk_before_the_next_step(tmp_path: Path) -> None:
    """A journal that only lived in memory would be worth nothing: the crash it
    protects against takes the process with it."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")

    reread = OperationJournal.load(tmp_path / "operations.json")

    assert reread.operations["op-1"].state == "dispatching"
    assert reread.operations["op-1"].money_may_have_been_spent is True


def test_a_second_reader_sees_what_the_first_wrote(tmp_path: Path) -> None:
    """Two journal objects on one path is the ordinary case — the CLI and the
    workbench both write here — so a transition must land on whatever the other
    left rather than on a stale snapshot."""
    first = _journal(tmp_path)
    second = _journal(tmp_path)
    _authorize(first, "op-1")

    _authorize(second, "op-2")

    both = OperationJournal.load(tmp_path / "operations.json")
    assert sorted(both.operations) == ["op-1", "op-2"]


def test_concurrent_authorizations_all_survive(tmp_path: Path) -> None:
    """The lock is what makes the file additive under real concurrency."""
    path = tmp_path / "operations.json"
    barrier = threading.Barrier(4)

    def authorize(index: int) -> None:
        barrier.wait()
        OperationJournal.load(path).authorize(
            f"op-{index}",
            kind="extract",
            source_file=f"{index}.pdf",
            source_sha256="a" * 64,
            request_fp="f" * 64,
            model="claude-opus-5",
        )

    threads = [threading.Thread(target=authorize, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(OperationJournal.load(path).operations) == [
        "op-0",
        "op-1",
        "op-2",
        "op-3",
    ]


# --- what a resumed run has to deal with ------------------------------------


def test_unfinished_lists_only_live_operations(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal, "live")
    _authorize(journal, "done")
    journal.advance("done", "canceled_before_send")

    assert [op.operation_id for op in journal.unfinished()] == ["live"]
    assert all(op.state in LIVE_STATES for op in journal.unfinished())


def test_an_unknown_outcome_is_what_a_person_must_look_at(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal, "lost")
    journal.advance("lost", "dispatching")
    journal.advance("lost", "outcome_unknown", detail="no response")
    _authorize(journal, "fine")

    assert [op.operation_id for op in journal.needing_attention()] == ["lost"]


def test_only_a_committed_operation_can_be_forgotten(tmp_path: Path) -> None:
    """Everything else is either live or a record of money whose outcome
    someone still needs to see."""
    journal = _journal(tmp_path)
    _authorize(journal, "lost")
    journal.advance("lost", "dispatching")
    journal.advance("lost", "outcome_unknown")

    with pytest.raises(OperationError, match="not\n?\\s*committed|not committed"):
        journal.forget(["lost"])

    assert "lost" in OperationJournal.load(tmp_path / "operations.json").operations


def test_forgetting_a_committed_operation_removes_it(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    for step in ("dispatching", "running", "result_captured"):
        journal.advance("op-1", step, artifact=".pending/op-1.json")
    journal.advance("op-1", "committed")

    assert journal.forget(["op-1"]) == 1
    assert OperationJournal.load(tmp_path / "operations.json").operations == {}


# --- a journal that cannot be read is not silently empty ---------------------


def test_an_unreadable_journal_refuses_rather_than_looking_empty(
    tmp_path: Path,
) -> None:
    """Reporting "no pending operations" for a corrupt file would be the worst
    possible answer: it says nothing was spent."""
    path = tmp_path / "operations.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(OperationError, match="Could not read"):
        OperationJournal.load(path)


def test_an_unknown_state_refuses_rather_than_being_guessed(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    path.write_text(
        json.dumps({"version": 1, "operations": {"op": {"state": "probably-fine"}}}),
        encoding="utf-8",
    )

    with pytest.raises(OperationError, match="unknown state"):
        OperationJournal.load(path)


def test_a_missing_journal_is_simply_empty(tmp_path: Path) -> None:
    """Absent is not corrupt: a corpus that has never run a paid call has no
    file, and that is not an error."""
    assert OperationJournal.load(tmp_path / "nope.json").operations == {}
