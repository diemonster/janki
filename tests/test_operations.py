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
    advance_refusal,
    answer_text,
    capture_artifact,
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
    first.advance("op-1", "canceled_before_send")

    _authorize(second, "op-2")

    both = OperationJournal.load(tmp_path / "operations.json")
    assert sorted(both.operations) == ["op-1", "op-2"]


def test_exactly_one_of_four_racing_authorizations_wins(tmp_path: Path) -> None:
    """The double-spend window, closed under real threads.

    Four processes deciding to spend at the same instant is the case a display
    cannot cover: each would render "nothing is running" and each would then
    authorize. Only the lock can arbitrate, and this is the proof that it
    does — one entry on disk, three callers told why.
    """
    path = tmp_path / "operations.json"
    barrier = threading.Barrier(4)
    refused: list[str] = []
    lock = threading.Lock()

    def authorize(index: int) -> None:
        barrier.wait()
        try:
            OperationJournal.load(path).authorize(
                f"op-{index}",
                kind="extract",
                source_file=f"{index}.pdf",
                source_sha256="a" * 64,
                request_fp="f" * 64,
                model="claude-opus-5",
            )
        except OperationError as exc:
            with lock:
                refused.append(str(exc))

    threads = [threading.Thread(target=authorize, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(OperationJournal.load(path).operations) == 1
    assert len(refused) == 3
    assert all("will not start another paid call" in message for message in refused)


# --- what a resumed run has to deal with ------------------------------------


def test_unfinished_lists_only_live_operations(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal, "done")
    journal.advance("done", "canceled_before_send")
    _authorize(journal, "live")

    assert [op.operation_id for op in journal.unfinished()] == ["live"]
    assert all(op.state in LIVE_STATES for op in journal.unfinished())


def test_an_unknown_outcome_is_what_a_person_must_look_at(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal, "lost")
    journal.advance("lost", "dispatching")
    journal.advance("lost", "outcome_unknown", detail="no response")

    assert [op.operation_id for op in journal.needing_attention()] == ["lost"]


def test_an_unfinished_operation_cannot_be_forgotten(tmp_path: Path) -> None:
    """A call still in flight is not something anybody has decided about, and
    dropping the entry would delete the only record that money may be moving.
    Ending it is a separate, deliberate statement."""
    journal = _journal(tmp_path)
    _authorize(journal, "lost")
    journal.advance("lost", "dispatching")

    with pytest.raises(OperationError, match="not finished"):
        journal.forget(["lost"])

    assert "lost" in OperationJournal.load(tmp_path / "operations.json").operations


def test_an_accepted_unknown_outcome_can_be_forgotten(tmp_path: Path) -> None:
    """The other half, and the reason the refusal above is not the whole rule.
    An `outcome_unknown` a person has looked at and accepted is finished — and
    refusing to drop it was how one lost call blocked every later one for
    ever."""
    journal = _journal(tmp_path)
    _authorize(journal, "lost")
    journal.advance("lost", "dispatching")
    journal.advance("lost", "outcome_unknown", detail="no response")

    assert journal.forget(["lost"]) == 1
    assert not OperationJournal.load(tmp_path / "operations.json").operations


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


# --- a captured response is not the same as a captured answer ---------------


def test_a_thinking_only_reply_reports_no_answer(tmp_path: Path) -> None:
    """What a real `janki extract` run produced: `stop_reason: max_tokens` with
    a single thinking block and no text at all. The reply is real and paid for
    and contains nothing to recover, so saying "your answer was saved" would
    send someone looking for cards in a file that has none."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json",
        "op-1",
        json.dumps(
            {
                "stop_reason": "max_tokens",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "x" * 200}
                ],
            }
        ).encode(),
    )
    operation = journal.advance("op-1", "result_captured", artifact=artifact)

    assert operation.money_may_have_been_spent is True
    assert answer_text(tmp_path / "operations.json", operation) == ""


def test_a_reply_carrying_text_reports_the_answer(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json",
        "op-1",
        json.dumps(
            {
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "x"},
                    {"type": "text", "text": '{"candidates": []}'},
                ],
            }
        ).encode(),
    )
    operation = journal.advance("op-1", "result_captured", artifact=artifact)

    assert answer_text(tmp_path / "operations.json", operation) == '{"candidates": []}'


def test_an_unreadable_artifact_reports_no_answer_rather_than_raising(
    tmp_path: Path,
) -> None:
    """This runs on the error path, where raising would replace a readable
    failure with a worse one."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = journal.advance(
        "op-1", "result_captured", artifact=".pending/missing.json"
    )

    assert answer_text(tmp_path / "operations.json", operation) == ""


def test_forgetting_an_operation_takes_its_artifact_with_it(tmp_path: Path) -> None:
    """A captured answer is a recovery buffer, not an archive. Once the answer
    has become staging, `data/staging/done/` holds the durable copy — leaving
    the blob behind accumulates an unreferenced megabyte per paid call in a
    repository whose whole point is being portable."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.advance("op-1", "result_captured", artifact=artifact)
    journal.advance("op-1", "committed")
    blob = tmp_path / artifact
    assert blob.exists()

    journal.forget(["op-1"])

    assert not blob.exists()


def test_an_unfinished_operation_keeps_its_artifact(tmp_path: Path) -> None:
    """The buffer for work someone still has to decide about cannot be swept
    away: this reply was paid for and never became staging."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.advance("op-1", "result_captured", artifact=artifact)

    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"])

    assert (tmp_path / artifact).exists()


def test_a_captured_answer_is_not_something_to_end(tmp_path: Path) -> None:
    """"End this" means "stop waiting on a call that will never finish".
    Nothing about a captured answer is unknown — it is on disk — and recording
    it as an unknown outcome would unwrite the one fact that matters about
    it."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.advance("op-1", "result_captured", artifact=artifact)

    with pytest.raises(OperationError, match="its reply arrived"):
        journal.end("op-1")

    assert (
        OperationJournal.load(tmp_path / "operations.json")
        .operations["op-1"]
        .state
        == "result_captured"
    )


def test_a_paid_reply_is_not_dropped_by_accident(tmp_path: Path) -> None:
    """Discarding a captured answer is one deliberate step, not a laundering
    of it through "unknown outcome" first. It still takes saying so twice: the
    reply was paid for and nobody read it."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.advance("op-1", "result_captured", artifact=artifact)

    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"])
    assert (tmp_path / artifact).exists()

    assert journal.forget(["op-1"], force=True) == 1
    assert not (tmp_path / artifact).exists()


def test_a_state_the_journal_does_not_know_is_refused_rather_than_raising(
    tmp_path: Path,
) -> None:
    """`advance_refusal` is exported for callers about to do something they
    cannot take back, which means it is asked about states it has not
    validated. The answer to "may this move?" is no — a traceback instead
    would send a caller holding a paid answer down its own error path."""
    assert advance_refusal("op-1", "not-a-state", "committed", "a.json")
    assert advance_refusal("op-1", "result_captured", "not-a-state", "a.json")
    # And the two it exists to answer, so the refusals above are not merely
    # this function refusing everything.
    assert advance_refusal("op-1", "result_captured", "committed", "a.json") == ""
    assert "captured provider answer" in advance_refusal(
        "op-1", "result_captured", "committed", ""
    )


# --- one paid call at a time ------------------------------------------------


def test_a_call_in_flight_refuses_a_second_authorization(tmp_path: Path) -> None:
    """The rule, not a warning about it. A page that only displayed this would
    be read when it rendered and acted on when a button was clicked; two
    callers could pass that and both spend. This refuses under the lock."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")

    with pytest.raises(OperationError, match="will not start another paid call"):
        _authorize(journal, "second")

    assert "second" not in OperationJournal.load(
        tmp_path / "operations.json"
    ).operations


def test_a_paid_answer_nobody_settled_refuses_a_second_authorization(
    tmp_path: Path,
) -> None:
    """Money already moved and nobody has decided what it bought. Starting
    another call would bury the answer under a second charge."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")
    journal.advance("first", "result_captured", artifact=".pending/first.json")

    with pytest.raises(OperationError, match="will not start another paid call"):
        _authorize(journal, "second")


def test_an_unknown_outcome_refuses_a_second_authorization(
    tmp_path: Path,
) -> None:
    """The case the whole journal exists for: janki does not know whether that
    call was billed, so it will not make another until a person says so."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")
    journal.advance("first", "outcome_unknown", detail="connection lost")

    with pytest.raises(OperationError, match="will not start another paid call"):
        _authorize(journal, "second")


def test_a_committed_call_does_not_block_the_next_one(tmp_path: Path) -> None:
    """The ordinary case, and the one that must not regress: a finished run is
    history, and a batch of three files is three sequential calls."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")
    journal.advance("first", "result_captured", artifact=".pending/first.json")
    journal.advance("first", "committed")

    _authorize(journal, "second")

    assert "second" in OperationJournal.load(
        tmp_path / "operations.json"
    ).operations


def test_an_authority_that_never_sent_anything_still_blocks(
    tmp_path: Path,
) -> None:
    """It has to. Writing the authority and marking it dispatched are two
    writes, so letting `authorized` through leaves a window where two runs both
    authorize, each sees the other resting in a state that does not block, and
    both dispatch — the exact double-spend this set exists to close."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")

    with pytest.raises(OperationError, match="will not start another paid call"):
        _authorize(journal, "second")


def test_an_orphaned_authority_is_retired_as_never_sent(tmp_path: Path) -> None:
    """The cost of blocking it, and the reason that cost is affordable: a
    process killed between the two writes leaves an authority nobody used.
    Calling that an unknown outcome would invent a charge — nothing was
    sent — so it retires as cancelled, and the next call can start."""
    journal = _journal(tmp_path)
    _authorize(journal, "orphan")

    ended = journal.end("orphan")

    assert ended.state == "canceled_before_send"
    assert "nothing had been sent" in ended.detail
    journal.forget(["orphan"])
    _authorize(journal, "next")


def test_ending_a_stuck_call_lets_the_next_one_start(tmp_path: Path) -> None:
    """The way out of a wedge that used to be permanent. Nothing can settle a
    killed process automatically — only a person can say it is gone — so this
    is that statement, and it lands on `outcome_unknown` because that is what
    is true about the money."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")

    ended = journal.end("first")

    assert ended.state == "outcome_unknown"
    assert "the process was gone" in ended.detail
    # Still blocking, because the record of possible spending survives being
    # ended. Forgetting it is the separate, deliberate second step.
    with pytest.raises(OperationError, match="will not start another paid call"):
        _authorize(journal, "second")

    journal.forget(["first"])
    _authorize(journal, "second")


def test_a_finished_call_cannot_be_ended(tmp_path: Path) -> None:
    """"End this" means "stop waiting on it". A committed call is not being
    waited on, and moving it anywhere would unwrite a fact."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "result_captured", artifact=".pending/op-1.json")
    journal.advance("op-1", "committed")

    with pytest.raises(OperationError, match="nothing left to end"):
        journal.end("op-1")


def test_ending_a_call_the_journal_never_had_is_refused(tmp_path: Path) -> None:
    journal = _journal(tmp_path)

    with pytest.raises(OperationError, match="No operation"):
        journal.end("never-existed")
