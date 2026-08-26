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

import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import japanese_anki.io as io_module
from japanese_anki import operations
from japanese_anki.io import DataError
from japanese_anki.operations import (
    LIVE_STATES,
    TERMINAL_STATES,
    ArtifactReceipt,
    Operation,
    OperationError,
    OperationJournal,
    advance_refusal,
    capture_artifact,
    response_answer_text,
    serialize_response,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _receipt(operation_id: str = "op-1") -> ArtifactReceipt:
    """Syntactically valid proof for tests concerned only with state moves."""
    return ArtifactReceipt(
        relative_name=f".pending/{operation_id}.json",
        directory_identity=(1, 2),
        entry_state=(1, 3, 4, 5, 6),
        content_sha256="a" * 64,
        terminal_marker=None,
    )


def _capture_result(
    journal: OperationJournal,
    payload: bytes,
    operation_id: str = "op-1",
) -> Operation:
    return journal.capture_result(
        operation_id,
        lambda: capture_artifact(journal.path, operation_id, payload),
    )


# --- what the states mean about money ---------------------------------------


def test_authorizing_spends_nothing(tmp_path: Path) -> None:
    """The point of journaling *before* dispatch: at this moment the decision
    exists on disk and the money does not."""
    operation = _authorize(_journal(tmp_path))

    assert operation.state == "authorized"
    assert operation.money_may_have_been_spent is False


def test_authorization_refuses_if_its_journal_parent_detaches_at_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    path = data / "operations.json"
    journal = OperationJournal.load(path)
    held = tmp_path / "held-data"
    real_link = io_module.os.link
    swapped = False

    def detach_then_link(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        data.rename(held)
        data.mkdir()
        swapped = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(io_module.os, "link", detach_then_link)

    with pytest.raises(OperationError, match="directory changed"):
        _authorize(journal)

    assert swapped
    assert OperationJournal.load(path).operations == {}
    assert set(OperationJournal.load(held / path.name).operations) == {"op-1"}


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
        if step == "result_captured":
            _capture_result(journal, b"{}")
        else:
            journal.advance("op-1", step)

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


def test_result_capture_requires_the_receipt_binding_route(tmp_path: Path) -> None:
    """Committing means "the exact answer became staging". With no artifact
    there is nothing that could have, so the claim would be empty."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    with pytest.raises(OperationError, match="capture_result"):
        journal.advance("op-1", "result_captured")

    captured = _capture_result(journal, b"{}")
    committed = journal.advance("op-1", "committed")

    assert committed.artifact == captured.artifact


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


def _operation_wire(
    state: str = "result_captured", *, receipt: bool = True
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": "extract",
        "state": state,
        "source_file": "lesson.pdf",
        "source_sha256": "a" * 64,
        "request_fp": "b" * 64,
        "model": "claude-opus-5",
        "authorized_at": "now",
        "updated_at": "now",
    }
    if receipt:
        value["artifact"] = _receipt().to_dict()
    return value


@pytest.mark.parametrize("state", ["result_captured", "committed"])
def test_captured_wire_states_require_an_exact_receipt(
    tmp_path: Path, state: str
) -> None:
    with pytest.raises(OperationError, match="requires an artifact receipt"):
        Operation.from_dict(
            tmp_path / "operations.json",
            "op-1",
            _operation_wire(state, receipt=False),
        )


@pytest.mark.parametrize(
    "state",
    sorted(set(operations.STATES) - {"result_captured", "committed"}),
)
def test_noncaptured_wire_states_forbid_an_artifact_receipt(
    tmp_path: Path, state: str
) -> None:
    with pytest.raises(OperationError, match="cannot hold an artifact receipt"):
        Operation.from_dict(
            tmp_path / "operations.json", "op-1", _operation_wire(state)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_name", ".pending/op-2.json"),
        ("relative_name", "/tmp/op-1.json"),
        ("directory_identity", [True, 2]),
        ("directory_identity", [-1, 2]),
        ("directory_identity", [1]),
        ("directory_identity", [1, 2, 3]),
        ("entry_state", [1, 2, 3, True, 5]),
        ("entry_state", [1, 2, 3, 4, -1]),
        ("entry_state", [9, 2, 3, 4, 5]),
        ("entry_state", [1, 2, 3, 4]),
        ("entry_state", [1, 2, 3, 4, 5, 6]),
        ("sha256", "not-a-digest"),
        ("sha256", "A" * 64),
    ],
)
def test_artifact_receipt_parser_rejects_unscoped_or_malformed_fields(
    tmp_path: Path, field: str, value: Any
) -> None:
    raw = _operation_wire()
    artifact = raw["artifact"]
    assert isinstance(artifact, dict)
    artifact[field] = value

    with pytest.raises(OperationError, match="invalid artifact receipt"):
        Operation.from_dict(tmp_path / "operations.json", "op-1", raw)


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_artifact_receipt_parser_requires_exact_keys(
    tmp_path: Path, edit: str
) -> None:
    raw = _operation_wire()
    artifact = raw["artifact"]
    assert isinstance(artifact, dict)
    if edit == "missing":
        del artifact["sha256"]
    else:
        artifact["unexpected"] = None

    with pytest.raises(OperationError, match="invalid artifact receipt"):
        Operation.from_dict(tmp_path / "operations.json", "op-1", raw)


def _terminal_marker_wire() -> dict[str, Any]:
    return {
        "name": (
            ".op-1.json.0123456789abcdef.1-3.janki-cas.validated"
        ),
        "entry_state": [1, 7, 8, 9, 10],
        "sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "name",
            ".op-2.json.0123456789abcdef.1-3.janki-cas.validated",
        ),
        (
            "name",
            ".op-1.json.0123456789abcdef.1-3.janki-cas.recovered",
        ),
        ("entry_state", [1, 7, 8, True, 10]),
        ("entry_state", [9, 7, 8, 9, 10]),
        ("entry_state", [1, 7, 8, 9]),
        ("sha256", "not-a-digest"),
        ("sha256", "B" * 64),
    ],
)
def test_artifact_receipt_parser_rejects_an_invalid_terminal_marker(
    tmp_path: Path, field: str, value: Any
) -> None:
    raw = _operation_wire()
    artifact = raw["artifact"]
    assert isinstance(artifact, dict)
    marker = _terminal_marker_wire()
    marker[field] = value
    artifact["terminal_marker"] = marker

    with pytest.raises(OperationError, match="invalid artifact receipt"):
        Operation.from_dict(tmp_path / "operations.json", "op-1", raw)


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_artifact_receipt_terminal_marker_requires_exact_keys(
    tmp_path: Path, edit: str
) -> None:
    raw = _operation_wire()
    artifact = raw["artifact"]
    assert isinstance(artifact, dict)
    marker = _terminal_marker_wire()
    if edit == "missing":
        del marker["sha256"]
    else:
        marker["unexpected"] = None
    artifact["terminal_marker"] = marker

    with pytest.raises(OperationError, match="invalid artifact receipt"):
        Operation.from_dict(tmp_path / "operations.json", "op-1", raw)


def test_receipt_and_cleanup_shapes_cannot_be_swapped(tmp_path: Path) -> None:
    receipt_as_cleanup = _operation_wire("committed")
    receipt_as_cleanup["cleanup"] = receipt_as_cleanup["artifact"]
    with pytest.raises(OperationError, match="invalid cleanup intent"):
        Operation.from_dict(
            tmp_path / "operations.json", "op-1", receipt_as_cleanup
        )

    cleanup_as_receipt = _operation_wire()
    cleanup_as_receipt["artifact"] = {
        "artifact": None,
        "write_ahead": None,
        "forced": True,
    }
    with pytest.raises(OperationError, match="invalid artifact receipt"):
        Operation.from_dict(
            tmp_path / "operations.json", "op-1", cleanup_as_receipt
        )


def test_committed_repository_operation_journal_loads() -> None:
    loaded = OperationJournal.load(ROOT / "data" / "operations.json")

    assert "1a85e6fd-bb25-43fd-aabd-658895bdbaf5" in loaded.operations


def test_an_existing_journal_write_is_bound_to_the_bytes_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal, "finished")
    journal.advance("finished", "canceled_before_send")
    real_write = operations.atomic_write_text_bound
    injected = False

    def interfere(target: Path, text: str, **kwargs: object) -> None:
        nonlocal injected
        Path(target).write_bytes(Path(target).read_bytes() + b" ")
        injected = True
        real_write(target, text, **kwargs)

    monkeypatch.setattr(operations, "atomic_write_text_bound", interfere)

    with pytest.raises(OperationError, match="changed content"):
        _authorize(journal, "next")

    assert injected
    assert path.read_bytes().endswith(b"\n ")
    assert "next" not in OperationJournal.load(path).operations


def test_a_bound_writer_failure_is_reported_as_an_operation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DataError("bound journal write failed")

    monkeypatch.setattr(operations, "atomic_write_text_bound", fail)

    with pytest.raises(OperationError, match="bound journal write failed") as caught:
        _authorize(_journal(tmp_path))

    assert type(caught.value) is OperationError


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
    for step in ("dispatching", "running"):
        journal.advance("op-1", step)
    _capture_result(journal, b"{}")
    journal.advance("op-1", "committed")

    assert journal.forget(["op-1"]) == 1
    assert OperationJournal.load(tmp_path / "operations.json").operations == {}


def test_artifact_store_probe_failure_does_not_leave_a_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = io_module.os.fsync
    calls = 0

    def fail_probe_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("probe sync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(io_module.os, "fsync", fail_probe_sync)

    with pytest.raises(OperationError, match="pending answer store"):
        operations.prepare_artifact_store(tmp_path / "operations.json")

    assert calls >= 1
    assert not list((tmp_path / operations.PENDING_DIR).glob(".janki-write-probe.*"))


def test_artifact_store_preflight_leaves_no_private_probe_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io_module, "_path_lock_root", lambda: lock_root)

    operations.prepare_artifact_store(tmp_path / "operations.json")

    assert list((lock_root / "write-probes").iterdir()) == []


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
    operation = journal.capture_result("op-1", lambda: artifact)

    assert operation.money_may_have_been_spent is True
    observed = operations.reply_observation(tmp_path / "operations.json", operation)
    assert response_answer_text(observed.payload) == ""


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
    operation = journal.capture_result("op-1", lambda: artifact)

    observed = operations.reply_observation(tmp_path / "operations.json", operation)
    assert response_answer_text(observed.payload) == '{"candidates": []}'


def test_capture_refuses_a_symlinked_pending_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".pending").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DataError, match="target directory"):
        capture_artifact(
            tmp_path / "operations.json", "op-1", b"paid provider answer"
        )

    assert not (outside / "op-1.json").exists()


def test_capture_refuses_a_symlinked_pending_artifact(tmp_path: Path) -> None:
    pending = tmp_path / ".pending"
    pending.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"keep this")
    (pending / "op-1.json").symlink_to(outside)

    with pytest.raises(DataError, match="non-regular target"):
        capture_artifact(
            tmp_path / "operations.json", "op-1", b"paid provider answer"
        )

    assert outside.read_bytes() == b"keep this"


def test_capture_never_overwrites_an_answer_already_saved_for_the_operation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    capture_artifact(path, "op-1", b"first paid answer")

    with pytest.raises(DataError, match="incomplete bound write"):
        capture_artifact(path, "op-1", b"second paid answer")

    assert (tmp_path / ".pending" / "op-1.json").read_bytes() == b"first paid answer"


def test_capture_publication_cannot_overwrite_an_answer_created_at_its_last_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".pending" / "op-1.json"
    first = b"first captured answer"
    real_link = io_module.os.link
    injected = False

    def create_then_link(*args: object, **kwargs: object) -> None:
        nonlocal injected
        target.write_bytes(first)
        injected = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(io_module.os, "link", create_then_link)

    with pytest.raises(DataError, match="changed before replace"):
        capture_artifact(
            tmp_path / "operations.json", "op-1", b"second captured answer"
        )

    assert injected
    assert target.read_bytes() == first


def test_capture_refuses_when_the_journal_ancestor_was_replaced_by_a_link(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    journal = data / "operations.json"
    operations.prepare_artifact_store(journal)
    held = tmp_path / "held-data"
    outside = tmp_path / "outside"
    data.rename(held)
    outside.mkdir()
    data.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DataError, match="directory|symlink|safely"):
        capture_artifact(journal, "op-1", b"paid provider answer")

    assert not (outside / ".pending" / "op-1.json").exists()
    assert not (held / ".pending" / "op-1.json").exists()


def test_capture_refuses_if_its_bound_pending_directory_is_detached_at_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "operations.json"
    operations.prepare_artifact_store(journal)
    pending = tmp_path / ".pending"
    held = tmp_path / "held-pending"
    real_link = io_module.os.link
    swapped = False

    def detach_then_link(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        pending.rename(held)
        pending.mkdir()
        swapped = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(io_module.os, "link", detach_then_link)

    with pytest.raises(DataError, match="directory changed"):
        capture_artifact(journal, "op-1", b"paid provider answer")

    assert swapped
    assert not (pending / "op-1.json").exists()
    assert (held / "op-1.json").read_bytes() == b"paid provider answer"


def test_receipt_read_does_not_follow_a_pending_artifact_symlink(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b'{"content": []}')
    target = tmp_path / operation.artifact.relative_name  # type: ignore[union-attr]
    other = target.with_name("other.json")
    other.write_text(
        '{"content": [{"type": "text", "text": "another answer"}]}',
        encoding="utf-8",
    )
    target.unlink()
    target.symlink_to(other.name)

    observed = operations.reply_observation(journal.path, operation)

    assert observed.recorded
    assert observed.payload is None
    assert other.is_file()


def test_receipt_read_refuses_a_link_swapped_at_the_read_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b'{"content": []}')
    assert operation.artifact is not None
    target = tmp_path / operation.artifact.relative_name
    other = target.with_name("other.json")
    other.write_text(
        '{"content": [{"type": "text", "text": "another answer"}]}',
        encoding="utf-8",
    )
    real_read = operations._read_artifact_binding
    swapped = False

    def swap_then_read(binding: object) -> bytes | None:
        nonlocal swapped
        target.unlink()
        target.symlink_to(other.name)
        swapped = True
        return real_read(binding)  # type: ignore[arg-type]

    monkeypatch.setattr(operations, "_read_artifact_binding", swap_then_read)

    observed = operations.reply_observation(journal.path, operation)

    assert observed.recorded
    assert observed.payload is None
    assert swapped
    assert other.is_file()


def test_receipt_read_refuses_a_regular_replacement_at_the_read_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b'{"content": []}')
    assert operation.artifact is not None
    target = tmp_path / operation.artifact.relative_name
    replacement = target.with_name("replacement.json")
    replacement.write_text(
        '{"content": [{"type": "text", "text": "wrong answer"}]}',
        encoding="utf-8",
    )
    real_read = operations._read_artifact_binding
    swapped = False

    def swap_then_read(binding: object) -> bytes | None:
        nonlocal swapped
        target.unlink()
        replacement.rename(target)
        swapped = True
        return real_read(binding)  # type: ignore[arg-type]

    monkeypatch.setattr(operations, "_read_artifact_binding", swap_then_read)

    observed = operations.reply_observation(journal.path, operation)

    assert observed.recorded
    assert observed.payload is None
    assert swapped
    assert target.is_file()


def test_a_missing_receipted_artifact_is_recorded_but_unavailable(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b'{"content": []}')
    assert operation.artifact is not None
    (tmp_path / operation.artifact.relative_name).unlink()

    observed = operations.reply_observation(journal.path, operation)

    assert observed.recorded
    assert observed.payload is None


@pytest.mark.parametrize(
    "replacement",
    [b"different answer", b"paid provider answer"],
    ids=["different-bytes", "byte-identical"],
)
def test_a_receipt_never_adopts_a_same_name_replacement_before_first_read_or_forget(
    tmp_path: Path, replacement: bytes
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b"paid provider answer")
    assert operation.artifact is not None
    target = tmp_path / operation.artifact.relative_name
    original_identity = (target.stat().st_dev, target.stat().st_ino)
    raced = target.with_name("replacement.json")
    raced.write_bytes(replacement)
    replacement_identity = (raced.stat().st_dev, raced.stat().st_ino)
    assert replacement_identity != original_identity
    target.unlink()
    raced.rename(target)

    with pytest.raises(OperationError, match="exact recovery bytes are unavailable"):
        OperationJournal.load(path).read_reply("op-1")
    with pytest.raises(OperationError, match="--force"):
        OperationJournal.load(path).forget(["op-1"])

    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1
    assert target.read_bytes() == replacement
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity


def test_a_receipt_never_rebinds_a_checkout_replacement_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b"paid provider answer")
    assert operation.artifact is not None
    pending = tmp_path / ".pending"
    detached = tmp_path / "detached-pending"
    pending.rename(detached)
    pending.mkdir()
    replacement = pending / "op-1.json"
    replacement.write_bytes(b"paid provider answer")

    observation = operations.reply_observation(path, operation)

    assert observation.recorded
    assert observation.payload is None
    with pytest.raises(OperationError, match="--force"):
        journal.forget(["op-1"])
    assert journal.forget(["op-1"], force=True) == 1
    assert replacement.read_bytes() == b"paid provider answer"
    assert (detached / "op-1.json").read_bytes() == b"paid provider answer"


def test_forgetting_an_operation_takes_its_artifact_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A captured answer is a recovery buffer, not an archive. Once the answer
    has become staging, `data/staging/done/` holds the durable copy — leaving
    the blob behind accumulates an unreferenced megabyte per paid call in a
    repository whose whole point is being portable."""
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io_module, "_path_lock_root", lambda: lock_root)
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    blob = tmp_path / artifact.relative_name
    assert blob.exists()

    journal.forget(["op-1"])

    assert not blob.exists()
    assert list((lock_root / "retired-writes").iterdir()) == []


def test_forget_resumes_after_process_death_with_a_durable_cleanup_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    blob = tmp_path / artifact.relative_name

    script = r"""
import os
import sys
from pathlib import Path

from japanese_anki import operations
from japanese_anki.operations import OperationJournal

path = Path(sys.argv[1])

def die_before_cleanup(_binding):
    os._exit(101)

operations._retire_artifact = die_before_cleanup
OperationJournal.load(path).forget(["op-1"], force=True)
raise AssertionError("cleanup was reached without the injected process death")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env=environment,
        check=False,
        timeout=10,
    )

    assert result.returncode == 101
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.forced is True
    assert blob.read_bytes() == b"captured provider answer"

    resumed = OperationJournal.load(path)
    with pytest.raises(OperationError, match="being forgotten"):
        resumed.advance("op-1", "committed")

    def rediscovered_evidence(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cleanup retry rediscovered filesystem evidence")

    monkeypatch.setattr(
        operations, "_pending_answer_evidence", rediscovered_evidence
    )
    assert resumed.forget(["op-1"]) == 1
    assert not blob.exists()
    assert OperationJournal.load(path).operations == {}


@pytest.mark.parametrize(
    "namespace_state",
    ["unchanged", "missing", "renamed", "replaced"],
)
def test_forget_resumes_a_crash_after_the_exact_artifact_was_moved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace_state: str,
) -> None:
    """The no-clobber retirement move is itself a crash seam.

    Once the public name is gone, a retry cannot rediscover a randomly named
    private destination from the journal.  The destination therefore has to be
    derivable from the already-durable binding, and the retry has to remove it
    before dropping the cleanup tombstone.
    """
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    captured = journal.capture_result(
        "op-1",
        lambda: capture_artifact(
            path, "op-1", b"captured provider answer"
        ),
    )
    assert captured.artifact is not None
    artifact = captured.artifact
    OperationJournal.load(path).advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    lock_root = tmp_path / "private-locks"

    script = r"""
import os
import sys
from pathlib import Path

from japanese_anki import io
from japanese_anki.operations import OperationJournal

path = Path(sys.argv[1])
lock_root = Path(sys.argv[2])
io._path_lock_root = lambda: lock_root
real_move = io._rename_entry_exclusive_between

def die_after_move(source_fd, source, destination_fd, destination):
    real_move(source_fd, source, destination_fd, destination)
    if source == "op-1.json":
        os._exit(102)

io._rename_entry_exclusive_between = die_after_move
OperationJournal.load(path).forget(["op-1"])
raise AssertionError("the retirement move did not reach the injected crash")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(lock_root)],
        cwd=ROOT,
        env=environment,
        check=False,
        timeout=10,
    )

    assert result.returncode == 102
    assert not target.exists()
    retirement = lock_root / "retired-writes"
    retired = list(retirement.iterdir())
    assert len(retired) == 1
    assert retired[0].read_bytes() == b"captured provider answer"
    assert OperationJournal.load(path).operations["op-1"].cleanup is not None

    pending = target.parent
    detached = tmp_path / "detached-pending"
    if namespace_state == "missing":
        pending.rmdir()
    elif namespace_state in {"renamed", "replaced"}:
        pending.rename(detached)
        if namespace_state == "replaced":
            pending.mkdir()
            (pending / "keep.txt").write_bytes(b"replacement namespace")

    monkeypatch.setattr(io_module, "_path_lock_root", lambda: lock_root)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert list(retirement.iterdir()) == []
    assert not OperationJournal.load(path).operations
    if namespace_state == "renamed":
        assert list(detached.iterdir()) == []
    elif namespace_state == "replaced":
        assert (pending / "keep.txt").read_bytes() == b"replacement namespace"


def test_a_durable_cleanup_decision_does_not_block_fresh_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once forget is durable, the person has accounted for the old call.

    Filesystem retirement can still need a retry, but that mechanical cleanup
    must not wedge every later paid call or describe the old one as needing a
    second money decision.
    """
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)

    def leave_the_cleanup_pending(_binding: Any) -> None:
        raise OperationError("injected cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_artifact", leave_the_cleanup_pending
    )
    with pytest.raises(OperationError, match="injected cleanup pause"):
        journal.forget(["op-1"], force=True)

    reloaded = OperationJournal.load(path)
    assert reloaded.operations["op-1"].cleanup is not None
    assert reloaded.blocking() == []
    assert reloaded.unfinished() == []
    assert reloaded.needing_attention() == []

    fresh = _authorize(reloaded, "op-2")
    assert fresh.state == "authorized"


def test_commit_result_holds_the_journal_lock_across_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forget must linearize wholly before or after the output write. It may
    not make cleanup durable in the seam between commit's state check and its
    persistence callback."""
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)

    real_lock = operations.exclusive_path_lock
    persist_entered = threading.Event()
    release_persist = threading.Event()
    forget_attempted = threading.Event()
    allow_forget_attempt = threading.Event()
    forget_acquired = threading.Event()
    failures: list[BaseException] = []

    @contextlib.contextmanager
    def observed_lock(lock_path: Path):
        forgetting = threading.current_thread().name == "forget-operation"
        if forgetting:
            forget_attempted.set()
            if not allow_forget_attempt.wait(5):
                raise AssertionError("forget lock attempt was never released")
        with real_lock(lock_path):
            if forgetting:
                forget_acquired.set()
            yield

    monkeypatch.setattr(operations, "exclusive_path_lock", observed_lock)

    def persist() -> None:
        persist_entered.set()
        if not release_persist.wait(5):
            raise AssertionError("commit persistence was never released")

    def commit() -> None:
        try:
            OperationJournal.load(path).commit_result("op-1", persist)
        except BaseException as exc:  # noqa: BLE001 - asserted across the thread
            failures.append(exc)

    def forget() -> None:
        try:
            OperationJournal.load(path).forget(["op-1"], force=True)
        except BaseException as exc:  # noqa: BLE001 - asserted across the thread
            failures.append(exc)

    committing = threading.Thread(target=commit, name="commit-operation")
    committing.start()
    assert persist_entered.wait(5)

    forgetting = threading.Thread(target=forget, name="forget-operation")
    forgetting.start()
    assert forget_attempted.wait(5)
    allow_forget_attempt.set()
    assert not forget_acquired.wait(0.5)

    release_persist.set()
    committing.join(5)
    forgetting.join(5)

    assert not committing.is_alive()
    assert not forgetting.is_alive()
    assert forget_acquired.is_set()
    assert failures == []
    assert OperationJournal.load(path).operations == {}
    assert not (tmp_path / artifact.relative_name).exists()


def test_capture_result_validates_the_transition_before_its_callback(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    _authorize(journal)
    captured = False

    def capture() -> str:
        nonlocal captured
        captured = True
        return ".pending/op-1.json"

    with pytest.raises(OperationError, match="cannot move"):
        journal.capture_result("op-1", capture)

    assert captured is False


def test_forget_retries_after_cleanup_wins_but_final_journal_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    blob = tmp_path / artifact.relative_name
    real_write = OperationJournal._write
    writes = 0

    def fail_the_final_write(current: OperationJournal) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OperationError("injected final journal write failure")
        real_write(current)

    monkeypatch.setattr(OperationJournal, "_write", fail_the_final_write)
    with pytest.raises(OperationError, match="final journal write failure"):
        journal.forget(["op-1"])

    assert writes == 2
    assert not blob.exists()
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None

    monkeypatch.setattr(OperationJournal, "_write", real_write)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert OperationJournal.load(path).operations == {}


def test_forget_attempts_every_cleanup_and_keeps_only_the_failed_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    artifacts: dict[str, Path] = {}
    for operation_id in ("op-1", "op-2"):
        _authorize(journal, operation_id)
        journal.advance(operation_id, "dispatching")
        artifact = capture_artifact(
            path, operation_id, f"answer for {operation_id}".encode()
        )
        journal.capture_result(operation_id, lambda artifact=artifact: artifact)
        journal.advance(operation_id, "committed")
        artifacts[operation_id] = tmp_path / artifact.relative_name

    real_retire = operations._retire_artifact
    attempted: list[str] = []

    def fail_only_the_first(binding: Any) -> None:
        attempted.append(binding.path.name)
        if binding.path.name == "op-1.json":
            raise OperationError("injected op-1 cleanup failure")
        real_retire(binding)

    monkeypatch.setattr(operations, "_retire_artifact", fail_only_the_first)

    with pytest.raises(OperationError, match="injected op-1 cleanup failure"):
        journal.forget(["op-1", "op-2"])

    assert attempted == ["op-1.json", "op-2.json"]
    held = OperationJournal.load(path).operations
    assert set(held) == {"op-1"}
    assert held["op-1"].cleanup is not None
    assert artifacts["op-1"].exists()
    assert not artifacts["op-2"].exists()

    monkeypatch.setattr(operations, "_retire_artifact", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert OperationJournal.load(path).operations == {}
    assert not artifacts["op-1"].exists()


def test_cleanup_tombstone_does_not_adopt_a_same_name_artifact_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    real_retire = operations._retire_artifact

    def leave_the_intent_pending(_binding: Any) -> None:
        raise OperationError("injected cleanup pause")

    monkeypatch.setattr(operations, "_retire_artifact", leave_the_intent_pending)
    with pytest.raises(OperationError, match="injected cleanup pause"):
        journal.forget(["op-1"])

    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(b"later answer at the same name")
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    target.unlink()
    replacement.rename(target)

    monkeypatch.setattr(operations, "_retire_artifact", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1

    assert target.read_bytes() == b"later answer at the same name"
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert OperationJournal.load(path).operations == {}


def _leave_artifact_cleanup_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    real_retire = operations._retire_artifact

    def pause_cleanup(_binding: Any) -> None:
        raise OperationError("injected cleanup pause")

    monkeypatch.setattr(operations, "_retire_artifact", pause_cleanup)
    with pytest.raises(OperationError, match="injected cleanup pause"):
        journal.forget(["op-1"])
    monkeypatch.setattr(operations, "_retire_artifact", real_retire)
    target = tmp_path / artifact.relative_name
    return path, target, target.parent


def test_artifact_cleanup_preserves_an_entry_whose_ctime_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, target, _pending = _leave_artifact_cleanup_tombstone(
        tmp_path, monkeypatch
    )
    before = target.stat()

    os.chmod(target, before.st_mode ^ 0o100)

    after = target.stat()
    assert (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns

    assert OperationJournal.load(path).forget(["op-1"]) == 1

    assert target.read_bytes() == b"captured provider answer"
    assert not OperationJournal.load(path).operations


@pytest.mark.parametrize("replacement_kind", ["symlink", "fifo", "directory"])
def test_artifact_cleanup_preserves_a_nonregular_same_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    """A non-regular occupant proves the old regular binding is gone.

    Cleanup neither follows nor removes that replacement, and it must not keep
    an otherwise-settled operation permanently tombstoned.
    """
    path, target, _pending = _leave_artifact_cleanup_tombstone(
        tmp_path, monkeypatch
    )
    target.unlink()
    referent = tmp_path / "outside-answer"
    if replacement_kind == "symlink":
        referent.write_bytes(b"outside replacement")
        target.symlink_to(referent)
    elif replacement_kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    replacement = os.lstat(target)

    assert OperationJournal.load(path).forget(["op-1"]) == 1

    current = os.lstat(target)
    assert (current.st_dev, current.st_ino, current.st_mode) == (
        replacement.st_dev,
        replacement.st_ino,
        replacement.st_mode,
    )
    if replacement_kind == "symlink":
        assert referent.read_bytes() == b"outside replacement"
    assert not OperationJournal.load(path).operations


def test_artifact_cleanup_clears_when_its_bound_namespace_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed checkout can recreate directories but not their inodes.
    Once the old lexical namespace is absent, retry must not wedge forever or
    search elsewhere for the inode it once bound."""
    path, target, pending = _leave_artifact_cleanup_tombstone(
        tmp_path, monkeypatch
    )
    detached = tmp_path / "detached-pending"
    pending.rename(detached)

    def unexpected_retirement(*_args: Any, **_kwargs: Any) -> bool:
        pytest.fail("cleanup searched outside its missing bound namespace")

    monkeypatch.setattr(operations, "_retire_exact_entry", unexpected_retirement)

    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not OperationJournal.load(path).operations
    assert not pending.exists()
    assert (detached / target.name).read_bytes() == b"captured provider answer"


@pytest.mark.parametrize("replacement_kind", ["directory", "file", "symlink"])
def test_artifact_cleanup_never_adopts_a_replaced_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    """A same-path namespace in another checkout is a replacement even when
    it carries copied names and bytes. Clear the unreachable old decision
    without inspecting or deleting any replacement occupant."""
    path, target, pending = _leave_artifact_cleanup_tombstone(
        tmp_path, monkeypatch
    )
    detached = tmp_path / "detached-pending"
    pending.rename(detached)
    replacement_root = pending
    replacement_target: Path | None = None
    if replacement_kind == "directory":
        replacement_root.mkdir()
        replacement_target = replacement_root / target.name
        replacement_target.write_bytes(b"captured provider answer")
        (replacement_root / "keep.txt").write_bytes(b"replacement sentinel")
    elif replacement_kind == "file":
        replacement_root.write_bytes(b"replacement non-directory")
    else:
        outside = tmp_path / "outside-pending"
        outside.mkdir()
        replacement_target = outside / target.name
        replacement_target.write_bytes(b"outside replacement answer")
        replacement_root.symlink_to(outside, target_is_directory=True)
    replacement_identity = (
        (replacement_target.stat().st_dev, replacement_target.stat().st_ino)
        if replacement_target is not None
        else None
    )

    def unexpected_retirement(*_args: Any, **_kwargs: Any) -> bool:
        pytest.fail("cleanup inspected names in a replacement namespace")

    monkeypatch.setattr(operations, "_retire_exact_entry", unexpected_retirement)

    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not OperationJournal.load(path).operations
    assert (detached / target.name).read_bytes() == b"captured provider answer"
    if replacement_kind == "directory":
        assert replacement_target is not None
        assert replacement_target.read_bytes() == b"captured provider answer"
        assert (replacement_root / "keep.txt").read_bytes() == b"replacement sentinel"
    elif replacement_kind == "file":
        assert replacement_root.read_bytes() == b"replacement non-directory"
    else:
        assert replacement_target is not None
        assert replacement_target.read_bytes() == b"outside replacement answer"
    if replacement_target is not None:
        assert (
            replacement_target.stat().st_dev,
            replacement_target.stat().st_ino,
        ) == replacement_identity


def test_exact_artifact_that_cannot_retire_keeps_its_cleanup_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    real_retire = operations._retire_exact_entry

    monkeypatch.setattr(
        operations,
        "_retire_exact_entry",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(OperationError, match="Exact artifact remained"):
        journal.forget(["op-1"])

    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert target.read_bytes() == b"captured provider answer"

    monkeypatch.setattr(operations, "_retire_exact_entry", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not target.exists()


def test_malformed_cleanup_json_never_grants_artifact_deletion_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name

    def leave_the_intent_pending(_binding: Any) -> None:
        raise OperationError("injected cleanup pause")

    monkeypatch.setattr(operations, "_retire_artifact", leave_the_intent_pending)
    with pytest.raises(OperationError, match="injected cleanup pause"):
        journal.forget(["op-1"])

    wire = json.loads(path.read_text(encoding="utf-8"))
    cleanup = wire["operations"]["op-1"]["cleanup"]
    cleanup["artifact"]["directory_identity"][1] = True
    path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OperationError, match="invalid cleanup binding"):
        OperationJournal.load(path)

    assert target.read_bytes() == b"captured provider answer"


def test_null_cleanup_cannot_erase_a_durable_forget_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"captured provider answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name

    def leave_the_intent_pending(_binding: Any) -> None:
        raise OperationError("injected cleanup pause")

    monkeypatch.setattr(operations, "_retire_artifact", leave_the_intent_pending)
    with pytest.raises(OperationError, match="injected cleanup pause"):
        journal.forget(["op-1"])

    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["operations"]["op-1"]["cleanup"] = None
    path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OperationError, match="invalid cleanup intent"):
        OperationJournal.load(path)
    assert target.read_bytes() == b"captured provider answer"


def test_artifact_reads_open_direct_names_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular-file stat followed by a FIFO swap must refuse, not hang."""
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b'{"content": []}')
    assert operation.artifact is not None
    target = tmp_path / operation.artifact.relative_name
    real_open = operations.os.open
    observed = 0

    def require_nonblocking(name: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal observed
        if name == target.name and kwargs.get("dir_fd") is not None:
            observed += 1
            assert flags & getattr(operations.os, "O_NONBLOCK", 0)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(operations.os, "open", require_nonblocking)

    assert operations.reply_observation(path, operation).payload == b'{"content": []}'
    assert observed == 1


def test_forgetting_does_not_unlink_through_a_symlinked_pending_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b"original answer")
    assert operation.artifact is not None
    journal.advance("op-1", "committed")
    pending = tmp_path / ".pending"
    held = tmp_path / "held-pending"
    pending.rename(held)
    outside = tmp_path / "outside"
    outside.mkdir()
    answer = outside / "op-1.json"
    answer.write_bytes(b"external answer")
    pending.symlink_to(outside, target_is_directory=True)

    journal.forget(["op-1"])

    assert answer.read_bytes() == b"external answer"
    assert (held / "op-1.json").read_bytes() == b"original answer"
    assert "op-1" not in OperationJournal.load(path).operations


def test_forgetting_does_not_follow_a_pending_artifact_symlink(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    operation = _capture_result(journal, b"original answer")
    assert operation.artifact is not None
    journal.advance("op-1", "committed")
    pending = tmp_path / ".pending"
    target = tmp_path / operation.artifact.relative_name
    target.unlink()
    other = pending / "other.json"
    other.write_bytes(b"another operation's answer")
    target.symlink_to(other.name)

    journal.forget(["op-1"])

    assert other.read_bytes() == b"another operation's answer"
    assert "op-1" not in OperationJournal.load(path).operations


def test_a_receipt_cannot_name_another_operations_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    with pytest.raises(OperationError, match="invalid artifact receipt"):
        journal.capture_result("op-1", lambda: _receipt("op-2"))

    assert OperationJournal.load(path).operations["op-1"].state == "dispatching"


def test_capture_result_refuses_a_receipt_without_its_terminal_wal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")

    with pytest.raises(OperationError, match="not proven"):
        journal.capture_result("op-1", _receipt)

    held = OperationJournal.load(path).operations["op-1"]
    assert held.state == "dispatching"
    assert held.artifact is None


def test_forgetting_preserves_a_regular_file_swapped_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    replacement = target.parent / "replacement.json"
    replacement.write_bytes(b"regular replacement")
    real_read = operations._read_artifact_binding
    swapped = False

    def swap_during_reply_read(binding: object) -> bytes | None:
        nonlocal swapped
        if not swapped:
            target.unlink()
            replacement.rename(target)
            swapped = True
        return real_read(binding)  # type: ignore[arg-type]

    monkeypatch.setattr(
        operations,
        "_read_artifact_binding",
        swap_during_reply_read,
    )

    journal.forget(["op-1"])

    assert swapped
    assert target.read_bytes() == b"regular replacement"
    assert "op-1" not in OperationJournal.load(path).operations


def test_forgetting_preserves_a_regular_file_swapped_at_the_delete_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    replacement = target.parent / "replacement.json"
    replacement.write_bytes(b"regular replacement")
    real_unlink = operations.os.unlink
    real_move = io_module._rename_entry_exclusive_between
    swapped = False
    retired: Path | None = None

    def install_replacement(directory_fd: int) -> None:
        nonlocal swapped
        real_unlink(target.name, dir_fd=directory_fd)
        operations.os.rename(
            replacement.name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        swapped = True

    def swap_at_retirement_move(
        source_directory_fd: int,
        left: str,
        destination_directory_fd: int,
        right: str,
    ) -> None:
        nonlocal retired
        if left == target.name and not swapped:
            install_replacement(source_directory_fd)
            retired = (
                Path(operations.os.path.realpath(io_module._path_lock_root()))
                / "retired-writes"
                / right
            )
        real_move(
            source_directory_fd,
            left,
            destination_directory_fd,
            right,
        )

    monkeypatch.setattr(
        io_module,
        "_rename_entry_exclusive_between",
        swap_at_retirement_move,
    )

    journal.forget(["op-1"])

    assert swapped
    assert target.read_bytes() == b"regular replacement"
    assert retired is not None
    assert not retired.exists()
    assert "op-1" not in OperationJournal.load(path).operations


def test_forgetting_reports_a_raced_file_that_cannot_be_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    replacement = target.parent / "replacement.json"
    replacement.write_bytes(b"regular replacement")
    real_move = io_module._rename_entry_exclusive_between
    moved_replacement = False
    restoration_refused = False
    retired: Path | None = None

    def race_then_refuse_restoration(
        source_directory_fd: int,
        left: str,
        destination_directory_fd: int,
        right: str,
    ) -> None:
        nonlocal moved_replacement, restoration_refused, retired
        if left == target.name and not moved_replacement:
            operations.os.unlink(target.name, dir_fd=source_directory_fd)
            operations.os.rename(
                replacement.name,
                target.name,
                src_dir_fd=source_directory_fd,
                dst_dir_fd=source_directory_fd,
            )
            real_move(
                source_directory_fd,
                left,
                destination_directory_fd,
                right,
            )
            retired = (
                Path(operations.os.path.realpath(io_module._path_lock_root()))
                / "retired-writes"
                / right
            )
            moved_replacement = True
            return
        if moved_replacement and right == target.name:
            restoration_refused = True
            raise OSError("injected restoration failure")
        real_move(
            source_directory_fd,
            left,
            destination_directory_fd,
            right,
        )

    monkeypatch.setattr(
        io_module,
        "_rename_entry_exclusive_between",
        race_then_refuse_restoration,
    )

    with pytest.raises(OperationError, match="could not be retired safely"):
        journal.forget(["op-1"])

    assert moved_replacement and restoration_refused
    assert not target.exists()
    assert retired is not None
    assert retired.read_bytes() == b"regular replacement"
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.forced is False
    assert journal.forget(["op-1"]) == 1
    assert "op-1" not in OperationJournal.load(path).operations
    retired.unlink()


def test_forgetting_never_adopts_a_preexisting_retirement_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    real_move = io_module._rename_entry_exclusive_between
    swapped = False
    occupied: Path | None = None

    def preoccupy_retirement_name(
        source_directory_fd: int,
        left: str,
        destination_directory_fd: int,
        right: str,
    ) -> None:
        nonlocal occupied, swapped
        if left == target.name and not swapped:
            descriptor = operations.os.open(
                right,
                operations.os.O_WRONLY | operations.os.O_CREAT | operations.os.O_EXCL,
                0o600,
                dir_fd=destination_directory_fd,
            )
            try:
                operations.os.write(descriptor, b"must preserve")
            finally:
                operations.os.close(descriptor)
            occupied = (
                Path(operations.os.path.realpath(io_module._path_lock_root()))
                / "retired-writes"
                / right
            )
            swapped = True
        real_move(
            source_directory_fd,
            left,
            destination_directory_fd,
            right,
        )

    monkeypatch.setattr(
        io_module,
        "_rename_entry_exclusive_between",
        preoccupy_retirement_name,
    )

    journal.forget(["op-1"])

    assert swapped
    assert occupied is not None
    assert occupied.read_bytes() == b"must preserve"
    occupied.unlink()
    assert "op-1" not in OperationJournal.load(path).operations


def test_forgetting_preserves_a_same_size_edit_before_the_retirement_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"old-content")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    expected_mtime = target.stat().st_mtime_ns
    real_move = io_module._rename_entry_exclusive_between
    changed = False
    retired: Path | None = None

    def edit_then_move(
        source_directory_fd: int,
        left: str,
        destination_directory_fd: int,
        right: str,
    ) -> None:
        nonlocal changed, retired
        if left == target.name and not changed:
            descriptor = operations.os.open(
                target.name,
                operations.os.O_WRONLY,
                dir_fd=source_directory_fd,
            )
            try:
                operations.os.write(descriptor, b"new-content")
                operations.os.fsync(descriptor)
            finally:
                operations.os.close(descriptor)
            operations.os.utime(
                target.name,
                ns=(expected_mtime, expected_mtime),
                dir_fd=source_directory_fd,
                follow_symlinks=False,
            )
            changed = True
            retired = (
                Path(operations.os.path.realpath(io_module._path_lock_root()))
                / "retired-writes"
                / right
            )
        real_move(
            source_directory_fd,
            left,
            destination_directory_fd,
            right,
        )

    monkeypatch.setattr(
        io_module,
        "_rename_entry_exclusive_between",
        edit_then_move,
    )

    journal.forget(["op-1"])

    assert changed
    assert target.read_bytes() == b"new-content"
    assert retired is not None
    assert not retired.exists()
    assert "op-1" not in OperationJournal.load(path).operations


def test_successful_forget_does_not_unlink_a_private_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    real_unlink = operations.os.unlink
    private_unlinks: list[str] = []

    def observe_unlink(name: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(name, str) and name.startswith(f".{target.name}."):
            private_unlinks.append(name)
        real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(operations.os, "unlink", observe_unlink)

    journal.forget(["op-1"])

    retained = [
        candidate
        for candidate in target.parent.glob(f".{target.name}.*")
        if candidate.is_file()
    ]
    assert not target.exists()
    assert private_unlinks == []
    assert retained == []
    assert "op-1" not in OperationJournal.load(path).operations


def test_forgetting_preserves_an_artifact_whose_link_count_changed(tmp_path: Path) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    payload = b"original captured answer"
    artifact = capture_artifact(path, "op-1", payload)
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    backup = tmp_path / "answer-backup.json"
    operations.os.link(target, backup)

    journal.forget(["op-1"])

    # Adding the backup changes ctime after the durable receipt. Cleanup has no
    # authority to infer that this is a benign change, so it preserves both
    # names rather than weakening the five-field identity proof.
    assert target.read_bytes() == payload
    assert backup.read_bytes() == payload


def test_forgetting_does_not_rollback_over_a_replacement_after_move_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.capture_result("op-1", lambda: artifact)
    journal.advance("op-1", "committed")
    target = tmp_path / artifact.relative_name
    replacement = target.parent / "replacement.json"
    replacement.write_bytes(b"regular replacement")
    real_move = io_module._rename_entry_exclusive_between
    swapped = False
    retired: Path | None = None

    def replace_then_fail(
        source_directory_fd: int,
        left: str,
        destination_directory_fd: int,
        right: str,
    ) -> None:
        nonlocal retired, swapped
        if left == target.name and not swapped:
            real_move(
                source_directory_fd,
                left,
                destination_directory_fd,
                right,
            )
            operations.os.rename(
                replacement.name,
                target.name,
                src_dir_fd=source_directory_fd,
                dst_dir_fd=source_directory_fd,
            )
            retired = (
                Path(operations.os.path.realpath(io_module._path_lock_root()))
                / "retired-writes"
                / right
            )
            swapped = True
            raise OSError("exclusive move failed")
        real_move(
            source_directory_fd,
            left,
            destination_directory_fd,
            right,
        )

    monkeypatch.setattr(
        io_module,
        "_rename_entry_exclusive_between",
        replace_then_fail,
    )

    journal.forget(["op-1"])

    assert swapped
    assert target.read_bytes() == b"regular replacement"
    assert retired is not None
    assert not retired.exists()
    assert "op-1" not in OperationJournal.load(path).operations


def test_an_unfinished_operation_keeps_its_artifact(tmp_path: Path) -> None:
    """The buffer for work someone still has to decide about cannot be swept
    away: this reply was paid for and never became staging."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(
        tmp_path / "operations.json", "op-1", b'{"content": []}'
    )
    journal.capture_result("op-1", lambda: artifact)

    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"])

    assert (tmp_path / artifact.relative_name).exists()


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
    journal.capture_result("op-1", lambda: artifact)

    with pytest.raises(OperationError, match="its reply arrived") as refusal:
        journal.end("op-1")

    assert "'janki operations --forget op-1 --force'" in str(refusal.value)
    assert "operations --forget --force" not in str(refusal.value)

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
    journal.capture_result("op-1", lambda: artifact)

    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"])
    assert (tmp_path / artifact.relative_name).exists()

    assert journal.forget(["op-1"], force=True) == 1
    assert not (tmp_path / artifact.relative_name).exists()


def test_a_state_the_journal_does_not_know_is_refused_rather_than_raising(
    tmp_path: Path,
) -> None:
    """`advance_refusal` is exported for callers about to do something they
    cannot take back, which means it is asked about states it has not
    validated. The answer to "may this move?" is no — a traceback instead
    would send a caller holding a paid answer down its own error path."""
    assert advance_refusal("op-1", "not-a-state", "committed", _receipt())
    assert advance_refusal("op-1", "result_captured", "not-a-state", _receipt())
    # And the two it exists to answer, so the refusals above are not merely
    # this function refusing everything.
    assert advance_refusal(
        "op-1", "result_captured", "committed", _receipt()
    ) == ""
    assert "captured provider answer" in advance_refusal(
        "op-1", "result_captured", "committed", None
    )


# --- one paid call at a time ------------------------------------------------


def test_a_call_in_flight_refuses_a_second_authorization(tmp_path: Path) -> None:
    """The rule, not a warning about it. A page that only displayed this would
    be read when it rendered and acted on when a button was clicked; two
    callers could pass that and both spend. This refuses under the lock."""
    journal = _journal(tmp_path)
    _authorize(journal, "first")
    journal.advance("first", "dispatching")

    with pytest.raises(
        OperationError, match="will not start another paid call"
    ) as refusal:
        _authorize(journal, "second")

    assert "operations --end" not in str(refusal.value)
    assert "shows it and the action that settles it" in str(refusal.value)

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
    _capture_result(journal, b"{}", "first")

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
    _capture_result(journal, b"{}", "first")
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
    _capture_result(journal, b"{}")
    journal.advance("op-1", "committed")

    with pytest.raises(OperationError, match="nothing left to end"):
        journal.end("op-1")


def test_ending_a_call_the_journal_never_had_is_refused(tmp_path: Path) -> None:
    journal = _journal(tmp_path)

    with pytest.raises(OperationError, match="No operation"):
        journal.end("never-existed")


# --- the edges where a paid answer could be forgotten ------------------------


def test_a_captured_answer_has_nowhere_to_go_but_committed(tmp_path: Path) -> None:
    """The table is the enforcement, not the caller. `end` refuses a captured
    answer, but `advance` re-checks only this table under the lock — so a
    reply landing between end's read and its write would be recorded as an
    unknown outcome while its bytes sat on disk. A state that must never be
    reachable has to be unreachable here."""
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    _capture_result(journal, b"{}")

    with pytest.raises(OperationError, match="cannot move"):
        journal.advance("op-1", "outcome_unknown", detail="giving up")


def test_a_reply_that_arrived_unrecorded_is_not_declared_lost(
    tmp_path: Path,
) -> None:
    """`capture_artifact` writes the bytes and *then* the journal names them.
    A process dying in that gap leaves the answer on disk under the operation's
    own id with the entry still saying the call is in flight — and ending the
    call there would send somebody to buy an answer they already have."""
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    # The write landed; the advance that would have recorded it did not.
    capture_artifact(path, "op-1", b'{"content": [{"type": "text", "text": "hi"}]}')
    assert OperationJournal.load(path).operations["op-1"].artifact is None

    with pytest.raises(OperationError, match="its reply arrived"):
        journal.end("op-1")


def test_an_unrecorded_reply_is_not_dropped_without_saying_so(
    tmp_path: Path,
) -> None:
    """And `forget` sees it under the same name, so the answer cannot be swept
    away by someone tidying up an entry that looks empty."""
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    capture_artifact(path, "op-1", b'{"content": []}')
    journal.advance("op-1", "outcome_unknown", detail="gave up waiting")

    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"], force=False)


def test_a_complete_wal_whose_bound_read_races_is_recorded_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    capture_artifact(path, "op-1", b"paid provider answer")
    monkeypatch.setattr(operations, "_read_bound_write_evidence", lambda _value: None)

    with pytest.raises(OperationError, match="records a captured reply"):
        journal.end("op-1")
    with pytest.raises(OperationError, match="--force"):
        journal.forget(["op-1"])

    assert OperationJournal.load(path).operations["op-1"].state == "dispatching"


@pytest.mark.parametrize(
    "occupant", [b"different bytes", b"paid provider answer"],
    ids=["different-bytes", "byte-identical"],
)
def test_a_no_wal_lexical_name_is_never_fresh_bound(
    tmp_path: Path, occupant: bytes
) -> None:
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "outcome_unknown", detail="gave up waiting")
    pending = tmp_path / ".pending"
    pending.mkdir()
    target = pending / "op-1.json"
    target.write_bytes(occupant)

    observation = operations.reply_observation(
        path, OperationJournal.load(path).operations["op-1"]
    )

    assert not observation.recorded
    assert observation.payload is None
    assert journal.forget(["op-1"]) == 1
    assert target.read_bytes() == occupant


@pytest.mark.parametrize(
    "replacement_bytes",
    [b"different replacement", b"original captured answer"],
    ids=["different-bytes", "byte-identical"],
)
def test_forgetting_an_unrecorded_reply_never_rebinds_its_replacement(
    tmp_path: Path, replacement_bytes: bytes
) -> None:
    """The recovery lookup and deletion must share one file identity.

    A reply can land before its journal advance. If the public pending name is
    replaced after discovering that reply, a forced forget may remove only the
    discovered file, never grant fresh deletion authority to the replacement.
    """
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    artifact = capture_artifact(path, "op-1", b"original captured answer")
    journal.advance("op-1", "outcome_unknown", detail="gave up waiting")
    target = tmp_path / artifact.relative_name
    original_identity = (target.stat().st_dev, target.stat().st_ino)
    replacement = target.parent / "replacement.json"
    replacement.write_bytes(replacement_bytes)
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    assert replacement_identity != original_identity
    target.unlink()
    replacement.rename(target)

    journal.forget(["op-1"])

    assert target.read_bytes() == replacement_bytes
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert "op-1" not in OperationJournal.load(path).operations


@pytest.mark.parametrize(
    "relative_name", ["normalized/vocabulary.json", "../escaped.json"]
)
def test_a_journal_receipt_naming_outside_the_pending_store_is_rejected(
    tmp_path: Path,
    relative_name: str,
) -> None:
    """`data/operations.json` is committed, so it arrives here after merges and
    hand edits. Both readers of this path either open it or *delete* it, and an
    artifact of `../normalized/vocabulary.json` would make forgetting a call a
    way to remove the collection."""
    path = tmp_path / "operations.json"
    # The real shape, and it needs no `..` to be dangerous: the journal lives
    # at `data/operations.json`, so the collection at
    # `data/normalized/vocabulary.json` is already outside `.pending/` and one
    # ordinary relative path away.
    outside = tmp_path / "normalized" / "vocabulary.json"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text('["do not delete me"]', encoding="utf-8")
    escapes = tmp_path.parent / "escaped.json"
    escapes.write_text("also not yours", encoding="utf-8")
    journal = _journal(tmp_path)
    _authorize(journal)
    journal.advance("op-1", "dispatching")
    invalid = ArtifactReceipt(
        relative_name=relative_name,
        directory_identity=(1, 2),
        entry_state=(1, 3, 4, 5, 6),
        content_sha256="a" * 64,
        terminal_marker=None,
    )
    with pytest.raises(OperationError, match="invalid artifact receipt"):
        journal.capture_result("op-1", lambda: invalid)

    assert OperationJournal.load(path).operations["op-1"].state == "dispatching"

    assert outside.read_text(encoding="utf-8") == '["do not delete me"]'
    assert escapes.exists()


@pytest.mark.parametrize(
    "operation_id",
    [pytest.param("../outside", id="traversal"), pytest.param("bad\0id", id="nul")],
)
def test_a_hand_edited_operation_id_never_builds_an_evidence_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_id: str,
) -> None:
    """Committed journal keys cannot authorize an unsafe filesystem name."""
    journal = _journal(tmp_path)
    _authorize(journal, operation_id)
    journal.advance(operation_id, "dispatching")

    def escaped_evidence_path(_path: Path) -> object:
        raise AssertionError("an invalid operation id reached the filesystem")

    monkeypatch.setattr(
        operations, "_bound_write_evidence", escaped_evidence_path
    )

    ended = journal.end(operation_id)
    assert ended.state == "outcome_unknown"
    assert journal.forget([operation_id], force=True) == 1


def test_a_forget_that_refuses_partway_deletes_nothing(tmp_path: Path) -> None:
    """The journal is written before any blob is unlinked. Deleting first
    leaves an entry naming an answer that is gone, which is the one thing
    `capture_artifact`'s ordering exists to prevent — in reverse."""
    path = tmp_path / "operations.json"
    journal = _journal(tmp_path)
    _authorize(journal, "done")
    journal.advance("done", "dispatching")
    kept = capture_artifact(path, "done", b'{"content": []}')
    journal.capture_result("done", lambda: kept)
    journal.advance("done", "committed")
    _authorize(journal, "live")
    journal.advance("live", "dispatching")

    with pytest.raises(OperationError):
        journal.forget(["done", "live"], force=True)

    # Neither the entry nor its blob went anywhere.
    assert (tmp_path / kept.relative_name).exists()
    assert "done" in OperationJournal.load(path).operations


def test_a_reply_object_that_hands_back_parsed_data_is_still_saved(
    tmp_path: Path,
) -> None:
    """Half the HTTP clients in existence return parsed data from `.json()`.
    Converting outside the guard made that either raise — at the one moment
    losing the answer is most expensive — or, for a number, fabricate NUL
    bytes and store them as somebody's paid answer."""

    class ParsedJson:
        def json(self) -> dict:
            return {"content": [{"type": "text", "text": "the answer"}]}

    class NumericJson:
        def to_json(self) -> int:
            return 7

    assert b"the answer" in serialize_response(ParsedJson())
    assert serialize_response(NumericJson()) != b"\x00" * 7
    assert b"7" in serialize_response(NumericJson())
