"""One finite extraction batch, reserved atomically and consumed once.

The ordinary rule stays exactly what it was: one authority, one call, and a
second paid call waits until the first settles. This file is about the single
exception the owner authorized — an exact, finite set of extraction requests
whose authorities are written together, before any of them is dispatched, and
then consumed one at a time under a stored concurrency limit.

Every test here asks the same question the rest of the journal asks: *what do
we know about the money?* A batch is only allowed to exist because reserving
the whole childset in one atomic write answers it no worse than a single
authorization does — the journal never holds half a batch, a child never
dispatches without consuming its own one-use authority, and an unknown outcome
keeps its slot forever rather than being quietly redispatched.

Nothing here calls a provider.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import operations
from japanese_anki.operations import (
    BATCH_OCCUPIED_STATES,
    Operation,
    OperationAuthorization,
    OperationBatch,
    OperationError,
    OperationJournal,
    capture_artifact,
)

MANIFEST = "9" * 64
OTHER_MANIFEST = "8" * 64
SOURCE = "c" * 64


def _journal(tmp_path: Path) -> OperationJournal:
    return OperationJournal.load(tmp_path / "operations.json")


def _child(
    index: int,
    *,
    request_fp: str | None = None,
    source_sha256: str = SOURCE,
    kind: str = "extract",
    operation_id: str | None = None,
    source_file: str | None = None,
) -> OperationAuthorization:
    return OperationAuthorization(
        operation_id=operation_id or f"op-{index}",
        kind=kind,
        source_file=source_file or "lesson-8.pdf",
        source_sha256=source_sha256,
        request_fp=f"{index:064x}" if request_fp is None else request_fp,
        model="claude-opus-5",
    )


def _authorize_batch(
    journal: OperationJournal,
    children: tuple[OperationAuthorization, ...],
    *,
    batch_id: str = "batch-1",
    concurrency_limit: int = 2,
    manifest_sha256: str = MANIFEST,
) -> OperationBatch:
    return journal.authorize_batch(
        batch_id,
        children,
        concurrency_limit=concurrency_limit,
        manifest_sha256=manifest_sha256,
    )


def _claim(
    journal: OperationJournal,
    child: OperationAuthorization,
    *,
    batch_id: str = "batch-1",
    manifest_sha256: str = MANIFEST,
) -> Operation:
    return journal.claim_batch_dispatch(
        child.operation_id,
        batch_id=batch_id,
        request_fp=child.request_fp,
        manifest_sha256=manifest_sha256,
    )


def _settled_journal(tmp_path: Path) -> OperationJournal:
    """A journal file that exists and blocks nothing, so bytes can be compared."""
    journal = _journal(tmp_path)
    journal.authorize(
        "already-done",
        kind="extract",
        source_file="lesson-1.pdf",
        source_sha256="a" * 64,
        request_fp="f" * 64,
        model="claude-opus-5",
    )
    journal.advance("already-done", "canceled_before_send")
    return journal


# --- reserving the whole childset, or none of it ----------------------------


def test_authorize_batch_reserves_every_child_in_one_atomic_write(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    children = (_child(1), _child(2), _child(3))

    batch = _authorize_batch(journal, children, concurrency_limit=2)

    assert batch == OperationBatch(
        batch_id="batch-1",
        child_operation_ids=("op-1", "op-2", "op-3"),
        concurrency_limit=2,
        manifest_sha256=MANIFEST,
    )
    loaded = OperationJournal.load(journal.path)
    assert loaded.batches == {"batch-1": batch}
    assert set(loaded.operations) == {"op-1", "op-2", "op-3"}
    assert [op.state for op in loaded.operations.values()] == ["authorized"] * 3
    assert {op.batch_id for op in loaded.operations.values()} == {"batch-1"}
    assert loaded.operations["op-2"].request_fp == children[1].request_fp
    assert loaded.operations["op-2"].money_may_have_been_spent is False


def test_a_bad_final_child_reserves_nothing_and_changes_no_journal_byte(
    tmp_path: Path,
) -> None:
    """A batch is one decision. A childset that cannot be authorized in full
    has to leave the journal exactly as it found it, or the caller is left
    holding authorities for requests it never agreed to make."""
    journal = _settled_journal(tmp_path)
    before = journal.path.read_bytes()
    children = (_child(1), _child(2), _child(3, source_sha256="not-a-digest"))

    with pytest.raises(OperationError, match="extraction authorization"):
        _authorize_batch(journal, children)

    assert journal.path.read_bytes() == before
    reloaded = OperationJournal.load(journal.path)
    assert reloaded.batches == {}
    assert set(reloaded.operations) == {"already-done"}


def test_a_batch_naming_one_operation_twice_reserves_nothing(
    tmp_path: Path,
) -> None:
    journal = _settled_journal(tmp_path)
    before = journal.path.read_bytes()

    with pytest.raises(OperationError, match="names operation 'op-1' twice"):
        _authorize_batch(journal, (_child(1), _child(2), _child(1, request_fp="e" * 64)))

    assert journal.path.read_bytes() == before
    assert OperationJournal.load(journal.path).batches == {}


def test_a_batch_reserving_one_exact_request_twice_reserves_nothing(
    tmp_path: Path,
) -> None:
    """Two children with the same exact request are the same paid call twice.
    Inside one batch that is a mistake, not a retry: nothing has settled."""
    journal = _settled_journal(tmp_path)
    before = journal.path.read_bytes()
    shared = "d" * 64

    with pytest.raises(OperationError, match="reserves one exact request twice"):
        _authorize_batch(
            journal,
            (_child(1, request_fp=shared), _child(2, request_fp=shared)),
        )

    assert journal.path.read_bytes() == before
    assert OperationJournal.load(journal.path).operations.get("op-1") is None


def test_identical_source_bytes_may_carry_distinct_exact_requests(
    tmp_path: Path,
) -> None:
    """The point of a table extraction batch: one PDF, many exact page requests.
    Refusing an equal source digest would refuse the only case this exists for."""
    journal = _journal(tmp_path)
    children = (
        _child(1, source_sha256=SOURCE, source_file="lesson-8.pdf"),
        _child(2, source_sha256=SOURCE, source_file="lesson-8.pdf"),
    )

    batch = _authorize_batch(journal, children)

    assert batch.child_operation_ids == ("op-1", "op-2")
    loaded = OperationJournal.load(journal.path)
    assert {op.source_sha256 for op in loaded.operations.values()} == {SOURCE}
    assert len({op.request_fp for op in loaded.operations.values()}) == 2


def test_a_settled_batch_lets_the_same_exact_request_be_retried_by_a_new_batch(
    tmp_path: Path,
) -> None:
    """Equal request fingerprints across *different* retired batches are a
    legitimate fresh attempt. Fresh operation ids are what make it one."""
    journal = _journal(tmp_path)
    first = _child(1, request_fp="a" * 64)
    _authorize_batch(journal, (first,), concurrency_limit=1)
    _claim(journal, first)
    journal.end("op-1")
    assert journal.forget(["op-1"]) == 1

    retry = _child(2, request_fp="a" * 64)
    batch = _authorize_batch(
        journal, (retry,), batch_id="batch-2", concurrency_limit=1
    )

    assert batch.batch_id == "batch-2"
    loaded = OperationJournal.load(journal.path)
    assert set(loaded.batches) == {"batch-1", "batch-2"}
    assert set(loaded.operations) == {"op-2"}
    assert loaded.operations["op-2"].request_fp == "a" * 64


# --- races -------------------------------------------------------------------


def test_exactly_one_of_a_batch_reservation_and_an_ordinary_authorization_wins(
    tmp_path: Path,
) -> None:
    """A batch is not a way around the one-call rule. Two callers deciding to
    spend at the same instant still get one winner, and a losing batch leaves
    no child behind."""
    path = tmp_path / "operations.json"
    barrier = threading.Barrier(2)
    refused: list[str] = []
    lock = threading.Lock()

    def reserve_batch() -> None:
        barrier.wait()
        try:
            OperationJournal.load(path).authorize_batch(
                "batch-1",
                (_child(1), _child(2), _child(3)),
                concurrency_limit=2,
                manifest_sha256=MANIFEST,
            )
        except OperationError as exc:
            with lock:
                refused.append(str(exc))

    def authorize_one() -> None:
        barrier.wait()
        try:
            OperationJournal.load(path).authorize(
                "solo",
                kind="extract",
                source_file="lesson-9.pdf",
                source_sha256="a" * 64,
                request_fp="f" * 64,
                model="claude-opus-5",
            )
        except OperationError as exc:
            with lock:
                refused.append(str(exc))

    threads = [
        threading.Thread(target=reserve_batch, name="batch"),
        threading.Thread(target=authorize_one, name="ordinary"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(refused) == 1
    loaded = OperationJournal.load(path)
    if loaded.batches:
        assert set(loaded.operations) == {"op-1", "op-2", "op-3"}
    else:
        assert set(loaded.operations) == {"solo"}


def test_concurrent_child_claims_never_exceed_the_stored_concurrency_limit(
    tmp_path: Path,
) -> None:
    """The whole reason a limit is stored rather than counted by the caller:
    four separate handles claiming at one instant must still leave two."""
    path = tmp_path / "operations.json"
    children = tuple(_child(index) for index in (1, 2, 3, 4))
    _authorize_batch(_journal(tmp_path), children, concurrency_limit=2)
    barrier = threading.Barrier(4)
    claimed: list[str] = []
    refused: list[str] = []
    lock = threading.Lock()

    def claim(child: OperationAuthorization) -> None:
        barrier.wait()
        try:
            operation = OperationJournal.load(path).claim_batch_dispatch(
                child.operation_id,
                batch_id="batch-1",
                request_fp=child.request_fp,
                manifest_sha256=MANIFEST,
            )
        except OperationError as exc:
            with lock:
                refused.append(str(exc))
        else:
            with lock:
                claimed.append(operation.operation_id)

    threads = [
        threading.Thread(target=claim, args=(child,), name=child.operation_id)
        for child in children
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(claimed) == 2
    assert len(refused) == 2
    assert all("concurrency limit" in message for message in refused)
    loaded = OperationJournal.load(path)
    live = [op for op in loaded.operations.values() if op.state in BATCH_OCCUPIED_STATES]
    assert sorted(op.operation_id for op in live) == sorted(claimed)
    assert len(loaded.operations) == 4


def test_exactly_one_of_two_identical_claims_for_one_child_wins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    child = _child(1)
    _authorize_batch(_journal(tmp_path), (child, _child(2)), concurrency_limit=2)
    barrier = threading.Barrier(2)
    won: list[Operation] = []
    refused: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        barrier.wait()
        try:
            operation = OperationJournal.load(path).claim_batch_dispatch(
                "op-1",
                batch_id="batch-1",
                request_fp=child.request_fp,
                manifest_sha256=MANIFEST,
            )
        except OperationError as exc:
            with lock:
                refused.append(str(exc))
        else:
            with lock:
                won.append(operation)

    threads = [threading.Thread(target=claim, name=f"claim-{i}") for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(won) == 1
    assert len(refused) == 1
    assert "cannot claim a dispatch slot" in refused[0]
    assert OperationJournal.load(path).operations["op-1"].state == "dispatching"


# --- a claim proves exactly what it is claiming ------------------------------


@pytest.mark.parametrize(
    ("keywords", "arguments"),
    [
        ("No operation batch", {"batch_id": "batch-9"}),
        ("different exact request", {"request_fp": "b" * 64}),
        ("different exact manifest", {"manifest_sha256": OTHER_MANIFEST}),
    ],
)
def test_a_claim_that_does_not_match_its_batch_is_refused_unchanged(
    tmp_path: Path,
    keywords: str,
    arguments: dict[str, str],
) -> None:
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child, _child(2)))
    before = journal.path.read_bytes()
    call: dict[str, str] = {
        "batch_id": "batch-1",
        "request_fp": child.request_fp,
        "manifest_sha256": MANIFEST,
        **arguments,
    }

    with pytest.raises(OperationError, match=keywords):
        journal.claim_batch_dispatch("op-1", **call)

    assert journal.path.read_bytes() == before
    assert OperationJournal.load(journal.path).operations["op-1"].state == "authorized"


def test_a_claim_for_an_operation_outside_the_batch_is_refused(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    _authorize_batch(journal, (_child(1),), concurrency_limit=1)
    # The first batch has to settle before a second may be reserved — the
    # global one-call gate is untouched by this capability. Its membership
    # stays durable, which is exactly what makes op-2 a non-member of it.
    journal.end("op-1")
    _authorize_batch(
        journal, (_child(2),), batch_id="batch-2", concurrency_limit=1
    )
    before = journal.path.read_bytes()

    with pytest.raises(OperationError, match="not a member of operation batch"):
        journal.claim_batch_dispatch(
            "op-2",
            batch_id="batch-1",
            request_fp=_child(2).request_fp,
            manifest_sha256=MANIFEST,
        )

    assert journal.path.read_bytes() == before
    assert OperationJournal.load(journal.path).operations["op-2"].state == "authorized"


def test_a_forgotten_batch_never_returns_its_identifiers(tmp_path: Path) -> None:
    """Membership outlives the rows. A retired child id is still spent, and so
    is the batch id — otherwise a replayed request rides on a decision that was
    already used once."""
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child,), concurrency_limit=1)
    _claim(journal, child)
    journal.end("op-1")
    assert journal.forget(["op-1"]) == 1
    reloaded = OperationJournal.load(journal.path)
    assert "op-1" not in reloaded.operations
    assert reloaded.batches["batch-1"].child_operation_ids == ("op-1",)

    with pytest.raises(OperationError, match="batch authority is one-use"):
        _authorize_batch(reloaded, (_child(9),), batch_id="batch-1")
    with pytest.raises(OperationError, match="retained by operation batch"):
        _authorize_batch(
            reloaded, (_child(1, request_fp="b" * 64),), batch_id="batch-3"
        )
    with pytest.raises(OperationError, match="retained by operation batch"):
        reloaded.authorize(
            "op-1",
            kind="extract",
            source_file="lesson-8.pdf",
            source_sha256=SOURCE,
            request_fp="b" * 64,
            model="claude-opus-5",
        )

    assert OperationJournal.load(journal.path).operations == {}


# --- the claim is the only door ----------------------------------------------


def test_advance_cannot_dispatch_a_batched_child_that_never_claimed(
    tmp_path: Path,
) -> None:
    """`advance` is what every ordinary caller uses, so leaving it open would
    make the concurrency limit advisory."""
    batched = tmp_path / "batched"
    plain = tmp_path / "plain"
    batched.mkdir()
    plain.mkdir()
    journal = _journal(batched)
    _authorize_batch(journal, (_child(1), _child(2)), concurrency_limit=1)
    before = journal.path.read_bytes()

    with pytest.raises(OperationError, match="must dispatch through its batch claim"):
        journal.advance("op-1", "dispatching")

    assert journal.path.read_bytes() == before
    assert OperationJournal.load(journal.path).operations["op-1"].state == "authorized"

    ordinary = _journal(plain)
    ordinary.authorize(
        "solo",
        kind="extract",
        source_file="lesson-9.pdf",
        source_sha256="a" * 64,
        request_fp="f" * 64,
        model="claude-opus-5",
    )
    assert ordinary.advance("solo", "dispatching").state == "dispatching"


def test_a_batched_child_may_still_be_canceled_or_ended_by_hand(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child, _child(2)), concurrency_limit=2)

    canceled = journal.advance("op-2", "canceled_before_send")
    _claim(journal, child)
    ended = journal.end("op-1")

    assert canceled.state == "canceled_before_send"
    assert canceled.batch_id == "batch-1"
    assert ended.state == "outcome_unknown"
    assert ended.batch_id == "batch-1"


@pytest.mark.parametrize("settled", ["dispatching", "running", "outcome_unknown", "committed"])
def test_a_claim_can_never_be_repeated_from_a_later_state(
    tmp_path: Path,
    settled: str,
) -> None:
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child, _child(2)), concurrency_limit=2)
    _claim(journal, child)
    if settled == "running":
        journal.advance("op-1", "running")
    elif settled == "outcome_unknown":
        journal.end("op-1")
    elif settled == "committed":
        journal.capture_result(
            "op-1", lambda: capture_artifact(journal.path, "op-1", b"{}")
        )
        journal.advance("op-1", "committed")
    assert OperationJournal.load(journal.path).operations["op-1"].state == settled

    with pytest.raises(OperationError, match="cannot claim a dispatch slot"):
        _claim(journal, child)

    assert OperationJournal.load(journal.path).operations["op-1"].state == settled


def test_an_unknown_outcome_holds_its_slot_and_a_captured_reply_frees_one(
    tmp_path: Path,
) -> None:
    """The rule this batch exists under: nothing redispatches an unknown
    outcome, so its slot is spent for good. A complete reply is a finished
    call, so its slot comes back."""
    journal = _journal(tmp_path)
    children = tuple(_child(index) for index in (1, 2, 3))
    _authorize_batch(journal, children, concurrency_limit=2)
    _claim(journal, children[0])
    _claim(journal, children[1])
    journal.end("op-1")

    assert OperationJournal.load(journal.path).operations["op-1"].state == (
        "outcome_unknown"
    )
    with pytest.raises(OperationError, match="concurrency limit"):
        _claim(journal, children[2])

    journal.capture_result(
        "op-2", lambda: capture_artifact(journal.path, "op-2", b"{}")
    )
    claimed = _claim(journal, children[2])

    assert claimed.state == "dispatching"
    assert claimed.batch_id == "batch-1"


@pytest.mark.parametrize(
    "member_state", ["authorized", "running", "result_captured", "outcome_unknown"]
)
def test_an_ordinary_authorization_is_blocked_by_every_unsettled_child(
    tmp_path: Path,
    member_state: str,
) -> None:
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child,), concurrency_limit=1)
    if member_state != "authorized":
        _claim(journal, child)
    if member_state == "running":
        journal.advance("op-1", "running")
    elif member_state == "result_captured":
        journal.capture_result(
            "op-1", lambda: capture_artifact(journal.path, "op-1", b"{}")
        )
    elif member_state == "outcome_unknown":
        journal.end("op-1")

    with pytest.raises(OperationError, match="will not start another paid call"):
        journal.authorize(
            "solo",
            kind="extract",
            source_file="lesson-9.pdf",
            source_sha256="a" * 64,
            request_fp="f" * 64,
            model="claude-opus-5",
        )

    assert "solo" not in OperationJournal.load(journal.path).operations


# --- the capture machinery keeps working, unchanged --------------------------


def test_the_whole_lifecycle_keeps_one_childs_batch_identity(
    tmp_path: Path,
) -> None:
    """Response capture is prepared while the child is still `authorized`,
    before it claims — exactly as an unbatched operation does — and the batch
    identity survives every reconstruction after it."""
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child, _child(2)), concurrency_limit=1)

    receipt = journal.begin_response_capture("op-1")
    prepared = OperationJournal.load(journal.path).operations["op-1"]
    assert prepared.state == "authorized"
    assert prepared.batch_id == "batch-1"

    claimed = _claim(journal, child)
    assert claimed.state == "dispatching"
    assert claimed.batch_id == "batch-1"
    assert claimed.response_spool == receipt

    journal.append_response_frame("op-1", '{"type": "result", "result": "ok"}')
    appended = OperationJournal.load(journal.path).operations["op-1"]
    assert appended.batch_id == "batch-1"
    assert appended.response_spool is not None
    assert appended.response_spool.frame_count == 1
    assert journal.read_response_frames("op-1") == (
        '{"type": "result", "result": "ok"}',
    )

    captured = journal.capture_result(
        "op-1",
        lambda: capture_artifact(journal.path, "op-1", b'{"content": []}'),
    )
    assert captured.batch_id == "batch-1"
    committed = journal.advance("op-1", "committed")
    assert committed.batch_id == "batch-1"

    assert journal.forget(["op-1"]) == 1
    reloaded = OperationJournal.load(journal.path)
    assert "op-1" not in reloaded.operations
    assert reloaded.operations["op-2"].batch_id == "batch-1"
    assert reloaded.batches["batch-1"].child_operation_ids == ("op-1", "op-2")


def test_an_interrupted_cleanup_intent_preserves_its_batch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _journal(tmp_path)
    child = _child(1)
    _authorize_batch(journal, (child, _child(2)), concurrency_limit=1)
    _claim(journal, child)
    journal.capture_result(
        "op-1", lambda: capture_artifact(journal.path, "op-1", b"paid answer")
    )

    def refuse_retirement(_binding: object) -> None:
        raise OperationError("injected cleanup failure")

    monkeypatch.setattr(operations, "_retire_artifact", refuse_retirement)
    with pytest.raises(OperationError, match="cleanup remains recorded"):
        journal.forget(["op-1"], force=True)

    held = OperationJournal.load(journal.path).operations["op-1"]
    assert held.cleanup is not None
    assert held.batch_id == "batch-1"

    monkeypatch.undo()
    assert OperationJournal.load(journal.path).forget(["op-1"]) == 1
    assert OperationJournal.load(journal.path).batches["batch-1"].child_operation_ids == (
        "op-1",
        "op-2",
    )


# --- what an unreadable journal must refuse ----------------------------------


def _wire(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _corrupt_manifest(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-1"]["manifest_sha256"] = "not-a-digest"


def _corrupt_limit(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-1"]["concurrency_limit"] = 0


def _boolean_limit(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-1"]["concurrency_limit"] = True


def _empty_childset(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-1"]["children"] = []


def _duplicate_child(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-1"]["children"] = ["op-1", "op-1"]


def _missing_batch(wire: dict[str, Any]) -> None:
    wire["operations"]["op-1"]["batch_id"] = "batch-404"


def _stripped_membership(wire: dict[str, Any]) -> None:
    del wire["operations"]["op-1"]["batch_id"]


def _shared_child(wire: dict[str, Any]) -> None:
    wire["batches"]["batch-2"] = {
        "children": ["op-2"],
        "concurrency_limit": 1,
        "manifest_sha256": OTHER_MANIFEST,
    }


def _batches_are_a_list(wire: dict[str, Any]) -> None:
    wire["batches"] = []


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_manifest,
        _corrupt_limit,
        _boolean_limit,
        _empty_childset,
        _duplicate_child,
        _missing_batch,
        _stripped_membership,
        _shared_child,
        _batches_are_a_list,
    ],
)
def test_a_malformed_batch_journal_is_refused_mechanically(
    tmp_path: Path,
    corrupt: Callable[[dict[str, Any]], None],
) -> None:
    journal = _journal(tmp_path)
    _authorize_batch(journal, (_child(1), _child(2)))
    wire = _wire(journal.path)
    corrupt(wire)
    journal.path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OperationError):
        OperationJournal.load(journal.path)


def test_an_ordinary_journal_keeps_its_exact_wire_shape(tmp_path: Path) -> None:
    """No batch, no trace of one: the ordinary journal a released janki writes
    must not change shape because this capability exists."""
    journal = _journal(tmp_path)
    journal.authorize(
        "op-1",
        kind="extract",
        source_file="lesson-8.pdf",
        source_sha256="a" * 64,
        request_fp="f" * 64,
        model="claude-opus-5",
    )
    journal.advance("op-1", "dispatching")

    wire = _wire(journal.path)

    assert set(wire) == {"version", "operations"}
    assert "batch_id" not in wire["operations"]["op-1"]
    loaded = OperationJournal.load(journal.path)
    assert loaded.batches == {}
    assert loaded.operations["op-1"].batch_id == ""
    assert loaded.operations["op-1"].to_dict() == wire["operations"]["op-1"]
