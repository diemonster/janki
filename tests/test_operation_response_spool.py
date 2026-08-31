"""Crash-safe response-frame capture for streaming paid providers."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import operations
from japanese_anki.operations import (
    OperationError,
    OperationJournal,
    capture_artifact,
)


def _journal(tmp_path: Path) -> OperationJournal:
    journal = OperationJournal.load(tmp_path / "operations.json")
    journal.authorize(
        "op-1",
        kind="audio-realtime",
        source_file="record/example-0",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="gpt-realtime-1.5",
    )
    return journal


def _spool_path(tmp_path: Path) -> Path:
    return tmp_path / ".pending" / "op-1.frames"


def _record(payload: bytes) -> bytes:
    length = struct.pack(">Q", len(payload))
    return length + payload + hashlib.sha256(length + payload).digest()


def test_begin_response_capture_persists_only_an_exact_empty_spool_receipt(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)

    receipt = journal.begin_response_capture("op-1")

    spool = _spool_path(tmp_path)
    assert spool.read_bytes() == b""
    assert receipt.relative_name == ".pending/op-1.frames"
    assert receipt.file_identity == (spool.stat().st_dev, spool.stat().st_ino)
    assert receipt.committed_size == 0
    assert receipt.committed_sha256 == hashlib.sha256(b"").hexdigest()
    assert receipt.frame_count == 0
    loaded = OperationJournal.load(journal.path).operations["op-1"]
    assert loaded.state == "authorized"
    assert loaded.response_spool == receipt
    wire = journal.path.read_text(encoding="utf-8")
    assert "response_spool" in wire
    assert "frames" not in loaded.response_spool.to_dict()
    assert "gpt-realtime-1.5" in wire


def test_begin_response_capture_is_idempotent_after_its_receipt_is_durable(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    first = journal.begin_response_capture("op-1")

    second = OperationJournal.load(journal.path).begin_response_capture("op-1")

    assert second == first
    assert list((tmp_path / ".pending").glob("op-1.frames")) == [
        _spool_path(tmp_path)
    ]


def test_begin_response_capture_recovers_its_exact_prejournal_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    real_write = OperationJournal._write

    def fail_receipt_write(current: OperationJournal) -> None:
        if current.operations["op-1"].response_spool is not None:
            raise OperationError("injected journal receipt failure")
        real_write(current)

    monkeypatch.setattr(OperationJournal, "_write", fail_receipt_write)
    with pytest.raises(OperationError, match="journal receipt failure"):
        journal.begin_response_capture("op-1")

    spool = _spool_path(tmp_path)
    identity = (spool.stat().st_dev, spool.stat().st_ino)
    assert OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool is None

    monkeypatch.setattr(OperationJournal, "_write", real_write)
    receipt = OperationJournal.load(journal.path).begin_response_capture("op-1")

    assert receipt.file_identity == identity
    assert receipt == OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool
    assert not list((tmp_path / ".pending").glob(".op-1.frames.*.janki-cas.*"))


def test_a_new_spool_cannot_be_prepared_after_dispatch(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.advance("op-1", "dispatching")

    with pytest.raises(OperationError, match="before dispatch"):
        journal.begin_response_capture("op-1")

    assert not _spool_path(tmp_path).exists()


def test_begin_response_capture_never_builds_a_path_from_an_invalid_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = OperationJournal.load(tmp_path / "operations.json")
    journal.authorize(
        "../outside",
        kind="audio-realtime",
        source_file="record/example-0",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="gpt-realtime-1.5",
    )

    def path_was_used(_path: Path) -> object:
        raise AssertionError("invalid operation id reached the filesystem")

    monkeypatch.setattr(operations, "_bound_write_evidence", path_was_used)

    with pytest.raises(OperationError, match="Invalid operation ID"):
        journal.begin_response_capture("../outside")


def test_response_spool_receipt_survives_every_provider_transition(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    receipt = journal.begin_response_capture("op-1")

    for state in ("dispatching", "running"):
        moved = journal.advance("op-1", state)
        assert moved.response_spool == receipt
        assert OperationJournal.load(journal.path).operations[
            "op-1"
        ].response_spool == receipt

    captured = journal.capture_result(
        "op-1",
        lambda: capture_artifact(journal.path, "op-1", b"terminal"),
    )
    assert captured.response_spool == receipt


def test_response_spool_receipt_survives_an_unknown_outcome(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    receipt = journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")

    ended = journal.end("op-1", detail="socket closed")

    assert ended.state == "outcome_unknown"
    assert ended.response_spool == receipt
    assert OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool == receipt


def test_append_fsyncs_each_exact_length_payload_and_sha_record_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    receipt = journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    real_fsync = os.fsync
    fsynced_spool = False

    def observe_fsync(descriptor: int) -> None:
        nonlocal fsynced_spool
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) == receipt.file_identity:
            fsynced_spool = True
        real_fsync(descriptor)

    monkeypatch.setattr(operations.os, "fsync", observe_fsync)

    journal.append_response_frame("op-1", "雨、まだ止まないの？")

    payload = "雨、まだ止まないの？".encode()
    assert fsynced_spool
    assert _spool_path(tmp_path).read_bytes() == _record(payload)


def test_response_spool_round_trips_complete_utf8_frames_without_parsing_json(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")

    frames = ('{"type":"session.created"}', "not JSON 日本語", "")
    for frame in frames:
        journal.append_response_frame("op-1", frame)

    assert journal.read_response_frames("op-1") == frames


def test_interrupted_partial_append_is_never_reported_as_a_complete_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    receipt = journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    real_write = os.write
    wrote_partial = False

    def interrupt_write(descriptor: int, payload: bytes) -> int:
        nonlocal wrote_partial
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) != receipt.file_identity:
            return real_write(descriptor, payload)
        if wrote_partial:
            raise OSError("injected append interruption")
        wrote_partial = True
        return real_write(descriptor, payload[:5])

    monkeypatch.setattr(operations.os, "write", interrupt_write)

    with pytest.raises(OperationError, match="append interruption"):
        journal.append_response_frame("op-1", "incomplete")
    with pytest.raises(OperationError, match="torn"):
        journal.read_response_frames("op-1")


def test_same_inode_rollback_to_a_valid_record_boundary_is_refused(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    receipt = journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    journal.append_response_frame("op-1", "first")
    first_record_size = _spool_path(tmp_path).stat().st_size
    journal.append_response_frame("op-1", "second")
    committed = OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool
    assert committed is not None
    assert committed.file_identity == receipt.file_identity
    assert committed.frame_count == 2

    with _spool_path(tmp_path).open("r+b") as handle:
        handle.truncate(first_record_size)

    with pytest.raises(OperationError, match="rolled back"):
        journal.read_response_frames("op-1")
    with pytest.raises(OperationError, match="rolled back"):
        journal.append_response_frame("op-1", "third")


def test_fsynced_frame_extending_an_old_head_recovers_after_journal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    real_write = OperationJournal._write

    def fail_first_frame_receipt(current: OperationJournal) -> None:
        receipt = current.operations["op-1"].response_spool
        if receipt is not None and receipt.frame_count == 1:
            raise OperationError("injected frame-head journal failure")
        real_write(current)

    monkeypatch.setattr(OperationJournal, "_write", fail_first_frame_receipt)
    with pytest.raises(OperationError, match="frame-head journal failure"):
        journal.append_response_frame("op-1", "first")

    held = OperationJournal.load(journal.path).operations["op-1"]
    assert held.response_spool is not None
    assert held.response_spool.frame_count == 0
    monkeypatch.setattr(OperationJournal, "_write", real_write)
    assert OperationJournal.load(journal.path).read_response_frames("op-1") == (
        "first",
    )
    adopted = OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool
    assert adopted is not None
    assert adopted.frame_count == 1
    assert adopted.committed_size == _spool_path(tmp_path).stat().st_size

    OperationJournal.load(journal.path).append_response_frame("op-1", "second")

    recovered = OperationJournal.load(journal.path).operations[
        "op-1"
    ].response_spool
    assert recovered is not None
    assert recovered.frame_count == 2
    assert OperationJournal.load(journal.path).read_response_frames("op-1") == (
        "first",
        "second",
    )


def test_recovery_read_prevents_later_rollback_to_its_old_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    real_write = OperationJournal._write

    def fail_frame_head(current: OperationJournal) -> None:
        receipt = current.operations["op-1"].response_spool
        if receipt is not None and receipt.frame_count == 1:
            raise OperationError("injected frame-head journal failure")
        real_write(current)

    monkeypatch.setattr(OperationJournal, "_write", fail_frame_head)
    with pytest.raises(OperationError, match="frame-head journal failure"):
        journal.append_response_frame("op-1", "recovered")
    monkeypatch.setattr(OperationJournal, "_write", real_write)

    assert OperationJournal.load(journal.path).read_response_frames("op-1") == (
        "recovered",
    )
    with _spool_path(tmp_path).open("r+b") as handle:
        handle.truncate(0)

    with pytest.raises(OperationError, match="rolled back"):
        OperationJournal.load(journal.path).read_response_frames("op-1")


def test_recovery_refuses_more_than_one_uncommitted_frame(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    _spool_path(tmp_path).write_bytes(_record(b"one") + _record(b"two"))

    with pytest.raises(OperationError, match="more than one uncommitted"):
        journal.read_response_frames("op-1")
    with pytest.raises(OperationError, match="more than one uncommitted"):
        journal.append_response_frame("op-1", "three")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda wire: wire[:-1], "torn"),
        (
            lambda wire: wire[:8] + bytes([wire[8] ^ 1]) + wire[9:],
            "checksum",
        ),
    ],
)
def test_response_spool_refuses_torn_or_corrupt_records(
    tmp_path: Path,
    damage: Callable[[bytes], bytes],
    message: str,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    journal.append_response_frame("op-1", "first")
    spool = _spool_path(tmp_path)
    spool.write_bytes(damage(spool.read_bytes()))

    with pytest.raises(OperationError, match=message):
        journal.read_response_frames("op-1")
    with pytest.raises(OperationError, match=message):
        journal.append_response_frame("op-1", "second")


def test_response_spool_refuses_validly_checksummed_non_utf8_payload(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    _spool_path(tmp_path).write_bytes(_record(b"\xff"))

    with pytest.raises(OperationError, match="UTF-8"):
        journal.read_response_frames("op-1")


def test_response_spool_never_reads_or_appends_to_a_same_name_replacement(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    journal.append_response_frame("op-1", "original")
    spool = _spool_path(tmp_path)
    original = tmp_path / "original.frames"
    spool.rename(original)
    replacement = _record(b"replacement")
    spool.write_bytes(replacement)

    with pytest.raises(OperationError, match="exact response spool"):
        journal.read_response_frames("op-1")
    with pytest.raises(OperationError, match="exact response spool"):
        journal.append_response_frame("op-1", "new")

    assert spool.read_bytes() == replacement
    assert original.read_bytes() == _record(b"original")


def test_response_spool_never_follows_a_same_name_symlink(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    spool = _spool_path(tmp_path)
    spool.unlink()
    outside = tmp_path / "outside.frames"
    outside.write_bytes(_record(b"outside"))
    spool.symlink_to(outside)

    with pytest.raises(OperationError, match="exact response spool"):
        journal.read_response_frames("op-1")
    with pytest.raises(OperationError, match="exact response spool"):
        journal.append_response_frame("op-1", "new")

    assert outside.read_bytes() == _record(b"outside")
    journal.end("op-1")
    assert journal.forget(["op-1"]) == 1
    assert spool.is_symlink()
    assert outside.read_bytes() == _record(b"outside")


def test_end_adopts_one_fsynced_frame_before_settling_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    frame = (
        '{"type":"session.created","session":'
        '{"model":"gpt-realtime-1.5"}}'
    )
    real_write = OperationJournal._write

    def fail_frame_head(current: OperationJournal) -> None:
        receipt = current.operations["op-1"].response_spool
        if receipt is not None and receipt.frame_count == 1:
            raise OperationError("injected frame-head journal failure")
        real_write(current)

    monkeypatch.setattr(OperationJournal, "_write", fail_frame_head)
    with pytest.raises(OperationError, match="frame-head journal failure"):
        journal.append_response_frame("op-1", frame)
    assert _spool_path(tmp_path).read_bytes() == _record(frame.encode())
    held = OperationJournal.load(journal.path).operations["op-1"]
    assert held.state == "running"
    assert held.response_spool is not None
    assert held.response_spool.frame_count == 0
    monkeypatch.setattr(OperationJournal, "_write", real_write)

    ended = OperationJournal.load(journal.path).end("op-1")

    assert ended.state == "outcome_unknown"
    assert ended.response_spool is not None
    assert ended.response_spool.frame_count == 1
    assert OperationJournal.load(journal.path).read_response_frames("op-1") == (
        frame,
    )
    inspection = json.loads(
        OperationJournal.load(journal.path).read_inspectable_reply("op-1")
    )
    assert inspection["complete"] is False
    assert inspection["frames"] == [
        {"encoding": "utf-8", "payload": frame}
    ]


def test_nonterminal_frames_require_force_and_retry_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    frame = '{"type":"response.output_audio.delta","delta":"AAAA"}'
    journal.append_response_frame("op-1", frame)
    ended = journal.end("op-1")
    assert ended.state == "outcome_unknown"
    spool = _spool_path(tmp_path)
    exact_spool = spool.read_bytes()

    with pytest.raises(OperationError, match="force|frame"):
        OperationJournal.load(journal.path).forget(["op-1"])

    preserved = OperationJournal.load(journal.path).operations["op-1"]
    assert preserved.cleanup is None
    assert spool.read_bytes() == exact_spool
    real_retire = operations._retire_response_spool

    def retire_then_interrupt(binding: Any) -> None:
        real_retire(binding)
        raise OperationError("injected pause after exact spool retirement")

    monkeypatch.setattr(
        operations, "_retire_response_spool", retire_then_interrupt
    )
    with pytest.raises(OperationError, match="cleanup remains"):
        OperationJournal.load(journal.path).forget(["op-1"], force=True)

    cleaning = OperationJournal.load(journal.path).operations["op-1"]
    assert cleaning.cleanup is not None
    assert cleaning.cleanup.forced is True
    assert cleaning.cleanup.response_spool is not None
    assert not spool.exists()
    replacement = _record(b"replacement must survive")
    spool.write_bytes(replacement)

    monkeypatch.setattr(operations, "_retire_response_spool", real_retire)
    assert OperationJournal.load(journal.path).forget(["op-1"]) == 1

    assert spool.read_bytes() == replacement
    assert OperationJournal.load(journal.path).operations == {}


def test_force_forget_refuses_a_live_running_response_spool_until_end(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "running")
    frame = '{"type":"response.output_audio.delta","delta":"AAAA"}'
    journal.append_response_frame("op-1", frame)
    spool = _spool_path(tmp_path)
    exact_spool = spool.read_bytes()

    with pytest.raises(OperationError, match="running|live|end"):
        OperationJournal.load(journal.path).forget(["op-1"], force=True)

    preserved = OperationJournal.load(journal.path).operations["op-1"]
    assert preserved.state == "running"
    assert preserved.cleanup is None
    assert spool.read_bytes() == exact_spool

    ended = OperationJournal.load(journal.path).end("op-1")
    assert ended.state == "outcome_unknown"
    assert OperationJournal.load(journal.path).forget(
        ["op-1"], force=True
    ) == 1
    assert OperationJournal.load(journal.path).operations == {}
    assert not spool.exists()


def test_forget_durably_binds_and_retires_the_exact_response_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "canceled_before_send")
    spool = _spool_path(tmp_path)
    real_retire = operations._retire_response_spool

    def pause_cleanup(binding: object) -> None:
        del binding
        raise OperationError("injected response spool cleanup pause")

    monkeypatch.setattr(operations, "_retire_response_spool", pause_cleanup)
    with pytest.raises(OperationError, match="cleanup remains"):
        journal.forget(["op-1"])

    held = OperationJournal.load(journal.path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.response_spool is not None
    original = tmp_path / "original.frames"
    spool.rename(original)
    replacement = b"replacement must survive"
    spool.write_bytes(replacement)

    monkeypatch.setattr(operations, "_retire_response_spool", real_retire)
    assert OperationJournal.load(journal.path).forget(["op-1"]) == 1

    assert spool.read_bytes() == replacement
    assert original.exists()
    assert OperationJournal.load(journal.path).operations == {}


def test_forget_retires_a_response_spool_only_through_its_cleanup_intent(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.begin_response_capture("op-1")
    journal.advance("op-1", "canceled_before_send")
    spool = _spool_path(tmp_path)

    assert journal.forget(["op-1"]) == 1

    assert not spool.exists()
    assert OperationJournal.load(journal.path).operations == {}
