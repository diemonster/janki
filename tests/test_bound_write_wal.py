"""Crash recovery for exact guarded whole-file replacement."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import io, operations, patterns, staging
from japanese_anki import status as status_module
from japanese_anki.io import DataError
from japanese_anki.operations import (
    OperationError,
    OperationJournal,
)

ROOT = Path(__file__).resolve().parents[1]


def _crash_guarded_write(path: Path, replacement: bytes, crash_point: str) -> None:
    script = r"""
import hashlib
import os
import sys
from pathlib import Path

from japanese_anki import io

target = Path(sys.argv[1])
replacement = sys.argv[2].encode("utf-8")
crash_point = sys.argv[3]
expected = hashlib.sha256(target.read_bytes()).hexdigest()

real_exchange = io._exchange_entries
real_move = io._move_cas_marker

if crash_point == "before_exchange":
    def crash_before_exchange(*_args, **_kwargs):
        os._exit(81)
    io._exchange_entries = crash_before_exchange
elif crash_point == "after_exchange":
    def crash_before_commit(binding, marker, suffix, **kwargs):
        if suffix == io._CAS_VALIDATED_SUFFIX:
            os._exit(82)
        return real_move(binding, marker, suffix, **kwargs)
    io._move_cas_marker = crash_before_commit
elif crash_point == "after_commit":
    def crash_after_commit(binding, marker, suffix, **kwargs):
        result = real_move(binding, marker, suffix, **kwargs)
        if suffix == io._CAS_VALIDATED_SUFFIX:
            os._exit(83)
        return result
    io._move_cas_marker = crash_after_commit
else:
    raise AssertionError(crash_point)

io.atomic_write_text_bound(
    target,
    replacement.decode("utf-8"),
    expected_revision=expected,
)
raise AssertionError("the injected crash point was not reached")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            replacement.decode("utf-8"),
            crash_point,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        timeout=10,
    )
    assert result.returncode == {
        "before_exchange": 81,
        "after_exchange": 82,
        "after_commit": 83,
    }[crash_point]


def _crash_write_once(journal: Path, payload: bytes, crash_point: str) -> Path:
    script = r"""
import os
import sys
from pathlib import Path

from japanese_anki import io
from japanese_anki.operations import capture_artifact

journal = Path(sys.argv[1])
payload = sys.argv[2].encode("utf-8")
crash_point = sys.argv[3]
real_create = io._create_cas_marker
real_commit = io._commit_bound_target
real_link = io._link_entry_exclusive
real_move = io._move_cas_marker
real_rename = io._rename_entry_exclusive
real_move_between = io._rename_entry_exclusive_between
real_retire = io._retire_exact_entry
real_allocate = io._allocate_bound_temporary
real_write = io.os.write

if crash_point in {"during_retirement_append", "after_retirement_move"}:
    io._path_lock_root = lambda: journal.parent / "private-locks"

if crash_point == "after_draft_open":
    def crash_before_draft_write(descriptor, payload):
        if b'"version":3' in payload:
            os._exit(99)
        return real_write(descriptor, payload)
    io.os.write = crash_before_draft_write
elif crash_point == "during_draft_write":
    def crash_during_draft_write(descriptor, payload):
        if b'"version":3' in payload:
            real_write(descriptor, payload[:24])
            os.fsync(descriptor)
            os._exit(100)
        return real_write(descriptor, payload)
    io.os.write = crash_during_draft_write
elif crash_point == "after_draft_fsync":
    def crash_before_draft_promotion(directory_fd, left, right):
        if left.endswith(io._CAS_DRAFT_SUFFIX):
            os._exit(97)
        return real_rename(directory_fd, left, right)
    io._rename_entry_exclusive = crash_before_draft_promotion
elif crash_point == "before_temp_write":
    def crash_after_intent(*args, **kwargs):
        marker = real_create(*args, **kwargs)
        os._exit(91)
    io._create_cas_marker = crash_after_intent
elif crash_point == "after_temp_allocation":
    def crash_after_allocation(*args, **kwargs):
        allocated = real_allocate(*args, **kwargs)
        os._exit(98)
    io._allocate_bound_temporary = crash_after_allocation
elif crash_point == "after_temp_fsync":
    def crash_before_publish(*_args, **_kwargs):
        os._exit(92)
    io._link_entry_exclusive = crash_before_publish
elif crash_point == "after_link":
    def crash_after_link(*args, **kwargs):
        real_link(*args, **kwargs)
        os._exit(93)
    io._link_entry_exclusive = crash_after_link
elif crash_point == "after_link_fsync":
    def crash_before_validation(binding, marker, suffix, **kwargs):
        if suffix == io._CAS_VALIDATED_SUFFIX:
            os._exit(94)
        return real_move(binding, marker, suffix, **kwargs)
    io._move_cas_marker = crash_before_validation
elif crash_point == "after_validation":
    def crash_after_validation(binding, marker, suffix, **kwargs):
        result = real_move(binding, marker, suffix, **kwargs)
        if suffix == io._CAS_VALIDATED_SUFFIX:
            os._exit(95)
        return result
    io._move_cas_marker = crash_after_validation
elif crash_point == "after_temp_cleanup":
    def crash_after_temp_cleanup(binding, name, *args, **kwargs):
        result = real_retire(binding, name, *args, **kwargs)
        if result and name.endswith(".tmp"):
            os._exit(96)
        return result
    io._retire_exact_entry = crash_after_temp_cleanup
elif crash_point == "during_retirement_append":
    def crash_during_retirement_append(descriptor, payload):
        if b'"temporary_snapshot"' in payload:
            real_write(descriptor, payload[: max(1, len(payload) // 2)])
            os.fsync(descriptor)
            os._exit(103)
        return real_write(descriptor, payload)
    io.os.write = crash_during_retirement_append
elif crash_point == "after_retirement_move":
    def crash_after_retirement_move(source_fd, source, destination_fd, destination):
        real_move_between(source_fd, source, destination_fd, destination)
        if source.endswith(".tmp") and destination.endswith(".retired"):
            os._exit(104)
    io._rename_entry_exclusive_between = crash_after_retirement_move
else:
    raise AssertionError(crash_point)

capture_artifact(journal, "op-1", payload)
raise AssertionError("the injected crash point was not reached")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(journal), payload.decode(), crash_point],
        cwd=ROOT,
        env=environment,
        check=False,
        timeout=10,
    )
    assert result.returncode == {
        "after_draft_open": 99,
        "during_draft_write": 100,
        "after_draft_fsync": 97,
        "before_temp_write": 91,
        "after_temp_allocation": 98,
        "after_temp_fsync": 92,
        "after_link": 93,
        "after_link_fsync": 94,
        "after_validation": 95,
        "after_temp_cleanup": 96,
        "during_retirement_append": 103,
        "after_retirement_move": 104,
    }[crash_point]
    return journal.parent / ".pending" / "op-1.json"


def _wire_values(reader: str) -> tuple[bytes, bytes]:
    if reader == "staging":
        return b"phase: old\nrecords: []\n", b"phase: new\nrecords: []\n"
    if reader == "patterns":
        return b"{}\n", b"{  }\n"
    if reader == "journal":
        return (
            b'{"version":1,"operations":{}}\n',
            b'{ "version": 1, "operations": {} }\n',
        )
    if reader == "records":
        return b"[]\n", b"[ ]\n"
    raise AssertionError(reader)


def _read_domain_file(reader: str, path: Path) -> None:
    if reader == "staging":
        staging.read_staging(path)
    elif reader == "patterns":
        patterns.load_store(path)
    elif reader == "journal":
        OperationJournal.load(path)
    elif reader == "records":
        io.load_records(path)
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(reader)


def _wal_entries(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.parent.glob(f".{path.name}.*")
        if candidate.is_file()
    )


def _identity_named_temp(target: Path, payload: bytes) -> Path:
    """Create a private name whose filename truthfully binds its own inode."""
    seed = target.parent / "identity-seed.tmp"
    seed.write_bytes(payload)
    details = seed.stat()
    named = target.with_name(
        f".{target.name}.{'0' * 16}.{details.st_dev:x}-{details.st_ino:x}.tmp"
    )
    seed.rename(named)
    return named


def _dispatching_journal(path: Path) -> OperationJournal:
    journal = OperationJournal.load(path)
    journal.authorize(
        "op-1",
        kind="extract",
        source_file="lesson.pdf",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )
    journal.advance("op-1", "dispatching")
    return journal


def test_capture_receipt_is_taken_after_private_retirement_and_before_wal_finalize(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    answer = b"paid provider answer"

    receipt = operations.capture_artifact(path, "op-1", answer)
    target = tmp_path / receipt.relative_name
    details = target.stat()

    assert receipt.directory_identity == (
        target.parent.stat().st_dev,
        target.parent.stat().st_ino,
    )
    assert receipt.entry_state == (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )
    assert receipt.content_sha256 == hashlib.sha256(answer).hexdigest()
    entries = _wal_entries(target)
    assert len(entries) == 1
    assert entries[0].name.endswith(io._CAS_VALIDATED_SUFFIX)
    assert receipt.terminal_marker is not None
    marker_details = entries[0].stat()
    assert receipt.terminal_marker.name == entries[0].name
    assert receipt.terminal_marker.entry_state == (
        marker_details.st_dev,
        marker_details.st_ino,
        marker_details.st_size,
        marker_details.st_mtime_ns,
        marker_details.st_ctime_ns,
    )
    assert receipt.terminal_marker.content_sha256 == hashlib.sha256(
        entries[0].read_bytes()
    ).hexdigest()


def test_journal_write_failure_leaves_terminal_wal_as_capture_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)

    def fail_journal_write(_journal: OperationJournal) -> None:
        raise OperationError("injected journal write failure")

    monkeypatch.setattr(OperationJournal, "_write", fail_journal_write)

    with pytest.raises(OperationError, match="journal write failure"):
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                path, "op-1", b"paid provider answer"
            ),
        )

    held = OperationJournal.load(path).operations["op-1"]
    assert held.state == "dispatching"
    assert held.artifact is None
    evidence = operations._pending_answer_evidence(path, "op-1")
    assert evidence.reply_complete
    assert operations.reply_observation(path, held).payload == b"paid provider answer"
    assert any(
        entry.name.endswith(io._CAS_VALIDATED_SUFFIX)
        for entry in _wal_entries(path.parent / ".pending" / "op-1.json")
    )


def test_finalize_failure_after_durable_receipt_is_readable_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    real_finalize = operations._finalize_artifact

    def fail_finalize(_path: Path, _receipt: object) -> None:
        raise OperationError("injected finalization failure")

    monkeypatch.setattr(operations, "_finalize_artifact", fail_finalize)
    with pytest.raises(OperationError, match="finalization failure"):
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                path, "op-1", b"paid provider answer"
            ),
        )

    held = OperationJournal.load(path).operations["op-1"]
    assert held.state == "result_captured"
    assert held.artifact is not None
    assert OperationJournal.load(path).read_reply("op-1") == b"paid provider answer"
    target = tmp_path / held.artifact.relative_name
    assert any(
        entry.name.endswith(io._CAS_VALIDATED_SUFFIX)
        for entry in _wal_entries(target)
    )

    monkeypatch.setattr(operations, "_finalize_artifact", real_finalize)

    def must_not_recapture() -> operations.ArtifactReceipt:
        raise AssertionError("a durable result must not invoke capture again")

    retried = OperationJournal.load(path).capture_result(
        "op-1", must_not_recapture
    )

    assert retried.artifact == held.artifact
    assert _wal_entries(target) == []
    assert OperationJournal.load(path).read_reply("op-1") == b"paid provider answer"


def test_durable_receipt_never_adopts_or_finalizes_a_replacement_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    real_finalize = operations._finalize_artifact

    def fail_finalize(_path: Path, _receipt: object) -> None:
        raise OperationError("injected finalization failure")

    monkeypatch.setattr(operations, "_finalize_artifact", fail_finalize)
    with pytest.raises(OperationError, match="finalization failure"):
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                path, "op-1", b"paid provider answer"
            ),
        )

    held = OperationJournal.load(path).operations["op-1"]
    assert held.artifact is not None
    assert held.artifact.terminal_marker is not None
    marker = tmp_path / ".pending" / held.artifact.terminal_marker.name
    replacement = marker.with_name("replacement-marker")
    replacement.write_bytes(marker.read_bytes())
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    marker.unlink()
    replacement.rename(marker)
    assert replacement_identity != held.artifact.terminal_marker.entry_state[:2]

    assert OperationJournal.load(path).read_reply("op-1") == b"paid provider answer"
    monkeypatch.setattr(operations, "_finalize_artifact", real_finalize)

    def must_not_recapture() -> operations.ArtifactReceipt:
        raise AssertionError("a durable result must not invoke capture again")

    OperationJournal.load(path).capture_result("op-1", must_not_recapture)

    assert (marker.stat().st_dev, marker.stat().st_ino) == replacement_identity
    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1
    assert (marker.stat().st_dev, marker.stat().st_ino) == replacement_identity


def test_direct_forget_finishes_a_receipted_marker_moved_before_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)
    receipt = operations.capture_artifact(path, "op-1", b"paid provider answer")
    assert receipt.terminal_marker is not None
    marker = tmp_path / ".pending" / receipt.terminal_marker.name
    real_move = io._rename_entry_exclusive_between

    class InjectedCrash(BaseException):
        pass

    def crash_after_marker_move(
        source_fd: int,
        source: str,
        destination_fd: int,
        destination: str,
    ) -> None:
        real_move(source_fd, source, destination_fd, destination)
        if source == marker.name and destination.endswith(".retired"):
            raise InjectedCrash

    monkeypatch.setattr(
        io, "_rename_entry_exclusive_between", crash_after_marker_move
    )
    with pytest.raises(InjectedCrash):
        journal.capture_result("op-1", lambda: receipt)

    retirement = lock_root / "retired-writes"
    assert not marker.exists()
    assert len(list(retirement.iterdir())) == 1
    monkeypatch.setattr(io, "_rename_entry_exclusive_between", real_move)

    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1

    assert not OperationJournal.load(path).operations
    assert list(retirement.iterdir()) == []


def test_unreadable_regular_marker_replacement_falls_back_and_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)

    def fail_finalize(_path: Path, _receipt: object) -> None:
        raise OperationError("injected finalization failure")

    monkeypatch.setattr(operations, "_finalize_artifact", fail_finalize)
    with pytest.raises(OperationError, match="finalization failure"):
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                path, "op-1", b"paid provider answer"
            ),
        )

    held = OperationJournal.load(path).operations["op-1"]
    assert held.artifact is not None
    assert held.artifact.terminal_marker is not None
    marker = tmp_path / ".pending" / held.artifact.terminal_marker.name
    replacement = marker.with_name("unreadable-marker-replacement")
    replacement.write_bytes(b"unrelated regular replacement")
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    marker.unlink()
    replacement.rename(marker)
    real_read = io._read_cleanup_bound_entry

    def refuse_replacement_read(
        directory_fd: int, name: str
    ) -> tuple[tuple[int, int, int, int, int], str]:
        if name == marker.name:
            raise OSError("injected unreadable regular marker")
        return real_read(directory_fd, name)

    monkeypatch.setattr(io, "_read_cleanup_bound_entry", refuse_replacement_read)

    assert OperationJournal.load(path).read_reply("op-1") == b"paid provider answer"
    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1

    assert not OperationJournal.load(path).operations
    assert marker.read_bytes() == b"unrelated regular replacement"
    assert (marker.stat().st_dev, marker.stat().st_ino) == replacement_identity


@pytest.mark.parametrize(
    "crash_point",
    [
        "after_temp_fsync",
        "after_link",
        "after_link_fsync",
        "after_temp_cleanup",
    ],
)
def test_paid_answer_write_once_crashes_are_read_without_public_recovery(
    tmp_path: Path,
    crash_point: str,
) -> None:
    journal = tmp_path / "operations.json"
    held = _dispatching_journal(journal)
    payload = b'{"content":[{"type":"text","text":"paid answer"}]}'

    target = _crash_write_once(journal, payload, crash_point)

    public_before = (
        (target.stat().st_dev, target.stat().st_ino, target.read_bytes())
        if target.exists()
        else None
    )
    wal_before = {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    }
    operation = OperationJournal.load(journal).operations["op-1"]

    observation = operations.reply_observation(journal, operation)

    assert observation.payload == payload
    public_after = (
        (target.stat().st_dev, target.stat().st_ino, target.read_bytes())
        if target.exists()
        else None
    )
    assert public_after == public_before
    assert {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    } == wal_before

    assert held.forget(["op-1"], force=True) == 1
    assert not target.exists()
    assert _wal_entries(target) == []


def test_partial_retirement_binding_keeps_the_live_private_answer_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    _dispatching_journal(path)
    answer = b"paid provider answer"
    target = _crash_write_once(path, answer, "during_retirement_append")
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)

    held = OperationJournal.load(path).operations["op-1"]
    assert operations.reply_observation(path, held).payload == answer
    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1

    assert not target.exists()
    assert _wal_entries(target) == []
    retirement = lock_root / "retired-writes"
    assert not retirement.exists() or list(retirement.iterdir()) == []


def test_capture_cleanup_finishes_a_private_answer_moved_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    _dispatching_journal(path)
    answer = b"paid provider answer"
    target = _crash_write_once(path, answer, "after_retirement_move")
    lock_root = tmp_path / "private-locks"
    retirement = lock_root / "retired-writes"
    retired = list(retirement.iterdir())
    assert len(retired) == 1
    assert retired[0].read_bytes() == answer
    assert not any(entry.name.endswith(".tmp") for entry in _wal_entries(target))
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)

    held = OperationJournal.load(path).operations["op-1"]
    assert operations.reply_observation(path, held).payload == answer
    assert OperationJournal.load(path).forget(["op-1"], force=True) == 1

    assert not target.exists()
    assert _wal_entries(target) == []
    assert list(retirement.iterdir()) == []


def test_incomplete_write_once_temp_is_refused_with_its_evidence_intact(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "operations.json"
    payload = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(journal, payload, "before_temp_write")
    evidence = {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    }

    with pytest.raises(DataError, match="absent or incomplete"):
        io.read_bytes_bound(target)

    assert not target.exists()
    assert {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    } == evidence


def test_prepared_marker_cannot_publish_a_temp_its_filename_does_not_own(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "operations.json"
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(journal, answer, "after_temp_fsync")
    original_temp = next(target.parent.glob(f".{target.name}.*.tmp"))
    marker = next(
        target.parent.glob(f".{target.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    decoy = _identity_named_temp(target, b"unrelated private bytes")
    replacement_marker = decoy.with_suffix("").with_name(
        decoy.name[: -len(".tmp")] + io._CAS_PREPARED_SUFFIX
    )
    marker.rename(replacement_marker)
    before = {
        entry.name: entry.read_bytes() for entry in _wal_entries(target)
    }

    with pytest.raises(DataError, match="bound name|owns"):
        io.read_bytes_bound(target)

    assert not target.exists()
    assert original_temp.read_bytes() == answer
    assert decoy.read_bytes() == b"unrelated private bytes"
    assert {
        entry.name: entry.read_bytes() for entry in _wal_entries(target)
    } == before


def test_terminal_marker_cannot_retire_a_temp_its_filename_does_not_own(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.json"
    old = b"old bytes\n"
    new = b"new bytes\n"
    path.write_bytes(old)
    _crash_guarded_write(path, new, "after_commit")
    marker = next(path.parent.glob(f".{path.name}.*{io._CAS_VALIDATED_SUFFIX}"))
    decoy = _identity_named_temp(path, b"unrelated private bytes")
    decoy_details = decoy.stat()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["temporary"] = decoy.name
    payload["temporary_identity"] = [decoy_details.st_dev, decoy_details.st_ino]
    payload["expected_state"] = [
        decoy_details.st_dev,
        decoy_details.st_ino,
        decoy_details.st_size,
        decoy_details.st_mtime_ns,
    ]
    payload["expected_sha256"] = hashlib.sha256(decoy.read_bytes()).hexdigest()
    marker.write_text(json.dumps(payload), encoding="utf-8")
    before_marker = marker.read_bytes()

    with pytest.raises(DataError, match="bound name|owns"):
        io.read_bytes_bound(path)

    assert path.read_bytes() == new
    assert decoy.read_bytes() == b"unrelated private bytes"
    assert marker.read_bytes() == before_marker


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temporary_identity", [True, 2]),
        ("temporary_identity", [-1, 2]),
        ("expected_state", [1, 2, True, 4]),
        ("generated_state", [1, 2, 3, -1]),
    ],
)
def test_live_marker_parser_rejects_boolean_and_negative_identity_state(
    field: str,
    value: list[int | bool],
) -> None:
    payload: dict[str, Any] = {
        "version": 3,
        "target": "target.json",
        "temporary": ".target.json.0000000000000000.1-2.tmp",
        "temporary_identity": [1, 2],
        "expected_absent": False,
        "expected_state": [1, 2, 3, 4],
        "expected_sha256": "a" * 64,
        "generated_state": [1, 2, 3, 4],
        "generated_sha256": "b" * 64,
        "retirement_snapshot_required": False,
    }
    payload[field] = value

    with pytest.raises(DataError, match="marker.*invalid"):
        io._parse_cas_marker(
            json.dumps(payload).encode("utf-8"), target_name="target.json"
        )


@pytest.mark.parametrize(
    ("crash_point", "keeps_unproven_temp"),
    [
        ("after_draft_open", True),
        ("during_draft_write", True),
        ("after_draft_fsync", False),
        ("before_temp_write", False),
    ],
)
def test_incomplete_answer_capture_is_settled_then_forgotten_with_its_wal(
    tmp_path: Path,
    crash_point: str,
    keeps_unproven_temp: bool,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", crash_point)

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    listed = "\n".join(
        status_module.format_operations(
            journal.tracked(),
            journal_path=path,
            noun="janki is still tracking",
        )
    )
    assert "answer capture was interrupted" in listed
    assert "no answer came back" not in listed
    ended = journal.end("op-1")

    assert ended.state == "outcome_unknown"
    assert "answer capture was interrupted" in ended.detail
    assert _wal_entries(target)
    assert journal.forget(["op-1"]) == 1
    assert OperationJournal.load(path).operations == {}
    assert not target.exists()
    remaining = _wal_entries(target)
    if keeps_unproven_temp:
        assert len(remaining) == 1
        assert remaining[0].name.endswith(".tmp")
    else:
        assert remaining == []


def test_valid_draft_with_mismatched_body_can_be_settled_as_marker_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(
        path, b"paid provider answer", "after_draft_fsync"
    )
    draft = next(target.parent.glob(f".{target.name}.*{io._CAS_DRAFT_SUFFIX}"))
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["temporary_identity"][1] += 1
    draft.write_text(json.dumps(payload), encoding="utf-8")

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    ended = journal.end("op-1")

    assert ended.state == "outcome_unknown"
    assert "answer capture was interrupted" in ended.detail
    assert journal.forget(["op-1"]) == 1
    assert not draft.exists()
    assert temporary.exists()
    assert OperationJournal.load(path).operations == {}


def test_temp_allocation_crash_preserves_unproven_markerless_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(
        path, b"paid provider answer", "after_temp_allocation"
    )

    entries = _wal_entries(target)
    assert len(entries) == 1
    assert entries[0].name.endswith(".tmp")
    assert entries[0].read_bytes() == b""

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    assert _wal_entries(target) == entries
    ended = journal.end("op-1")
    assert ended.state == "outcome_unknown"
    assert "answer capture was interrupted" not in ended.detail
    assert journal.forget(["op-1"]) == 1
    assert _wal_entries(target) == entries


def test_unproven_exact_pattern_temp_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    pending = tmp_path / ".pending"
    pending.mkdir()
    target = pending / "op-1.json"
    unrelated = pending / ".op-1.json.0123456789abcdef.tmp"
    unrelated.write_bytes(b"unrelated private file")

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    assert unrelated.read_bytes() == b"unrelated private file"
    assert not target.exists()


@pytest.mark.parametrize("same_bytes", [False, True], ids=["different", "same"])
def test_complete_private_answer_requires_force_and_forget_cleans_exact_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_bytes: bool,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b'{"content":[{"type":"text","text":"paid provider answer"}]}'
    target = _crash_write_once(path, answer, "after_validation")
    target.unlink()
    replacement = answer if same_bytes else b"unrelated public replacement"
    target.write_bytes(replacement)

    assert operations._pending_answer_evidence(path, "op-1").reply_complete
    held = OperationJournal.load(path).operations["op-1"]
    observed = operations.reply_observation(path, held)
    assert operations.response_answer_text(observed.payload) == "paid provider answer"
    with pytest.raises(OperationError, match="its reply arrived"):
        journal.end("op-1")
    with pytest.raises(OperationError, match="paid for and never became"):
        journal.forget(["op-1"])

    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"], force=True)
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.write_ahead is not None
    assert held.cleanup.write_ahead.public_snapshot is None

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert OperationJournal.load(path).operations == {}
    assert target.read_bytes() == replacement
    assert _wal_entries(target) == []


def test_complete_private_answer_is_read_without_adopting_public_replacement(
    tmp_path: Path,
) -> None:
    """A WAL-bound answer is the paid reply even when its public name is now
    occupied by somebody else's file. Reading it must use the private binding,
    not make the lexical public path into evidence or disturb its occupant.
    """
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(path, answer, "after_validation")
    replacement = target.with_name("replacement-public.json")
    replacement.write_bytes(b'{"content":[{"type":"text","text":"not the answer"}]}')
    target.unlink()
    replacement.rename(target)
    replacement_identity = (target.stat().st_dev, target.stat().st_ino)

    assert journal.read_reply("op-1") == answer
    assert target.read_bytes() != answer
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert OperationJournal.load(path).operations["op-1"].state == "dispatching"


def test_forget_public_retirement_failure_keeps_private_answer_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b"paid provider answer"
    target = _crash_write_once(path, answer, "after_validation")
    real_retire = io._retire_exact_entry

    # The operation has no receipt yet, so the still-active WAL is its only
    # reply evidence.
    def refuse_public_retirement(
        binding: Any,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        if name == target.name:
            return False
        return real_retire(binding, name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            io, "_retire_exact_entry", refuse_public_retirement
        )

        with pytest.raises(OperationError, match="public answer"):
            journal.forget(["op-1"], force=True)

    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.forced is True
    assert target.read_bytes() == answer
    entries = _wal_entries(target)
    assert any(entry.name.endswith(".tmp") for entry in entries)
    assert any(entry.name.endswith(io._CAS_VALIDATED_SUFFIX) for entry in entries)

    assert journal.forget(["op-1"]) == 1
    assert not target.exists()
    assert _wal_entries(target) == []


@pytest.mark.parametrize("failure", ["false", "raise", "raise_after_private"])
def test_paired_answer_cleanup_failure_keeps_its_tombstone_until_both_names_retire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """The public/private answer names are one cleanup decision.

    The public move's reversible callback must not turn a failed private
    retirement into success.  A retry must discover any already-moved public
    link, remove both exact links, and only then clear the journal entry.
    """
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b"paid provider answer"
    target = _crash_write_once(path, answer, "after_validation")
    private_name = next(
        entry.name for entry in _wal_entries(target) if entry.name.endswith(".tmp")
    )
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)
    real_retire = io._retire_exact_entry
    injected = False

    def fail_private_retirement(
        binding: Any,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        nonlocal injected
        confirmation = kwargs.get("confirm_while_reversible")
        if (
            failure == "raise_after_private"
            and name == target.name
            and confirmation is not None
            and not injected
        ):
            injected = True

            def retire_private_then_raise() -> bool:
                assert callable(confirmation)
                assert confirmation() is True
                raise OSError("injected failure after private retirement")

            kwargs["confirm_while_reversible"] = retire_private_then_raise
        if name == private_name and not injected:
            injected = True
            if failure == "raise":
                raise OSError("injected private retirement failure")
            return False
        return real_retire(binding, name, *args, **kwargs)

    monkeypatch.setattr(io, "_retire_exact_entry", fail_private_retirement)

    expected_failure = (
        "Paired retirement did not"
        if failure == "false"
        else "Paired retirement failed"
    )
    with pytest.raises(OperationError, match=expected_failure):
        journal.forget(["op-1"], force=True)

    assert injected
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None and held.cleanup.write_ahead is not None
    temporary = target.parent / held.cleanup.write_ahead.temporary_name
    marker = target.parent / held.cleanup.write_ahead.marker_name
    assert target.exists() or list((lock_root / "retired-writes").iterdir())
    assert temporary.exists() is (failure != "raise_after_private")

    monkeypatch.setattr(io, "_retire_exact_entry", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not target.exists()
    assert not temporary.exists()
    assert not marker.exists()
    assert list((lock_root / "retired-writes").iterdir()) == []
    assert not OperationJournal.load(path).operations


def test_public_only_terminal_answer_remains_complete_when_marker_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(path, answer, "after_temp_cleanup")
    real_retire = io._retire_exact_entry

    def refuse_marker_cleanup(
        binding: Any,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        if name.endswith(io._CAS_VALIDATED_SUFFIX):
            return False
        return real_retire(binding, name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(io, "_retire_exact_entry", refuse_marker_cleanup)
        assert operations._pending_answer_evidence(path, "op-1").reply_complete
        held = OperationJournal.load(path).operations["op-1"]
        observed = operations.reply_observation(path, held)
        assert operations.response_answer_text(observed.payload) == "paid answer"
        with pytest.raises(OperationError, match="its reply arrived"):
            journal.end("op-1")
        with pytest.raises(OperationError, match="paid for and never became"):
            journal.forget(["op-1"])
        with pytest.raises(OperationError, match="marker"):
            journal.forget(["op-1"], force=True)

    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.forced is True
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not target.exists()
    assert _wal_entries(target) == []


@pytest.mark.parametrize(
    "bad_character",
    [pytest.param("\0", id="nul"), pytest.param("\ud800", id="surrogate")],
)
def test_unencodable_private_name_can_be_settled_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_character: str,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    marker = next(
        target.parent.glob(f".{target.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["temporary"] = f".{target.name}.{bad_character}.tmp"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    before = marker.read_bytes()

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    ended = journal.end("op-1")
    assert "answer capture was interrupted" in ended.detail
    assert marker.read_bytes() == before
    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.write_ahead is not None
    assert held.cleanup.write_ahead.temporary_owned is False
    assert held.cleanup.write_ahead.temporary_name == ""

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert temporary.exists()
    assert _wal_entries(target) == [temporary]


def test_replaced_private_name_cannot_adopt_matching_answer_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b"paid provider answer"
    target = _crash_write_once(path, answer, "before_temp_write")
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    temporary.unlink()
    temporary.write_bytes(answer)

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    ended = journal.end("op-1")
    assert "answer capture was interrupted" in ended.detail
    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.write_ahead is not None
    assert held.cleanup.write_ahead.temporary_owned is False

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert temporary.read_bytes() == answer
    assert all(
        not candidate.name.endswith(
            (
                io._CAS_DRAFT_SUFFIX,
                io._CAS_PREPARED_SUFFIX,
                io._CAS_VALIDATED_SUFFIX,
                io._CAS_RECOVERED_SUFFIX,
            )
        )
        for candidate in _wal_entries(target)
    )


def test_cleanup_tombstone_preserves_a_same_name_marker_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    marker = next(
        target.parent.glob(f".{target.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    journal.end("op-1")
    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])

    replacement = marker.with_name("replacement-marker")
    replacement.write_bytes(b"unrelated marker replacement")
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    marker.unlink()
    replacement.rename(marker)

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not temporary.exists()
    assert marker.read_bytes() == b"unrelated marker replacement"
    assert (marker.stat().st_dev, marker.stat().st_ino) == replacement_identity


def test_cleanup_tombstone_preserves_a_same_name_private_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    marker = next(
        target.parent.glob(f".{target.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    journal.end("op-1")
    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])

    replacement = temporary.with_name("replacement-private")
    replacement.write_bytes(b"unrelated private replacement")
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    temporary.unlink()
    replacement.rename(temporary)

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert temporary.read_bytes() == b"unrelated private replacement"
    assert (temporary.stat().st_dev, temporary.stat().st_ino) == replacement_identity
    assert not marker.exists()


@pytest.mark.parametrize("entry_kind", ["marker", "private"])
@pytest.mark.parametrize("replacement_kind", ["symlink", "fifo", "directory"])
def test_wal_cleanup_preserves_a_nonregular_same_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
    replacement_kind: str,
) -> None:
    path, target = _leave_wal_cleanup_tombstone(tmp_path, monkeypatch)
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None and held.cleanup.write_ahead is not None
    evidence = held.cleanup.write_ahead
    name = (
        evidence.marker_name
        if entry_kind == "marker"
        else evidence.temporary_name
    )
    replaced = target.parent / name
    replaced.unlink()
    referent = tmp_path / f"outside-{entry_kind}"
    if replacement_kind == "symlink":
        referent.write_bytes(b"outside replacement")
        replaced.symlink_to(referent)
    elif replacement_kind == "fifo":
        os.mkfifo(replaced)
    else:
        replaced.mkdir()
    replacement = os.lstat(replaced)

    assert OperationJournal.load(path).forget(["op-1"]) == 1

    current = os.lstat(replaced)
    assert (current.st_dev, current.st_ino, current.st_mode) == (
        replacement.st_dev,
        replacement.st_ino,
        replacement.st_mode,
    )
    if replacement_kind == "symlink":
        assert referent.read_bytes() == b"outside replacement"
    assert not OperationJournal.load(path).operations


def test_cleanup_tombstone_preserves_a_same_bytes_public_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed hard-link sibling also moves the private ctime out of scope.

    The replacement public name is plainly outside the old authority. Removing
    the original public hard link changes the remaining private link's ctime;
    a durable retry must not widen its old exact binding to adopt that entry.
    """
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(path, answer, "after_validation")
    original_identity = (target.stat().st_dev, target.stat().st_ino)
    real_retire = operations._retire_write_ahead

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"], force=True)
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None and held.cleanup.write_ahead is not None
    write_ahead = held.cleanup.write_ahead
    assert write_ahead.temporary_snapshot is not None
    bound_private_ctime = write_ahead.temporary_snapshot[0][4]
    temporary = target.parent / write_ahead.temporary_name
    marker = target.parent / write_ahead.marker_name

    replacement = target.with_name("replacement-public")
    replacement.write_bytes(answer)
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    assert replacement_identity != original_identity
    target.unlink()
    replacement.rename(target)
    assert temporary.stat().st_ctime_ns != bound_private_ctime

    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert target.read_bytes() == answer
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert temporary.read_bytes() == answer
    assert _wal_entries(target) == [temporary]
    assert not marker.exists()
    assert not OperationJournal.load(path).operations


def _leave_wal_cleanup_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    journal.end("op-1")
    real_retire = operations._retire_write_ahead

    def pause_cleanup(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(operations, "_retire_write_ahead", pause_cleanup)
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])
    monkeypatch.setattr(operations, "_retire_write_ahead", real_retire)
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None
    assert held.cleanup.artifact is None
    assert held.cleanup.write_ahead is not None
    return path, target


@pytest.mark.parametrize("entry_kind", ["marker", "private"])
def test_wal_cleanup_preserves_an_entry_whose_ctime_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    path, target = _leave_wal_cleanup_tombstone(tmp_path, monkeypatch)
    evidence = (
        OperationJournal.load(path)
        .operations["op-1"]
        .cleanup
    )
    assert evidence is not None and evidence.write_ahead is not None
    write_ahead = evidence.write_ahead
    name = (
        write_ahead.marker_name
        if entry_kind == "marker"
        else write_ahead.temporary_name
    )
    changed = target.parent / name
    before = changed.stat()

    os.chmod(changed, before.st_mode ^ 0o100)

    after = changed.stat()
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

    assert changed.exists()
    assert _wal_entries(target) == [changed]
    assert not OperationJournal.load(path).operations


@pytest.mark.parametrize("entry_kind", ["marker", "private"])
def test_wal_cleanup_rechecks_ctime_at_the_retirement_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    path, target = _leave_wal_cleanup_tombstone(tmp_path, monkeypatch)
    cleanup = OperationJournal.load(path).operations["op-1"].cleanup
    assert cleanup is not None and cleanup.write_ahead is not None
    write_ahead = cleanup.write_ahead
    name = (
        write_ahead.marker_name
        if entry_kind == "marker"
        else write_ahead.temporary_name
    )
    changed = target.parent / name
    real_retire = io._retire_exact_entry
    injected = False

    def change_ctime_before_retirement(
        binding: Any,
        candidate: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        nonlocal injected
        if candidate == name and not injected:
            os.chmod(changed, changed.stat().st_mode ^ 0o100)
            injected = True
        return real_retire(binding, candidate, *args, **kwargs)

    monkeypatch.setattr(io, "_retire_exact_entry", change_ctime_before_retirement)

    assert OperationJournal.load(path).forget(["op-1"]) == 1

    assert injected
    assert changed.exists()
    assert _wal_entries(target) == [changed]
    assert not OperationJournal.load(path).operations


def test_wal_cleanup_clears_when_its_bound_namespace_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, target = _leave_wal_cleanup_tombstone(tmp_path, monkeypatch)
    pending = target.parent
    before = {entry.name: entry.read_bytes() for entry in pending.iterdir()}
    detached = tmp_path / "detached-pending"
    pending.rename(detached)
    real_snapshot = io._bound_snapshot

    def unexpected_snapshot(directory_fd: int, name: str) -> Any:
        if name in before:
            pytest.fail("cleanup searched outside its missing bound namespace")
        return real_snapshot(directory_fd, name)

    monkeypatch.setattr(io, "_bound_snapshot", unexpected_snapshot)

    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not OperationJournal.load(path).operations
    assert not pending.exists()
    assert {entry.name: entry.read_bytes() for entry in detached.iterdir()} == before


def test_wal_cleanup_never_adopts_a_replaced_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, target = _leave_wal_cleanup_tombstone(tmp_path, monkeypatch)
    pending = target.parent
    before = {entry.name: entry.read_bytes() for entry in pending.iterdir()}
    detached = tmp_path / "detached-pending"
    pending.rename(detached)
    pending.mkdir()
    for name, payload in before.items():
        (pending / name).write_bytes(payload)
    replacement_identities = {
        entry.name: (entry.stat().st_dev, entry.stat().st_ino)
        for entry in pending.iterdir()
    }
    real_snapshot = io._bound_snapshot

    def unexpected_snapshot(directory_fd: int, name: str) -> Any:
        if name in before:
            pytest.fail("cleanup inspected WAL names in a replacement namespace")
        return real_snapshot(directory_fd, name)

    monkeypatch.setattr(io, "_bound_snapshot", unexpected_snapshot)

    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert not OperationJournal.load(path).operations
    assert {entry.name: entry.read_bytes() for entry in detached.iterdir()} == before
    assert {entry.name: entry.read_bytes() for entry in pending.iterdir()} == before
    assert {
        entry.name: (entry.stat().st_dev, entry.stat().st_ino)
        for entry in pending.iterdir()
    } == replacement_identities


def test_write_ahead_cleanup_retries_after_final_journal_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    journal.end("op-1")
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
    assert _wal_entries(target) == []
    held = OperationJournal.load(path).operations["op-1"]
    assert held.cleanup is not None

    monkeypatch.setattr(OperationJournal, "_write", real_write)
    assert OperationJournal.load(path).forget(["op-1"]) == 1
    assert OperationJournal.load(path).operations == {}


@pytest.mark.parametrize(
    "damage",
    [
        "marker_nul",
        "marker_surrogate",
        "temporary_traversal",
        "bool_state",
        "short_state",
        "bad_digest",
        "empty_generated",
        "device_mismatch",
    ],
)
def test_malformed_write_ahead_cleanup_never_grants_deletion_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(path, b"paid provider answer", "before_temp_write")
    temporary = next(target.parent.glob(f".{target.name}.*.tmp"))
    marker = next(
        target.parent.glob(f".{target.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    journal.end("op-1")

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])

    wire = json.loads(path.read_text(encoding="utf-8"))
    cleanup = wire["operations"]["op-1"]["cleanup"]["write_ahead"]
    if damage == "marker_nul":
        cleanup["marker_name"] = f".{target.name}.bad\0marker.janki-cas.prepared"
        cleanup["temporary_name"] = ""
        cleanup["temporary_identity"] = None
        cleanup["temporary_snapshot"] = None
        cleanup["temporary_owned"] = False
    elif damage == "marker_surrogate":
        cleanup["marker_name"] = f".{target.name}.\ud800.janki-cas.prepared"
        cleanup["temporary_name"] = ""
        cleanup["temporary_identity"] = None
        cleanup["temporary_snapshot"] = None
        cleanup["temporary_owned"] = False
    elif damage == "temporary_traversal":
        cleanup["temporary_name"] = "../outside.tmp"
    elif damage == "bool_state":
        cleanup["marker_state"][2] = True
    elif damage == "short_state":
        cleanup["marker_state"].pop()
        assert len(cleanup["marker_state"]) == 4
    elif damage == "bad_digest":
        cleanup["marker_sha256"] = "G" * 64
    elif damage == "empty_generated":
        cleanup["generated_sha256"] = ""
    elif damage == "device_mismatch":
        cleanup["directory_identity"][0] += 1
    else:  # pragma: no cover - parametrization is the exhaustive list
        raise AssertionError(damage)
    if damage == "short_state":
        with pytest.raises(DataError, match="entry state is invalid"):
            io._BoundWriteEvidence.from_cleanup_dict(target, cleanup)
    path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OperationError, match="invalid write-ahead cleanup"):
        OperationJournal.load(path)

    assert temporary.exists()
    assert marker.exists()


def test_marker_only_cleanup_rejects_a_four_field_destructive_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict arity guard must stand on its own.

    A cleanup carrying a private or public snapshot can still be rejected by
    that other five-field value after the marker parser is weakened.  Bind a
    marker-only draft so shortening its state is the only defect; accepting it
    would silently discard the ctime that distinguishes an inode reuse.
    """
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    target = _crash_write_once(
        path, b"paid provider answer", "after_draft_fsync"
    )
    draft = next(target.parent.glob(f".{target.name}.*{io._CAS_DRAFT_SUFFIX}"))
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["temporary_identity"][1] += 1
    draft.write_text(json.dumps(payload), encoding="utf-8")
    journal.end("op-1")

    def pause_after_cleanup_intent(_evidence: Any) -> None:
        raise OperationError("injected write-ahead cleanup pause")

    monkeypatch.setattr(
        operations, "_retire_write_ahead", pause_after_cleanup_intent
    )
    with pytest.raises(OperationError, match="cleanup pause"):
        journal.forget(["op-1"])

    wire = json.loads(path.read_text(encoding="utf-8"))
    cleanup = wire["operations"]["op-1"]["cleanup"]["write_ahead"]
    assert cleanup["temporary_snapshot"] is None
    assert cleanup["public_snapshot"] is None
    cleanup["marker_state"].pop()
    path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OperationError, match="invalid write-ahead cleanup"):
        OperationJournal.load(path)

    assert draft.exists()


def test_write_once_temp_retirement_restores_answer_if_public_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "op-1.json"
    answer = b"paid provider answer"
    replacement = b"unrelated public replacement"
    real_retire = io._retire_exact_entry
    injected = False

    def replace_during_temp_retirement(
        binding: Any,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        nonlocal injected
        if name.endswith(".tmp") and not injected:
            os.unlink(path.name, dir_fd=binding.descriptor)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=binding.descriptor,
            )
            try:
                os.write(descriptor, replacement)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(binding.descriptor)
            injected = True
        return real_retire(binding, name, *args, **kwargs)

    monkeypatch.setattr(
        io, "_retire_exact_entry", replace_during_temp_retirement
    )

    with pytest.raises(DataError, match="public name"):
        io.atomic_write_bytes_bound(path, answer, expected_absent=True)

    assert injected
    assert path.read_bytes() == replacement
    entries = _wal_entries(path)
    assert any(entry.name.endswith(io._CAS_VALIDATED_SUFFIX) for entry in entries)
    assert [entry.read_bytes() for entry in entries if entry.name.endswith(".tmp")] == [
        answer
    ]


def test_bound_bytes_refuses_a_different_planned_directory(
    tmp_path: Path,
) -> None:
    planned = tmp_path / "planned"
    replacement = tmp_path / "replacement"
    planned.mkdir()
    replacement.mkdir()
    replacement_details = replacement.stat()
    path = planned / "answer.json"

    with pytest.raises(DataError, match="Bound target directory changed"):
        io.atomic_write_bytes_bound(
            path,
            b"paid answer",
            expected_absent=True,
            expected_directory_identity=(
                replacement_details.st_dev,
                replacement_details.st_ino,
            ),
        )

    assert list(planned.iterdir()) == []


def test_bound_bytes_does_not_recreate_a_missing_planned_directory(
    tmp_path: Path,
) -> None:
    planned = tmp_path / "planned"
    moved = tmp_path / "moved"
    planned.mkdir()
    planned_details = planned.stat()
    expected_identity = (planned_details.st_dev, planned_details.st_ino)
    planned.rename(moved)

    with pytest.raises(DataError):
        io.atomic_write_bytes_bound(
            planned / "answer.json",
            b"paid answer",
            expected_absent=True,
            expected_directory_identity=expected_identity,
        )

    assert not planned.exists()
    assert list(moved.iterdir()) == []


def test_write_once_cleanup_refuses_a_public_replacement_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "op-1.json"
    answer = b"paid provider answer"
    replacement = b"unrelated public replacement"
    real_move = io._move_cas_marker
    injected = False

    def replace_after_validation(
        binding: Any,
        marker: str,
        suffix: str,
        **kwargs: Any,
    ) -> str:
        nonlocal injected
        moved = real_move(binding, marker, suffix, **kwargs)
        if suffix == io._CAS_VALIDATED_SUFFIX and not injected:
            os.unlink(path.name, dir_fd=binding.descriptor)
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=binding.descriptor,
            )
            try:
                os.write(descriptor, replacement)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(binding.descriptor)
            injected = True
        return moved

    monkeypatch.setattr(io, "_move_cas_marker", replace_after_validation)

    with pytest.raises(DataError, match="public name no longer holds"):
        io.atomic_write_bytes_bound(path, answer, expected_absent=True)

    assert injected
    assert path.read_bytes() == replacement
    entries = _wal_entries(path)
    assert any(entry.name.endswith(io._CAS_VALIDATED_SUFFIX) for entry in entries)
    assert [entry.read_bytes() for entry in entries if entry.name.endswith(".tmp")] == [
        answer
    ]


@pytest.mark.parametrize("public_change", ["removed", "replaced"])
def test_validated_write_once_recovery_retains_answer_if_public_name_changes(
    tmp_path: Path,
    public_change: str,
) -> None:
    journal = tmp_path / "operations.json"
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(journal, answer, "after_validation")
    before = {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    }
    assert any(name.endswith(io._CAS_VALIDATED_SUFFIX) for name in before)
    target.unlink()
    if public_change == "replaced":
        target.write_bytes(b"unrelated public replacement")

    with pytest.raises(DataError, match="public name no longer holds"):
        io.read_bytes_bound(target)

    assert {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    } == before
    temporary_answers = [
        payload for name, payload in before.items() if name.endswith(".tmp")
    ]
    assert temporary_answers == [answer]
    if public_change == "removed":
        assert not target.exists()
    else:
        assert target.read_bytes() == b"unrelated public replacement"


@pytest.mark.parametrize("public_change", ["removed", "replaced"])
def test_terminal_write_once_marker_survives_a_later_public_change(
    tmp_path: Path,
    public_change: str,
) -> None:
    journal = tmp_path / "operations.json"
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(journal, answer, "after_temp_cleanup")
    before = {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    }
    assert len(before) == 1
    assert next(iter(before)).endswith(io._CAS_VALIDATED_SUFFIX)
    target.unlink()
    if public_change == "replaced":
        target.write_bytes(b"unrelated public replacement")

    with pytest.raises(DataError, match="public name no longer holds"):
        io.read_bytes_bound(target)

    assert {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(target)
    } == before


def test_public_only_same_bytes_replacement_is_not_the_captured_answer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operations.json"
    journal = _dispatching_journal(path)
    answer = b'{"content":[{"type":"text","text":"paid answer"}]}'
    target = _crash_write_once(path, answer, "after_temp_cleanup")
    original_identity = (target.stat().st_dev, target.stat().st_ino)
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(answer)
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    assert replacement_identity != original_identity
    target.unlink()
    replacement.rename(target)

    assert not operations._pending_answer_evidence(path, "op-1").reply_complete
    held = OperationJournal.load(path).operations["op-1"]
    assert operations.reply_observation(path, held).payload is None
    ended = journal.end("op-1")
    assert ended.state == "outcome_unknown"
    assert "answer capture was interrupted" in ended.detail

    assert journal.forget(["op-1"]) == 1
    assert target.read_bytes() == answer
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert _wal_entries(target) == []


@pytest.mark.parametrize("publication", ["expected_absent", "guarded"])
def test_failed_marker_draft_retires_every_private_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    path = tmp_path / "target.json"
    old = b"old bytes\n"
    if publication == "guarded":
        path.write_bytes(old)
    real_write = io.os.write
    injected = False

    def fail_marker_write(descriptor: int, payload: bytes) -> int:
        nonlocal injected
        if not injected and b'"version":3' in payload:
            injected = True
            raise OSError("injected marker write failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(io.os, "write", fail_marker_write)

    with pytest.raises(DataError, match="injected marker write failure"):
        if publication == "expected_absent":
            io.atomic_write_bytes_bound(path, b"new bytes\n", expected_absent=True)
        else:
            io.atomic_write_text_bound(
                path,
                "new bytes\n",
                expected_revision=hashlib.sha256(old).hexdigest(),
            )

    assert injected
    if publication == "guarded":
        assert path.read_bytes() == old
    else:
        assert not path.exists()
    assert _wal_entries(path) == []


@pytest.mark.parametrize("reader", ["staging", "patterns", "journal", "records"])
@pytest.mark.parametrize(
    "crash_point", ["before_exchange", "after_exchange", "after_commit"]
)
def test_domain_readers_recover_each_guarded_write_crash_point(
    tmp_path: Path,
    reader: str,
    crash_point: str,
) -> None:
    old, new = _wire_values(reader)
    path = tmp_path / {
        "staging": "candidates.yaml",
        "patterns": "patterns.json",
        "journal": "operations.json",
        "records": "vocabulary.json",
    }[reader]
    path.write_bytes(old)

    _crash_guarded_write(path, new, crash_point)

    prepared = list(
        path.parent.glob(f".{path.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    validated = list(
        path.parent.glob(f".{path.name}.*{io._CAS_VALIDATED_SUFFIX}")
    )
    temporary = list(path.parent.glob(f".{path.name}.*.tmp"))
    if crash_point == "before_exchange":
        assert path.read_bytes() == old
        assert len(prepared) == len(temporary) == 1
        assert prepared[0].stat().st_size > 0
        assert temporary[0].read_bytes() == new
    elif crash_point == "after_exchange":
        assert path.read_bytes() == new
        assert len(prepared) == len(temporary) == 1
        assert prepared[0].stat().st_size > 0
        assert temporary[0].read_bytes() == old
    else:
        assert path.read_bytes() == new
        assert len(validated) == len(temporary) == 1
        assert validated[0].stat().st_size > 0
        assert temporary[0].read_bytes() == old

    expected = new if crash_point == "after_commit" else old
    _read_domain_file(reader, path)
    assert path.read_bytes() == expected
    _read_domain_file(reader, path)
    assert path.read_bytes() == expected
    assert _wal_entries(path) == []


def test_records_revision_recovers_before_binding_collection_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vocabulary.json"
    old = b"[]\n"
    new = b"[ ]\n"
    path.write_bytes(old)
    _crash_guarded_write(path, new, "after_exchange")

    revision = io.records_revision(path)

    assert revision.text == old.decode("utf-8")
    assert path.read_bytes() == old
    assert _wal_entries(path) == []


@pytest.mark.parametrize(
    "damage",
    ["truncated", "wrong_target", "missing_temp", "replaced_temp", "second_marker"],
)
def test_damaged_prepared_evidence_refuses_without_moving_names(
    tmp_path: Path,
    damage: str,
) -> None:
    path = tmp_path / "candidates.yaml"
    old = b"phase: old\nrecords: []\n"
    new = b"phase: new\nrecords: []\n"
    path.write_bytes(old)
    _crash_guarded_write(path, new, "after_exchange")
    marker = next(path.parent.glob(f".{path.name}.*{io._CAS_PREPARED_SUFFIX}"))
    temporary = next(path.parent.glob(f".{path.name}.*.tmp"))

    if damage == "truncated":
        marker.write_bytes(marker.read_bytes()[:12])
    elif damage == "wrong_target":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["target"] = "other.yaml"
        marker.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "missing_temp":
        temporary.unlink()
    elif damage == "replaced_temp":
        temporary.unlink()
        temporary.write_bytes(b"unrelated private bytes\n")
    elif damage == "second_marker":
        marker.with_name(
            f".{path.name}.second{io._CAS_PREPARED_SUFFIX}"
        ).write_bytes(marker.read_bytes())
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(damage)

    evidence = {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(path)
    }
    with pytest.raises(DataError):
        io.read_bytes_bound(path)

    assert path.read_bytes() == new
    assert {
        candidate.name: candidate.read_bytes()
        for candidate in _wal_entries(path)
    } == evidence


def test_guarded_replacement_does_not_truncate_a_hard_link_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)
    path = tmp_path / "candidates.yaml"
    backup = tmp_path / "owner-backup.yaml"
    old = b"phase: old\nrecords: []\n"
    path.write_bytes(old)
    os.link(path, backup)

    io.atomic_write_text_bound(
        path,
        "phase: new\nrecords: []\n",
        expected_revision=hashlib.sha256(old).hexdigest(),
    )

    assert path.read_bytes() == b"phase: new\nrecords: []\n"
    assert backup.read_bytes() == old
    assert list((lock_root / "retired-writes").iterdir()) == []


@pytest.mark.parametrize("move_reports_error", [False, True])
def test_marker_transition_refuses_a_replacement_left_at_the_prepared_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_reports_error: bool,
) -> None:
    path = tmp_path / "candidates.yaml"
    old = b"phase: old\nrecords: []\n"
    path.write_bytes(old)
    real_move = io._rename_entry_exclusive
    injected = False

    def move_then_replace_source(
        directory_fd: int,
        left: str,
        right: str,
    ) -> None:
        nonlocal injected
        real_move(directory_fd, left, right)
        if (
            left.endswith(io._CAS_PREPARED_SUFFIX)
            and right.endswith(io._CAS_VALIDATED_SUFFIX)
            and not injected
        ):
            descriptor = os.open(
                left,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b"unrelated marker replacement")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            injected = True
            if move_reports_error:
                raise OSError("rename reported an ambiguous failure")

    monkeypatch.setattr(io, "_rename_entry_exclusive", move_then_replace_source)

    with pytest.raises(DataError):
        io.atomic_write_text_bound(
            path,
            "phase: new\nrecords: []\n",
            expected_revision=hashlib.sha256(old).hexdigest(),
        )

    assert injected
    replacements = list(
        path.parent.glob(f".{path.name}.*{io._CAS_PREPARED_SUFFIX}")
    )
    assert len(replacements) == 1
    assert replacements[0].read_bytes() == b"unrelated marker replacement"


@pytest.mark.parametrize(
    "suffix", [io._CAS_VALIDATED_SUFFIX, io._CAS_RECOVERED_SUFFIX]
)
def test_empty_terminal_marker_refuses_without_touching_public_bytes(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / "candidates.yaml"
    public = b"phase: owner\nrecords: []\n"
    path.write_bytes(public)
    marker = path.with_name(f".{path.name}.damaged{suffix}")
    marker.write_bytes(b"")

    with pytest.raises(DataError, match="marker"):
        io.read_bytes_bound(path)

    assert path.read_bytes() == public
    assert marker.read_bytes() == b""


@pytest.mark.parametrize("publication", ["expected_absent", "guarded"])
def test_parent_detached_during_final_cleanup_cannot_report_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    directory = tmp_path / "live"
    directory.mkdir()
    held = tmp_path / "detached"
    path = directory / "target.json"
    old = b"old bytes\n"
    new = b"new bytes\n"
    if publication == "guarded":
        path.write_bytes(old)
    real_retire = io._retire_exact_entry
    detached = False

    def detach_then_retire(*args: object, **kwargs: object) -> bool:
        nonlocal detached
        if not detached:
            directory.rename(held)
            directory.mkdir()
            detached = True
        return real_retire(*args, **kwargs)

    monkeypatch.setattr(io, "_retire_exact_entry", detach_then_retire)

    with pytest.raises(DataError, match="directory changed"):
        if publication == "expected_absent":
            io.atomic_write_bytes_bound(path, new, expected_absent=True)
        else:
            io.atomic_write_text_bound(
                path,
                new.decode("utf-8"),
                expected_revision=hashlib.sha256(old).hexdigest(),
            )

    assert detached
    assert not path.exists()
    assert (held / path.name).read_bytes() == new


def test_private_temp_replaced_after_exchange_is_never_made_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.json"
    old = b"old bytes\n"
    new = b"new bytes\n"
    raced = b"raced private bytes\n"
    path.write_bytes(old)
    real_exchange = io._exchange_entries
    raced_name = ""

    def exchange_then_replace_temp(
        directory_fd: int,
        temporary: str,
        target: str,
    ) -> None:
        nonlocal raced_name
        real_exchange(directory_fd, temporary, target)
        os.unlink(temporary, dir_fd=directory_fd)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, raced)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raced_name = temporary

    monkeypatch.setattr(io, "_exchange_entries", exchange_then_replace_temp)

    with pytest.raises(DataError, match="both names were retained"):
        io.atomic_write_text_bound(
            path,
            new.decode("utf-8"),
            expected_revision=hashlib.sha256(old).hexdigest(),
        )

    assert raced_name
    assert path.read_bytes() == new
    assert (path.parent / raced_name).read_bytes() == raced
    with pytest.raises(DataError, match="bound evidence"):
        io.read_bytes_bound(path)
    assert path.read_bytes() == new
    assert (path.parent / raced_name).read_bytes() == raced
