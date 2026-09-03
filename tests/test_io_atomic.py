from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from japanese_anki import cli, io
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError
from japanese_anki.models import VocabularyRecord


def _record(expression: str = "話す", reading: str = "はなす") -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["to speak"],
    )


def _no_temp_files(directory: Path) -> bool:
    return not list(directory.rglob("*.tmp"))


def test_atomic_write_creates_parents_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "vocabulary.json"

    io.atomic_write_text(target, "first\n")
    io.atomic_write_text(target, "second\n")

    assert target.read_text(encoding="utf-8") == "second\n"
    assert _no_temp_files(tmp_path)


def test_bound_write_accepts_the_standard_macos_private_var_alias(
    tmp_path: Path,
) -> None:
    private = tmp_path.absolute()
    if private.parts[1:3] != ("private", "var"):
        pytest.skip("this platform does not expose temporary files through /private/var")
    alias = Path("/var").joinpath(*private.parts[3:]) / "bound.bin"

    io.atomic_write_bytes_bound(alias, b"answer", expected_absent=True)

    assert (private / "bound.bin").read_bytes() == b"answer"


def test_atomic_write_failure_mid_write_keeps_previous_contents(tmp_path: Path) -> None:
    # A lone surrogate cannot be encoded as UTF-8, so the failure happens
    # inside handle.write() — genuinely mid-write, with the temp file open.
    target = tmp_path / "vocabulary.json"
    target.write_text("original\n", encoding="utf-8")

    with pytest.raises(UnicodeEncodeError):
        io.atomic_write_text(target, "prefix \ud800 suffix")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert _no_temp_files(tmp_path)


def test_guarded_write_opens_the_observed_target_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular-file stat followed by a FIFO swap must refuse, not hang."""
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nonblock is None:
        pytest.skip("this platform has no nonblocking open flag")
    target = tmp_path / "vocabulary.json"
    old = b"original\n"
    target.write_bytes(old)
    real_open = io.os.open
    observed = 0

    def require_nonblocking(
        name: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal observed
        if name == target.name and kwargs.get("dir_fd") is not None:
            observed += 1
            assert flags & nonblock
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(io.os, "open", require_nonblocking)

    io.atomic_write_text_bound(
        target,
        "replacement\n",
        expected_revision=hashlib.sha256(old).hexdigest(),
    )

    assert observed >= 2


def test_exact_retirement_unlink_failure_is_retryable_after_the_source_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed unlink must not turn the private move into apparent success."""
    target = tmp_path / "captured-answer.json"
    payload = b"paid provider answer"
    target.write_bytes(payload)
    details = target.stat()
    expected_state = (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    )
    revision = hashlib.sha256(payload).hexdigest()
    lock_root = tmp_path / "private-locks"
    monkeypatch.setattr(io, "_path_lock_root", lambda: lock_root)
    real_unlink = io.os.unlink

    def refuse_private_unlink(
        name: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if isinstance(name, str) and name.endswith(".retired"):
            raise OSError("injected retirement unlink failure")
        real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(io.os, "unlink", refuse_private_unlink)
    with (
        io._open_bound_directory(tmp_path, create=False) as binding,
        pytest.raises(DataError, match="could not be removed"),
    ):
        io._retire_exact_entry(
            binding,
            target.name,
            expected_state,
            revision,
            expected_ctime_ns=details.st_ctime_ns,
        )

    retirement = lock_root / "retired-writes"
    assert not target.exists()
    assert len(list(retirement.iterdir())) == 1

    monkeypatch.setattr(io.os, "unlink", real_unlink)
    with io._open_bound_directory(tmp_path, create=False) as binding:
        assert io._retire_exact_entry(
            binding,
            target.name,
            expected_state,
            revision,
            expected_ctime_ns=details.st_ctime_ns,
        )
    assert list(retirement.iterdir()) == []


@pytest.mark.parametrize("replacement_kind", ["symlink", "fifo", "directory"])
def test_cleanup_classifies_a_nonregular_replacement_without_opening_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    target = tmp_path / "captured-answer.json"
    if replacement_kind == "symlink":
        target.symlink_to(tmp_path / "outside-answer")
    elif replacement_kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    real_open = io.os.open

    with io._open_bound_directory(tmp_path, create=False) as binding:

        def refuse_target_open(
            name: object,
            flags: int,
            *args: object,
            **kwargs: object,
        ) -> int:
            if name == target.name and kwargs.get("dir_fd") == binding.descriptor:
                pytest.fail("cleanup opened a non-regular replacement")
            return real_open(name, flags, *args, **kwargs)

        monkeypatch.setattr(io.os, "open", refuse_target_open)
        assert io._cleanup_bound_snapshot(binding.descriptor, target.name) is None


def test_atomic_byte_write_failed_replace_keeps_previous_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "clip.mp3"
    target.write_bytes(b"OLD AUDIO")

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(io.os, "replace", fail_replace)

    with pytest.raises(DataError, match="Could not write"):
        io.atomic_write_bytes(target, b"NEW AUDIO")

    assert target.read_bytes() == b"OLD AUDIO"
    assert _no_temp_files(tmp_path)


def test_atomic_write_failed_rename_is_a_clean_error_and_cleans_up(tmp_path: Path) -> None:
    # Renaming a file over a non-empty directory fails at the os.replace
    # step with a real kernel error — no monkeypatching of os.replace.
    target = tmp_path / "vocabulary.json"
    target.mkdir()
    (target / "child.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(DataError):
        io.atomic_write_text(target, "new content\n")

    assert (target / "child.txt").read_text(encoding="utf-8") == "keep me\n"
    assert _no_temp_files(tmp_path)


def test_atomic_write_writes_through_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real" / "actual.json"
    real.parent.mkdir()
    real.write_text("OLD\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)

    io.atomic_write_text(link, "NEW\n")

    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "NEW\n"


def test_atomic_write_preserves_permission_bits(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"
    target.write_text("original\n", encoding="utf-8")
    os.chmod(target, 0o600)

    io.atomic_write_text(target, "replacement\n")

    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


def test_atomic_write_never_emits_carriage_returns(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"

    io.atomic_write_text(target, "line one\nline two\n")

    assert b"\r" not in target.read_bytes()


def test_save_records_json_untouched_when_a_record_cannot_serialize(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"
    io.save_records_json(target, [_record()])
    original = target.read_text(encoding="utf-8")

    class ExplodingRecord:
        id = "word:壊れる:こわれる"

        def to_dict(self) -> dict[str, str]:
            raise ValueError("cannot serialize this record")

    with pytest.raises(ValueError):
        io.save_records_json(target, [ExplodingRecord()])  # type: ignore[list-item]

    assert target.read_text(encoding="utf-8") == original
    assert _no_temp_files(tmp_path)


def test_save_records_json_survives_the_mid_encoding_failure_that_truncated_v1(
    tmp_path: Path,
) -> None:
    # The old writer streamed json.dump straight into the open target, so an
    # encoder error partway through left truncated garbage behind. A set is
    # not JSON-serializable and only fails during encoding, after to_dict
    # has succeeded — this test fails against the pre-M1.2 implementation.
    target = tmp_path / "vocabulary.json"
    io.save_records_json(target, [_record()])
    original = target.read_text(encoding="utf-8")

    class UnencodableRecord:
        id = "word:囲む:かこむ"

        def to_dict(self) -> dict[str, object]:
            return {"id": self.id, "tags": {1, 2}}

    with pytest.raises(TypeError):
        io.save_records_json(target, [_record(), UnencodableRecord()])  # type: ignore[list-item]

    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(original)
    assert _no_temp_files(tmp_path)


def test_write_failure_message_names_the_target_not_the_temp_file(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    target = locked / "vocabulary.json"
    os.chmod(locked, 0o500)
    try:
        with pytest.raises(DataError) as excinfo:
            io.atomic_write_text(target, "content\n")
    finally:
        os.chmod(locked, 0o700)

    message = str(excinfo.value)
    assert str(target) in message
    assert ".tmp" not in message


def test_load_reports_clean_error_for_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "vocabulary.json"
    directory.mkdir()

    with pytest.raises(DataError):
        io.load_records(directory)


def test_load_records_rejects_non_mapping_items(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"
    target.write_text('["just a string"]\n', encoding="utf-8")

    with pytest.raises(DataError):
        io.load_records(target)


def test_save_records_json_writes_sorted_records_that_load_back(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"

    io.save_records_json(target, [_record("食べる", "たべる"), _record()])

    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "話す" in text  # not escaped to 話
    assert [record.expression for record in io.load_records(target)] == ["話す", "食べる"]


def test_two_record_writers_cannot_silently_discard_each_other(tmp_path: Path) -> None:
    target = tmp_path / "vocabulary.json"
    io.save_records_json(target, [_record()])

    first_revision = io.records_revision(target)
    first_records = io.load_records(target)
    second_revision = io.records_revision(target)
    second_records = io.load_records(target)

    first_records.append(_record("食べる", "たべる"))
    io.save_records_json(target, first_records, expected=first_revision)
    second_records.append(_record("見る", "みる"))

    with pytest.raises(DataError) as caught:
        io.save_records_json(target, second_records, expected=second_revision)

    assert "changed on disk since it was read" in str(caught.value)
    assert "re-run" in str(caught.value)
    assert [record.expression for record in io.load_records(target)] == ["話す", "食べる"]


def test_revision_check_and_replace_are_one_locked_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "vocabulary.json"
    io.save_records_json(target, [_record()])
    first_revision = io.records_revision(target)
    second_revision = io.records_revision(target)
    real_atomic_write = io.atomic_write_text
    first_in_writer = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    failures: dict[str, BaseException] = {}

    def held_write(path: Path, text: str) -> None:
        if "食べる" in text:
            first_in_writer.set()
            release_first.wait(timeout=2)
        real_atomic_write(path, text)

    def save_first() -> None:
        try:
            io.save_records_json(
                target, [_record(), _record("食べる", "たべる")], expected=first_revision
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures["first"] = exc

    def save_second() -> None:
        try:
            io.save_records_json(
                target, [_record(), _record("見る", "みる")], expected=second_revision
            )
        except BaseException as exc:
            failures["second"] = exc
        finally:
            second_done.set()

    monkeypatch.setattr(io, "atomic_write_text", held_write)
    first = threading.Thread(target=save_first)
    second = threading.Thread(target=save_second)
    first.start()
    assert first_in_writer.wait(timeout=2)
    second.start()
    second_finished_before_release = second_done.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not second_finished_before_release
    assert "first" not in failures
    assert isinstance(failures.get("second"), DataError)
    assert [record.expression for record in io.load_records(target)] == ["話す", "食べる"]


@pytest.mark.skipif(os.name == "nt", reason="Windows temp directories are per-user")
def test_locks_avoid_a_shared_temp_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_temp = tmp_path / "shared-temp"
    shared_temp.mkdir(mode=0o700)
    shared_temp.chmod(0o1777)
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    monkeypatch.setattr(io.tempfile, "gettempdir", lambda: str(shared_temp))
    monkeypatch.setattr(io, "_user_home", lambda: user_home)

    with io.exclusive_path_lock(tmp_path / "vocabulary.json"):
        pass

    lock_root = user_home / ".cache" / "janki" / "file-locks"
    assert lock_root.is_dir()
    assert stat.S_IMODE(lock_root.stat().st_mode) == 0o700
    assert not (shared_temp / "janki-file-locks").exists()


def test_main_reports_any_janki_error_without_registration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class BrandNewFeatureError(JankiError):
        """Defined here and deliberately never added to any except clause."""

    def boom(args: object) -> int:
        raise BrandNewFeatureError("the new feature failed")

    monkeypatch.setattr(cli, "command_inspect", boom)

    exit_code = cli.main(["inspect", str(tmp_path / "export.csv")])

    assert exit_code == 1
    assert "the new feature failed" in capsys.readouterr().err


def test_a_file_that_is_not_utf8_is_a_data_error_not_a_crash(
    tmp_path: Path,
) -> None:
    """`UnicodeDecodeError` comes from the *read*, before either parser sees
    anything, so it is not caught by the parse clause it sits beside — and
    being a `ValueError` rather than a `JankiError` it escapes every caller
    that guards for janki's own errors.

    That is not theoretical: the workbench walks staging files and reads the
    collection to answer whether a paid call is safe to start, and a single
    non-UTF-8 byte took the whole page down with no error at all.
    """
    for name, blob in (("bad.json", b"[\xff]"), ("bad.yaml", b"a: \xff")):
        path = tmp_path / name
        path.write_bytes(blob)

        with pytest.raises(DataError) as caught:
            io.load_structured(path)

        assert name in str(caught.value)
