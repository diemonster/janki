"""Preparing PDFs and photos for the API.

The one impure dependency — ``sips`` — is faked, so nothing here shells out or
depends on running on macOS.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki.inputs import InputError, PreparedInput, prepare_inputs

PNG = b"\x89PNG\r\n\x1a\n fake png bytes"
PDF = b"%PDF-1.7 fake pdf bytes"
HEIC = b"\x00\x00\x00\x18ftypheic fake"
JPEG = b"\xff\xd8\xff fake jpeg bytes"


class FakeSips:
    """Writes the JPEG ``sips`` would have written, and records the argv."""

    def __init__(self, returncode: int = 0, stderr: str = "", write: bool = True) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.write = write
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(argv)
        if self.write and self.returncode == 0:
            Path(argv[argv.index("--out") + 1]).write_bytes(JPEG)
        return SimpleNamespace(returncode=self.returncode, stderr=self.stderr, stdout="")


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def decoded(item: PreparedInput) -> bytes:
    return base64.standard_b64decode(item.data_b64)


# --- formats -----------------------------------------------------------------


def test_a_pdf_becomes_a_document_block(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)

    [item] = prepare_inputs([source], tmp_path / "inbox")

    assert item.kind == "document"
    assert item.media_type == "application/pdf"
    assert decoded(item) == PDF
    assert item.source_sha256 == hashlib.sha256(PDF).hexdigest()


@pytest.mark.parametrize(
    ("name", "media_type"),
    [("a.jpg", "image/jpeg"), ("a.JPEG", "image/jpeg"), ("a.png", "image/png")],
)
def test_photos_become_image_blocks(tmp_path: Path, name: str, media_type: str) -> None:
    # Suffix matching is case-insensitive: a phone writes .JPG and .HEIC.
    source = write(tmp_path / "desk" / name, PNG)

    [item] = prepare_inputs([source], tmp_path / "inbox")

    assert item.kind == "image"
    assert item.media_type == media_type


def test_the_content_block_is_the_shape_the_api_takes(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)

    [item] = prepare_inputs([source], tmp_path / "inbox")

    assert item.content_block() == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": item.data_b64,
        },
    }


def test_the_payload_carries_no_line_breaks(tmp_path: Path) -> None:
    # A wrapped base64 payload is rejected by the API.
    source = write(tmp_path / "desk" / "big.pdf", PDF * 200)

    [item] = prepare_inputs([source], tmp_path / "inbox")

    assert "\n" not in item.data_b64


def test_an_unsupported_file_stops_the_batch_and_lists_what_works(
    tmp_path: Path,
) -> None:
    source = write(tmp_path / "desk" / "notes.txt", b"hello")

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox")

    message = str(excinfo.value)
    assert "notes.txt" in message
    assert ".pdf" in message and ".heic" in message


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(InputError) as excinfo:
        prepare_inputs([tmp_path / "nope.pdf"], tmp_path / "inbox")

    assert "nope.pdf" in str(excinfo.value)


# --- the inbox copy ----------------------------------------------------------


def test_a_file_from_outside_is_copied_in_and_the_copy_is_the_provenance(
    tmp_path: Path,
) -> None:
    # The path the user typed is on a desktop or a phone; it will not be there
    # in six months, and a record extracted from it has to stay checkable.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)

    [item] = prepare_inputs([source], inbox)

    assert item.origin_path == inbox / "lesson.pdf"
    assert item.origin_path.read_bytes() == PDF
    assert source.exists()


def test_a_file_already_in_the_inbox_is_used_where_it_lies(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    source = write(inbox / "lesson.pdf", PDF)

    [item] = prepare_inputs([source], inbox)

    assert item.origin_path == source
    assert list(inbox.iterdir()) == [source]


def test_a_file_in_the_parent_inbox_is_used_where_it_lies(tmp_path: Path) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    source = write(inbox / "lesson.pdf", PDF)

    [item] = prepare_inputs([source], scan_inbox, inbox_root=inbox)

    assert item.origin_path == source
    assert not scan_inbox.exists()


def test_a_relative_parent_inbox_path_is_used_where_it_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    write(inbox / "lesson.pdf", PDF)
    monkeypatch.chdir(tmp_path)

    [item] = prepare_inputs(
        [Path("data/inbox/lesson.pdf")], scan_inbox, inbox_root=inbox
    )

    assert item.origin_path == Path("data/inbox/lesson.pdf")
    assert not scan_inbox.exists()


def test_parent_traversal_does_not_make_an_outside_file_look_durable(
    tmp_path: Path,
) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    actual = write(tmp_path / "data" / "desk" / "lesson.pdf", PDF)
    inbox.mkdir(parents=True)
    traversing = inbox / ".." / "desk" / "lesson.pdf"
    assert traversing.is_file()

    [item] = prepare_inputs([traversing], scan_inbox, inbox_root=inbox)

    assert item.origin_path == scan_inbox / "lesson.pdf"
    assert item.origin_path.read_bytes() == actual.read_bytes()


def test_an_inbox_symlink_to_an_outside_file_gets_a_durable_copy(
    tmp_path: Path,
) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    outside = write(tmp_path / "desk" / "lesson.pdf", PDF)
    inbox.mkdir(parents=True)
    source = inbox / "linked.pdf"
    source.symlink_to(outside)

    [item] = prepare_inputs([source], scan_inbox, inbox_root=inbox)

    assert item.origin_path == scan_inbox / "linked.pdf"
    assert item.origin_path.read_bytes() == PDF


def test_a_scan_inbox_symlink_to_an_outside_file_gets_a_fingerprinted_copy(
    tmp_path: Path,
) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    outside = write(tmp_path / "desk" / "lesson.pdf", PDF)
    scan_inbox.mkdir(parents=True)
    source = scan_inbox / "lesson.pdf"
    source.symlink_to(outside)

    [item] = prepare_inputs([source], scan_inbox, inbox_root=inbox)

    assert item.origin_path.parent == scan_inbox
    assert item.origin_path != source
    assert item.origin_path.name.startswith("lesson-")
    assert item.origin_path.read_bytes() == PDF


def test_a_broken_scan_inbox_symlink_is_not_followed_or_overwritten(
    tmp_path: Path,
) -> None:
    inbox = tmp_path / "data" / "inbox"
    scan_inbox = inbox / "scans"
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)
    outside = tmp_path / "outside" / "missing.pdf"
    scan_inbox.mkdir(parents=True)
    collision = scan_inbox / "lesson.pdf"
    collision.symlink_to(outside)

    [item] = prepare_inputs([source], scan_inbox, inbox_root=inbox)

    assert collision.is_symlink()
    assert not outside.exists()
    assert item.origin_path != collision
    assert item.origin_path.read_bytes() == PDF


def test_re_preparing_the_same_file_copies_it_once(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)

    prepare_inputs([source], inbox)
    prepare_inputs([source], inbox)

    assert [path.name for path in inbox.iterdir()] == ["lesson.pdf"]


def test_two_different_photos_with_one_name_both_survive(tmp_path: Path) -> None:
    # Every phone writes IMG_0001. Overwriting one with the other would destroy
    # the evidence behind every record extracted from it.
    inbox = tmp_path / "inbox"
    first = write(tmp_path / "trip" / "IMG_0001.png", PNG)
    second = write(tmp_path / "class" / "IMG_0001.png", PNG + b" different")

    [one] = prepare_inputs([first], inbox)
    [two] = prepare_inputs([second], inbox)

    assert one.origin_path != two.origin_path
    assert one.origin_path.read_bytes() == PNG
    assert two.origin_path.read_bytes() == PNG + b" different"


def test_nothing_already_in_the_inbox_is_overwritten(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    existing = write(inbox / "lesson.pdf", b"%PDF the original")
    incoming = write(tmp_path / "desk" / "lesson.pdf", PDF)

    prepare_inputs([incoming], inbox)

    assert existing.read_bytes() == b"%PDF the original"


def test_duplicates_in_one_call_are_kept_not_silently_dropped(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)

    prepared = prepare_inputs([source, source], tmp_path / "inbox")

    assert len(prepared) == 2


def test_order_is_the_callers(tmp_path: Path) -> None:
    first = write(tmp_path / "desk" / "a.pdf", PDF)
    second = write(tmp_path / "desk" / "b.png", PNG)

    prepared = prepare_inputs([second, first], tmp_path / "inbox")

    assert [item.origin_path.name for item in prepared] == ["b.png", "a.pdf"]


# --- HEIC --------------------------------------------------------------------


def test_a_heic_is_converted_and_sent_as_jpeg(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "IMG_0001.HEIC", HEIC)
    sips = FakeSips()

    [item] = prepare_inputs([source], inbox, run=sips, platform="darwin")

    assert item.kind == "image"
    assert item.media_type == "image/jpeg"
    assert decoded(item) == JPEG
    argv = sips.calls[0]
    assert argv[:4] == ["sips", "-s", "format", "jpeg"]


def test_the_heic_original_is_what_the_inbox_keeps(tmp_path: Path) -> None:
    # The camera's file is the evidence; the JPEG is a rendering of it, and a
    # derived file beside the original would read as a second source.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "IMG_0001.HEIC", HEIC)

    [item] = prepare_inputs([source], inbox, run=FakeSips(), platform="darwin")

    assert item.origin_path == inbox / "IMG_0001.HEIC"
    assert item.origin_path.read_bytes() == HEIC
    assert [path.name for path in inbox.iterdir()] == ["IMG_0001.HEIC"]


def test_heic_off_macos_names_the_package_that_would_fix_it(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "a.heic", HEIC)
    sips = FakeSips()

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox", run=sips, platform="linux")

    assert "pillow-heif" in str(excinfo.value)
    assert sips.calls == []


def test_a_failed_conversion_reports_what_sips_said(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "a.heic", HEIC)
    sips = FakeSips(returncode=1, stderr="Error: unsupported file")

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox", run=sips, platform="darwin")

    assert "unsupported file" in str(excinfo.value)


def test_a_conversion_that_writes_nothing_is_an_error_not_empty_bytes(
    tmp_path: Path,
) -> None:
    # A silent success that produced no file would send an empty image.
    source = write(tmp_path / "desk" / "a.heic", HEIC)
    sips = FakeSips(write=False)

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox", run=sips, platform="darwin")

    assert "no JPEG" in str(excinfo.value)


def test_a_missing_sips_binary_is_reported_clearly(tmp_path: Path) -> None:
    source = write(tmp_path / "desk" / "a.heic", HEIC)

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("No such file or directory: 'sips'")

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox", run=missing, platform="darwin")

    assert "sips" in str(excinfo.value)


def test_no_temporary_jpeg_is_left_behind(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "a.heic", HEIC)
    sips = FakeSips()

    prepare_inputs([source], inbox, run=sips, platform="darwin")

    out = Path(sips.calls[0][sips.calls[0].index("--out") + 1])
    assert not out.exists()
    assert not out.parent.exists()


# --- the collision fingerprint is of the bytes, not the path ----------------


def test_a_reused_path_holding_a_new_photo_is_stored_not_confused_for_the_old(
    tmp_path: Path,
) -> None:
    # The bug this guards: a path is not an identity. ~/Downloads/IMG_0001.png
    # holds a different photo after the next AirDrop, and a path-derived
    # suffix pointed the new photo at the old one's copy — sending the wrong
    # image and citing it as provenance for a file never stored at all.
    inbox = tmp_path / "inbox"
    reused = tmp_path / "downloads" / "IMG_0001.png"

    write(tmp_path / "trip" / "IMG_0001.png", PNG)
    [first] = prepare_inputs([tmp_path / "trip" / "IMG_0001.png"], inbox)

    write(reused, PNG + b" photo B")
    [second] = prepare_inputs([reused], inbox)

    write(reused, PNG + b" photo C")
    [third] = prepare_inputs([reused], inbox)

    # Three distinct photos, three distinct copies, each holding its own bytes.
    paths = {first.origin_path, second.origin_path, third.origin_path}
    assert len(paths) == 3
    assert third.origin_path.read_bytes() == PNG + b" photo C"
    assert decoded(third) == PNG + b" photo C"
    assert second.origin_path.read_bytes() == PNG + b" photo B"


def test_the_same_bytes_from_a_third_path_reuse_the_existing_copy(
    tmp_path: Path,
) -> None:
    # The other direction: a content-addressed name means identical bytes are
    # stored once however many paths they arrive from.
    inbox = tmp_path / "inbox"
    write(tmp_path / "a" / "IMG_0001.png", PNG)
    write(tmp_path / "b" / "IMG_0001.png", PNG + b" same")
    write(tmp_path / "c" / "IMG_0001.png", PNG + b" same")

    prepare_inputs([tmp_path / "a" / "IMG_0001.png"], inbox)
    [second] = prepare_inputs([tmp_path / "b" / "IMG_0001.png"], inbox)
    [third] = prepare_inputs([tmp_path / "c" / "IMG_0001.png"], inbox)

    assert second.origin_path == third.origin_path
    assert len(list(inbox.iterdir())) == 2


# --- failures leave the inbox alone -----------------------------------------


def test_an_unsupported_file_is_not_copied_into_the_inbox_first(
    tmp_path: Path,
) -> None:
    # data/inbox is committed and nothing here removes files, so a copy made
    # before the check would permanently pollute the provenance directory.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "notes.txt", b"hello")

    with pytest.raises(InputError):
        prepare_inputs([source], inbox)

    assert not inbox.exists() or list(inbox.iterdir()) == []


def test_a_heic_on_another_platform_keeps_its_copy(tmp_path: Path) -> None:
    # Unlike an unsupported suffix: HEIC *is* supported, so the copy is right
    # to keep — only this machine cannot convert it.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "a.heic", HEIC)

    with pytest.raises(InputError):
        prepare_inputs([source], inbox, run=FakeSips(), platform="linux")

    assert (inbox / "a.heic").read_bytes() == HEIC


def test_an_unreadable_file_is_a_janki_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # is_file() says a path exists, not that it can be read. A raw OSError
    # would reach the user as a traceback — cli.main formats JankiError only.
    source = write(tmp_path / "desk" / "locked.pdf", PDF)
    real = Path.read_bytes

    def deny(self: Path) -> bytes:
        if self.name == "locked.pdf":
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", deny)

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], tmp_path / "inbox")

    assert "locked.pdf" in str(excinfo.value)


def test_what_is_sent_is_what_is_stored_even_if_the_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The paths this module is pointed at are the volatile ones: a
    # half-finished AirDrop, an iCloud sync, a re-export into Downloads. Two
    # independent reads of the same path could send one image and store
    # another, citing provenance that is not what was extracted from.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "downloads" / "IMG_0001.png", PNG)
    real = Path.read_bytes
    seen: list[str] = []

    def changing(self: Path) -> bytes:
        data = real(self)
        if self == source and not seen:
            seen.append("read")
            # Replaced the instant *after* janki reads it, which is the window
            # a second independent read would fall into.
            write(source, PNG + b" replaced mid-run")
        return data

    monkeypatch.setattr(Path, "read_bytes", changing)

    [item] = prepare_inputs([source], inbox)

    monkeypatch.undo()
    assert decoded(item) == item.origin_path.read_bytes()


def test_a_failed_copy_leaves_nothing_behind_under_the_real_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A partial write surviving under the authentic name would pose as the real
    # file forever, with the genuine bytes hidden behind a fingerprint suffix.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)
    real = Path.write_bytes

    def fail(self: Path, data: bytes) -> int:
        real(self, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", fail)

    with pytest.raises(InputError):
        prepare_inputs([source], inbox)

    monkeypatch.undo()
    assert not (inbox / "lesson.pdf").exists()


def test_a_cleanup_that_also_fails_still_reports_a_janki_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failures that break a write mid-way — a disconnected volume, a dying
    # disk — are the same ones that can break the unlink. Letting that
    # propagate would swap the formatted error for a traceback and leave the
    # partial file unnamed.
    inbox = tmp_path / "inbox"
    source = write(tmp_path / "desk" / "lesson.pdf", PDF)
    real_write = Path.write_bytes

    def fail_write(self: Path, data: bytes) -> int:
        real_write(self, data[: len(data) // 2])
        raise OSError(5, "Input/output error")

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        raise OSError(19, "No such device")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(InputError) as excinfo:
        prepare_inputs([source], inbox)

    monkeypatch.undo()
    message = str(excinfo.value)
    assert "Input/output error" in message
    # The survivor is named, since nothing prunes the inbox.
    assert "partial file may remain" in message
    assert str(inbox / "lesson.pdf") in message
