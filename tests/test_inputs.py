"""Preparing PDFs and photos for the API.

The one impure dependency — ``sips`` — is faked, so nothing here shells out or
depends on running on macOS.
"""

from __future__ import annotations

import base64
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
