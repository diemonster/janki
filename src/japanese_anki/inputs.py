"""Turning files a human dropped on the desk into content blocks for the API.

``janki extract`` (M3.3) accepts PDFs and phone photos. This module is the step
between "a path the user typed" and "a block the Claude API accepts": it works
out what kind of file it is, converts the one format the API cannot read, and —
importantly — makes sure a copy of the original lives under ``data/inbox/``
before anything reads it.

**The inbox copy is the provenance.** A record extracted from a photo is only
as trustworthy as the ability to go back and look at the photo, and the path
the user typed is on their desktop, in a Downloads folder, or on a phone —
somewhere that will not exist in six months. So a file from outside the inbox
is copied into it first and the *copy* is what every downstream record points
at. Nothing here ever writes to a file that already exists in the inbox
(AGENTS.md: "Never modify files under ``data/inbox/``"); a name collision with
different content gets a fingerprint suffix rather than an overwrite.

HEIC is the one format that cannot be sent as-is. It converts through ``sips``,
which is macOS-only — an accepted scope limit (IMPLEMENTATION_PLAN dependency
policy), with an error naming ``pillow-heif`` elsewhere. The converted JPEG is
transient: it is read into the request and thrown away, so the inbox keeps the
original the camera produced rather than a derived file beside it.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint

__all__ = [
    "DOCUMENT_TYPES",
    "HEIC_SUFFIXES",
    "IMAGE_TYPES",
    "InputError",
    "PreparedInput",
    "prepare_inputs",
]


class InputError(JankiError):
    pass


#: Suffixes the API reads as a ``document`` block, and the media type each one
#: is sent under.
DOCUMENT_TYPES: dict[str, str] = {".pdf": "application/pdf"}

#: Suffixes the API reads as an ``image`` block. PDFs are vision-backed too, but
#: the block type differs, which is why the two tables are separate.
IMAGE_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

#: Converted to JPEG before sending. Both spellings occur in the wild — a phone
#: writes ``.HEIC``, some tools write ``.heif``.
HEIC_SUFFIXES: tuple[str, ...] = (".heic", ".heif")

_CONVERTED_MEDIA_TYPE = "image/jpeg"


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """One file, ready to send, and the path a record should cite for it.

    ``origin_path`` is the copy inside the inbox — never the path the user
    typed — because that is the one that will still be there when someone asks
    where a word came from. For a HEIC it points at the *original*, not the
    JPEG that was actually sent: the JPEG is a rendering, and the camera's file
    is the evidence.
    """

    kind: str
    media_type: str
    data_b64: str
    origin_path: Path

    def content_block(self) -> dict[str, Any]:
        """The API content block for this input.

        Lives here rather than in :mod:`claude_client` on purpose: that module
        is the single owner of *the client*, and giving it a media vocabulary
        would mean two places to change when a format is added.
        """
        return {
            "type": self.kind,
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data_b64,
            },
        }


def _encode(data: bytes) -> str:
    # No line breaks: the API rejects a wrapped base64 payload, and
    # `standard_b64encode` is the variant that does not insert them.
    return base64.standard_b64encode(data).decode("ascii")


def _copy_into_inbox(source: Path, scan_inbox: Path) -> Path:
    """Return the inbox copy of ``source``, making one if it is not there yet.

    A file already under ``scan_inbox`` is used where it lies — re-running an
    extraction over an inbox file must not copy it again. Otherwise the copy
    lands under its own name, and a name already taken by *different* content
    earns a fingerprint suffix: two photos both called ``IMG_0001.HEIC`` are
    two photos, and overwriting one with the other would destroy the evidence
    behind every record extracted from it.
    """
    scan_inbox = Path(scan_inbox)
    if source.is_relative_to(scan_inbox):
        return source

    scan_inbox.mkdir(parents=True, exist_ok=True)
    target = scan_inbox / source.name
    if target.exists():
        if target.read_bytes() == source.read_bytes():
            return target
        stamp = short_fingerprint(str(source.resolve()), length=8)
        target = scan_inbox / f"{source.stem}-{stamp}{source.suffix}"
        if target.exists():
            return target
    shutil.copy2(source, target)
    return target


def _heic_to_jpeg(
    source: Path,
    run: Callable[..., Any],
    platform: str,
) -> bytes:
    """The JPEG bytes for a HEIC, via ``sips``.

    macOS-only, and said so plainly rather than half-working elsewhere: the
    accepted scope of this project is a single-user macOS tool, and a caller on
    another platform gets the name of the package that would fix it rather than
    a confusing "command not found".
    """
    if not platform.startswith("darwin"):
        raise InputError(
            f"Converting {source.name} needs macOS's 'sips', and this is "
            f"{platform}. Install 'pillow-heif' and convert the file to JPEG "
            "first, or pass a JPEG or PNG instead."
        )
    with tempfile.TemporaryDirectory() as work:
        # Converted into a temporary directory, never beside the original:
        # data/inbox holds what the camera produced, and a derived JPEG sitting
        # next to it would look like a second source file.
        out = Path(work) / f"{source.stem}.jpg"
        try:
            result = run(
                ["sips", "-s", "format", "jpeg", str(source), "--out", str(out)],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise InputError(f"Could not run 'sips' to convert {source.name}: {exc}") from exc
        if getattr(result, "returncode", 0) != 0:
            detail = (getattr(result, "stderr", "") or "").strip()
            raise InputError(
                f"'sips' could not convert {source.name}"
                + (f": {detail}" if detail else "")
            )
        try:
            return out.read_bytes()
        except OSError as exc:
            raise InputError(
                f"'sips' reported success but wrote no JPEG for {source.name}: {exc}"
            ) from exc


def prepare_inputs(
    paths: Sequence[Path],
    scan_inbox: Path,
    *,
    run: Callable[..., Any] = subprocess.run,
    platform: str | None = None,
) -> list[PreparedInput]:
    """Every path, copied into the inbox and encoded for the API.

    Order is the caller's, and duplicates are kept: passing the same photo
    twice costs tokens, but silently dropping the second one is the kind of
    quiet discard this project refuses everywhere else. An unreadable path or
    an unsupported suffix stops the whole batch rather than extracting a subset
    the user would have to notice was short.

    ``run`` and ``platform`` are injectable so tests never shell out
    (IMPLEMENTATION_PLAN rule 6 in spirit: the one impure dependency is faked).
    """
    where = platform if platform is not None else sys.platform
    prepared: list[PreparedInput] = []
    for raw in paths:
        source = Path(raw).expanduser()
        if not source.is_file():
            raise InputError(f"Not a readable file: {source}")

        suffix = source.suffix.lower()
        stored = _copy_into_inbox(source, scan_inbox)

        if suffix in DOCUMENT_TYPES:
            kind, media_type = "document", DOCUMENT_TYPES[suffix]
            data = stored.read_bytes()
        elif suffix in IMAGE_TYPES:
            kind, media_type = "image", IMAGE_TYPES[suffix]
            data = stored.read_bytes()
        elif suffix in HEIC_SUFFIXES:
            kind, media_type = "image", _CONVERTED_MEDIA_TYPE
            data = _heic_to_jpeg(stored, run, where)
        else:
            known = ", ".join(
                sorted({*DOCUMENT_TYPES, *IMAGE_TYPES, *HEIC_SUFFIXES})
            )
            raise InputError(
                f"janki cannot read {source.name} ('{suffix or 'no suffix'}'). "
                f"Supported: {known}."
            )

        prepared.append(
            PreparedInput(
                kind=kind,
                media_type=media_type,
                data_b64=_encode(data),
                origin_path=stored,
            )
        )
    return prepared
