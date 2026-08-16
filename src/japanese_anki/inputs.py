"""Turning files a human dropped on the desk into content blocks for the API.

``janki extract`` (M3.3) accepts PDFs and phone photos. This module is the step
between "a path the user typed" and "a block the Claude API accepts": it works
out what kind of file it is, converts the one format the API cannot read, and —
importantly — makes sure the original lives under ``data/inbox/`` before
anything reads it.

**The inbox file is the provenance.** A record extracted from a photo is only
as trustworthy as the ability to go back and look at the photo, and the path
the user typed is on their desktop, in a Downloads folder, or on a phone —
somewhere that will not exist in six months. So a file from outside the durable
inbox root is copied into its scan directory first. A file already anywhere in
that root is used where it lies. Nothing here ever writes to a file that
already exists in the inbox (AGENTS.md: "Never modify files under
``data/inbox/``"); a name collision with different content gets a fingerprint
suffix rather than an overwrite.

HEIC is the one format that cannot be sent as-is. It converts through ``sips``,
which is macOS-only — an accepted scope limit (IMPLEMENTATION_PLAN dependency
policy), with an error naming ``pillow-heif`` elsewhere. The converted JPEG is
transient: it is read into the request and thrown away, so the inbox keeps the
original the camera produced rather than a derived file beside it.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
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
    "inside",
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

    ``origin_path`` is the durable file inside the inbox. It can be the path the
    user supplied when that path is already in the inbox. Otherwise it is the
    new scan-inbox copy. For a HEIC it points at the *original*, not the JPEG
    that was actually sent: the JPEG is a rendering, and the camera's file is
    the evidence.
    """

    kind: str
    media_type: str
    data_b64: str
    origin_path: Path
    source_sha256: str = ""
    #: Whether *this* run wrote the file at ``origin_path`` into the inbox.
    #: False when the source was already durable and used in place, or when it
    #: matched a file the inbox already held. A caller reporting what a run left
    #: behind needs this: the path alone cannot distinguish the two.
    copied: bool = False

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


def _classify(source: Path) -> tuple[str, str, bool]:
    """``(block kind, media type, needs conversion)`` for a path, or an error.

    An unsupported suffix stops the whole batch rather than skipping the file:
    extracting a subset would leave the user to notice it came up short.
    """
    suffix = source.suffix.lower()
    if suffix in DOCUMENT_TYPES:
        return "document", DOCUMENT_TYPES[suffix], False
    if suffix in IMAGE_TYPES:
        return "image", IMAGE_TYPES[suffix], False
    if suffix in HEIC_SUFFIXES:
        return "image", _CONVERTED_MEDIA_TYPE, True
    known = ", ".join(sorted({*DOCUMENT_TYPES, *IMAGE_TYPES, *HEIC_SUFFIXES}))
    raise InputError(
        f"janki cannot read {source.name} ('{suffix or 'no suffix'}'). "
        f"Supported: {known}."
    )


def _read(path: Path) -> bytes:
    """The file's bytes, with any OS failure named as a janki error.

    ``is_file()`` says a path exists and is a file; it does not say it can be
    read. A no-permission file, or an iCloud placeholder that was never
    downloaded, gets past that check and fails here — and a raw ``OSError``
    would reach the user as a traceback, because ``cli.main`` formats
    ``JankiError`` and nothing else.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc


def inside(source: Path, root: Path) -> bool:
    """Return whether an existing source resolves inside ``root``."""
    try:
        return source.resolve(strict=True).is_relative_to(root.resolve())
    except OSError:
        return False


def _occupied(path: Path) -> bool:
    """Return whether a path entry exists, including a broken symlink."""
    return path.exists() or path.is_symlink()


def _durable_namesakes(
    name: str,
    durable_root: Path,
    data: bytes,
    *,
    exclude: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Durable files with this case-insensitive basename, split by content."""
    try:
        excluded_real = exclude.resolve(strict=True) if exclude is not None else None
        candidates = tuple(durable_root.rglob("*"))
    except OSError as exc:
        raise InputError(
            f"Could not check durable inbox names for {name}: {exc}"
        ) from exc
    matches: list[Path] = []
    conflicts: list[Path] = []
    for candidate in candidates:
        if candidate.name.casefold() != name.casefold():
            continue
        try:
            same_file = (
                excluded_real is not None
                and candidate.resolve(strict=True) == excluded_real
            )
        except OSError:
            continue
        if same_file or not candidate.is_file() or not inside(candidate, durable_root):
            continue
        if _read(candidate) == data:
            matches.append(candidate)
        else:
            conflicts.append(candidate)
    return (
        sorted(matches, key=lambda path: path.as_posix()),
        sorted(conflicts, key=lambda path: path.as_posix()),
    )


def _durable_name_collision(source: Path, conflicts: list[Path]) -> InputError:
    names = ", ".join(str(path) for path in conflicts)
    return InputError(
        f"Cannot use {source}: it and durable inbox file(s) {names} would be "
        "stored under one name because they have the same basename but "
        "different content. Give each source a unique filename before you put "
        "it in the inbox."
    )


def _copy_into_inbox(
    source: Path,
    scan_inbox: Path,
    data: bytes,
    *,
    inbox_root: Path | None = None,
) -> tuple[Path, bool]:
    """Return the inbox path for ``source`` and whether this call wrote it.

    The flag is not cosmetic. Four of the five ways out of here return a file
    that was *already* in the inbox — used in place, matched by name, matched
    by fingerprint, or already holding these exact bytes — and only the last
    one copies. A caller that wants to tell a user "this is now in your inbox"
    cannot work that out from the returned path, because a path under the inbox
    is what every branch returns.

    A file already under the durable ``inbox_root`` is used where it lies —
    re-running an extraction over any inbox file must not copy it again. The
    root defaults to ``scan_inbox`` for callers that have only one inbox
    directory. Otherwise the copy lands under its own name, and a name already
    taken by *different* content earns a suffix: two photos both called
    ``IMG_0001.HEIC`` are two photos, and overwriting one with the other would
    destroy the evidence behind every record extracted from it.

    **The suffix is a fingerprint of the bytes, never of the source path.** A
    path is not an identity: the same Downloads filename holds a different
    photo after the next AirDrop, so a path-derived suffix would point a second
    photo at the first one's copy — sending the wrong image to the API and
    citing it as the provenance for records extracted from a file that was
    never stored at all. Fingerprinting the content instead makes the target
    name a property of what is being stored, so an existing target under that
    name already holds exactly these bytes.
    """
    scan_inbox = Path(scan_inbox)
    durable_root = Path(inbox_root) if inbox_root is not None else scan_inbox
    if inside(source, durable_root):
        _, conflicts = _durable_namesakes(
            source.name, durable_root, data, exclude=source
        )
        if conflicts:
            raise _durable_name_collision(source, conflicts)
        return source, False

    scan_inbox.mkdir(parents=True, exist_ok=True)
    target = scan_inbox / source.name
    matches, conflicts = _durable_namesakes(source.name, durable_root, data)
    if matches and conflicts:
        raise _durable_name_collision(source, conflicts)
    if matches:
        return (target if target in matches else matches[0]), False
    if conflicts or _occupied(target):
        # Content-addressed, via the project's one fingerprint helper: the hex
        # digest goes through it rather than a second hashing scheme.
        stamp = short_fingerprint(hashlib.sha256(data).hexdigest(), length=8)
        target = scan_inbox / f"{source.stem}-{stamp}{source.suffix}"
        fingerprint_matches, fingerprint_conflicts = _durable_namesakes(
            target.name, durable_root, data
        )
        if fingerprint_conflicts:
            names = ", ".join(str(path) for path in fingerprint_conflicts)
            raise InputError(
                f"Cannot store {source}: durable inbox file(s) {names} have "
                "different content under the fingerprint of these bytes. Move "
                "those files aside; janki will not overwrite inbox evidence."
            )
        if fingerprint_matches:
            return (
                target if target in fingerprint_matches else fingerprint_matches[0]
            ), False
        if _occupied(target):
            if inside(target, durable_root) and _read(target) == data:
                return target, False
            raise InputError(
                f"Cannot store {source}: {target} already holds different content "
                "under the fingerprint of these bytes. Move that file aside — janki "
                "will not overwrite anything in the inbox."
            )
    # Written from the bytes already in hand rather than re-read from the
    # source. `shutil.copy2` would open the file a second time, and the paths
    # this module is pointed at are exactly the volatile ones — a half-finished
    # AirDrop, an iCloud sync, a re-export into Downloads. If the file changed
    # in between, the API would be sent one image while the inbox stored
    # another, and every record would cite provenance that is not what it was
    # extracted from: the same wrong-image bug the content fingerprint above
    # exists to prevent, arriving by a different door.
    try:
        target.write_bytes(data)
        # Timestamps and mode, the part of copy2 worth keeping. Best-effort:
        # failing to carry an mtime across is not worth losing the copy over.
        with contextlib.suppress(OSError):
            shutil.copystat(source, target)
    except OSError as exc:
        # A partial write must not survive under the authentic name — it would
        # pose as the real file forever while the genuine bytes hid behind a
        # fingerprint suffix, and nothing prunes the inbox. The cleanup is
        # best-effort because the failures that break a write mid-way — a
        # disconnected volume, a dying disk — are exactly the ones that can
        # break the unlink too, and letting *that* propagate would replace this
        # error with a traceback the CLI cannot format.
        removed = True
        try:
            target.unlink(missing_ok=True)
        except OSError:
            removed = False
        leftover = (
            ""
            if removed
            else f" A partial file may remain at {target} — remove it before retrying."
        )
        raise InputError(
            f"Could not copy {source} into {scan_inbox}: {exc}.{leftover}"
        ) from exc
    return target, True


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
    inbox_root: Path | None = None,
    run: Callable[..., Any] = subprocess.run,
    platform: str | None = None,
) -> list[PreparedInput]:
    """Every path, made durable in the inbox and encoded for the API.

    Order is the caller's, and duplicates are kept: passing the same photo
    twice costs tokens, but silently dropping the second one is the kind of
    quiet discard this project refuses everywhere else. An unreadable path or
    an unsupported suffix stops the whole batch rather than extracting a subset
    the user would have to notice was short.

    ``inbox_root`` names the full durable root when ``scan_inbox`` is one
    subdirectory of it. It defaults to ``scan_inbox``. ``run`` and ``platform``
    are injectable so tests never shell out
    (IMPLEMENTATION_PLAN rule 6 in spirit: the one impure dependency is faked).
    """
    where = platform if platform is not None else sys.platform
    prepared: list[PreparedInput] = []
    for raw in paths:
        source = Path(raw).expanduser()
        if not source.is_file():
            raise InputError(f"Not a readable file: {source}")

        # Classified *before* anything is copied: `data/inbox/` is committed
        # and nothing here removes files, so copying first would leave a file
        # janki cannot read sitting permanently in the provenance directory
        # after a run that failed. (A HEIC on a non-macOS machine is different
        # — that is a supported format, and keeping its copy is right.)
        kind, media_type, convert = _classify(source)
        data = _read(source)
        source_sha256 = hashlib.sha256(data).hexdigest()
        stored, copied = _copy_into_inbox(
            source, scan_inbox, data, inbox_root=inbox_root
        )
        if convert:
            data = _heic_to_jpeg(stored, run, where)

        prepared.append(
            PreparedInput(
                kind=kind,
                media_type=media_type,
                data_b64=_encode(data),
                origin_path=stored,
                source_sha256=source_sha256,
                copied=copied,
            )
        )
    return prepared
