"""Local two-phase source uploads for the isolated Assistant sidecar.

ChatKit first registers attachment metadata and then follows the returned PUT
descriptor.  This module keeps those uploaded bytes in an owned temporary
directory.  They enter the repository only when the containing user message is
accepted, through :func:`japanese_anki.inputs.receive_upload`, the same immutable
intake boundary used by the ordinary workbench.

The module deliberately does not import ChatKit at import time.  The Assistant
extra remains optional for every command that never constructs this store.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import io
import os
import secrets
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit, urlunsplit

from japanese_anki import inputs
from japanese_anki.errors import JankiError

MAX_ASSISTANT_ATTACHMENT_BYTES = 128 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_ATTACHMENT_STATES = 256


class AssistantAttachmentError(JankiError):
    """A source attachment could not cross the local intake boundary."""


@dataclass(frozen=True, slots=True)
class ReadyAssistantAttachment:
    """Metadata for a complete temporary upload that may now be committed."""

    attachment_id: str
    name: str
    mime_type: str
    size: int


@dataclass(frozen=True, slots=True)
class CommittedAssistantAttachment:
    """The immutable corpus result produced for one ready attachment."""

    name: str
    path: Path
    stored: bool


@dataclass(slots=True)
class _AttachmentState:
    attachment_id: str
    name: str
    mime_type: str
    size: int
    capability: str | None
    temporary_path: Path
    uploading: bool = False
    committing: bool = False
    uploaded: bool = False
    temporary_device: int | None = None
    temporary_inode: int | None = None
    content_sha256: bytes | None = None
    committed: CommittedAssistantAttachment | None = None


def _canonical_media_type(name: str) -> str:
    """Classify a browser name using the source tables owned by ``inputs``."""

    suffix = Path(name).suffix.lower()
    if suffix in inputs.DOCUMENT_TYPES:
        return inputs.DOCUMENT_TYPES[suffix]
    if suffix in inputs.IMAGE_TYPES:
        return inputs.IMAGE_TYPES[suffix]
    if suffix in inputs.HEIC_SUFFIXES:
        return "image/heic" if suffix == ".heic" else "image/heif"
    supported = ", ".join(
        sorted({*inputs.DOCUMENT_TYPES, *inputs.IMAGE_TYPES, *inputs.HEIC_SUFFIXES})
    )
    raise AssistantAttachmentError(
        f"janki cannot read {name} ({suffix!r} suffix). Supported: {supported}."
    )


def _normalise_upload_prefix(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("upload_prefix must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("upload_prefix must not contain URL credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("upload_prefix must not contain a query or fragment")
    path = parsed.path.rstrip("/") + "/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _upload_prefix_for_context(default: str, context: Any) -> str:
    """Use the already-authorized browser origin for its upload descriptor."""

    request_origin = getattr(context, "request_origin", None)
    if request_origin is None:
        return default
    parsed_origin = urlsplit(request_origin)
    if (
        parsed_origin.scheme not in {"http", "https"}
        or not parsed_origin.netloc
        or parsed_origin.username is not None
        or parsed_origin.password is not None
        or parsed_origin.path not in {"", "/"}
        or parsed_origin.query
        or parsed_origin.fragment
    ):
        raise AssistantAttachmentError("The Assistant request origin is invalid.")
    path = urlsplit(default).path
    return urlunsplit((parsed_origin.scheme, parsed_origin.netloc, path, "", ""))


class LocalAssistantAttachmentStore:
    """One-use, temporary implementation of ChatKit's ``AttachmentStore``.

    It intentionally relies on structural typing instead of inheriting from
    ChatKit's abstract base class.  ``ChatKitServer`` calls these two async
    methods directly, while keeping this module importable without the optional
    Assistant dependencies installed.
    """

    def __init__(self, *, inbox_root: Path, upload_prefix: str) -> None:
        self._inbox_root = Path(inbox_root)
        self._upload_prefix = _normalise_upload_prefix(upload_prefix)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="janki-assistant-attachments-"
        )
        self._temporary_root = Path(self._temporary_directory.name)
        self._states: dict[str, _AttachmentState] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def temporary_root(self) -> Path:
        """The exact owned directory, exposed for lifecycle diagnostics."""

        return self._temporary_root

    def _assert_open(self) -> None:
        if self._closed:
            raise AssistantAttachmentError("The Assistant attachment store is closed.")

    def _state(self, attachment_id: str) -> _AttachmentState:
        self._assert_open()
        try:
            return self._states[attachment_id]
        except KeyError as exc:
            raise AssistantAttachmentError(
                f"Unknown attachment: {attachment_id}"
            ) from exc

    def _evict_oldest_inactive_state(self) -> None:
        """Retire one stale session state without racing active file work."""

        candidate = next(
            (
                state
                for state in self._states.values()
                if not state.uploading and not state.committing
            ),
            None,
        )
        if candidate is None:
            raise AssistantAttachmentError(
                "The Assistant's attachment limit is occupied by uploads or "
                "commits in progress. Finish one before adding another source."
            )
        try:
            candidate.temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise AssistantAttachmentError(
                f"Could not retire the abandoned temporary upload for "
                f"{candidate.name}: {exc}"
            ) from exc
        self._states.pop(candidate.attachment_id)

    def generate_attachment_id(self, mime_type: str, context: Any) -> str:
        """Return a URL-segment-safe identifier without importing ChatKit."""

        del mime_type, context
        return f"atc_{secrets.token_hex(16)}"

    async def create_attachment(self, input: Any, context: Any) -> Any:
        """Reserve metadata and return a one-use same-origin PUT descriptor."""

        try:
            name = inputs.safe_upload_name(input.name)
        except inputs.InputError as exc:
            raise AssistantAttachmentError(str(exc)) from exc
        media_type = _canonical_media_type(name)
        size = input.size
        if isinstance(size, bool) or not isinstance(size, int):
            raise AssistantAttachmentError(f"{name} did not report a valid byte size.")
        if size <= 0:
            raise AssistantAttachmentError(f"{name} is empty; there is nothing to upload.")
        if size > MAX_ASSISTANT_ATTACHMENT_BYTES:
            limit_mib = MAX_ASSISTANT_ATTACHMENT_BYTES // (1024 * 1024)
            raise AssistantAttachmentError(
                f"{name} is too large for the Assistant's {limit_mib} MiB limit."
            )

        try:
            from chatkit.types import AttachmentUploadDescriptor, FileAttachment
        except ImportError as exc:  # pragma: no cover - exercised by install boundary
            raise RuntimeError(
                "Assistant attachments require the optional 'assistant' dependencies."
            ) from exc

        with self._lock:
            self._assert_open()
            # Match the workbench's 256-entry session capability bound. A tab
            # may reload or abandon a completed PUT forever; retiring its
            # oldest inactive state also retires any uncommitted temp bytes.
            while len(self._states) >= _MAX_ATTACHMENT_STATES:
                self._evict_oldest_inactive_state()
            attachment_id = self.generate_attachment_id(media_type, None)
            while attachment_id in self._states:
                attachment_id = self.generate_attachment_id(media_type, None)
            capability = secrets.token_urlsafe(32)
            state = _AttachmentState(
                attachment_id=attachment_id,
                name=name,
                mime_type=media_type,
                size=size,
                capability=capability,
                temporary_path=self._temporary_root / f"{attachment_id}.upload",
            )
            upload_prefix = _upload_prefix_for_context(self._upload_prefix, context)
            upload_url = (
                f"{upload_prefix}{quote(attachment_id, safe='')}/"
                f"{quote(capability, safe='')}"
            )
            attachment = FileAttachment(
                id=attachment_id,
                name=name,
                mime_type=media_type,
                upload_descriptor=AttachmentUploadDescriptor(
                    url=upload_url,
                    method="PUT",
                    headers={},
                ),
            )
            self._states[attachment_id] = state
            return attachment

    def accept_upload(
        self,
        attachment_id: str,
        capability: str,
        data: bytes,
        *,
        declared_content_length: int,
    ) -> None:
        """Consume one bound capability and retain the exact bytes temporarily."""

        if not isinstance(data, bytes):
            raise AssistantAttachmentError("The attachment upload body must be bytes.")
        if len(data) != declared_content_length:
            raise AssistantAttachmentError(
                f"The attachment declared {declared_content_length} bytes but the "
                f"upload received {len(data)} bytes."
            )
        self.accept_upload_stream(
            attachment_id,
            capability,
            io.BytesIO(data),
            declared_content_length=declared_content_length,
        )

    def accept_upload_stream(
        self,
        attachment_id: str,
        capability: str,
        stream: BinaryIO,
        *,
        declared_content_length: int,
    ) -> None:
        """Stream one authorized body into a byte- and inode-bound temp file."""

        with self._lock:
            state = self._state(attachment_id)
            if state.capability is None:
                raise AssistantAttachmentError(
                    f"The upload capability for {attachment_id} has already been used."
                )
            if (
                not isinstance(capability, str)
                or not capability.isascii()
                or not hmac.compare_digest(capability, state.capability)
            ):
                raise AssistantAttachmentError("The attachment upload capability is invalid.")
            if (
                isinstance(declared_content_length, bool)
                or not isinstance(declared_content_length, int)
                or declared_content_length < 0
            ):
                raise AssistantAttachmentError("The upload needs a valid Content-Length.")
            if declared_content_length != state.size:
                raise AssistantAttachmentError(
                    f"{state.name} declared {declared_content_length} bytes, but its "
                    f"attachment reservation expects {state.size}."
                )
            if state.uploading:
                raise AssistantAttachmentError(
                    f"The upload capability for {attachment_id} has already been used "
                    "or is currently in use."
                )
            state.uploading = True

        completed = False
        try:
            digest = hashlib.sha256()
            with state.temporary_path.open("xb") as handle:
                remaining = declared_content_length
                received = 0
                while remaining:
                    chunk = stream.read(min(remaining, _UPLOAD_CHUNK_BYTES))
                    if not isinstance(chunk, bytes):
                        raise AssistantAttachmentError(
                            "The attachment upload stream did not return bytes."
                        )
                    if not chunk:
                        raise AssistantAttachmentError(
                            f"{state.name} ended after {received} of "
                            f"{declared_content_length} bytes."
                        )
                    if len(chunk) > remaining:
                        raise AssistantAttachmentError(
                            f"{state.name} exceeded its declared byte size."
                        )
                    written = handle.write(chunk)
                    if written != len(chunk):
                        raise OSError(f"wrote {written} of {len(chunk)} upload bytes")
                    digest.update(chunk)
                    received += len(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
                retained = os.fstat(handle.fileno())
                if not stat.S_ISREG(retained.st_mode):
                    raise AssistantAttachmentError(
                        f"The temporary upload for {state.name} is not a regular file."
                    )

            with self._lock:
                if (
                    self._closed
                    or self._states.get(attachment_id) is not state
                    or not state.uploading
                ):
                    raise AssistantAttachmentError(
                        f"The upload claim for {state.name} changed before completion."
                    )
                state.capability = None
                state.uploading = False
                state.uploaded = True
                state.temporary_device = retained.st_dev
                state.temporary_inode = retained.st_ino
                state.content_sha256 = digest.digest()
                completed = True
        except AssistantAttachmentError:
            raise
        except OSError as exc:
            raise AssistantAttachmentError(
                f"Could not retain the temporary upload for {state.name}: {exc}"
            ) from exc
        finally:
            if not completed:
                with contextlib.suppress(OSError):
                    state.temporary_path.unlink(missing_ok=True)
                with self._lock:
                    if self._states.get(attachment_id) is state:
                        state.uploading = False

    def assert_ready(self, attachment_id: str) -> ReadyAssistantAttachment:
        """Return bound metadata only after all declared bytes were received."""

        with self._lock:
            state = self._state(attachment_id)
            if not state.uploaded:
                raise AssistantAttachmentError(
                    f"Attachment {state.name} has not finished uploading."
                )
            return ReadyAssistantAttachment(
                attachment_id=state.attachment_id,
                name=state.name,
                mime_type=state.mime_type,
                size=state.size,
            )

    def commit_attachment(self, attachment_id: str) -> CommittedAssistantAttachment:
        """Move ready bytes through the existing immutable corpus intake."""

        with self._lock:
            state = self._state(attachment_id)
            if state.committed is not None:
                return state.committed
            if state.committing:
                raise AssistantAttachmentError(
                    f"The commit is in progress for {state.name}."
                )
            if not state.uploaded:
                raise AssistantAttachmentError(
                    f"Attachment {state.name} has not finished uploading."
                )
            if (
                state.temporary_device is None
                or state.temporary_inode is None
                or state.content_sha256 is None
            ):
                raise AssistantAttachmentError(
                    f"The temporary upload for {state.name} changed before commit."
                )
            state.committing = True
            name = state.name
            size = state.size
            temporary_path = state.temporary_path
            temporary_device = state.temporary_device
            temporary_inode = state.temporary_inode
            content_sha256 = state.content_sha256

        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = -1
            try:
                descriptor = os.open(temporary_path, flags)
                retained = os.fstat(descriptor)
                identity_matches = (
                    stat.S_ISREG(retained.st_mode)
                    and retained.st_dev == temporary_device
                    and retained.st_ino == temporary_inode
                    and retained.st_size == size
                )
                if not identity_matches:
                    raise AssistantAttachmentError(
                        f"The temporary upload for {name} changed before commit."
                    )
                with os.fdopen(descriptor, "rb", closefd=True) as handle:
                    descriptor = -1
                    data = handle.read()
            except AssistantAttachmentError:
                raise
            except OSError as exc:
                raise AssistantAttachmentError(
                    f"The temporary upload for {name} changed before commit."
                ) from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if len(data) != size or not hmac.compare_digest(
                hashlib.sha256(data).digest(), content_sha256
            ):
                raise AssistantAttachmentError(
                    f"The temporary upload for {name} changed before commit."
                )
            try:
                intake = inputs.receive_upload(
                    name,
                    data,
                    inbox_root=self._inbox_root,
                )
            except inputs.InputError as exc:
                raise AssistantAttachmentError(str(exc)) from exc
            result = CommittedAssistantAttachment(
                name=name,
                path=intake.path,
                stored=intake.stored,
            )
            with self._lock:
                if (
                    self._closed
                    or self._states.get(attachment_id) is not state
                    or not state.committing
                ):
                    raise AssistantAttachmentError(
                        f"The commit claim for {name} changed before completion."
                    )
                state.committed = result
                state.committing = False
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)
            return result
        finally:
            with self._lock:
                if self._states.get(attachment_id) is state:
                    state.committing = False

    async def delete_attachment(self, attachment_id: str, context: Any) -> None:
        """Forget Assistant metadata and pending bytes, never a corpus source."""

        del context
        with self._lock:
            state = self._state(attachment_id)
            if state.uploading:
                raise AssistantAttachmentError(
                    f"Cannot delete {state.name} while its upload is in progress."
                )
            if state.committing:
                raise AssistantAttachmentError(
                    f"Cannot delete {state.name} while its commit is in progress."
                )
            self._states.pop(attachment_id)
            state.temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Forget every state and remove exactly this store's owned temp tree."""

        with self._lock:
            if self._closed:
                return
            uploading = next(
                (state for state in self._states.values() if state.uploading), None
            )
            if uploading is not None:
                raise AssistantAttachmentError(
                    f"Cannot close the attachment store while {uploading.name}'s "
                    "upload is in progress."
                )
            committing = next(
                (state for state in self._states.values() if state.committing), None
            )
            if committing is not None:
                raise AssistantAttachmentError(
                    f"Cannot close the attachment store while {committing.name}'s "
                    "commit is in progress."
                )
            self._closed = True
            self._states.clear()
            temporary_directory = self._temporary_directory
        temporary_directory.cleanup()

    def __enter__(self) -> LocalAssistantAttachmentStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        del exc
        self.close()
