"""The Assistant's local, two-phase source attachment boundary."""

from __future__ import annotations

import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from japanese_anki.workbench import assistant_attachments as attachments_module
from japanese_anki.workbench.assistant_attachments import (
    MAX_ASSISTANT_ATTACHMENT_BYTES,
    AssistantAttachmentError,
    LocalAssistantAttachmentStore,
)


@dataclass(frozen=True, slots=True)
class _CreateParams:
    name: str
    size: int
    mime_type: str


def _create(
    store: LocalAssistantAttachmentStore,
    *,
    name: str = "lesson.pdf",
    size: int = 4,
    mime_type: str = "application/octet-stream",
):
    return asyncio.run(
        store.create_attachment(
            _CreateParams(name=name, size=size, mime_type=mime_type),
            context=None,
        )
    )


def _upload_coordinates(attachment: object) -> tuple[str, str]:
    descriptor = attachment.upload_descriptor
    assert descriptor is not None
    parts = urlsplit(str(descriptor.url)).path.rstrip("/").split("/")
    return unquote(parts[-2]), unquote(parts[-1])


def test_attachment_stays_temporary_until_explicit_commit(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/session/attachments/",
    )
    try:
        attachment = _create(
            store,
            name=r"C:\fakepath\lesson.PDF",
            mime_type="application/octet-stream",
        )

        assert attachment.type == "file"
        assert attachment.name == "lesson.PDF"
        assert attachment.mime_type == "application/pdf"
        assert attachment.upload_descriptor.method == "PUT"
        assert str(attachment.upload_descriptor.url).startswith(
            "http://127.0.0.1:4321/session/attachments/"
        )
        assert not inbox.exists()
        with pytest.raises(AssistantAttachmentError, match="has not finished uploading"):
            store.assert_ready(attachment.id)

        attachment_id, capability = _upload_coordinates(attachment)
        assert attachment_id == attachment.id
        store.accept_upload(
            attachment_id,
            capability,
            b"%PDF",
            declared_content_length=4,
        )

        ready = store.assert_ready(attachment.id)
        assert ready.name == "lesson.PDF"
        assert ready.mime_type == "application/pdf"
        assert ready.size == 4
        assert not inbox.exists()

        committed = store.commit_attachment(attachment.id)
        assert committed.name == "lesson.PDF"
        assert committed.path == inbox / "lesson.PDF"
        assert committed.stored is True
        assert committed.path.read_bytes() == b"%PDF"
        assert store.commit_attachment(attachment.id) == committed

        with pytest.raises(AssistantAttachmentError, match="already been used"):
            store.accept_upload(
                attachment_id,
                capability,
                b"%PDF",
                declared_content_length=4,
            )
    finally:
        store.close()


@pytest.mark.parametrize("name", ["notes.txt", "archive.zip", "no-suffix"])
def test_create_refuses_unsupported_source_suffix(tmp_path: Path, name: str) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        with pytest.raises(AssistantAttachmentError, match="Supported"):
            _create(store, name=name)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("name", "mime_type"),
    [
        ("source.pdf", "application/pdf"),
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("scan.png", "image/png"),
        ("camera.heic", "image/heic"),
        ("camera.heif", "image/heif"),
    ],
)
def test_create_accepts_each_corpus_source_suffix(
    tmp_path: Path,
    name: str,
    mime_type: str,
) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store, name=name, size=1)

        assert attachment.name == name
        assert attachment.mime_type == mime_type
    finally:
        store.close()


@pytest.mark.parametrize("size", [0, MAX_ASSISTANT_ATTACHMENT_BYTES + 1])
def test_create_refuses_empty_or_oversize_source(tmp_path: Path, size: int) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        with pytest.raises(AssistantAttachmentError, match="128 MiB|empty"):
            _create(store, size=size)
    finally:
        store.close()


def test_create_accepts_existing_large_genki_pdf_size(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store, size=78_805_435)

        assert attachment.name == "lesson.pdf"
        assert MAX_ASSISTANT_ATTACHMENT_BYTES == 128 * 1024 * 1024
    finally:
        store.close()


def test_completed_upload_accumulation_evicts_oldest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attachments_module, "_MAX_ATTACHMENT_STATES", 2)
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    large_bytes = b"x" * (2 * 1024 * 1024 + 17)
    try:
        oldest = _create(store, name="oldest.pdf", size=len(large_bytes))
        oldest_id, oldest_capability = _upload_coordinates(oldest)
        store.accept_upload_stream(
            oldest_id,
            oldest_capability,
            io.BytesIO(large_bytes),
            declared_content_length=len(large_bytes),
        )
        oldest_path = store.temporary_root / f"{oldest_id}.upload"
        assert oldest_path.stat().st_size == len(large_bytes)

        newer = _create(store, name="newer.png", size=3)
        newer_id, newer_capability = _upload_coordinates(newer)
        store.accept_upload(
            newer_id,
            newer_capability,
            b"PNG",
            declared_content_length=3,
        )

        newest = _create(store, name="newest.jpg", size=3)

        assert not oldest_path.exists()
        with pytest.raises(AssistantAttachmentError, match="Unknown attachment"):
            store.assert_ready(oldest_id)
        assert store.assert_ready(newer.id).size == 3
        with pytest.raises(
            AssistantAttachmentError, match="has not finished uploading"
        ):
            store.assert_ready(newest.id)
        assert not (tmp_path / "inbox").exists()
    finally:
        store.close()


def test_bare_reservation_accumulation_keeps_only_session_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert attachments_module._MAX_ATTACHMENT_STATES == 256
    monkeypatch.setattr(attachments_module, "_MAX_ATTACHMENT_STATES", 3)
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        reservations = [
            _create(store, name=f"page-{index}.png", size=1)
            for index in range(8)
        ]

        for reservation in reservations[:-3]:
            with pytest.raises(AssistantAttachmentError, match="Unknown attachment"):
                store.assert_ready(reservation.id)
        for reservation in reservations[-3:]:
            with pytest.raises(
                AssistantAttachmentError, match="has not finished uploading"
            ):
                store.assert_ready(reservation.id)
        assert list(store.temporary_root.iterdir()) == []
    finally:
        store.close()


def test_upload_requires_its_bound_capability(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)

        with pytest.raises(AssistantAttachmentError, match="capability"):
            store.accept_upload(
                attachment_id,
                "wrong-capability",
                b"%PDF",
                declared_content_length=4,
            )
        store.accept_upload(
            attachment_id,
            capability,
            b"%PDF",
            declared_content_length=4,
        )
        assert store.assert_ready(attachment_id).size == 4
    finally:
        store.close()


def test_upload_refuses_non_ascii_capability_as_invalid(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)

        with pytest.raises(AssistantAttachmentError, match="capability is invalid"):
            store.accept_upload(
                attachment.id,
                "capability-☃",
                b"%PDF",
                declared_content_length=4,
            )
    finally:
        store.close()


def test_upload_length_must_match_attachment_reservation(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)

        with pytest.raises(AssistantAttachmentError, match="declared 3 bytes"):
            store.accept_upload(
                attachment_id,
                capability,
                b"PDF",
                declared_content_length=3,
            )

        store.accept_upload(
            attachment_id,
            capability,
            b"%PDF",
            declared_content_length=4,
        )
        assert store.assert_ready(attachment_id).size == 4
    finally:
        store.close()


def test_upload_body_must_match_declared_content_length(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)

        with pytest.raises(AssistantAttachmentError, match="received 3 bytes"):
            store.accept_upload(
                attachment_id,
                capability,
                b"PDF",
                declared_content_length=4,
            )

        store.accept_upload(
            attachment_id,
            capability,
            b"%PDF",
            declared_content_length=4,
        )
        assert store.assert_ready(attachment_id).size == 4
    finally:
        store.close()


class _BoundedReadStream(io.BytesIO):
    def __init__(self, data: bytes, *, largest_allowed_read: int) -> None:
        super().__init__(data)
        self.largest_allowed_read = largest_allowed_read
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0 or size > self.largest_allowed_read:
            raise AssertionError(f"unbounded upload read: {size}")
        return super().read(size)


class _PausedReadStream(io.BytesIO):
    def __init__(
        self,
        data: bytes,
        *,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(data)
        self._entered = entered
        self._release = release

    def read(self, size: int = -1) -> bytes:
        self._entered.set()
        if not self._release.wait(3):
            raise AssertionError("the test never released the upload stream")
        return super().read(size)


def _uploaded_store(
    tmp_path: Path,
) -> tuple[LocalAssistantAttachmentStore, str]:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    attachment = _create(store)
    attachment_id, capability = _upload_coordinates(attachment)
    store.accept_upload(
        attachment_id,
        capability,
        b"GOOD",
        declared_content_length=4,
    )
    return store, attachment_id


def _pause_commit_intake(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    entered = threading.Event()
    release = threading.Event()
    original = attachments_module.inputs.receive_upload

    def paused(*args: object, **kwargs: object):
        entered.set()
        if not release.wait(3):
            raise AssertionError("the test never released the corpus intake")
        return original(*args, **kwargs)

    monkeypatch.setattr(attachments_module.inputs, "receive_upload", paused)
    return entered, release


def test_upload_stream_is_consumed_in_bounded_chunks(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    data = b"x" * (2 * 1024 * 1024 + 17)
    try:
        attachment = _create(store, size=len(data))
        attachment_id, capability = _upload_coordinates(attachment)
        stream = _BoundedReadStream(data, largest_allowed_read=1024 * 1024)

        store.accept_upload_stream(
            attachment_id,
            capability,
            stream,
            declared_content_length=len(data),
        )

        assert len(stream.read_sizes) == 3
        assert max(stream.read_sizes) <= 1024 * 1024
        assert store.commit_attachment(attachment_id).path.read_bytes() == data
    finally:
        store.close()


def test_incomplete_stream_does_not_consume_its_one_use_capability(
    tmp_path: Path,
) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)

        with pytest.raises(AssistantAttachmentError, match="ended after 3 of 4 bytes"):
            store.accept_upload_stream(
                attachment_id,
                capability,
                io.BytesIO(b"BAD"),
                declared_content_length=4,
            )

        store.accept_upload_stream(
            attachment_id,
            capability,
            io.BytesIO(b"GOOD"),
            declared_content_length=4,
        )
        with pytest.raises(AssistantAttachmentError, match="already been used"):
            store.accept_upload_stream(
                attachment_id,
                capability,
                io.BytesIO(b"GOOD"),
                declared_content_length=4,
            )
    finally:
        store.close()


def test_assert_ready_returns_promptly_while_upload_stream_is_paused(
    tmp_path: Path,
) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    attachment = _create(store)
    attachment_id, capability = _upload_coordinates(attachment)
    entered = threading.Event()
    release = threading.Event()
    stream = _PausedReadStream(b"GOOD", entered=entered, release=release)
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload = pool.submit(
            store.accept_upload_stream,
            attachment_id,
            capability,
            stream,
            declared_content_length=4,
        )
        assert entered.wait(1)
        readiness = pool.submit(store.assert_ready, attachment_id)
        try:
            with pytest.raises(
                AssistantAttachmentError, match="has not finished uploading"
            ):
                readiness.result(timeout=0.5)
        finally:
            release.set()
        upload.result(timeout=1)
    try:
        assert store.assert_ready(attachment_id).size == 4
    finally:
        store.close()


def test_upload_claim_refuses_concurrent_upload_without_waiting(
    tmp_path: Path,
) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    attachment = _create(store)
    attachment_id, capability = _upload_coordinates(attachment)
    entered = threading.Event()
    release = threading.Event()
    stream = _PausedReadStream(b"GOOD", entered=entered, release=release)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            store.accept_upload_stream,
            attachment_id,
            capability,
            stream,
            declared_content_length=4,
        )
        assert entered.wait(1)
        second = pool.submit(
            store.accept_upload,
            attachment_id,
            capability,
            b"EVIL",
            declared_content_length=4,
        )
        try:
            with pytest.raises(AssistantAttachmentError, match="already been used"):
                second.result(timeout=0.5)
        finally:
            release.set()
        first.result(timeout=1)
    try:
        assert store.commit_attachment(attachment_id).path.read_bytes() == b"GOOD"
    finally:
        store.close()


def test_delete_and_close_refuse_while_upload_claim_is_active(
    tmp_path: Path,
) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    attachment = _create(store)
    attachment_id, capability = _upload_coordinates(attachment)
    entered = threading.Event()
    release = threading.Event()
    stream = _PausedReadStream(b"GOOD", entered=entered, release=release)
    with ThreadPoolExecutor(max_workers=3) as pool:
        upload = pool.submit(
            store.accept_upload_stream,
            attachment_id,
            capability,
            stream,
            declared_content_length=4,
        )
        assert entered.wait(1)
        deletion = pool.submit(
            asyncio.run, store.delete_attachment(attachment_id, context=None)
        )
        closing = pool.submit(store.close)
        try:
            with pytest.raises(AssistantAttachmentError, match="upload is in progress"):
                deletion.result(timeout=0.5)
            with pytest.raises(AssistantAttachmentError, match="upload is in progress"):
                closing.result(timeout=0.5)
        finally:
            release.set()
        upload.result(timeout=1)
    try:
        assert store.assert_ready(attachment_id).size == 4
    finally:
        store.close()


def test_state_bound_never_evicts_active_upload_or_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attachments_module, "_MAX_ATTACHMENT_STATES", 2)
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    committing = _create(store, name="committing.pdf")
    committing_id, committing_capability = _upload_coordinates(committing)
    store.accept_upload(
        committing_id,
        committing_capability,
        b"GOOD",
        declared_content_length=4,
    )
    commit_entered, commit_release = _pause_commit_intake(monkeypatch)

    uploading = _create(store, name="uploading.png")
    uploading_id, uploading_capability = _upload_coordinates(uploading)
    upload_entered = threading.Event()
    upload_release = threading.Event()
    stream = _PausedReadStream(
        b"GOOD",
        entered=upload_entered,
        release=upload_release,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(store.commit_attachment, committing_id)
        assert commit_entered.wait(1)
        upload = pool.submit(
            store.accept_upload_stream,
            uploading_id,
            uploading_capability,
            stream,
            declared_content_length=4,
        )
        assert upload_entered.wait(1)
        try:
            with pytest.raises(AssistantAttachmentError, match="in progress"):
                _create(store, name="refused.jpg")

            assert store.assert_ready(committing_id).size == 4
            with pytest.raises(
                AssistantAttachmentError, match="has not finished uploading"
            ):
                store.assert_ready(uploading_id)

            upload_release.set()
            upload.result(timeout=1)
            replacement = _create(store, name="replacement.jpg")

            with pytest.raises(AssistantAttachmentError, match="Unknown attachment"):
                store.assert_ready(uploading_id)
            assert store.assert_ready(committing_id).size == 4
            with pytest.raises(
                AssistantAttachmentError, match="has not finished uploading"
            ):
                store.assert_ready(replacement.id)
        finally:
            upload_release.set()
            commit_release.set()
        assert commit.result(timeout=1).path.read_bytes() == b"GOOD"
    store.close()


def test_assert_ready_returns_promptly_while_commit_intake_is_paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, attachment_id = _uploaded_store(tmp_path)
    entered, release = _pause_commit_intake(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(store.commit_attachment, attachment_id)
        assert entered.wait(1)
        readiness = pool.submit(store.assert_ready, attachment_id)
        try:
            assert readiness.result(timeout=0.5).size == 4
        finally:
            release.set()
        assert commit.result(timeout=1).path.read_bytes() == b"GOOD"
    store.close()


def test_commit_claim_refuses_a_concurrent_commit_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, attachment_id = _uploaded_store(tmp_path)
    entered, release = _pause_commit_intake(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.commit_attachment, attachment_id)
        assert entered.wait(1)
        second = pool.submit(store.commit_attachment, attachment_id)
        try:
            with pytest.raises(AssistantAttachmentError, match="commit is in progress"):
                second.result(timeout=0.5)
        finally:
            release.set()
        assert first.result(timeout=1).path.read_bytes() == b"GOOD"
    store.close()


def test_failed_commit_releases_its_claim_for_an_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, attachment_id = _uploaded_store(tmp_path)
    original = attachments_module.inputs.receive_upload
    calls = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise attachments_module.inputs.InputError("temporary intake failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(attachments_module.inputs, "receive_upload", fail_once)
    try:
        with pytest.raises(AssistantAttachmentError, match="temporary intake failure"):
            store.commit_attachment(attachment_id)

        result = store.commit_attachment(attachment_id)

        assert result.path.read_bytes() == b"GOOD"
        assert calls == 2
    finally:
        store.close()


def test_delete_refuses_promptly_while_commit_claim_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, attachment_id = _uploaded_store(tmp_path)
    entered, release = _pause_commit_intake(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(store.commit_attachment, attachment_id)
        assert entered.wait(1)
        deletion = pool.submit(
            asyncio.run,
            store.delete_attachment(attachment_id, context=None),
        )
        try:
            with pytest.raises(AssistantAttachmentError, match="commit is in progress"):
                deletion.result(timeout=0.5)
        finally:
            release.set()
        assert commit.result(timeout=1).path.read_bytes() == b"GOOD"
    store.close()


def test_close_refuses_promptly_while_commit_claim_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, attachment_id = _uploaded_store(tmp_path)
    entered, release = _pause_commit_intake(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(store.commit_attachment, attachment_id)
        assert entered.wait(1)
        closing = pool.submit(store.close)
        try:
            with pytest.raises(AssistantAttachmentError, match="commit is in progress"):
                closing.result(timeout=0.5)
        finally:
            release.set()
        assert commit.result(timeout=1).path.read_bytes() == b"GOOD"
    store.close()


def test_commit_refuses_same_size_mutation_of_uploaded_bytes(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)
        store.accept_upload(
            attachment_id,
            capability,
            b"GOOD",
            declared_content_length=4,
        )
        temporary_path = store.temporary_root / f"{attachment_id}.upload"
        temporary_path.write_bytes(b"EVIL")

        with pytest.raises(AssistantAttachmentError, match="changed before commit"):
            store.commit_attachment(attachment_id)

        assert not inbox.exists()
    finally:
        store.close()


def test_commit_refuses_same_size_replacement_of_uploaded_file(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)
        store.accept_upload(
            attachment_id,
            capability,
            b"GOOD",
            declared_content_length=4,
        )
        temporary_path = store.temporary_root / f"{attachment_id}.upload"
        replacement = store.temporary_root / "replacement.upload"
        replacement.write_bytes(b"GOOD")
        temporary_path.unlink()
        replacement.replace(temporary_path)

        with pytest.raises(AssistantAttachmentError, match="changed before commit"):
            store.commit_attachment(attachment_id)

        assert not inbox.exists()
    finally:
        store.close()


def test_commit_never_follows_replacement_symlink(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"EVIL")
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)
        store.accept_upload(
            attachment_id,
            capability,
            b"GOOD",
            declared_content_length=4,
        )
        temporary_path = store.temporary_root / f"{attachment_id}.upload"
        temporary_path.unlink()
        temporary_path.symlink_to(outside)

        with pytest.raises(AssistantAttachmentError, match="changed before commit"):
            store.commit_attachment(attachment_id)

        assert outside.read_bytes() == b"EVIL"
        assert not inbox.exists()
    finally:
        store.close()


def test_capability_can_win_only_one_concurrent_upload(tmp_path: Path) -> None:
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)

        def upload() -> str:
            try:
                store.accept_upload(
                    attachment_id,
                    capability,
                    b"%PDF",
                    declared_content_length=4,
                )
            except AssistantAttachmentError as exc:
                return str(exc)
            return "accepted"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: upload(), range(2)))

        assert outcomes.count("accepted") == 1
        refusal = next(outcome for outcome in outcomes if outcome != "accepted")
        assert "already been used" in refusal
    finally:
        store.close()


def test_delete_discards_only_pending_bytes(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        pending = _create(store, name="pending.png", size=3)
        pending_id, pending_capability = _upload_coordinates(pending)
        store.accept_upload(
            pending_id,
            pending_capability,
            b"PNG",
            declared_content_length=3,
        )
        asyncio.run(store.delete_attachment(pending_id, context=None))
        with pytest.raises(AssistantAttachmentError, match="Unknown attachment"):
            store.assert_ready(pending_id)
        assert not inbox.exists()

        durable = _create(store, name="durable.jpg", size=3)
        durable_id, durable_capability = _upload_coordinates(durable)
        store.accept_upload(
            durable_id,
            durable_capability,
            b"JPG",
            declared_content_length=3,
        )
        result = store.commit_attachment(durable_id)
        asyncio.run(store.delete_attachment(durable_id, context=None))

        assert result.path.read_bytes() == b"JPG"
    finally:
        store.close()


def test_close_removes_only_its_owned_temporary_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("mine", encoding="utf-8")
    store = LocalAssistantAttachmentStore(
        inbox_root=tmp_path / "inbox",
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    attachment = _create(store)
    attachment_id, capability = _upload_coordinates(attachment)
    store.accept_upload(
        attachment_id,
        capability,
        b"%PDF",
        declared_content_length=4,
    )
    owned_root = store.temporary_root

    store.close()

    assert not owned_root.exists()
    assert outside.read_text(encoding="utf-8") == "mine"
    store.close()


def test_exact_duplicate_commit_reports_existing_corpus_source(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "lesson.pdf").write_bytes(b"%PDF")
    store = LocalAssistantAttachmentStore(
        inbox_root=inbox,
        upload_prefix="http://127.0.0.1:4321/uploads/",
    )
    try:
        attachment = _create(store)
        attachment_id, capability = _upload_coordinates(attachment)
        store.accept_upload(
            attachment_id,
            capability,
            b"%PDF",
            declared_content_length=4,
        )

        result = store.commit_attachment(attachment_id)

        assert result.path == inbox / "lesson.pdf"
        assert result.stored is False
    finally:
        store.close()
