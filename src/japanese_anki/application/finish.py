"""Recover one exact post-promotion scope from its durable archive receipt.

W5 carries only the records that one promotion transaction actually landed.
The browser process is not that authority: it may restart after promotion, and
query-string counts do not identify records.  The done archive does.  This
service scans that fixed repository namespace, validates each receipt through
the promotion service that wrote it, and re-proves that the canonical records
still have the exact study-deck owners recorded at commit time.

Nothing here writes.  A later finish action re-resolves the receipt and compares
``FinishScope.fingerprint`` before acting, so a rendered scope cannot silently
widen when canonical records or deck ownership change.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from japanese_anki import staging
from japanese_anki import status as status_module
from japanese_anki.application.assignment import require_exact_deck_ownership
from japanese_anki.application.promotion import PromotionBatch, promotion_batches
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    RecordsRevision,
    exclusive_path_lock,
    load_records_snapshot,
    read_bytes_bound,
)

__all__ = [
    "FinishOwnerScope",
    "FinishReceipt",
    "FinishScope",
    "FinishScopeError",
    "list_finish_receipts",
    "records_revision_fingerprint",
    "resolve_finish_scope",
]


class FinishScopeError(JankiError):
    """A durable promotion receipt cannot be resolved to one exact finish job."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class FinishReceipt:
    """One validated durable handle discovered in a done archive."""

    source_file: str
    receipt_id: str
    record_count: int

    def __post_init__(self) -> None:
        if not self.source_file.strip():
            raise ValueError("A finish receipt needs a nonblank source file")
        if not _is_sha256(self.receipt_id):
            raise ValueError("A finish receipt must be lowercase SHA-256 text")
        if self.record_count < 1:
            raise ValueError("A finish receipt must name at least one record")


@dataclass(frozen=True, slots=True)
class FinishOwnerScope:
    """The receipted records whose commit-time owner was one study deck."""

    stem: str
    deck_path: Path
    record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.stem.strip():
            raise ValueError("A finish owner group needs a nonblank deck stem")
        if not self.deck_path.is_absolute():
            raise ValueError("A finish owner group needs an absolute deck path")
        if (
            not self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or any(not record_id for record_id in self.record_ids)
        ):
            raise ValueError("A finish owner group needs unique nonempty record ids")


@dataclass(frozen=True, slots=True)
class FinishScope:
    """One immutable, restart-safe finish scope and its rendered-state digest."""

    receipt_id: str
    source_file: str
    archive_path: Path
    archive_run_fingerprint: str
    archive_start_index: int
    review_run_id: str | None
    canonical_path: Path
    canonical_revision: str
    deck_configuration_revision: str
    record_ids: tuple[str, ...]
    #: Parallel to ``record_ids`` so the receipt's per-record owner binding is
    #: retained even after the convenience grouping below.
    owner_stems: tuple[str, ...]
    owner_groups: tuple[FinishOwnerScope, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.receipt_id):
            raise ValueError("A finish scope receipt must be lowercase SHA-256 text")
        if not self.source_file.strip():
            raise ValueError("A finish scope needs a nonblank source file")
        if not self.archive_path.is_absolute() or not self.canonical_path.is_absolute():
            raise ValueError("Finish scope repository paths must be absolute")
        if not _is_sha256(self.archive_run_fingerprint):
            raise ValueError("A finish archive run must be lowercase SHA-256 text")
        if self.archive_start_index < 0:
            raise ValueError("A finish archive start index cannot be negative")
        if not _is_sha256(self.canonical_revision):
            raise ValueError("A finish canonical revision must be lowercase SHA-256 text")
        if not _is_sha256(self.deck_configuration_revision):
            raise ValueError(
                "A finish deck-configuration revision must be lowercase SHA-256 text"
            )
        if not _is_sha256(self.fingerprint):
            raise ValueError("A finish scope fingerprint must be lowercase SHA-256 text")
        if (
            not self.record_ids
            or len(set(self.record_ids)) != len(self.record_ids)
            or any(not record_id for record_id in self.record_ids)
        ):
            raise ValueError("A finish scope needs unique nonempty record ids")
        if len(self.owner_stems) != len(self.record_ids) or any(
            not stem.strip() for stem in self.owner_stems
        ):
            raise ValueError("A finish scope needs one nonblank owner per record")

        expected: dict[str, list[str]] = {}
        for record_id, stem in zip(self.record_ids, self.owner_stems, strict=True):
            expected.setdefault(stem, []).append(record_id)
        if tuple(group.stem for group in self.owner_groups) != tuple(expected):
            raise ValueError("Finish owner groups must preserve first-owner order")
        for group in self.owner_groups:
            if group.record_ids != tuple(expected[group.stem]):
                raise ValueError(
                    "Finish owner groups must exactly partition the receipted ids"
                )


@dataclass(frozen=True, slots=True)
class _LocatedBatch:
    archive_path: Path
    batch: PromotionBatch


def records_revision_fingerprint(revision: RecordsRevision) -> str:
    """Digest one exact present-or-missing canonical records snapshot."""
    marker = b"missing\0" if revision.text is None else b"present\0"
    content = b"" if revision.text is None else revision.text.encode("utf-8")
    return hashlib.sha256(marker + content).hexdigest()


def _scope_fingerprint(scope: FinishScope) -> str:
    """Bind every durable and freshly re-proved claim a form may render."""
    payload = {
        "version": 1,
        "receipt_id": scope.receipt_id,
        "source_file": scope.source_file,
        "archive": {
            "path": str(scope.archive_path),
            "run_fingerprint": scope.archive_run_fingerprint,
            "start_index": scope.archive_start_index,
            "review_run_id": scope.review_run_id,
        },
        "canonical": {
            "path": str(scope.canonical_path),
            "revision": scope.canonical_revision,
        },
        "deck_configuration_revision": scope.deck_configuration_revision,
        "records": [
            {"id": record_id, "owner_stem": owner_stem}
            for record_id, owner_stem in zip(
                scope.record_ids, scope.owner_stems, strict=True
            )
        ],
        "owner_groups": [
            {
                "stem": group.stem,
                "deck_path": str(group.deck_path),
                "record_ids": list(group.record_ids),
            }
            for group in scope.owner_groups
        ],
    }
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _entry_state(details: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _read_bound_archive(directory_fd: int, name: str) -> str | None:
    """Capture one unchanged direct regular entry through its bound parent."""
    initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(initial.st_mode):
        return None
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FinishScopeError(
                f"[finish-archive-scan] done archive {name} is not a regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(named.st_mode) or not (
            _entry_state(initial)
            == _entry_state(opened)
            == _entry_state(after)
            == _entry_state(named)
        ):
            raise FinishScopeError(
                f"[finish-archive-scan] done archive {name} changed while read"
            )
        return b"".join(chunks).decode("utf-8")
    finally:
        os.close(descriptor)


def _done_archive_batches(config: ProjectConfig) -> tuple[_LocatedBatch, ...]:
    """Validate every direct supported archive within one bound namespace."""
    done = Path(os.path.abspath(config.staging_dir / "done"))
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(
        os, "O_DIRECTORY"
    ):
        raise FinishScopeError(
            f"[finish-archive-scan] this platform cannot safely bind {done}"
        )
    directory_fd = -1
    try:
        directory_fd = os.open(done, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        with os.scandir(directory_fd) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if Path(entry.name).suffix.lower() in staging.STAGING_SUFFIXES
            )
        located: list[_LocatedBatch] = []
        for name in names:
            path = done / name
            try:
                text = _read_bound_archive(directory_fd, name)
                if text is None:
                    continue
                archived, meta = staging.read_staging_text(text, source=str(path))
                located.extend(
                    _LocatedBatch(archive_path=path, batch=batch)
                    for batch in promotion_batches(
                        meta,
                        archived=archived,
                        archive_file=name,
                    )
                )
            except FinishScopeError:
                raise
            except (JankiError, UnicodeDecodeError, OSError) as exc:
                raise FinishScopeError(
                    f"[finish-archive-invalid] could not safely validate done "
                    f"archive {path}: {exc}"
                ) from exc
        return tuple(located)
    except FileNotFoundError:
        if directory_fd < 0:
            return ()
        raise FinishScopeError(
            f"[finish-archive-scan] done archive changed while read: {done}"
        ) from None
    except FinishScopeError:
        raise
    except (JankiError, UnicodeDecodeError, OSError) as exc:
        raise FinishScopeError(
            f"[finish-archive-invalid] could not safely validate done archive "
            f"under {done}: {exc}"
        ) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _validated_located_batches(config: ProjectConfig) -> tuple[_LocatedBatch, ...]:
    """Return every archive batch after rejecting duplicate durable handles."""
    located = _done_archive_batches(config)
    by_receipt: dict[str, Path] = {}
    for item in located:
        previous = by_receipt.get(item.batch.receipt_id)
        if previous is not None:
            raise FinishScopeError(
                f"[finish-receipt-ambiguous] promotion receipt "
                f"{item.batch.receipt_id} appears in more than one done archive: "
                f"{previous}, {item.archive_path}"
            )
        by_receipt[item.batch.receipt_id] = item.archive_path
    return located


def list_finish_receipts(config: ProjectConfig) -> tuple[FinishReceipt, ...]:
    """List every validated promotion receipt in deterministic archive order.

    Archive filenames are sorted by :func:`_done_archive_batches`; batches keep
    their durable append order within each archive.  Discovery deliberately
    does not inspect canonical records or current deck ownership.  Those are
    mutable inputs and are re-proved by :func:`resolve_finish_scope` when a
    receipt is opened or acted on.
    """
    return tuple(
        FinishReceipt(
            source_file=located.batch.source_file,
            receipt_id=located.batch.receipt_id,
            record_count=len(located.batch.promoted_ids),
        )
        for located in _validated_located_batches(config)
    )


def _locate_receipt(config: ProjectConfig, receipt_id: str) -> _LocatedBatch:
    for located in _validated_located_batches(config):
        if located.batch.receipt_id == receipt_id:
            return located
    raise FinishScopeError(
        f"[finish-receipt-not-found] no done archive contains promotion "
        f"receipt {receipt_id}"
    )


def _owner_groups(
    record_ids: tuple[str, ...],
    owner_stems: tuple[str, ...],
    deck_paths: dict[str, Path],
) -> tuple[FinishOwnerScope, ...]:
    grouped: dict[str, list[str]] = {}
    for record_id, stem in zip(record_ids, owner_stems, strict=True):
        grouped.setdefault(stem, []).append(record_id)
    return tuple(
        FinishOwnerScope(
            stem=stem,
            deck_path=deck_paths[stem],
            record_ids=tuple(ids),
        )
        for stem, ids in grouped.items()
    )


def _configured_deck_snapshot(
    config: ProjectConfig,
) -> tuple[tuple[Path, ...], str]:
    """Bind the configured deck-file set and exact bytes under its writer lock."""
    try:
        paths = tuple(path.resolve() for path in status_module.deck_files(config))
        digest = hashlib.sha256()
        for path in paths:
            wire = read_bytes_bound(path)
            encoded_path = str(path).encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(len(wire).to_bytes(8, "big"))
            digest.update(wire)
        current = tuple(path.resolve() for path in status_module.deck_files(config))
    except (JankiError, OSError) as exc:
        raise FinishScopeError(
            f"[finish-decks-unreadable] could not bind configured deck files: {exc}"
        ) from exc
    if current != paths:
        raise FinishScopeError(
            "[finish-decks-stale] the configured deck-file set changed while "
            "the finish scope was being read; reload it"
        )
    return paths, digest.hexdigest()


def resolve_finish_scope(config: ProjectConfig, receipt_id: str) -> FinishScope:
    """Resolve one exact durable receipt without writing or widening its IDs."""
    if not _is_sha256(receipt_id):
        raise FinishScopeError(
            "[finish-receipt-invalid] a finish receipt must be exactly lowercase "
            "SHA-256 text"
        )
    located = _locate_receipt(config, receipt_id)
    batch = located.batch
    record_ids = batch.promoted_ids
    owner_stems = tuple(batch.owner_stems[record_id] for record_id in record_ids)

    try:
        canonical, revision = load_records_snapshot(config.normalized_file)
    except JankiError as exc:
        raise FinishScopeError(
            f"[finish-canonical-unreadable] could not read the canonical "
            f"collection: {exc}"
        ) from exc
    counts = Counter(record.id for record in canonical)
    for record_id in record_ids:
        if counts[record_id] == 0:
            raise FinishScopeError(
                f"[finish-record-missing] receipted record {record_id!r} is "
                "missing from the canonical collection"
            )
        if counts[record_id] > 1:
            raise FinishScopeError(
                f"[finish-record-duplicate] receipted record {record_id!r} "
                "occurs more than once in the canonical collection"
            )
    by_id = {record.id: record for record in canonical}
    selected = tuple(by_id[record_id] for record_id in record_ids)

    with exclusive_path_lock(config.deck_dir):
        deck_paths_before, deck_revision_before = _configured_deck_snapshot(config)
        try:
            ownership = require_exact_deck_ownership(config, selected, canonical)
        except JankiError as exc:
            raise FinishScopeError(
                f"[finish-owner-unavailable] could not re-prove the receipted study "
                f"deck ownership: {exc}"
            ) from exc
        deck_paths_after, deck_revision_after = _configured_deck_snapshot(config)
        if (
            deck_paths_after != deck_paths_before
            or deck_revision_after != deck_revision_before
        ):
            raise FinishScopeError(
                "[finish-decks-stale] a configured deck changed while its "
                "ownership was being proved; reload the finish scope"
            )
    if tuple(item.record_id for item in ownership) != record_ids:
        raise FinishScopeError(
            "[finish-owner-invalid] deck ownership returned a different record scope"
        )
    for evaluation, stored_stem in zip(ownership, owner_stems, strict=True):
        current_stems = evaluation.owner_stems
        if current_stems != (stored_stem,):
            current = ", ".join(current_stems) or "none"
            raise FinishScopeError(
                f"[finish-owner-changed] {evaluation.record_id}'s study-deck "
                f"owner changed from {stored_stem!r} to {current!r}; reload only "
                "after resolving that deck change"
            )

    deck_paths: dict[str, Path] = {}
    for path in deck_paths_after:
        if path.stem in deck_paths:
            raise FinishScopeError(
                f"[finish-decks-ambiguous] more than one configured deck has "
                f"stem {path.stem!r}"
            )
        deck_paths[path.stem] = path.resolve()
    for stem in dict.fromkeys(owner_stems):
        if stem not in deck_paths:
            raise FinishScopeError(
                f"[finish-owner-missing] receipted owner deck {stem!r} is no "
                "longer configured"
            )

    canonical_path = Path(os.path.abspath(revision.path))
    groups = _owner_groups(record_ids, owner_stems, deck_paths)
    draft = FinishScope(
        receipt_id=batch.receipt_id,
        source_file=batch.source_file,
        archive_path=located.archive_path,
        archive_run_fingerprint=batch.archive_run_fingerprint,
        archive_start_index=batch.archive_start_index,
        review_run_id=batch.review_run_id,
        canonical_path=canonical_path,
        canonical_revision=records_revision_fingerprint(revision),
        deck_configuration_revision=deck_revision_after,
        record_ids=record_ids,
        owner_stems=owner_stems,
        owner_groups=groups,
        # Valid placeholder so dataclass invariants also cover construction;
        # replaced immediately with the complete digest below.
        fingerprint="0" * 64,
    )
    return replace(draft, fingerprint=_scope_fingerprint(draft))
