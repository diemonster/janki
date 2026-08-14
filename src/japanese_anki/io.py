from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.errors import JankiError
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    FURIGANA_UNVERIFIED_KEY,
    PROVISIONAL_FIELDS_KEY,
    ModelError,
    VocabularyRecord,
    add_example_flags,
    example_accepted,
    flag_entries,
    join_provisional_entries,
    provisional_entries,
)


class DataError(JankiError):
    pass


def _user_home() -> Path:
    """Return the current user's home so tests can isolate cache fallbacks."""
    return Path.home()


def _owned_private_directory(path: Path) -> bool:
    """Whether ``path`` is a real directory private to the current Unix user."""
    try:
        details = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(details.st_mode):
        return False
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        return False
    return stat.S_IMODE(details.st_mode) & 0o077 == 0


def _path_lock_root() -> Path:
    """Choose a stable lock namespace another local user cannot own first."""
    temporary = Path(tempfile.gettempdir())
    if os.name == "nt" or _owned_private_directory(temporary):
        return temporary / "janki-file-locks"
    return _user_home() / ".cache" / "janki" / "file-locks"


@contextlib.contextmanager
def exclusive_path_lock(path: Path) -> Iterable[None]:
    """Serialize whole-file transactions on ``path`` across processes.

    Atomic rename keeps readers from seeing a partial file, but it cannot make
    a preceding compare-and-swap check atomic: two writers can both compare the
    old content before either one renames.  The lock lives in a private per-user
    temp or cache directory rather than beside the target, so protecting tracked
    files never creates an untracked artifact under ``data/``.
    """
    target = Path(os.path.realpath(path))
    lock_root = _path_lock_root()
    digest = hashlib.sha256(os.fsencode(target)).hexdigest()
    lock_path = lock_root / f"{digest}.lock"
    try:
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt" and not _owned_private_directory(lock_root):
            raise DataError(
                f"Could not lock {target}: lock directory {lock_root} is not a "
                "private directory owned by the current user"
            )
        handle = lock_path.open("a+b")
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not lock {target}: {exc.strerror or exc}") from exc
    with handle:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            import msvcrt

            try:
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            except OSError as exc:
                raise DataError(
                    f"Could not lock {target}: {exc.strerror or exc}"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                raise DataError(
                    f"Could not lock {target}: {exc.strerror or exc}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive_path_locks(paths: Iterable[Path]) -> Iterable[None]:
    """Lock several targets in one canonical order.

    A transaction that touches records, staging, an archive, and a journal must
    never choose a different lock order from another transaction. Real paths
    are deduplicated before sorting, so aliases cannot acquire the same lock
    twice or reverse two locks.
    """
    targets = sorted(
        {Path(os.path.realpath(path)) for path in paths},
        key=lambda path: os.fsencode(path),
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(exclusive_path_lock(target))
        yield


@dataclass(frozen=True, slots=True)
class RecordsRevision:
    """The exact records-file content a read-modify-write pass started from."""

    path: Path
    text: str | None


def records_revision(path: Path) -> RecordsRevision:
    """Capture ``path`` for a later stale-writer check, including absence."""
    target = Path(os.path.realpath(path))
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = None
    except OSError as exc:
        raise DataError(
            f"Could not read records file {target}: {exc.strerror or exc}"
        ) from exc
    return RecordsRevision(target, text)


def _target_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


def _fsync_directory(directory: Path) -> None:
    # Best effort: not every filesystem lets you open or fsync a directory.
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader never observes a partial file.

    The text goes to a uniquely named temporary file in the target's
    directory (concurrent writers cannot collide), is fsynced, then renamed
    over the target. A failure mid-write leaves the previous contents
    untouched. Symlinked targets are written through to their real file, and
    an existing target keeps its permission bits. Filesystem failures are
    reported as ``DataError``.
    """
    path = Path(os.path.realpath(path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, _target_mode(path))
            os.replace(temp_path, path)
        except BaseException:
            # Cleanup must never replace the real error with its own.
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
        _fsync_directory(path.parent)
    except OSError as exc:
        # exc may name the random temp file; report the target the caller asked for.
        raise DataError(f"Could not write {path}: {exc.strerror or exc}") from exc


def atomic_write_text_bound(
    path: Path,
    text: str,
    *,
    expected_revision: str | None = None,
    expected_identity: tuple[int, int] | None = None,
    expected_absent: bool = False,
) -> None:
    """Atomically replace the named path without following the target symlink.

    Safe repair operations bind a repository path before they show a plan. They
    must replace that name, not a symlink target introduced after the plan. A
    directory file descriptor keeps the temporary file and final replace bound
    to the same parent directory.
    """
    if expected_revision is not None and expected_absent:
        raise DataError("A bound write cannot expect content and absence together")
    target = Path(path).absolute()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(target.parent, directory_flags)
    except OSError as exc:
        raise DataError(f"Could not open target directory for {target}: {exc}") from exc
    temporary_name = ""
    try:
        try:
            details = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_umask = os.umask(0)
            os.umask(current_umask)
            mode = 0o666 & ~current_umask
        else:
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise DataError(f"Refusing to replace non-regular target {target}")
            mode = stat.S_IMODE(details.st_mode)
        descriptor = -1
        for _attempt in range(20):
            temporary_name = f".{target.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    mode,
                    dir_fd=directory_fd,
                )
                break
            except FileExistsError:
                continue
        if descriptor < 0:
            raise DataError(f"Could not allocate a temporary file for {target}")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            bound_state: tuple[int, int, int, int] | None = None
            try:
                current_fd = os.open(
                    target.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not expected_absent and (
                    expected_revision is not None or expected_identity is not None
                ):
                    raise DataError(
                        f"Bound target changed before replace: {target}"
                    ) from None
            else:
                try:
                    current_details = os.fstat(current_fd)
                    if expected_absent or not stat.S_ISREG(current_details.st_mode):
                        raise DataError(f"Bound target changed before replace: {target}")
                    bound_state = (
                        current_details.st_dev,
                        current_details.st_ino,
                        current_details.st_size,
                        current_details.st_mtime_ns,
                    )
                    if (
                        expected_identity is not None
                        and bound_state[:2] != expected_identity
                    ):
                        raise DataError(f"Bound target changed identity: {target}")
                    if expected_revision is not None:
                        digest = hashlib.sha256()
                        while current_chunk := os.read(current_fd, 1024 * 1024):
                            digest.update(current_chunk)
                        if digest.hexdigest() != expected_revision:
                            raise DataError(f"Bound target changed content: {target}")
                    after_hash = os.fstat(current_fd)
                    if (
                        after_hash.st_dev,
                        after_hash.st_ino,
                        after_hash.st_size,
                        after_hash.st_mtime_ns,
                    ) != bound_state:
                        raise DataError(f"Bound target changed content: {target}")
                finally:
                    os.close(current_fd)
            try:
                final_details = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if bound_state is not None:
                    raise DataError(
                        f"Bound target changed before replace: {target}"
                    ) from None
            else:
                if bound_state is None or (
                    final_details.st_dev,
                    final_details.st_ino,
                    final_details.st_size,
                    final_details.st_mtime_ns,
                ) != bound_state:
                    raise DataError(f"Bound target changed before replace: {target}")
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_name = ""
            os.fsync(directory_fd)
        finally:
            if temporary_name:
                with contextlib.suppress(OSError):
                    os.unlink(temporary_name, dir_fd=directory_fd)
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not write {target}: {exc.strerror or exc}") from exc
    finally:
        os.close(directory_fd)


def unlink_path_bound(
    path: Path,
    *,
    expected_revision: str,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Remove one exact regular file without following a changed path."""
    target = Path(path).absolute()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(target.parent, directory_flags)
    except OSError as exc:
        raise DataError(f"Could not open target directory for {target}: {exc}") from exc
    try:
        try:
            descriptor = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise DataError(f"Bound removal target changed: {target}") from exc
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise DataError(f"Bound removal target is not regular: {target}")
            if expected_identity is not None and (
                details.st_dev,
                details.st_ino,
            ) != expected_identity:
                raise DataError(f"Bound removal target changed identity: {target}")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_revision:
                raise DataError(f"Bound removal target changed content: {target}")
            after_hash = os.fstat(descriptor)
            if (
                after_hash.st_dev,
                after_hash.st_ino,
                after_hash.st_size,
                after_hash.st_mtime_ns,
            ) != (
                details.st_dev,
                details.st_ino,
                details.st_size,
                details.st_mtime_ns,
            ):
                raise DataError(f"Bound removal target changed content: {target}")
        finally:
            os.close(descriptor)
        try:
            final_details = os.stat(
                target.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise DataError(f"Bound removal target changed: {target}") from exc
        if (
            final_details.st_dev,
            final_details.st_ino,
            final_details.st_size,
            final_details.st_mtime_ns,
        ) != (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
        ):
            raise DataError(f"Bound removal target changed before removal: {target}")
        os.unlink(target.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not remove {target}: {exc.strerror or exc}") from exc
    finally:
        os.close(directory_fd)


def load_structured(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                return json.load(handle)
            if suffix in {".yaml", ".yml"}:
                return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DataError(f"Could not parse {path}: {exc}") from exc
    raise DataError(f"Unsupported file type for {path}; expected JSON or YAML")


def load_records(path: Path) -> list[VocabularyRecord]:
    data = load_structured(path)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise DataError(f"Expected a list of vocabulary records in {path}")
    for item in data:
        if not isinstance(item, dict):
            raise DataError(
                f"Each record in {path} must be a mapping, got {type(item).__name__}"
            )
    try:
        return [VocabularyRecord.from_dict(item) for item in data]
    except ModelError as exc:
        # The constructor knows the field; only this frame knows the file.
        raise DataError(f"Could not read a record in {path}: {exc}") from exc


def save_records_json(
    path: Path,
    records: list[VocabularyRecord],
    *,
    expected: RecordsRevision | None = None,
) -> None:
    """Atomically save records, refusing to overwrite a newer collection.

    ``expected`` comes from :func:`records_revision` immediately before the
    command reads the collection. It includes a missing file as real state, so
    two first-time writers cannot silently replace one another either.
    """
    target = Path(os.path.realpath(path))
    if expected is not None and target != expected.path:
        raise DataError(
            f"Records revision for {expected.path} cannot guard a write to {target}."
        )
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with exclusive_path_lock(target):
        if expected is not None:
            current = records_revision(target).text
            if current != expected.text:
                raise DataError(
                    f"Records file {target} changed on disk since it was read; saving "
                    "now would discard those changes. Another janki command may still "
                    "be writing it. Let that command finish, then re-run this one."
                )
        atomic_write_text(target, text)


MERGE_LABELS: tuple[str, ...] = ("added", "filled", "unchanged", "conflicting")

# Fields nobody may opt into incoming-wins behavior for: ``tags`` is always a
# union, ``source`` belongs to whoever saw the record first, and
# id/expression/reading are the record's identity (the id is derived from the
# latter two, so overwriting them would orphan Anki's GUIDs).
PREFER_INCOMING_PROTECTED: frozenset[str] = frozenset(
    {"id", "expression", "reading", "tags", "source"}
)

# Derived from the dataclass so schema additions (M2.2) merge automatically.
_RECORD_FIELDS: tuple[str, ...] = tuple(
    item.name for item in dataclasses.fields(VocabularyRecord)
)
# Everything that resolves field-by-field: id is identity, tags are unioned and
# source sticks, so those three are handled separately.
_CONTENT_FIELDS: tuple[str, ...] = tuple(
    name for name in _RECORD_FIELDS if name not in {"id", "tags", "source"}
)
MERGEABLE_FIELDS: tuple[str, ...] = tuple(
    name for name in _RECORD_FIELDS if name not in PREFER_INCOMING_PROTECTED
)

# ``source.raw_fields`` keys that annotate a *content* field rather than the
# record's origin, mapped to the field they describe. ``source`` otherwise
# belongs to whoever saw the record first, which is right for provenance and
# wrong for these: ``furigana_unverified`` says "nobody checked the segmentation
# of these examples", so if the examples travel and the key does not, the store
# ends up holding unchecked sentences with nothing saying so, and M5.3
# reads exactly these keys before it generates audio. Carried only when the
# merge actually wrote the field, because a flag describing examples that were
# not kept is a lie in the other direction. Every key here holds a
# comma-joined fingerprint list, which is what lets the merge union them.
# (``provisional_fields`` travels too, but per field name rather than as a
# blob, and ``example_authority`` travels unconditionally — a bound acceptance
# matching no sentence blesses nothing, while dropping it un-accepts a
# reviewer's stamp. See ``_carried_provisional`` and ``_carried_authority``.)
CONTENT_ANNOTATIONS: dict[str, str] = {
    FURIGANA_UNVERIFIED_KEY: "examples",
}

_EMPTY_CONTAINERS = (str, bytes, list, tuple, set, frozenset, dict)


def _copy_value(value: Any) -> Any:
    """Detach a value so merged records never alias the *incoming* records.

    A shallow copy is not a detachment: an ``examples`` list holds
    ``ExampleSentence`` instances and ``conjugations`` holds dict values, and
    copying only the outer container leaves the importer able to mutate what the
    merged record holds inside it. The import is the side that has to be
    detached: it is freshly constructed data the caller may still be walking.

    The existing side is deliberately *not* copied. An untouched record is
    carried through by reference and a merged one keeps every container the
    merge did not write to, so ``merged[i].examples`` may be the very list
    ``existing[i].examples`` is. That is a deliberate trade — one deepcopy per
    untouched record on every merge, for an aliasing no caller has — and the
    rule it buys is: **treat the records you passed in as spent**. The same
    applies to ``MergeOutcome.conflicts``, which reports live values from both
    sides rather than snapshots of them.
    """
    return copy.deepcopy(value)


def is_empty(value: Any) -> bool:
    """Empty means ``""``, ``[]``, ``{}`` or ``None`` — and nothing else.

    Public because the merge's notion of "a hole an import may fill" is a
    project-wide rule, not a private detail of this module: ``migrate.py``
    decides which inline fields to prefer with the same test, and a second
    copy of it would drift.

    Zero and ``False`` are *values*: a ``frequency_rank`` of 0 or an explicit
    false flag must never look like a hole an import can fill.
    """
    if value is None:
        return True
    if isinstance(value, _EMPTY_CONTAINERS):
        return len(value) == 0
    return False


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """What a merge did to one incoming record.

    ``label`` is one of ``MERGE_LABELS``, with precedence
    conflicting > filled > unchanged. ``filled_fields`` lists every field the
    merge wrote — empty fields filled from the import, a grown ``tags`` union,
    and ``prefer_incoming`` overrides. ``conflicts`` holds
    ``(field, existing, incoming)`` for fields where both sides were non-empty
    and different; those keep the existing value and are reported, never
    silently resolved.
    """

    label: str
    filled_fields: list[str] = field(default_factory=list)
    conflicts: list[tuple[str, Any, Any]] = field(default_factory=list)


def validate_prefer_incoming(fields: Iterable[str]) -> tuple[str, ...]:
    """Check field names destined for ``merge_records(prefer_incoming=...)``.

    Messages are worded for library callers; ``parse_prefer_incoming`` names
    the CLI flag for people who actually typed one.
    """
    names = tuple(fields)
    for name in names:
        if name in PREFER_INCOMING_PROTECTED:
            raise DataError(
                f"'{name}' cannot be preferred from the import: tags are always "
                "unioned, source stays with the first import, and "
                "id/expression/reading are the record's identity."
            )
        if name not in MERGEABLE_FIELDS:
            raise DataError(
                f"Unknown field '{name}'. Valid fields: {', '.join(MERGEABLE_FIELDS)}"
            )
    return names


def parse_prefer_incoming(value: str | None) -> tuple[str, ...]:
    """Parse a ``--prefer-incoming FIELD[,FIELD]`` option value."""
    if not value:
        return ()
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        # Separators only: the flag was passed but names nothing. Silently
        # merging as if it were absent would hide the typo.
        raise DataError(f"--prefer-incoming got no field names in {value!r}")
    try:
        return validate_prefer_incoming(names)
    except DataError as exc:
        raise DataError(f"--prefer-incoming: {exc}") from exc


def _carried_annotations(
    old: VocabularyRecord, new: VocabularyRecord, filled: Sequence[str]
) -> dict[str, str]:
    """The ``CONTENT_ANNOTATIONS`` the incoming record's fields brought with them.

    Values are unioned rather than replaced: every key in this table holds a
    comma-joined list of example fingerprints, and a record can collect flagged
    examples across several passes.
    """
    carried: dict[str, str] = {}
    for key, field_name in CONTENT_ANNOTATIONS.items():
        if field_name not in filled:
            continue
        items = flag_entries(old, key) + flag_entries(new, key)
        if items:
            carried[key] = ",".join(dict.fromkeys(items))
    return carried


def _carried_provisional(
    old: VocabularyRecord, new: VocabularyRecord, filled: Sequence[str]
) -> str | None:
    """The provisional marker the merged record should carry.

    ``None`` leaves the existing marker untouched; a string — possibly empty,
    meaning *remove the key* — rewrites it. Per field name, not as a blob: a
    mark travels with the *value* it is bound to. A filled field takes the
    incoming side's mark, or none at all when the incoming value is unmarked
    (curated) — the old mark bound the old value, and surviving the overwrite
    would make it a standing false claim about text the model never wrote.
    A field the merge did not write keeps the existing side's entry.
    """
    old_entries = provisional_entries(old)
    incoming = [
        (name, fingerprint)
        for name, fingerprint in provisional_entries(new)
        if name in filled
    ]
    kept = [(name, fp) for name, fp in old_entries if name not in filled]
    if kept == old_entries and not incoming:
        # Nothing the merge wrote touches any marked field: leave the
        # existing marker byte-for-byte alone (an unconditional rewrite would
        # also drop unknown-name entries the parser filters).
        return None
    return join_provisional_entries(kept + incoming)


def _carried_authority(old: VocabularyRecord, new: VocabularyRecord) -> str:
    """The acceptance fingerprints the merged record should carry, or empty.

    United *unconditionally*, unlike the hold flags: a bound acceptance
    fingerprint that matches no sentence on the record blesses nothing, so
    carrying it cannot lie — while conditioning on "the examples field was
    written" silently un-accepted a reviewer's stamp whenever the accepted
    text already matched the store (re-reviewing a legacy record fills
    nothing, and that is the documented remediation path).
    """
    items = flag_entries(old, EXAMPLE_AUTHORITY_KEY) + flag_entries(
        new, EXAMPLE_AUTHORITY_KEY
    )
    return ",".join(dict.fromkeys(items))


def _merge_one(
    old: VocabularyRecord, new: VocabularyRecord, prefer_incoming: frozenset[str]
) -> tuple[VocabularyRecord, MergeOutcome]:
    changes: dict[str, Any] = {}
    filled: list[str] = []
    conflicts: list[tuple[str, Any, Any]] = []

    for name in _CONTENT_FIELDS:
        old_value = getattr(old, name)
        new_value = getattr(new, name)
        if (
            name == "examples"
            and new.source.type == "extract"
            and old.source.type != "extract"
        ):
            # The mirror of the minting rule below: a machine-era sentence on
            # an extract row that no reviewer accepted must not fill a
            # *curated-type* record's hole, where the merged record's source
            # type would silently bless it as the user's own data — the exact
            # false-reviewed laundering M7.6T exists to prevent. Accepted
            # sentences pass; their stamp travels via the authority carry.
            # An extract-typed store record needs no filter: the sentences
            # land preserved-but-unaccepted there, exactly the documented
            # posture, and dropping them would silently lose a reviewer's
            # typed sentence between promote and the store.
            new_value = [
                example
                for example in new.examples
                if example_accepted(new, example)
            ]
        if is_empty(new_value) or old_value == new_value:
            continue
        if is_empty(old_value) or name in prefer_incoming:
            # Copy containers: the merged record must not alias the caller's
            # input, or a later mutation of the import silently edits the store.
            changes[name] = _copy_value(new_value)
            filled.append(name)
            continue
        conflicts.append((name, old_value, new_value))

    # Only a genuinely new tag is a write. When the union adds nothing, the
    # hand-edited list is left exactly as it is — order included — so a merge
    # the summary reports as "unchanged" really did leave the file unchanged.
    tags = sorted(set(old.tags) | set(new.tags))
    if set(tags) != set(old.tags):
        changes["tags"] = tags
        filled.append("tags")

    merged = replace(old, **changes) if changes else old
    annotations = _carried_annotations(old, new, filled)
    if (marker := _carried_provisional(old, new, filled)) is not None:
        annotations[PROVISIONAL_FIELDS_KEY] = marker
    if authority := _carried_authority(old, new):
        annotations[EXAMPLE_AUTHORITY_KEY] = authority
    removals = {key for key, value in annotations.items() if not value}
    if annotations:
        raw_fields = {
            key: value
            for key, value in {**merged.source.raw_fields, **annotations}.items()
            if key not in removals
        }
        if raw_fields != merged.source.raw_fields:
            merged = replace(
                merged, source=replace(merged.source, raw_fields=raw_fields)
            )
    if "examples" in filled and new.source.type != "extract":
        # An incoming curated source's examples that filled the hole are the
        # user's own data — curated by arrival, HARDENING.md's words — but the
        # merged record keeps its first-seen extract origin, which demands a
        # stamp nobody could type. The fill event itself is the provenance, so
        # it mints acceptance for exactly those sentences, through the same
        # writer every other acceptance goes through.
        merged = add_example_flags(
            merged,
            EXAMPLE_AUTHORITY_KEY,
            [example.japanese for example in new.examples if example.japanese],
        )
    if conflicts:
        label = "conflicting"
    elif filled:
        label = "filled"
    else:
        label = "unchanged"
    return merged, MergeOutcome(label=label, filled_fields=filled, conflicts=conflicts)


def _combine_outcomes(first: MergeOutcome, second: MergeOutcome) -> MergeOutcome:
    """Fold two passes over one id (an import carrying the same word twice)."""
    filled = first.filled_fields + [
        name for name in second.filled_fields if name not in first.filled_fields
    ]
    conflicts = first.conflicts + [
        item for item in second.conflicts if item not in first.conflicts
    ]
    if conflicts:
        label = "conflicting"
    elif first.label == "added":
        # The store gained a record no pre-existing curation was touched for;
        # a later row filling one of its holes does not turn that into
        # "filled", which would report zero additions while the count grew.
        label = "added"
    elif filled:
        label = "filled"
    else:
        label = "unchanged"
    return MergeOutcome(label=label, filled_fields=filled, conflicts=conflicts)


def merge_records(
    existing: list[VocabularyRecord],
    incoming: list[VocabularyRecord],
    prefer_incoming: Iterable[str] = (),
) -> tuple[list[VocabularyRecord], dict[str, MergeOutcome]]:
    """Merge an import into stored records without destroying curation.

    Existing wins: an incoming value lands only where the existing field is
    empty (``""``/``[]``/``{}``/``None`` — zero is a value, not a hole). Where
    both sides are non-empty and differ, the existing value is kept and the
    disagreement is reported as a conflict. ``tags`` is a sorted union;
    ``source`` is the existing record's, unconditionally — the first sighting
    sticks and later ones belong in the ledger. Fields named in
    ``prefer_incoming`` fall back to incoming-wins for deliberate refreshes.

    Returns the merged records sorted by id, plus an outcome map covering
    **exactly the incoming record ids** — existing records the import never
    mentioned are carried through untouched and do not appear. An id the
    import carries twice gets one folded outcome, so nothing it reported on
    the first row is lost.
    """
    prefer = frozenset(validate_prefer_incoming(prefer_incoming))
    by_id: dict[str, VocabularyRecord] = {}
    for record in existing:
        if record.id in by_id:
            # Keying by id would drop one of them — with its curation — and the
            # outcome map would never mention it. Refuse rather than choose.
            raise DataError(
                f"{record.id} appears more than once in the existing records; "
                "merging would silently discard one. Run 'janki validate' and "
                "resolve the duplicate first."
            )
        by_id[record.id] = record
    outcomes: dict[str, MergeOutcome] = {}

    for new in incoming:
        old = by_id.get(new.id)
        if old is None:
            # A deep copy, not `replace(new)`: a shallow dataclass copy shares
            # every container with the import, so a caller clearing its record
            # after the merge would silently edit the stored one.
            by_id[new.id] = copy.deepcopy(new)
            outcomes[new.id] = MergeOutcome(label="added")
            continue
        merged, outcome = _merge_one(old, new, prefer)
        by_id[new.id] = merged
        seen = outcomes.get(new.id)
        outcomes[new.id] = _combine_outcomes(seen, outcome) if seen else outcome

    return sorted(by_id.values(), key=lambda item: item.id), outcomes
