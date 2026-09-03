from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

from japanese_anki.errors import JankiError
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
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
    except FileNotFoundError:
        raise
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


@dataclass(frozen=True, slots=True)
class RecordsRevision:
    """The exact records-file content a read-modify-write pass started from."""

    path: Path
    text: str | None


def records_revision(path: Path) -> RecordsRevision:
    """Capture ``path`` for a later stale-writer check, including absence."""
    target = Path(path).absolute()
    try:
        text = read_text_bound(target)
    except FileNotFoundError:
        text = None
    except (DataError, OSError) as exc:
        raise DataError(
            f"Could not read records file {target}: "
            f"{getattr(exc, 'strerror', None) or exc}"
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


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically, preserving an existing file on failure.

    Audio is rewritten in place under an identity-addressed name. A direct
    write truncates the old, still-current clip before the new bytes are
    durable; a full disk can therefore turn a re-voice into a silent card.
    This uses the same temp/fsync/replace transaction as atomic text writes.
    """
    path = Path(os.path.realpath(path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, _target_mode(path))
            os.replace(temp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
        _fsync_directory(path.parent)
    except OSError as exc:
        raise DataError(f"Could not write {path}: {exc.strerror or exc}") from exc


_EntryState = tuple[int, int, int, int]
_CleanupEntryState = tuple[int, int, int, int, int]
_BoundEntryState = TypeVar(
    "_BoundEntryState", _EntryState, _CleanupEntryState
)


def _entry_state(details: os.stat_result) -> _EntryState:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    )


def _cleanup_entry_state(details: os.stat_result) -> _CleanupEntryState:
    """Identity persisted before deletion, including inode-reuse evidence.

    Live CAS state deliberately excludes ctime: link, rename, and exchange
    change it as part of a legitimate transaction. Cleanup binds an already
    observed entry and therefore can use ctime to distinguish a later inode
    reuse that recreates the same dev/inode/size/mtime/content tuple.
    """
    return (*_entry_state(details), details.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class _DirectoryBinding:
    path: Path
    descriptor: int
    steps: tuple[tuple[int, str, tuple[int, int]], ...]


def _directory_identity(details: os.stat_result) -> tuple[int, int]:
    return details.st_dev, details.st_ino


def _validate_bound_directory(binding: _DirectoryBinding) -> None:
    """Prove every lexical directory entry still names the opened chain."""
    try:
        for parent_fd, component, identity in binding.steps:
            current = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or (
                _directory_identity(current) != identity
            ):
                raise DataError(
                    f"Bound target directory changed: {binding.path}"
                )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Bound target directory changed: {binding.path}: {exc}") from exc


def _canonical_macos_root_alias(directory: Path) -> Path:
    """Use the real root for macOS's fixed ``/var`` and ``/tmp`` aliases.

    Python may spell its temporary directory through ``/var`` even though that
    root entry is the system-owned ``/var -> private/var`` symlink.  The bound
    writer must still reject arbitrary symlinked ancestors, so only these two
    fixed aliases are accepted, after proving that the alias currently reaches
    the expected real directory.  The returned path is then opened component by
    component without following links as usual.
    """

    if sys.platform != "darwin" or len(directory.parts) < 2:
        return directory
    name = directory.parts[1]
    if name not in {"tmp", "var"}:
        return directory
    alias = Path("/") / name
    real = Path("/private") / name
    try:
        alias_entry = os.lstat(alias)
        alias_target = os.readlink(alias)
        followed = os.stat(alias)
        real_entry = os.stat(real, follow_symlinks=False)
    except OSError:
        return directory
    if (
        not stat.S_ISLNK(alias_entry.st_mode)
        or alias_target != f"private/{name}"
        or not stat.S_ISDIR(real_entry.st_mode)
        or _directory_identity(followed) != _directory_identity(real_entry)
    ):
        return directory
    return real.joinpath(*directory.parts[2:])


@contextlib.contextmanager
def _open_bound_directory(
    path: Path,
    *,
    create: bool,
) -> Iterator[_DirectoryBinding]:
    """Open a lexical directory one no-follow component at a time."""
    directory = _canonical_macos_root_alias(Path(path).absolute())
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise DataError(
            f"This platform cannot safely bind target directory {directory}"
        )
    if directory.anchor != "/":  # pragma: no cover - non-POSIX path shape
        raise DataError(f"Refusing unsupported target directory {directory}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        with contextlib.ExitStack() as stack:
            current_fd = os.open("/", flags)
            stack.callback(os.close, current_fd)
            steps: list[tuple[int, str, tuple[int, int]]] = []
            for component in directory.parts[1:]:
                try:
                    child_fd = os.open(component, flags, dir_fd=current_fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(component, dir_fd=current_fd)
                    child_fd = os.open(component, flags, dir_fd=current_fd)
                stack.callback(os.close, child_fd)
                opened = os.fstat(child_fd)
                entry = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(opened.st_mode) or (
                    _directory_identity(opened) != _directory_identity(entry)
                ):
                    raise DataError(
                        f"Refusing replaced target directory {directory}"
                    )
                steps.append(
                    (current_fd, component, _directory_identity(opened))
                )
                current_fd = child_fd
            binding = _DirectoryBinding(directory, current_fd, tuple(steps))
            _validate_bound_directory(binding)
            yield binding
    except FileNotFoundError:
        raise
    except DataError:
        raise
    except (NotImplementedError, OSError) as exc:
        raise DataError(
            f"Could not safely open target directory {directory}: "
            f"{getattr(exc, 'strerror', None) or exc}"
        ) from exc


@contextlib.contextmanager
def _open_cleanup_directory(
    path: Path,
    expected_identity: tuple[int, int],
) -> Iterator[_DirectoryBinding | None]:
    """Open the one directory a durable cleanup decision actually bound.

    A committed journal can be checked out on another filesystem, where the
    lexical ``.pending`` path is recreated with a different inode. Missing,
    symlinked, non-directory, and stably opened replacement namespaces cannot
    contain the old bound entries under this authority: preserve them, inspect
    no child names, and report the old namespace as unreachable. Other probe
    failures remain errors because they do not prove replacement.
    """
    directory = Path(path).absolute()
    try:
        observed = os.lstat(directory)
    except FileNotFoundError:
        yield None
        return
    except OSError as exc:
        raise DataError(
            f"Could not inspect cleanup directory {directory}: "
            f"{exc.strerror or exc}"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode):
        # No-follow inspection makes a symlink a replacement too. Never open
        # it and therefore never touch the directory it names.
        yield None
        return
    try:
        with _open_bound_directory(directory, create=False) as binding:
            parent = os.fstat(binding.descriptor)
            if _directory_identity(parent) != expected_identity:
                # The opened chain is stable, but it is not the namespace the
                # journal bound. Do not enumerate it or adopt copied names.
                _validate_bound_directory(binding)
                yield None
                return
            yield binding
    except FileNotFoundError as exc:
        # It existed at the no-follow probe and vanished while binding. That
        # is a race, not a stable proof that the old namespace is unreachable.
        raise DataError(
            f"Cleanup directory changed while it was being bound: {directory}"
        ) from exc


def _link_entry_exclusive(directory_fd: int, source: str, target: str) -> None:
    """Publish one same-directory hard link without replacing ``target``."""
    os.link(
        source,
        target,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
        follow_symlinks=False,
    )


def prepare_bound_directory(path: Path) -> Path:
    """Create, bind, and exercise the publication used by bound writers."""
    probe_cache = Path(os.path.realpath(_path_lock_root())) / "write-probes"
    try:
        probe_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise DataError(
            f"Could not prepare private write-probe store {probe_cache}: {exc}"
        ) from exc
    if not _owned_private_directory(probe_cache):
        raise DataError(
            f"Write-probe store is not private to this user: {probe_cache}"
        )
    with (
        _open_bound_directory(path, create=True) as binding,
        _open_bound_directory(probe_cache, create=False) as quarantine,
    ):
        if os.fstat(binding.descriptor).st_dev != os.fstat(
            quarantine.descriptor
        ).st_dev:
            raise DataError(
                f"Could not safely retire write probes from {binding.path}: "
                "the private probe store is on another filesystem"
            )
        probe_name = ""
        published_name = ""
        probe_state: _EntryState | None = None
        descriptor = -1

        def retire_probe(name: str) -> bool:
            if not name or probe_state is None:
                return False
            for _attempt in range(20):
                retired = f"probe-{secrets.token_hex(16)}.retired"
                try:
                    _rename_entry_exclusive_between(
                        binding.descriptor,
                        name,
                        quarantine.descriptor,
                        retired,
                    )
                except FileExistsError:
                    continue
                except OSError:
                    return False
                try:
                    moved = os.stat(
                        retired,
                        dir_fd=quarantine.descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    os.fsync(quarantine.descriptor)
                    _validate_bound_directory(quarantine)
                    return True
                except OSError:
                    return False
                if stat.S_ISREG(moved.st_mode) and _entry_state(moved) == probe_state:
                    try:
                        os.unlink(retired, dir_fd=quarantine.descriptor)
                        os.fsync(quarantine.descriptor)
                        _validate_bound_directory(quarantine)
                        os.stat(
                            retired,
                            dir_fd=quarantine.descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        return True
                    except (DataError, OSError):
                        return False
                    return False
                # A raced-in entry was moved, not our probe. Restore it only
                # while its original name is still absent; otherwise retain
                # both names rather than overwrite either one.
                with contextlib.suppress(OSError):
                    _rename_entry_exclusive_between(
                        quarantine.descriptor,
                        retired,
                        binding.descriptor,
                        name,
                    )
                return False
            return False

        try:
            for _attempt in range(20):
                probe_name = f".janki-write-probe.{secrets.token_hex(8)}.tmp"
                try:
                    descriptor = os.open(
                        probe_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=binding.descriptor,
                    )
                    break
                except FileExistsError:
                    continue
            if descriptor < 0:
                raise DataError(
                    f"Could not allocate a write probe in {binding.path}"
                )
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise DataError(
                        f"Write probe is not regular in {binding.path}"
                    )
                # Capture an identity immediately, so even a write or fsync
                # failure leaves enough authority to remove only our probe.
                probe_state = _entry_state(opened)
                if os.write(descriptor, b"\0") != 1:
                    raise OSError("Could not write the complete directory probe")
                probe_state = _entry_state(os.fstat(descriptor))
                os.fsync(descriptor)
                probe_state = _entry_state(os.fstat(descriptor))
            except BaseException:
                raise
            named = os.stat(
                probe_name,
                dir_fd=binding.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(named.st_mode)
                or _entry_state(named) != probe_state
            ):
                raise DataError(
                    f"Write probe changed in bound directory {binding.path}"
                )
            # Recovery answers use hard-link publication so an existing
            # operation artifact can never be overwritten. Exercise that
            # exact filesystem capability before a provider may be called.
            for _attempt in range(20):
                published_name = (
                    f".janki-write-probe.{secrets.token_hex(8)}.publish"
                )
                try:
                    _link_entry_exclusive(
                        binding.descriptor, probe_name, published_name
                    )
                    break
                except FileExistsError:
                    continue
            else:
                raise DataError(
                    f"Could not allocate a publication probe in {binding.path}"
                )
            original = os.stat(
                probe_name,
                dir_fd=binding.descriptor,
                follow_symlinks=False,
            )
            published = os.stat(
                published_name,
                dir_fd=binding.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(original.st_mode)
                or not stat.S_ISREG(published.st_mode)
                or _entry_state(original) != probe_state
                or _entry_state(published) != probe_state
            ):
                raise DataError(
                    f"Publication probe changed in bound directory {binding.path}"
                )
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
            if not retire_probe(published_name):
                raise DataError(
                    f"Could not safely retire publication probe in {binding.path}"
                )
            published_name = ""
            if not retire_probe(probe_name):
                raise DataError(
                    f"Could not safely retire write probe in {binding.path}"
                )
            probe_name = ""
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
            _validate_bound_directory(quarantine)
            return binding.path
        finally:
            for leftover in (published_name, probe_name):
                with contextlib.suppress(OSError):
                    retire_probe(leftover)
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.ftruncate(descriptor, 0)
                    os.fsync(descriptor)
                os.close(descriptor)


def _rename_entries_with_flags(
    source_directory_fd: int,
    left: str,
    destination_directory_fd: int,
    right: str,
    *,
    darwin_flags: int,
    linux_flags: int,
) -> None:
    """Run one flagged descriptor-bound rename, or fail closed."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:  # pragma: no cover - supported Darwin exports it
            raise OSError(errno.ENOTSUP, "renameatx_np is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_directory_fd,
            os.fsencode(left),
            destination_directory_fd,
            os.fsencode(right),
            darwin_flags,
        )
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:  # pragma: no cover - libc/kernel dependent
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_directory_fd,
            os.fsencode(left),
            destination_directory_fd,
            os.fsencode(right),
            linux_flags,
        )
    else:  # pragma: no cover - workbench deployment is macOS
        raise OSError(errno.ENOTSUP, "flagged atomic rename is unavailable")
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def _exchange_entries(directory_fd: int, left: str, right: str) -> None:
    """Atomically exchange two names, or fail closed on an unsupported host."""
    _rename_entries_with_flags(
        directory_fd,
        left,
        directory_fd,
        right,
        darwin_flags=2,  # RENAME_SWAP
        linux_flags=2,  # RENAME_EXCHANGE
    )


def _rename_entry_exclusive(directory_fd: int, left: str, right: str) -> None:
    """Atomically move ``left`` only while ``right`` remains absent."""
    _rename_entry_exclusive_between(
        directory_fd,
        left,
        directory_fd,
        right,
    )


def _rename_entry_exclusive_between(
    source_directory_fd: int,
    left: str,
    destination_directory_fd: int,
    right: str,
) -> None:
    """No-clobber move between two bound directories on one filesystem."""
    _rename_entries_with_flags(
        source_directory_fd,
        left,
        destination_directory_fd,
        right,
        darwin_flags=4,  # RENAME_EXCL
        linux_flags=1,  # RENAME_NOREPLACE
    )


def _read_bound_entry_with_state(
    directory_fd: int,
    name: str,
    state_of: Callable[[os.stat_result], _BoundEntryState],
) -> tuple[_BoundEntryState, str]:
    """Read one direct regular entry and prove its name stayed on that inode."""
    initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(initial.st_mode):
        raise DataError(f"Refusing non-regular bound target {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DataError(f"Refusing non-regular bound target {name}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or state_of(initial) != state_of(before)
            or state_of(before) != state_of(after)
            or state_of(before) != state_of(entry)
        ):
            raise DataError(f"Bound target changed content: {name}")
        return state_of(before), digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_bound_entry(
    directory_fd: int,
    name: str,
) -> tuple[_EntryState, str]:
    return _read_bound_entry_with_state(directory_fd, name, _entry_state)


def _read_cleanup_bound_entry(
    directory_fd: int,
    name: str,
) -> tuple[_CleanupEntryState, str]:
    return _read_bound_entry_with_state(
        directory_fd, name, _cleanup_entry_state
    )


_CAS_PREPARED_SUFFIX = ".janki-cas.prepared"
_CAS_VALIDATED_SUFFIX = ".janki-cas.validated"
_CAS_RECOVERED_SUFFIX = ".janki-cas.recovered"
_CAS_DRAFT_SUFFIX = ".janki-cas.draft"


@dataclass(frozen=True, slots=True)
class _BoundWriteEvidence:
    """One exact write-once transaction that still needs a decision.

    The marker binds the generated digest and the private name.  The entry
    snapshots bind the names seen by a journal decision so a later cleanup can
    remove only those exact entries, never replacements raced in afterward.
    """

    target: Path
    directory_identity: tuple[int, int]
    marker_name: str
    marker_state: _CleanupEntryState
    marker_revision: str
    temporary_name: str
    temporary_identity: tuple[int, int] | None
    temporary_snapshot: tuple[_CleanupEntryState, str] | None
    temporary_retirement_snapshot: tuple[_CleanupEntryState, str] | None
    temporary_owned: bool
    public_snapshot: tuple[_CleanupEntryState, str] | None
    generated_revision: str

    @property
    def private_reply_complete(self) -> bool:
        return (
            self.temporary_owned
            and self.temporary_identity is not None
            and self.temporary_snapshot is not None
            and self.temporary_snapshot[0][:2] == self.temporary_identity
            and self.temporary_snapshot[1] == self.generated_revision
        )

    @property
    def public_reply_complete(self) -> bool:
        return (
            self.marker_name.endswith(_CAS_VALIDATED_SUFFIX)
            and self.temporary_identity is not None
            and self.public_snapshot is not None
            and self.public_snapshot[0][:2] == self.temporary_identity
            and self.public_snapshot[1] == self.generated_revision
        )

    @property
    def reply_complete(self) -> bool:
        """Whether a marker-bound name holds the whole exact answer."""
        return self.private_reply_complete or self.public_reply_complete

    @property
    def public_answer_snapshot(
        self,
    ) -> tuple[_CleanupEntryState, str] | None:
        """The public hard link, only when the private answer proves it."""
        if (
            self.private_reply_complete
            and self.public_snapshot == self.temporary_snapshot
        ):
            return self.public_snapshot
        if self.public_reply_complete and self.temporary_snapshot is None:
            return self.public_snapshot
        return None

    def to_cleanup_dict(self) -> dict[str, Any]:
        """JSON-safe exact bindings for a durable journal cleanup intent."""

        def snapshot(
            value: tuple[_CleanupEntryState, str] | None,
        ) -> dict[str, Any] | None:
            if value is None:
                return None
            return {"state": list(value[0]), "sha256": value[1]}

        return {
            "directory_identity": list(self.directory_identity),
            "marker_name": self.marker_name,
            "marker_state": list(self.marker_state),
            "marker_sha256": self.marker_revision,
            "temporary_name": self.temporary_name,
            "temporary_identity": (
                list(self.temporary_identity)
                if self.temporary_identity is not None
                else None
            ),
            "temporary_snapshot": snapshot(self.temporary_snapshot),
            "temporary_retirement_snapshot": snapshot(
                self.temporary_retirement_snapshot
            ),
            "temporary_owned": self.temporary_owned,
            "public_snapshot": snapshot(self.public_answer_snapshot),
            "generated_sha256": self.generated_revision,
        }

    @classmethod
    def from_cleanup_dict(
        cls,
        target: Path,
        raw: Mapping[str, Any],
    ) -> _BoundWriteEvidence:
        """Rebuild only a strictly scoped binding written by ``to_cleanup_dict``.

        The journal is committed data and can be hand-edited or merged.  A
        cleanup record therefore grants authority only over the fixed target
        supplied by its operation id and direct private names that prove they
        belong to that target's bound-write namespace.
        """
        required = {
            "directory_identity",
            "marker_name",
            "marker_state",
            "marker_sha256",
            "temporary_name",
            "temporary_identity",
            "temporary_snapshot",
            "temporary_retirement_snapshot",
            "temporary_owned",
            "public_snapshot",
            "generated_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise DataError("Bound-write cleanup intent is invalid")

        def identity(value: Any, *, optional: bool = False) -> tuple[int, int] | None:
            if optional and value is None:
                return None
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                )
            ):
                raise DataError("Bound-write cleanup identity is invalid")
            return value[0], value[1]

        def entry_state(value: Any) -> _CleanupEntryState:
            if (
                not isinstance(value, list)
                or len(value) != 5
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                )
            ):
                raise DataError("Bound-write cleanup entry state is invalid")
            return value[0], value[1], value[2], value[3], value[4]

        def digest(value: Any, *, allow_empty: bool = False) -> str:
            if allow_empty and value == "":
                return ""
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DataError("Bound-write cleanup digest is invalid")
            return value

        def snapshot(
            value: Any,
        ) -> tuple[_CleanupEntryState, str] | None:
            if value is None:
                return None
            if not isinstance(value, Mapping) or set(value) != {"state", "sha256"}:
                raise DataError("Bound-write cleanup snapshot is invalid")
            return entry_state(value["state"]), digest(value["sha256"])

        marker_name = raw["marker_name"]
        temporary_name = raw["temporary_name"]
        try:
            encoded_marker = os.fsencode(marker_name)
            encoded_temporary = os.fsencode(temporary_name)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise DataError("Bound-write cleanup name is invalid") from exc
        marker_suffixes = (
            _CAS_DRAFT_SUFFIX,
            _CAS_PREPARED_SUFFIX,
            _CAS_VALIDATED_SUFFIX,
            _CAS_RECOVERED_SUFFIX,
        )
        if (
            not isinstance(marker_name, str)
            or not encoded_marker
            or b"\0" in encoded_marker
            or Path(marker_name).name != marker_name
            or "/" in marker_name
            or "\\" in marker_name
            or not marker_name.startswith(f".{target.name}.")
            or not marker_name.endswith(marker_suffixes)
            or not isinstance(temporary_name, str)
            or b"\0" in encoded_temporary
            or (temporary_name and Path(temporary_name).name != temporary_name)
            or "/" in temporary_name
            or "\\" in temporary_name
        ):
            raise DataError("Bound-write cleanup name is invalid")

        temporary_identity = identity(raw["temporary_identity"], optional=True)
        temporary_owned = raw["temporary_owned"]
        temporary_snapshot = snapshot(raw["temporary_snapshot"])
        temporary_retirement_snapshot = snapshot(
            raw["temporary_retirement_snapshot"]
        )
        if not isinstance(temporary_owned, bool) or (
            not temporary_owned and temporary_snapshot is not None
        ):
            raise DataError("Bound-write cleanup ownership is invalid")
        marker_owned = _marker_owned_temporary(marker_name, target.name)
        if not temporary_name and temporary_identity is None:
            if (
                temporary_snapshot is not None
                or temporary_retirement_snapshot is not None
                or temporary_owned
            ):
                raise DataError("Bound-write cleanup private binding is invalid")
        elif marker_owned != (temporary_name, temporary_identity):
            raise DataError("Bound-write cleanup private binding is invalid")
        if temporary_owned and (
            temporary_identity is None
            or not temporary_name
            or (
                temporary_snapshot is not None
                and temporary_snapshot[0][:2] != temporary_identity
            )
        ):
            raise DataError("Bound-write cleanup ownership is invalid")
        if temporary_retirement_snapshot is not None and (
            temporary_identity is None
            or not temporary_name
            or temporary_retirement_snapshot[0][:2] != temporary_identity
        ):
            raise DataError("Bound-write cleanup retirement binding is invalid")

        directory_identity = identity(raw["directory_identity"])
        assert directory_identity is not None
        marker_state = entry_state(raw["marker_state"])
        marker_revision = digest(raw["marker_sha256"])
        generated_revision = digest(
            raw["generated_sha256"], allow_empty=True
        )
        public_snapshot = snapshot(raw["public_snapshot"])
        if marker_state[0] != directory_identity[0]:
            raise DataError("Bound-write cleanup directory binding is invalid")
        for bound_snapshot in (
            temporary_snapshot,
            temporary_retirement_snapshot,
            public_snapshot,
        ):
            if (
                bound_snapshot is not None
                and bound_snapshot[0][0] != directory_identity[0]
            ):
                raise DataError("Bound-write cleanup directory binding is invalid")
        if temporary_name and not generated_revision:
            raise DataError("Bound-write cleanup generated digest is invalid")
        if (
            temporary_retirement_snapshot is not None
            and temporary_retirement_snapshot[1] != generated_revision
        ):
            raise DataError("Bound-write cleanup retirement binding is invalid")
        if public_snapshot is not None and (
            temporary_identity is None
            or public_snapshot[0][:2] != temporary_identity
            or public_snapshot[1] != generated_revision
            or (
                temporary_snapshot is not None
                and public_snapshot != temporary_snapshot
            )
        ):
            raise DataError("Bound-write cleanup public binding is invalid")
        return cls(
            target=Path(target).absolute(),
            directory_identity=directory_identity,
            marker_name=marker_name,
            marker_state=marker_state,
            marker_revision=marker_revision,
            temporary_name=temporary_name,
            temporary_identity=temporary_identity,
            temporary_snapshot=temporary_snapshot,
            temporary_retirement_snapshot=temporary_retirement_snapshot,
            temporary_owned=temporary_owned,
            public_snapshot=public_snapshot,
            generated_revision=generated_revision,
        )


@dataclass(frozen=True, slots=True)
class _BoundFileReceipt:
    """Exact public-file and terminal-marker proof returned by a capture.

    Unlike a path, this remains meaningful after the writer's WAL has been
    finalized: both the directory and the direct entry have to retain their
    captured identities and bytes.  The five-field state is taken only after
    the private hard link is retired, because removing that link changes the
    public inode's ctime.
    """

    target: Path
    directory_identity: tuple[int, int]
    entry_state: _CleanupEntryState
    content_sha256: str
    marker_name: str
    marker_state: _CleanupEntryState
    marker_revision: str


def _bound_capture_evidence(
    expected: _BoundFileReceipt,
) -> _BoundWriteEvidence:
    """Materialize marker-only cleanup authority from a validated receipt."""
    temporary_name = (
        expected.marker_name[: -len(_CAS_VALIDATED_SUFFIX)] + ".tmp"
    )
    return _BoundWriteEvidence(
        target=expected.target,
        directory_identity=expected.directory_identity,
        marker_name=expected.marker_name,
        marker_state=expected.marker_state,
        marker_revision=expected.marker_revision,
        temporary_name=temporary_name,
        temporary_identity=expected.entry_state[:2],
        temporary_snapshot=None,
        temporary_retirement_snapshot=None,
        temporary_owned=False,
        public_snapshot=None,
        generated_revision=expected.content_sha256,
    )


def _cas_lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.janki-cas-lock")


def _has_active_cas_marker(
    binding: _DirectoryBinding,
    target_name: str,
) -> bool:
    """Whether recovery evidence may still refer to this target's temp."""
    prefix = f".{target_name}."
    suffixes = (
        _CAS_PREPARED_SUFFIX,
        _CAS_VALIDATED_SUFFIX,
        _CAS_RECOVERED_SUFFIX,
        _CAS_DRAFT_SUFFIX,
    )
    try:
        return any(
            name.startswith(prefix) and name.endswith(suffixes)
            for name in os.listdir(binding.descriptor)
        )
    except OSError:
        # Cleanup is the destructive branch. If the directory cannot be
        # inspected, retain the temp rather than guessing that no WAL owns it.
        return True


def _read_bound_bytes_with_state(
    directory_fd: int,
    name: str,
    state_of: Callable[[os.stat_result], _BoundEntryState],
) -> tuple[_BoundEntryState, str, bytes]:
    initial = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(initial.st_mode):
        raise DataError(f"Refusing non-regular bound target {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DataError(f"Refusing non-regular bound target {name}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or state_of(initial) != state_of(before)
            or state_of(before) != state_of(after)
            or state_of(before) != state_of(named)
        ):
            raise DataError(f"Bound target changed content: {name}")
        return state_of(before), digest.hexdigest(), b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_bound_bytes(
    directory_fd: int, name: str
) -> tuple[_EntryState, str, bytes]:
    return _read_bound_bytes_with_state(directory_fd, name, _entry_state)


def _read_cleanup_bound_bytes(
    directory_fd: int, name: str
) -> tuple[_CleanupEntryState, str, bytes]:
    return _read_bound_bytes_with_state(
        directory_fd, name, _cleanup_entry_state
    )


def _bound_snapshot(
    directory_fd: int, name: str
) -> tuple[_EntryState, str] | None:
    try:
        return _read_bound_entry(directory_fd, name)
    except FileNotFoundError:
        return None


def _cleanup_bound_snapshot(
    directory_fd: int, name: str
) -> tuple[_CleanupEntryState, str] | None:
    """Read a destructive binding, treating non-regular names as replacements.

    Cleanup authority is always captured from a regular file.  A later
    symlink, FIFO, socket, or directory at that lexical name cannot be that
    entry and must neither be opened nor keep the old decision pending.  Other
    failures against a still-regular name remain real cleanup failures.
    """
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode):
            return None
        return _read_cleanup_bound_entry(directory_fd, name)
    except FileNotFoundError:
        return None
    except (DataError, OSError):
        try:
            current = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(current.st_mode):
            return None
        raise


def _cleanup_bound_entry_state(
    directory_fd: int,
    name: str,
) -> _CleanupEntryState | None:
    """Lstat one regular cleanup name without opening or reading its bytes."""
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(observed.st_mode):
        return None
    return _cleanup_entry_state(observed)


def _owned_temporary_identity(
    name: str,
    target_name: str,
) -> tuple[int, int] | None:
    """Identity persisted in a janki-private temp name, if it is well formed."""
    prefix = f".{target_name}."
    suffix = ".tmp"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    body = name[len(prefix) : -len(suffix)]
    try:
        token, identity = body.split(".", 1)
        device, inode = identity.split("-", 1)
    except ValueError:
        return None
    if (
        len(token) != 16
        or any(character not in "0123456789abcdef" for character in token)
        or not device
        or not inode
        or any(character not in "0123456789abcdef" for character in device)
        or any(character not in "0123456789abcdef" for character in inode)
    ):
        return None
    return int(device, 16), int(inode, 16)


def _marker_owned_temporary(
    marker: str,
    target_name: str,
) -> tuple[str, tuple[int, int]] | None:
    for suffix in (
        _CAS_DRAFT_SUFFIX,
        _CAS_PREPARED_SUFFIX,
        _CAS_VALIDATED_SUFFIX,
        _CAS_RECOVERED_SUFFIX,
    ):
        if marker.endswith(suffix):
            temporary = marker[: -len(suffix)] + ".tmp"
            identity = _owned_temporary_identity(temporary, target_name)
            return (temporary, identity) if identity is not None else None
    return None


def _require_marker_owns_transaction(
    marker: str,
    target_name: str,
    transaction: Mapping[str, Any],
) -> None:
    """Refuse a body copied under another identity-bearing marker name."""
    owned = _marker_owned_temporary(marker, target_name)
    declared = (
        transaction["temporary"],
        transaction["temporary_identity"],
    )
    if owned != declared:
        raise DataError(
            f"Incomplete bound-write marker disagrees with its bound name: {marker}"
        )


def _create_cas_marker(
    binding: _DirectoryBinding,
    target_name: str,
    temporary_name: str,
    *,
    temporary_identity: tuple[int, int],
    expected_absent: bool,
    expected_state: _EntryState | None,
    expected_revision: str | None,
    generated_state: _EntryState | None,
    generated_revision: str,
    retirement_snapshot_required: bool = False,
) -> tuple[str, _EntryState, str]:
    payload = json.dumps(
        {
            "version": 3,
            "target": target_name,
            "temporary": temporary_name,
            "temporary_identity": list(temporary_identity),
            "expected_absent": expected_absent,
            "expected_state": (
                list(expected_state) if expected_state is not None else None
            ),
            "expected_sha256": expected_revision,
            "generated_state": (
                list(generated_state) if generated_state is not None else None
            ),
            "generated_sha256": generated_revision,
            "retirement_snapshot_required": retirement_snapshot_required,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_revision = hashlib.sha256(payload).hexdigest()
    owned = _owned_temporary_identity(temporary_name, target_name)
    if owned != temporary_identity:
        raise DataError(
            f"Bound-write temp name does not bind its identity: {temporary_name}"
        )
    draft = temporary_name[: -len(".tmp")] + _CAS_DRAFT_SUFFIX
    marker = temporary_name[: -len(".tmp")] + _CAS_PREPARED_SUFFIX
    try:
        descriptor = os.open(
            draft,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=binding.descriptor,
        )
    except FileExistsError:
        raise DataError(f"Could not allocate a CAS marker for {target_name}") from None
    created = os.fstat(descriptor)
    draft_identity = created.st_dev, created.st_ino
    try:
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("Could not write the complete CAS marker")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        draft_state, draft_revision = _read_bound_entry(
            binding.descriptor, draft
        )
    except BaseException:
        cleaned = _retire_created_entry(
            binding,
            draft,
            draft_identity,
        )
        if not cleaned and _bound_snapshot(binding.descriptor, draft) is not None:
            raise DataError(
                f"A failed CAS marker draft could not be retired: {draft}"
            ) from None
        os.fsync(binding.descriptor)
        _validate_bound_directory(binding)
        raise
    if draft_revision != payload_revision:
        if not _retire_exact_entry(
            binding, draft, draft_state, draft_revision
        ):
            raise DataError(
                f"A damaged CAS marker draft could not be retired: {draft}"
            )
        os.fsync(binding.descriptor)
        _validate_bound_directory(binding)
        raise DataError(f"Could not verify a CAS marker for {target_name}")
    # The complete draft is itself crash evidence.  Make its directory entry
    # durable before the no-clobber rename so recovery can promote it.
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)
    try:
        _rename_entry_exclusive(binding.descriptor, draft, marker)
    except OSError:
        moved = _bound_snapshot(binding.descriptor, marker)
        source = _bound_snapshot(binding.descriptor, draft)
        if moved != (draft_state, draft_revision) or source is not None:
            raise
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)
    if _bound_snapshot(binding.descriptor, marker) != (
        draft_state,
        draft_revision,
    ):
        raise DataError(f"Could not verify a CAS marker for {target_name}")
    return marker, draft_state, draft_revision


def _append_cas_retirement_snapshot(
    binding: _DirectoryBinding,
    marker: str,
    target_name: str,
    *,
    expected_state: _EntryState,
    expected_revision: str,
    temporary_snapshot: tuple[_CleanupEntryState, str],
) -> tuple[_EntryState, str]:
    """Durably bind a temp's destructive snapshot before retiring its name.

    The base marker remains independently parseable if a process dies during
    the append.  Each complete record ends with a newline; recovery can append
    a later complete record after an unterminated fragment, but no terminal
    marker is accepted unless its last record is complete.
    """
    current_state, current_revision, payload = _read_bound_bytes(
        binding.descriptor, marker
    )
    if (current_state, current_revision) != (
        expected_state,
        expected_revision,
    ):
        raise DataError(
            f"Bound-write marker changed before retirement binding: {marker}"
        )
    transaction = _parse_cas_marker(payload, target_name=target_name)
    _require_marker_owns_transaction(marker, target_name, transaction)
    if not transaction["retirement_snapshot_required"]:
        raise DataError(
            f"Bound-write marker does not require retirement binding: {marker}"
        )
    if (
        transaction["temporary_retirement_snapshot"] == temporary_snapshot
        and not transaction["retirement_record_incomplete"]
    ):
        return current_state, current_revision

    base_payload = payload.partition(b"\n")[0]
    record = json.dumps(
        {
            "base_sha256": hashlib.sha256(base_payload).hexdigest(),
            "temporary_snapshot": {
                "state": list(temporary_snapshot[0]),
                "sha256": temporary_snapshot[1],
            },
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    frame = b"\n" + record + b"\n"
    descriptor = os.open(
        marker,
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=binding.descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _entry_state(before) != expected_state
        ):
            raise DataError(
                f"Bound-write marker changed before retirement binding: {marker}"
            )
        offset = 0
        while offset < len(frame):
            written = os.write(descriptor, frame[offset:])
            if written <= 0:
                raise OSError("Could not append the complete retirement binding")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    final_state, final_revision, final_payload = _read_bound_bytes(
        binding.descriptor, marker
    )
    final_transaction = _parse_cas_marker(
        final_payload, target_name=target_name
    )
    _require_marker_owns_transaction(marker, target_name, final_transaction)
    if (
        not final_payload.endswith(frame)
        or final_transaction["temporary_retirement_snapshot"]
        != temporary_snapshot
        or final_transaction["retirement_record_incomplete"]
    ):
        raise DataError(
            f"Could not verify bound-write retirement binding: {marker}"
        )
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)
    return final_state, final_revision


def _move_cas_marker(
    binding: _DirectoryBinding,
    marker: str,
    suffix: str,
    *,
    expected_state: _EntryState,
    expected_revision: str,
) -> str:
    destination = marker[: -len(_CAS_PREPARED_SUFFIX)] + suffix
    expected = expected_state, expected_revision
    if _bound_snapshot(binding.descriptor, marker) != expected:
        raise DataError(f"Bound-write marker changed before transition: {marker}")
    try:
        _rename_entry_exclusive(
            binding.descriptor,
            marker,
            destination,
        )
    except OSError:
        moved = _bound_snapshot(binding.descriptor, destination)
        source = _bound_snapshot(binding.descriptor, marker)
        if moved != expected or source is not None:
            raise
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)
    if (
        _bound_snapshot(binding.descriptor, destination) != expected
        or _bound_snapshot(binding.descriptor, marker) is not None
    ):
        raise DataError(
            f"Bound-write marker changed during transition: {destination}"
        )
    return destination


_RETIREMENT_SLOTS = 20


def _retirement_names(
    directory_identity: tuple[int, int],
    name: str,
    expected_state: _EntryState,
    expected_revision: str,
    expected_ctime_ns: int | None,
) -> tuple[str, ...]:
    """Destinations recoverable from the exact already-durable binding.

    The checkout's absolute path is deliberately absent.  A repository can be
    moved between a forget decision and its retry while the original directory
    inode remains bound.  Device/inode, the entry's full destructive snapshot,
    name, and digest are the facts the journal actually persisted.
    """
    full_state = (*expected_state, expected_ctime_ns)
    seed = b"\0".join(
        (
            str(directory_identity[0]).encode("ascii"),
            str(directory_identity[1]).encode("ascii"),
            os.fsencode(name),
            *(str(value).encode("ascii") for value in full_state),
            expected_revision.encode("ascii"),
        )
    )
    token = hashlib.sha256(seed).hexdigest()
    return tuple(
        f"entry-{token}-{slot:02d}.retired"
        for slot in range(_RETIREMENT_SLOTS)
    )


@contextlib.contextmanager
def _open_retirement_directory(
    *,
    create: bool,
) -> Iterator[_DirectoryBinding | None]:
    root = Path(os.path.realpath(_path_lock_root())) / "retired-writes"
    if create:
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise DataError(
                f"Could not prepare bound-write retirement store {root}: {exc}"
            ) from exc
    else:
        try:
            observed = os.lstat(root)
        except FileNotFoundError:
            yield None
            return
        except OSError as exc:
            raise DataError(
                f"Could not inspect bound-write retirement store {root}: {exc}"
            ) from exc
        if not stat.S_ISDIR(observed.st_mode):
            raise DataError(f"Bound-write retirement store is not private: {root}")
    if not _owned_private_directory(root):
        raise DataError(f"Bound-write retirement store is not private: {root}")
    with _open_bound_directory(root, create=False) as retirement:
        yield retirement


def _retirement_candidate_snapshot(
    retirement_fd: int,
    name: str,
) -> tuple[_EntryState, str] | None:
    """Read a regular candidate; an unrelated non-regular occupant is occupied."""
    try:
        details = os.stat(name, dir_fd=retirement_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode):
        return None
    return _bound_snapshot(retirement_fd, name)


def _retirement_candidate_exists(retirement_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=retirement_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


class _RetirementRestored(Exception):
    """Internal signal that a reversible confirmation restored its source."""


def _restore_retired_entry(
    retirement: _DirectoryBinding,
    retired_name: str,
    source: _DirectoryBinding,
    source_name: str,
    expected: tuple[_EntryState, str],
) -> None:
    """Restore a moved exact entry without overwriting a raced source name."""
    try:
        _rename_entry_exclusive_between(
            retirement.descriptor,
            retired_name,
            source.descriptor,
            source_name,
        )
    except OSError as exc:
        restored = _bound_snapshot(source.descriptor, source_name)
        retained = _retirement_candidate_snapshot(
            retirement.descriptor, retired_name
        )
        if restored != expected or retained is not None:
            raise DataError(
                "An exact entry could not be restored to "
                f"{source.path / source_name}; it remains under the private "
                f"retirement name {retired_name}"
            ) from exc
    os.fsync(source.descriptor)
    os.fsync(retirement.descriptor)
    _validate_bound_directory(source)
    _validate_bound_directory(retirement)
    if (
        _bound_snapshot(source.descriptor, source_name) != expected
        or _retirement_candidate_exists(retirement.descriptor, retired_name)
    ):
        raise DataError(
            "An exact entry changed while it was being restored to "
            f"{source.path / source_name}"
        )


def _finish_retired_entry(
    retirement: _DirectoryBinding,
    retired_name: str,
    expected_state: _EntryState,
    expected_revision: str,
    *,
    confirm_while_reversible: Callable[[], bool] | None,
    restore_to: tuple[_DirectoryBinding, str] | None,
) -> bool:
    """Revalidate and unlink one exact deterministic retirement entry."""
    descriptor = os.open(
        retired_name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=retirement.descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _entry_state(before) != expected_state:
            return False
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(
            retired_name,
            dir_fd=retirement.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or digest.hexdigest() != expected_revision
            or _entry_state(after) != expected_state
            or _entry_state(named) != expected_state
        ):
            return False
        if confirm_while_reversible is not None:
            try:
                confirmed = confirm_while_reversible()
            except BaseException as exc:
                os.fsync(retirement.descriptor)
                _validate_bound_directory(retirement)
                if restore_to is not None:
                    _restore_retired_entry(
                        retirement,
                        retired_name,
                        restore_to[0],
                        restore_to[1],
                        (expected_state, expected_revision),
                    )
                if isinstance(exc, Exception):
                    raise DataError(
                        f"Paired retirement failed: {exc}"
                    ) from exc
                raise
            if not confirmed:
                os.fsync(retirement.descriptor)
                _validate_bound_directory(retirement)
                if restore_to is not None:
                    _restore_retired_entry(
                        retirement,
                        retired_name,
                        restore_to[0],
                        restore_to[1],
                        (expected_state, expected_revision),
                    )
                    raise _RetirementRestored
                raise DataError(
                    "Paired retirement did not remove its second exact name; "
                    f"the first remains recoverable as {retired_name}"
                )
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmed_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            confirmed_digest.update(chunk)
        confirmed_held = os.fstat(descriptor)
        confirmed_named = os.stat(
            retired_name,
            dir_fd=retirement.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(confirmed_named.st_mode)
            or _entry_state(confirmed_held) != expected_state
            or _entry_state(confirmed_named) != expected_state
            or confirmed_digest.hexdigest() != expected_revision
        ):
            raise DataError(
                f"Exact private retirement entry changed: {retired_name}"
            )
        os.unlink(retired_name, dir_fd=retirement.descriptor)
        os.fsync(retirement.descriptor)
        _validate_bound_directory(retirement)
        if _retirement_candidate_exists(retirement.descriptor, retired_name):
            raise DataError(
                f"Exact private retirement entry remained: {retired_name}"
            )
        return True
    finally:
        os.close(descriptor)


def _retire_detached_entry(
    directory_identity: tuple[int, int],
    name: str,
    expected_state: _EntryState,
    expected_revision: str,
    *,
    expected_ctime_ns: int | None = None,
    confirm_while_reversible: Callable[[], bool] | None = None,
    restore_to: tuple[_DirectoryBinding, str] | None = None,
) -> bool:
    """Finish a prior move even when its original namespace is unreachable."""
    candidates = _retirement_names(
        directory_identity,
        name,
        expected_state,
        expected_revision,
        expected_ctime_ns,
    )
    with _open_retirement_directory(create=False) as retirement:
        if retirement is None:
            return False
        if os.fstat(retirement.descriptor).st_dev != expected_state[0]:
            return False
        expected = expected_state, expected_revision
        for candidate in candidates:
            if (
                _retirement_candidate_snapshot(retirement.descriptor, candidate)
                == expected
            ):
                finished = _finish_retired_entry(
                    retirement,
                    candidate,
                    expected_state,
                    expected_revision,
                    confirm_while_reversible=confirm_while_reversible,
                    restore_to=restore_to,
                )
                if not finished:
                    raise DataError(
                        f"Exact private retirement entry changed: {candidate}"
                    )
                return True
        return False


def _retire_exact_entry(
    binding: _DirectoryBinding,
    name: str,
    expected_state: _EntryState,
    expected_revision: str,
    *,
    expected_ctime_ns: int | None = None,
    retirement_ctime_ns: int | None = None,
    confirm_while_reversible: Callable[[], bool] | None = None,
    leave_retired_on_confirmation_failure: bool = False,
) -> bool:
    """Move one exact private entry out of the target directory.

    There is no POSIX unlink-if-inode operation.  The exclusive rename makes
    the name private; if a different entry wins the rename seam, both names are
    left untouched after inspection. Once the exact name is private, its link
    is removed rather than its inode being truncated, so an independent
    hard-link backup stays intact. A host whose private cache is on another
    filesystem leaves the private source name in place rather than deleting or
    truncating uncertain data.
    """
    directory_fd = binding.descriptor
    parent = os.fstat(directory_fd)
    directory_identity = _directory_identity(parent)
    durable_retirement_ctime = (
        expected_ctime_ns
        if retirement_ctime_ns is None
        else retirement_ctime_ns
    )
    candidates = _retirement_names(
        directory_identity,
        name,
        expected_state,
        expected_revision,
        durable_retirement_ctime,
    )

    def still_bound(details: os.stat_result) -> bool:
        return _entry_state(details) == expected_state and (
            expected_ctime_ns is None
            or details.st_ctime_ns == expected_ctime_ns
        )

    restore_to = (
        None
        if leave_retired_on_confirmation_failure
        else (binding, name)
    )
    try:
        retired_before_source = _retire_detached_entry(
            directory_identity,
            name,
            expected_state,
            expected_revision,
            expected_ctime_ns=durable_retirement_ctime,
            confirm_while_reversible=confirm_while_reversible,
            restore_to=restore_to,
        )
    except _RetirementRestored:
        return False
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except OSError:
        return retired_before_source
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not still_bound(before)
        ):
            return retired_before_source
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            digest.hexdigest() != expected_revision
            or not still_bound(before)
            or not still_bound(after)
            or not still_bound(named)
        ):
            return retired_before_source
        try:
            with _open_retirement_directory(create=True) as retirement:
                if retirement is None:  # pragma: no cover - create=True
                    return retired_before_source
                if os.fstat(directory_fd).st_dev == os.fstat(
                    retirement.descriptor
                ).st_dev:
                    # Re-prove the durable identity at the destructive seam.
                    # Once this descriptor is open, its inode cannot be reused;
                    # post-rename checks intentionally use the four stable CAS
                    # fields because rename itself changes ctime.
                    held_before_move = os.fstat(descriptor)
                    named_before_move = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if not still_bound(held_before_move) or not still_bound(
                        named_before_move
                    ):
                        return retired_before_source
                    retired_name = ""
                    for candidate in candidates:
                        if _retirement_candidate_exists(
                            retirement.descriptor, candidate
                        ):
                            continue
                        try:
                            _rename_entry_exclusive_between(
                                directory_fd,
                                name,
                                retirement.descriptor,
                                candidate,
                            )
                        except FileExistsError:
                            continue
                        except OSError:
                            moved = _retirement_candidate_snapshot(
                                retirement.descriptor, candidate
                            )
                            try:
                                os.stat(
                                    name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                source_absent = True
                            else:
                                source_absent = False
                            if moved == (expected_state, expected_revision):
                                retired_name = candidate
                                break
                            if moved is not None and source_absent:
                                retired_name = candidate
                                break
                            break
                        else:
                            retired_name = candidate
                            break
                    if retired_name:
                        moved = _retirement_candidate_snapshot(
                            retirement.descriptor, retired_name
                        )
                        held = _entry_state(os.fstat(descriptor))
                        if (
                            moved != (expected_state, expected_revision)
                            or held != expected_state
                        ):
                            try:
                                _rename_entry_exclusive_between(
                                    retirement.descriptor,
                                    retired_name,
                                    directory_fd,
                                    name,
                                )
                            except OSError as exc:
                                restored = _bound_snapshot(directory_fd, name)
                                retained = _retirement_candidate_snapshot(
                                    retirement.descriptor, retired_name
                                )
                                if restored != moved or retained is not None:
                                    raise DataError(
                                        "A raced entry could not be restored to "
                                        f"{binding.path / name}; it remains under "
                                        f"the private retirement name {retired_name}"
                                    ) from exc
                            os.fsync(directory_fd)
                            os.fsync(retirement.descriptor)
                            _validate_bound_directory(binding)
                            _validate_bound_directory(retirement)
                            if (
                                _bound_snapshot(directory_fd, name) != moved
                                or _retirement_candidate_snapshot(
                                    retirement.descriptor, retired_name
                                )
                                is not None
                            ):
                                raise DataError(
                                    "A raced entry changed while it was being "
                                    f"restored to {binding.path / name}"
                                )
                            return retired_before_source
                        try:
                            finished = _finish_retired_entry(
                                retirement,
                                retired_name,
                                expected_state,
                                expected_revision,
                                confirm_while_reversible=confirm_while_reversible,
                                restore_to=restore_to,
                            )
                        except _RetirementRestored:
                            return False
                        except DataError:
                            raise
                        except OSError as exc:
                            raise DataError(
                                "Exact private retirement entry could not be "
                                f"removed: {retired_name}: {exc}"
                            ) from exc
                        if not finished:
                            raise DataError(
                                "Exact private retirement entry changed before "
                                f"removal: {retired_name}"
                            )
                        os.fsync(directory_fd)
                        _validate_bound_directory(binding)
                        return True
        except DataError:
            raise
        except OSError:
            pass
        return False
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _retire_created_entry(
    binding: _DirectoryBinding,
    name: str,
    identity: tuple[int, int] | None,
) -> bool:
    """Retire a private entry only when it is still the inode we created."""
    if not name or identity is None:
        return False
    try:
        snapshot = _bound_snapshot(binding.descriptor, name)
    except (DataError, OSError):
        return False
    if snapshot is None or snapshot[0][:2] != identity:
        return False
    return _retire_exact_entry(
        binding,
        name,
        snapshot[0],
        snapshot[1],
    )


def _parse_cas_marker(payload: bytes, *, target_name: str) -> dict[str, Any]:
    base_payload, separator, trailing_payload = payload.partition(b"\n")
    try:
        raw = json.loads(base_payload)
    except (UnicodeError, ValueError) as exc:
        raise DataError(
            f"Incomplete bound-write marker for {target_name} is unreadable: {exc}"
        ) from exc
    required = {
        "version",
        "target",
        "temporary",
        "temporary_identity",
        "expected_absent",
        "expected_state",
        "expected_sha256",
        "generated_state",
        "generated_sha256",
        "retirement_snapshot_required",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise DataError(f"Incomplete bound-write marker for {target_name} is invalid")
    if raw.get("version") != 3 or raw.get("target") != target_name:
        raise DataError(f"Incomplete bound-write marker for {target_name} is invalid")
    temporary = raw.get("temporary")
    temporary_identity = raw.get("temporary_identity")
    expected_absent = raw.get("expected_absent")
    expected_state = raw.get("expected_state")
    expected = raw.get("expected_sha256")
    generated_state = raw.get("generated_state")
    generated = raw.get("generated_sha256")
    retirement_snapshot_required = raw.get("retirement_snapshot_required")
    try:
        encoded_temporary = os.fsencode(temporary)
    except (TypeError, UnicodeError, ValueError):
        encoded_temporary = b""
    if (
        not isinstance(temporary, str)
        or "/" in temporary
        or not encoded_temporary
        or b"\0" in encoded_temporary
        or not temporary.startswith(f".{target_name}.")
        or not temporary.endswith(".tmp")
        or not isinstance(temporary_identity, list)
        or len(temporary_identity) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in temporary_identity
        )
        or not isinstance(expected_absent, bool)
        or not isinstance(retirement_snapshot_required, bool)
        or not isinstance(generated, str)
        or len(generated) != 64
        or any(character not in "0123456789abcdef" for character in generated)
    ):
        raise DataError(f"Incomplete bound-write marker for {target_name} is invalid")
    def state_is_valid(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in value
            )
        )

    def cleanup_state_is_valid(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 5
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in value
            )
        )

    def digest_is_valid(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )
    if expected_absent:
        if (
            expected_state is not None
            or expected is not None
            or generated_state is not None
        ):
            raise DataError(
                f"Incomplete bound-write marker for {target_name} is invalid"
            )
    elif (
        not state_is_valid(expected_state)
        or not digest_is_valid(expected)
        or not state_is_valid(generated_state)
    ):
        raise DataError(f"Incomplete bound-write marker for {target_name} is invalid")
    if retirement_snapshot_required and not expected_absent:
        raise DataError(f"Incomplete bound-write marker for {target_name} is invalid")

    temporary_retirement_snapshot = None
    retirement_record_incomplete = False
    if separator:
        framed_records = trailing_payload.split(b"\n")
        incomplete_tail = (
            b"" if trailing_payload.endswith(b"\n") else framed_records.pop()
        )
        base_revision = hashlib.sha256(base_payload).hexdigest()
        for encoded_record in framed_records:
            if not encoded_record:
                continue
            try:
                record = json.loads(encoded_record)
                record_snapshot = record["temporary_snapshot"]
                record_state = record_snapshot["state"]
                record_revision = record_snapshot["sha256"]
                valid_record = (
                    isinstance(record, Mapping)
                    and set(record)
                    == {"version", "base_sha256", "temporary_snapshot"}
                    and record.get("version") == 1
                    and record.get("base_sha256") == base_revision
                    and isinstance(record_snapshot, Mapping)
                    and set(record_snapshot) == {"state", "sha256"}
                    and cleanup_state_is_valid(record_state)
                    and digest_is_valid(record_revision)
                    and tuple(record_state[:2]) == tuple(temporary_identity)
                    and record_revision == generated
                    and retirement_snapshot_required
                )
            except (KeyError, TypeError, UnicodeError, ValueError):
                valid_record = False
            if not valid_record:
                retirement_record_incomplete = True
                continue
            parsed_snapshot = tuple(record_state), record_revision
            if (
                temporary_retirement_snapshot is not None
                and parsed_snapshot != temporary_retirement_snapshot
            ):
                raise DataError(
                    f"Incomplete bound-write marker for {target_name} has "
                    "conflicting retirement bindings"
                )
            temporary_retirement_snapshot = parsed_snapshot
            retirement_record_incomplete = False
        if incomplete_tail:
            retirement_record_incomplete = True
    return {
        "temporary": temporary,
        "temporary_identity": tuple(temporary_identity),
        "expected_absent": expected_absent,
        "expected_state": (
            tuple(expected_state) if expected_state is not None else None
        ),
        "expected_sha256": expected,
        "generated_state": (
            tuple(generated_state) if generated_state is not None else None
        ),
        "generated_sha256": generated,
        "retirement_snapshot_required": retirement_snapshot_required,
        "temporary_retirement_snapshot": temporary_retirement_snapshot,
        "retirement_record_incomplete": retirement_record_incomplete,
    }


def _promote_complete_cas_draft_under_lock(
    binding: _DirectoryBinding,
    target: Path,
) -> None:
    """Turn a crash-durable marker draft into the active prepared name.

    The draft is written and fsynced before its exclusive rename.  A process
    death at that seam used to leave a valid intent that every recovery scan
    ignored.  Recovery validates the exact draft before promoting it; an
    incomplete or competing marker remains untouched and refuses the write.
    """
    prefix = f".{target.name}."
    suffixes = (
        _CAS_PREPARED_SUFFIX,
        _CAS_VALIDATED_SUFFIX,
        _CAS_RECOVERED_SUFFIX,
    )
    active = sorted(
        name
        for name in os.listdir(binding.descriptor)
        if name.startswith(prefix) and name.endswith(suffixes)
    )
    drafts = sorted(
        name
        for name in os.listdir(binding.descriptor)
        if name.startswith(prefix) and name.endswith(_CAS_DRAFT_SUFFIX)
    )
    if not drafts:
        return
    if active or len(drafts) != 1:
        raise DataError(
            f"Multiple incomplete bound writes exist for {target}; refusing to "
            "guess which one owns the public file"
        )
    draft = drafts[0]
    owned = _marker_owned_temporary(draft, target.name)
    if owned is None:
        raise DataError(f"Incomplete bound-write marker name is invalid: {draft}")
    draft_state, draft_revision, payload = _read_bound_bytes(
        binding.descriptor, draft
    )
    try:
        transaction = _parse_cas_marker(payload, target_name=target.name)
    except DataError:
        # The exact draft can be journalled and retired, but an unreadable body
        # does not prove that the temp derived from its filename belongs to the
        # same transaction. Preserve that separate name.
        _validate_bound_directory(binding)
        return
    if (
        transaction["temporary"] != owned[0]
        or transaction["temporary_identity"] != owned[1]
    ):
        # The body is valid JSON but still cannot authorize its filename's
        # private temp. Keep the draft as exact marker-only evidence, exactly
        # like an unreadable body, so end/forget can settle without publishing
        # or retiring the separate temp name.
        _validate_bound_directory(binding)
        return
    prepared = draft[: -len(_CAS_DRAFT_SUFFIX)] + _CAS_PREPARED_SUFFIX
    try:
        _rename_entry_exclusive(binding.descriptor, draft, prepared)
    except OSError:
        moved = _bound_snapshot(binding.descriptor, prepared)
        source = _bound_snapshot(binding.descriptor, draft)
        if moved != (draft_state, draft_revision) or source is not None:
            raise
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)
    if (
        _bound_snapshot(binding.descriptor, prepared)
        != (draft_state, draft_revision)
        or _bound_snapshot(binding.descriptor, draft) is not None
    ):
        raise DataError(
            f"Bound-write marker changed while recovering its draft: {prepared}"
        )


def _cleanup_terminal_cas_markers_under_lock(
    binding: _DirectoryBinding, target: Path
) -> None:
    """Finish cleanup after a durable commit or rollback decision.

    A crash after the prepared marker becomes ``validated`` must keep the
    generated public file.  A crash after it becomes ``recovered`` must keep
    the restored public file.  These terminal markers therefore authorize
    cleanup only; they never move the public name.
    """
    prefix = f".{target.name}."
    terminal_suffixes = (_CAS_VALIDATED_SUFFIX, _CAS_RECOVERED_SUFFIX)
    markers = sorted(
        name
        for name in os.listdir(binding.descriptor)
        if name.startswith(prefix) and name.endswith(terminal_suffixes)
    )
    for marker in markers:
        marker_state, marker_revision, payload = _read_bound_bytes(
            binding.descriptor, marker
        )
        transaction = _parse_cas_marker(payload, target_name=target.name)
        _require_marker_owns_transaction(marker, target.name, transaction)
        temporary = transaction["temporary"]
        temporary_identity = transaction["temporary_identity"]
        expected_absent = transaction["expected_absent"]
        expected_snapshot = (
            transaction["expected_state"],
            transaction["expected_sha256"],
        )
        generated_snapshot = (
            transaction["generated_state"],
            transaction["generated_sha256"],
        )
        public_snapshot = _bound_snapshot(binding.descriptor, target.name)
        temporary_snapshot = _bound_snapshot(binding.descriptor, temporary)
        cleaned_temporary = temporary_snapshot is None

        if expected_absent:
            if marker.endswith(_CAS_RECOVERED_SUFFIX):
                raise DataError(
                    f"Write-once bound write for {target} has an invalid "
                    "rollback marker"
                )
            generated_revision = transaction["generated_sha256"]
            retirement_snapshot = transaction[
                "temporary_retirement_snapshot"
            ]
            if transaction["retirement_snapshot_required"] and (
                retirement_snapshot is None
                or transaction["retirement_record_incomplete"]
            ):
                raise DataError(
                    f"Committed write-once bound write for {target} lacks its "
                    "complete private retirement binding"
                )
            if (
                public_snapshot is None
                or public_snapshot[0][:2] != temporary_identity
                or public_snapshot[1] != generated_revision
            ):
                raise DataError(
                    f"Committed write-once bound write for {target} cannot be "
                    "cleaned because its public name no longer holds the "
                    "captured answer"
                )
            if retirement_snapshot is not None:
                parent = os.fstat(binding.descriptor)
                _retire_detached_entry(
                    (parent.st_dev, parent.st_ino),
                    temporary,
                    retirement_snapshot[0][:4],
                    retirement_snapshot[1],
                    expected_ctime_ns=retirement_snapshot[0][4],
                )
                temporary_snapshot = _bound_snapshot(
                    binding.descriptor, temporary
                )
                cleaned_temporary = temporary_snapshot is None
            if temporary_snapshot is not None:
                if (
                    temporary_snapshot[0][:2] != temporary_identity
                    or temporary_snapshot[1] != generated_revision
                    or temporary_snapshot[0] != public_snapshot[0]
                ):
                    raise DataError(
                        f"Committed write-once bound write for {target} has "
                        "unrecognized private bytes"
                    )
                destructive_snapshot = _cleanup_bound_snapshot(
                    binding.descriptor, temporary
                )
                if (
                    retirement_snapshot is not None
                    and destructive_snapshot != retirement_snapshot
                ):
                    raise DataError(
                        f"Committed write-once bound write for {target} has "
                        "changed its private retirement binding"
                    )
                assert destructive_snapshot is not None

                def public_still_holds_answer(
                    expected_public: tuple[_EntryState, str] = public_snapshot,
                ) -> bool:
                    current = _bound_snapshot(
                        binding.descriptor, target.name
                    )
                    _validate_bound_directory(binding)
                    return current == expected_public

                cleaned_temporary = _retire_exact_entry(
                    binding,
                    temporary,
                    temporary_snapshot[0],
                    temporary_snapshot[1],
                    expected_ctime_ns=destructive_snapshot[0][4],
                    confirm_while_reversible=public_still_holds_answer,
                )
                if not cleaned_temporary:
                    raise DataError(
                        f"Committed write-once bound write for {target} "
                        "cannot retire its private answer while the public "
                        "name is changing"
                    )
            if _bound_snapshot(binding.descriptor, target.name) != public_snapshot:
                raise DataError(
                    f"Committed write-once bound write for {target} cannot be "
                    "cleaned because its public name changed during cleanup"
                )
            if not _retire_exact_entry(
                binding,
                marker,
                marker_state,
                marker_revision,
            ):
                raise DataError(
                    f"Committed write-once bound write for {target} cannot "
                    "retire its terminal marker"
                )
            continue

        if marker.endswith(_CAS_VALIDATED_SUFFIX):
            if temporary_snapshot == generated_snapshot:
                raise DataError(
                    f"Committed bound write for {target} has generated bytes "
                    "under its private name; refusing an ambiguous cleanup"
                )
            if temporary_snapshot == expected_snapshot:
                cleaned_temporary = _retire_exact_entry(
                    binding,
                    temporary,
                    expected_snapshot[0],
                    expected_snapshot[1],
                )
            # A public edit after the commit is authoritative, including an
            # edit back to the old bytes.  Terminal recovery never rewrites it.
        else:
            if public_snapshot == generated_snapshot:
                raise DataError(
                    f"Rolled-back bound write for {target} still has generated "
                    "bytes public; refusing an ambiguous cleanup"
                )
            if temporary_snapshot == generated_snapshot:
                cleaned_temporary = _retire_exact_entry(
                    binding,
                    temporary,
                    generated_snapshot[0],
                    generated_snapshot[1],
                )

        if cleaned_temporary:
            _retire_exact_entry(
                binding,
                marker,
                marker_state,
                marker_revision,
            )


def _recover_bound_target_under_lock(
    binding: _DirectoryBinding,
    target: Path,
) -> None:
    _promote_complete_cas_draft_under_lock(binding, target)
    _cleanup_terminal_cas_markers_under_lock(binding, target)
    prefix = f".{target.name}."
    prepared = sorted(
        name
        for name in os.listdir(binding.descriptor)
        if name.startswith(prefix) and name.endswith(_CAS_PREPARED_SUFFIX)
    )
    if len(prepared) > 1:
        raise DataError(
            f"Multiple incomplete bound writes exist for {target}; refusing to "
            "guess which one owns the public file"
        )
    if not prepared:
        return
    marker = prepared[0]
    _marker_state, _marker_revision, payload = _read_bound_bytes(
        binding.descriptor, marker
    )
    transaction = _parse_cas_marker(payload, target_name=target.name)
    _require_marker_owns_transaction(marker, target.name, transaction)
    temporary = transaction["temporary"]
    temporary_identity = transaction["temporary_identity"]
    expected_absent = transaction["expected_absent"]
    generated = transaction["generated_sha256"]
    public_snapshot = _bound_snapshot(binding.descriptor, target.name)
    temporary_snapshot = _bound_snapshot(binding.descriptor, temporary)

    if expected_absent:
        if (
            temporary_snapshot is None
            or temporary_snapshot[0][:2] != temporary_identity
            or temporary_snapshot[1] != generated
        ):
            raise DataError(
                f"Could not recover incomplete write-once publication for {target}: "
                "the private answer is absent or incomplete"
            )
        if public_snapshot is None:
            try:
                _link_entry_exclusive(
                    binding.descriptor,
                    temporary,
                    target.name,
                )
            except OSError:
                if _bound_snapshot(
                    binding.descriptor, target.name
                ) != temporary_snapshot:
                    raise
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
            public_snapshot = _bound_snapshot(
                binding.descriptor, target.name
            )
        if public_snapshot != temporary_snapshot:
            raise DataError(
                f"Could not recover incomplete write-once publication for {target}: "
                "the public name holds different bytes"
            )
        if transaction["retirement_snapshot_required"]:
            temporary_cleanup_snapshot = _cleanup_bound_snapshot(
                binding.descriptor, temporary
            )
            public_cleanup_snapshot = _cleanup_bound_snapshot(
                binding.descriptor, target.name
            )
            if (
                temporary_cleanup_snapshot is None
                or temporary_cleanup_snapshot != public_cleanup_snapshot
            ):
                raise DataError(
                    f"Could not bind private retirement for {target}: the "
                    "published hard links changed"
                )
            persisted_retirement = transaction[
                "temporary_retirement_snapshot"
            ]
            if (
                persisted_retirement is not None
                and persisted_retirement != temporary_cleanup_snapshot
            ):
                raise DataError(
                    f"Could not bind private retirement for {target}: its "
                    "durable snapshot changed"
                )
            if (
                persisted_retirement is None
                or transaction["retirement_record_incomplete"]
            ):
                _marker_state, _marker_revision = (
                    _append_cas_retirement_snapshot(
                        binding,
                        marker,
                        target.name,
                        expected_state=_marker_state,
                        expected_revision=_marker_revision,
                        temporary_snapshot=temporary_cleanup_snapshot,
                    )
                )
        validated = _move_cas_marker(
            binding,
            marker,
            _CAS_VALIDATED_SUFFIX,
            expected_state=_marker_state,
            expected_revision=_marker_revision,
        )
        if not validated.endswith(_CAS_VALIDATED_SUFFIX):  # pragma: no cover
            raise AssertionError(validated)
        _cleanup_terminal_cas_markers_under_lock(binding, target)
        os.fsync(binding.descriptor)
        _validate_bound_directory(binding)
        return

    expected_snapshot = (
        transaction["expected_state"],
        transaction["expected_sha256"],
    )
    generated_state = transaction["generated_state"]
    assert generated_state is not None
    generated_snapshot = generated_state, generated

    if public_snapshot == generated_snapshot:
        if temporary_snapshot != expected_snapshot:
            raise DataError(
                f"Could not recover incomplete bound write for {target}: "
                "the displaced file does not match bound evidence"
            )
        _exchange_entries(binding.descriptor, temporary, target.name)
        os.fsync(binding.descriptor)
        _validate_bound_directory(binding)
        if (
            _bound_snapshot(binding.descriptor, temporary) != generated_snapshot
            or _bound_snapshot(binding.descriptor, target.name)
            != temporary_snapshot
        ):
            raise DataError(f"Could not verify bound-write recovery for {target}")
        public_snapshot = temporary_snapshot
        temporary_snapshot = generated_snapshot
    elif temporary_snapshot != generated_snapshot:
        raise DataError(
            f"Could not recover incomplete bound write for {target}: neither "
            "name holds the generated bytes"
        )

    if public_snapshot is None:
        raise DataError(
            f"Could not recover incomplete bound write for {target}: the public "
            "file is missing"
        )
    _move_cas_marker(
        binding,
        marker,
        _CAS_RECOVERED_SUFFIX,
        expected_state=_marker_state,
        expected_revision=_marker_revision,
    )
    _cleanup_terminal_cas_markers_under_lock(binding, target)
    os.fsync(binding.descriptor)
    _validate_bound_directory(binding)


def _recover_bound_path(path: Path) -> None:
    """Finish a path's WAL transaction without reading its public payload."""
    target = Path(path).absolute()
    with (
        exclusive_path_lock(_cas_lock_path(target)),
        _open_bound_directory(target.parent, create=False) as binding,
    ):
        _recover_bound_target_under_lock(binding, target)
        _validate_bound_directory(binding)


def _bound_write_evidence(path: Path) -> _BoundWriteEvidence | None:
    """Bind exact write-once WAL names without requiring public recovery.

    A complete private answer can survive while recovery quite correctly
    refuses to overwrite a changed public name.  Journal settlement still has
    to see that answer, and later cleanup must retain the exact identities that
    authorized its removal.
    """
    target = Path(path).absolute()
    try:
        parent_details = os.lstat(target.parent)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DataError(
            f"Could not inspect bound-write evidence for {target}: "
            f"{exc.strerror or exc}"
        ) from exc
    if not stat.S_ISDIR(parent_details.st_mode):
        # A symlink/non-directory can never be janki's descriptor-bound
        # pending store and grants no authority over what it points at.
        return None
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=False) as binding,
        ):
            _promote_complete_cas_draft_under_lock(binding, target)
            prefix = f".{target.name}."
            suffixes = (
                _CAS_DRAFT_SUFFIX,
                _CAS_PREPARED_SUFFIX,
                _CAS_VALIDATED_SUFFIX,
                _CAS_RECOVERED_SUFFIX,
            )
            markers = sorted(
                name
                for name in os.listdir(binding.descriptor)
                if name.startswith(prefix) and name.endswith(suffixes)
            )
            if len(markers) > 1:
                raise DataError(
                    f"Multiple incomplete bound writes exist for {target}; "
                    "refusing to guess which one owns the public file"
                )
            if not markers:
                _validate_bound_directory(binding)
                return None
            marker = markers[0]
            marker_state, marker_revision, payload = _read_cleanup_bound_bytes(
                binding.descriptor, marker
            )
            marker_owned = _marker_owned_temporary(marker, target.name)
            try:
                transaction = _parse_cas_marker(
                    payload, target_name=target.name
                )
            except DataError:
                transaction = None
            if (
                transaction is not None
                and (
                    not transaction["expected_absent"]
                    or marker_owned is None
                    or transaction["temporary"] != marker_owned[0]
                    or transaction["temporary_identity"] != marker_owned[1]
                    or (
                        marker.endswith(
                            (_CAS_VALIDATED_SUFFIX, _CAS_RECOVERED_SUFFIX)
                        )
                        and transaction["retirement_snapshot_required"]
                        and (
                            transaction["temporary_retirement_snapshot"] is None
                            or transaction["retirement_record_incomplete"]
                        )
                    )
                )
            ):
                transaction = None
            proven_temporary = (
                marker_owned
                if transaction is not None and marker_owned is not None
                else None
            )
            temporary = proven_temporary[0] if proven_temporary is not None else ""
            temporary_identity = (
                proven_temporary[1] if proven_temporary is not None else None
            )
            temporary_snapshot = None
            temporary_retirement_snapshot = (
                transaction["temporary_retirement_snapshot"]
                if transaction is not None
                else None
            )
            temporary_owned = proven_temporary is not None
            if proven_temporary is not None:
                try:
                    observed_temporary = _cleanup_bound_snapshot(
                        binding.descriptor, temporary
                    )
                except (DataError, OSError, UnicodeError, ValueError):
                    temporary_owned = False
                else:
                    if (
                        observed_temporary is not None
                        and observed_temporary[0][:2] != temporary_identity
                    ):
                        temporary_owned = False
                    elif temporary_owned:
                        temporary_snapshot = observed_temporary
            try:
                public_snapshot = _cleanup_bound_snapshot(
                    binding.descriptor, target.name
                )
            except (DataError, OSError, UnicodeError, ValueError):
                public_snapshot = None
            parent = os.fstat(binding.descriptor)
            _validate_bound_directory(binding)
            return _BoundWriteEvidence(
                target=target,
                directory_identity=(parent.st_dev, parent.st_ino),
                marker_name=marker,
                marker_state=marker_state,
                marker_revision=marker_revision,
                temporary_name=temporary,
                temporary_identity=temporary_identity,
                temporary_snapshot=temporary_snapshot,
                temporary_retirement_snapshot=temporary_retirement_snapshot,
                temporary_owned=temporary_owned,
                public_snapshot=public_snapshot,
                generated_revision=(
                    transaction["generated_sha256"]
                    if transaction is not None
                    else ""
                ),
            )
    except FileNotFoundError:
        if not target.parent.exists():
            return None
        raise DataError(
            f"Bound-write evidence changed while it was being read: {target}"
        ) from None
    except DataError:
        raise
    except OSError as exc:
        raise DataError(
            f"Could not inspect bound-write evidence for {target}: "
            f"{exc.strerror or exc}"
        ) from exc


def _read_bound_write_evidence(expected: _BoundWriteEvidence) -> bytes | None:
    """Read the exact complete answer bound by write-ahead evidence."""
    if not expected.reply_complete:
        return None
    target = expected.target
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=False) as binding,
        ):
            parent = os.fstat(binding.descriptor)
            if (parent.st_dev, parent.st_ino) != expected.directory_identity:
                return None
            if _cleanup_bound_snapshot(
                binding.descriptor, expected.marker_name
            ) != (
                expected.marker_state,
                expected.marker_revision,
            ):
                return None
            if expected.private_reply_complete:
                name = expected.temporary_name
                expected_snapshot = expected.temporary_snapshot
            else:
                name = target.name
                expected_snapshot = expected.public_snapshot
            if expected_snapshot is None:
                return None
            state, revision, payload = _read_cleanup_bound_bytes(
                binding.descriptor, name
            )
            _validate_bound_directory(binding)
            if (
                (state, revision) != expected_snapshot
                or revision != expected.generated_revision
            ):
                return None
            return payload
    except (DataError, OSError, UnicodeError, ValueError):
        return None


def _retire_receipted_marker(
    binding: _DirectoryBinding,
    *,
    directory_identity: tuple[int, int],
    name: str,
    expected_state: _CleanupEntryState,
    expected_revision: str,
) -> None:
    """Retire a marker receipt without probing a replacement's bytes."""
    observed_state = _cleanup_bound_entry_state(binding.descriptor, name)
    if observed_state != expected_state:
        _retire_detached_entry(
            directory_identity,
            name,
            expected_state[:4],
            expected_revision,
            expected_ctime_ns=expected_state[4],
        )
        return
    retired = _retire_exact_entry(
        binding,
        name,
        expected_state[:4],
        expected_revision,
        expected_ctime_ns=expected_state[4],
    )
    if retired:
        return
    if _cleanup_bound_entry_state(binding.descriptor, name) == expected_state:
        raise DataError(f"Could not retire bound-write marker: {name}")
    _retire_detached_entry(
        directory_identity,
        name,
        expected_state[:4],
        expected_revision,
        expected_ctime_ns=expected_state[4],
    )


def _retire_bound_write_evidence(expected: _BoundWriteEvidence) -> None:
    """Retire only the marker/private names bound by a journal decision.

    A public name that still holds the bound answer is part of the forced
    discard.  A missing or replaced public name is preserved: binding the WAL
    never grants deletion authority over a later occupant of that name.
    """
    target = expected.target
    private_retirement_snapshot = (
        expected.temporary_retirement_snapshot
        or expected.temporary_snapshot
    )

    def retire_detached_names() -> None:
        """Remove exact prior destinations without opening the old namespace."""
        if private_retirement_snapshot is not None:
            _retire_detached_entry(
                expected.directory_identity,
                expected.temporary_name,
                private_retirement_snapshot[0][:4],
                private_retirement_snapshot[1],
                expected_ctime_ns=private_retirement_snapshot[0][4],
            )
        public_answer = expected.public_answer_snapshot
        if public_answer is not None:
            _retire_detached_entry(
                expected.directory_identity,
                target.name,
                public_answer[0][:4],
                public_answer[1],
                expected_ctime_ns=public_answer[0][4],
            )
        _retire_detached_entry(
            expected.directory_identity,
            expected.marker_name,
            expected.marker_state[:4],
            expected.marker_revision,
            expected_ctime_ns=expected.marker_state[4],
        )

    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_cleanup_directory(
                target.parent, expected.directory_identity
            ) as binding,
        ):
            if binding is None:
                # A checkout may make the lexical pending namespace
                # unreachable after an exact entry was already moved.  That
                # closes authority over its old child names, but not over the
                # deterministic private destinations derived from the durable
                # cleanup binding.
                retire_detached_names()
                return
            temporary_snapshot = None
            if expected.temporary_owned:
                observed_temporary = _cleanup_bound_snapshot(
                    binding.descriptor, expected.temporary_name
                )
                # Absence means an earlier cleanup attempt succeeded. A
                # different occupant is outside the persisted authority and
                # is preserved while the transaction's marker is retired.
                if observed_temporary == expected.temporary_snapshot:
                    temporary_snapshot = observed_temporary
            _validate_bound_directory(binding)

            # Only the public identity seen by the journal decision is in
            # scope, and only the private answer's exact hard link proves the
            # public name belongs to this transaction. If it changed, preserve
            # the replacement untouched. Public and private can be hard links;
            # retiring either one legitimately changes the other's ctime. When
            # both exact links remain, retire the private name while the moved
            # public link is still open and reversible. That held descriptor
            # prevents inode reuse and narrowly authorizes the sibling's new
            # ctime without weakening a later cleanup retry.
            public_answer = expected.public_answer_snapshot
            paired_private_snapshot = expected.temporary_snapshot
            paired_answer = (
                paired_private_snapshot is not None
                and public_answer == paired_private_snapshot
            )
            if (
                not paired_answer
                and expected.temporary_snapshot is None
                and expected.temporary_retirement_snapshot is not None
                and public_answer is not None
                and public_answer[0][:4]
                == expected.temporary_retirement_snapshot[0][:4]
                and public_answer[1]
                == expected.temporary_retirement_snapshot[1]
            ):
                paired_private_snapshot = (
                    expected.temporary_retirement_snapshot
                )
                paired_answer = True

            if private_retirement_snapshot is not None and not paired_answer:
                _retire_detached_entry(
                    expected.directory_identity,
                    expected.temporary_name,
                    private_retirement_snapshot[0][:4],
                    private_retirement_snapshot[1],
                    expected_ctime_ns=private_retirement_snapshot[0][4],
                )

            def retire_paired_temporary() -> bool:
                nonlocal temporary_snapshot
                persisted = paired_private_snapshot
                if persisted is None:
                    return True
                _retire_detached_entry(
                    expected.directory_identity,
                    expected.temporary_name,
                    persisted[0][:4],
                    persisted[1],
                    expected_ctime_ns=persisted[0][4],
                )
                observed = _cleanup_bound_snapshot(
                    binding.descriptor, expected.temporary_name
                )
                if observed is None:
                    temporary_snapshot = None
                    return True
                if (
                    observed[0][:4] != persisted[0][:4]
                    or observed[1] != persisted[1]
                ):
                    # While the exact public inode is held open in the
                    # retirement store, a different tuple proves the old
                    # private name is absent. Preserve that replacement.
                    temporary_snapshot = None
                    return True
                retired = _retire_exact_entry(
                    binding,
                    expected.temporary_name,
                    observed[0][:4],
                    observed[1],
                    expected_ctime_ns=observed[0][4],
                    retirement_ctime_ns=persisted[0][4],
                )
                current = _cleanup_bound_snapshot(
                    binding.descriptor, expected.temporary_name
                )
                if retired or current != observed:
                    temporary_snapshot = None
                    return True
                return False

            if public_answer is not None:
                retired_public = _retire_exact_entry(
                    binding,
                    target.name,
                    public_answer[0][:4],
                    public_answer[1],
                    expected_ctime_ns=public_answer[0][4],
                    confirm_while_reversible=(
                        retire_paired_temporary if paired_answer else None
                    ),
                    leave_retired_on_confirmation_failure=True,
                )
                if (
                    not retired_public
                    and _cleanup_bound_snapshot(
                        binding.descriptor, target.name
                    )
                    == public_answer
                ):
                    raise DataError(
                        "Could not retire bound-write public answer: "
                        f"{target.name}"
                    )
                if (
                    retired_public
                    and paired_answer
                    and temporary_snapshot is not None
                ):
                    current_temporary = _cleanup_bound_snapshot(
                        binding.descriptor, expected.temporary_name
                    )
                    if (
                        current_temporary is not None
                        and current_temporary[0][:4]
                        == temporary_snapshot[0][:4]
                        and current_temporary[1] == temporary_snapshot[1]
                    ):
                        raise DataError(
                            "Bound-write private answer remained after its "
                            f"paired public retirement: {expected.temporary_name}"
                        )
                    temporary_snapshot = None

            if temporary_snapshot is not None and not paired_answer:
                retired_temporary = _retire_exact_entry(
                    binding,
                    expected.temporary_name,
                    temporary_snapshot[0][:4],
                    temporary_snapshot[1],
                    expected_ctime_ns=temporary_snapshot[0][4],
                )
                if (
                    not retired_temporary
                    and _cleanup_bound_snapshot(
                        binding.descriptor, expected.temporary_name
                    )
                    == temporary_snapshot
                ):
                    raise DataError(
                        f"Could not retire bound-write private answer: "
                        f"{expected.temporary_name}"
                    )

            # A different current marker is a replacement outside the durable
            # intent. Preserve it exactly as we preserve a replaced public or
            # private name; only the old bound marker is considered gone.
            _retire_receipted_marker(
                binding,
                directory_identity=expected.directory_identity,
                name=expected.marker_name,
                expected_state=expected.marker_state,
                expected_revision=expected.marker_revision,
            )
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
            if (
                temporary_snapshot is not None
                and _cleanup_bound_snapshot(
                    binding.descriptor, expected.temporary_name
                )
                == temporary_snapshot
                or _cleanup_bound_entry_state(
                    binding.descriptor, expected.marker_name
                )
                == expected.marker_state
            ):
                raise DataError(
                    f"Bound-write evidence remained after cleanup: {target}"
                )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(
            f"Could not retire bound-write evidence for {target}: "
            f"{exc.strerror or exc}"
        ) from exc


def read_bytes_bound_snapshot(
    path: Path,
) -> tuple[_EntryState, str, bytes]:
    """Read bytes plus their stable state while holding the guarded-read lock."""
    target = Path(path).absolute()
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=False) as binding,
        ):
            _recover_bound_target_under_lock(binding, target)
            _state, _revision, payload = _read_bound_bytes(
                binding.descriptor, target.name
            )
            _validate_bound_directory(binding)
            return _state, _revision, payload
    except FileNotFoundError:
        raise
    except DataError:
        raise
    except OSError as exc:
        raise DataError(
            f"Could not safely read {target}: {exc.strerror or exc}"
        ) from exc


def read_bytes_bound(path: Path) -> bytes:
    """Read one direct file while excluding/recovering guarded publication."""
    _state, _revision, payload = read_bytes_bound_snapshot(path)
    return payload


def read_text_bound(path: Path) -> str:
    """UTF-8 text form of :func:`read_bytes_bound`."""
    target = Path(path).absolute()
    try:
        return read_bytes_bound(target).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise DataError(f"Could not safely read {target}: {exc}") from exc


def atomic_unlink_bound(path: Path, *, expected_revision: str) -> None:
    """Retire one exact regular file without deleting a raced replacement.

    The CAS lock serializes janki's bound writers. The bound directory and
    private retirement machinery close the remaining POSIX rename/unlink seam:
    if the named entry no longer has ``expected_revision``, its replacement is
    left at the original path.
    """
    target = Path(path).absolute()
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=False) as binding,
        ):
            _recover_bound_target_under_lock(binding, target)
            state, revision, _payload = _read_bound_bytes(
                binding.descriptor, target.name
            )
            if revision != expected_revision:
                raise DataError(f"Bound target changed content: {target}")
            if not _retire_exact_entry(
                binding,
                target.name,
                state,
                revision,
            ):
                raise DataError(f"Bound target changed before unlink: {target}")
            _validate_bound_directory(binding)
    except FileNotFoundError as exc:
        raise DataError(f"Bound target changed before unlink: {target}") from exc
    except DataError:
        raise
    except OSError as exc:
        raise DataError(
            f"Could not safely unlink {target}: {exc.strerror or exc}"
        ) from exc


def _allocate_bound_temporary(
    binding: _DirectoryBinding,
    target_name: str,
    mode: int,
) -> tuple[int, str, tuple[int, int]]:
    """Create a temp whose final private name persists its inode identity."""
    for _attempt in range(20):
        token = secrets.token_hex(8)
        allocating = f".{target_name}.{token}.janki-allocating"
        try:
            descriptor = os.open(
                allocating,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode,
                dir_fd=binding.descriptor,
            )
        except FileExistsError:
            continue
        created = os.fstat(descriptor)
        identity = created.st_dev, created.st_ino
        name = (
            f".{target_name}.{token}.{identity[0]:x}-{identity[1]:x}.tmp"
        )
        try:
            _rename_entry_exclusive(binding.descriptor, allocating, name)
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
            named = os.stat(
                name, dir_fd=binding.descriptor, follow_symlinks=False
            )
            if _entry_state(named)[:2] != identity:
                raise DataError(
                    f"Bound-write temp changed during allocation: {name}"
                )
        except BaseException:
            os.close(descriptor)
            for candidate in (allocating, name):
                _retire_created_entry(binding, candidate, identity)
            raise
        return descriptor, name, identity
    raise DataError(f"Could not allocate a temporary file for {target_name}")


def _commit_bound_target(
    binding: _DirectoryBinding,
    temporary_name: str,
    target: Path,
    *,
    bound_state: _EntryState | None,
    expected_revision: str | None,
    expected_identity: tuple[int, int] | None,
    expected_absent: bool,
    prepared_marker: tuple[str, _EntryState, str] | None = None,
) -> None:
    """Publish a temp without overwriting a state the caller did not bind."""
    directory_fd = binding.descriptor
    target_name = target.name
    generated_state, generated_revision = _read_bound_entry(
        directory_fd, temporary_name
    )
    temporary_exists = True
    retain_temporary = False
    published_hard_link = False
    try:
        _validate_bound_directory(binding)
        if expected_absent:
            if prepared_marker is None:
                raise DataError(
                    f"Write-once publication for {target} has no recovery marker"
                )
            marker, marker_state, marker_revision = prepared_marker
            if _bound_snapshot(directory_fd, marker) != (
                marker_state,
                marker_revision,
            ):
                raise DataError(
                    f"Bound-write marker changed before publication: {marker}"
                )
            retain_temporary = True
            try:
                _link_entry_exclusive(
                    directory_fd, temporary_name, target_name
                )
            except FileExistsError:
                raise DataError(
                    f"Bound target changed before replace: {target}"
                ) from None
            published_hard_link = True
            os.fsync(directory_fd)
            # If the parent was detached during publication, the captured
            # bytes remain in that held directory but the caller must not
            # journal a path that does not name them.
            _validate_bound_directory(binding)
            if _bound_snapshot(directory_fd, target_name) != (
                generated_state,
                generated_revision,
            ):
                raise DataError(
                    f"Could not verify write-once publication for {target}"
                )
            _move_cas_marker(
                binding,
                marker,
                _CAS_VALIDATED_SUFFIX,
                expected_state=marker_state,
                expected_revision=marker_revision,
            )
            _cleanup_terminal_cas_markers_under_lock(binding, target)
            temporary_exists = False
            published_hard_link = False
            retain_temporary = False
            os.fsync(directory_fd)
            _validate_bound_directory(binding)
            return

        guarded = expected_revision is not None or expected_identity is not None
        if not guarded:
            os.replace(
                temporary_name,
                target_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_exists = False
            os.fsync(directory_fd)
            _validate_bound_directory(binding)
            return

        if bound_state is None:
            raise DataError(f"Bound target changed before replace: {target}")
        observed_state, observed_revision = _read_bound_entry(
            directory_fd, target_name
        )
        if observed_state != bound_state:
            raise DataError(f"Bound target changed content: {target}")
        if expected_identity is not None and observed_state[:2] != expected_identity:
            raise DataError(f"Bound target changed identity: {target}")
        if (
            expected_revision is not None
            and observed_revision != expected_revision
        ):
            raise DataError(f"Bound target changed content: {target}")
        marker, marker_state, marker_revision = _create_cas_marker(
            binding,
            target_name,
            temporary_name,
            temporary_identity=generated_state[:2],
            expected_absent=False,
            expected_state=observed_state,
            expected_revision=observed_revision,
            generated_state=generated_state,
            generated_revision=generated_revision,
        )
        # Once the prepared marker exists, retain the generated temp on every
        # failure. Recovery needs both names.
        retain_temporary = True
        try:
            if _bound_snapshot(directory_fd, marker) != (
                marker_state,
                marker_revision,
            ):
                raise DataError(
                    f"Bound-write marker changed before publication: {marker}"
                )
            _exchange_entries(directory_fd, temporary_name, target_name)
            os.fsync(directory_fd)
            _validate_bound_directory(binding)
            displaced_state, displaced_revision = _read_bound_entry(
                directory_fd, temporary_name
            )
            live_state, live_revision = _read_bound_entry(
                directory_fd, target_name
            )
            if (live_state, live_revision) != (
                generated_state,
                generated_revision,
            ):
                raise DataError(f"Bound target changed before replace: {target}")
            if bound_state is None or displaced_state != bound_state:
                raise DataError(f"Bound target changed content: {target}")
            if (
                expected_identity is not None
                and displaced_state[:2] != expected_identity
            ):
                raise DataError(f"Bound target changed identity: {target}")
            if (
                expected_revision is not None
                and displaced_revision != expected_revision
            ):
                raise DataError(f"Bound target changed content: {target}")
            _move_cas_marker(
                binding,
                marker,
                _CAS_VALIDATED_SUFFIX,
                expected_state=marker_state,
                expected_revision=marker_revision,
            )
        except BaseException as cause:
            retain_temporary = True
            try:
                _recover_bound_target_under_lock(binding, target)
                temporary_exists = False
            except BaseException as recovery_error:
                raise DataError(
                    f"{cause}. Could not safely recover {target}; the active "
                    f"transaction and both names were retained: {recovery_error}"
                ) from cause
            raise
        # The validated marker is the crash-durable commit point. Anything
        # before its directory fsync rolls back on the next bound read;
        # anything after keeps the generated public bytes. Terminal cleanup
        # retains that marker whenever it cannot also retire the exact temp.
        _cleanup_terminal_cas_markers_under_lock(binding, target)
        temporary_exists = False
        retain_temporary = False
        os.fsync(directory_fd)
        _validate_bound_directory(binding)
    finally:
        if (
            temporary_exists
            and not retain_temporary
            and not published_hard_link
            and not _has_active_cas_marker(binding, target_name)
        ):
            _retire_exact_entry(
                binding,
                temporary_name,
                generated_state,
                generated_revision,
            )


def _capture_bytes_bound(path: Path, data: bytes) -> _BoundFileReceipt:
    """Publish one write-once paid reply while retaining its terminal WAL.

    The ordinary bound writer finalizes its WAL before returning.  A paid
    reply needs one additional durability seam: its journal receipt cannot be
    written until publication has succeeded, but publication must remain
    operation-bound until that receipt is durable.  This writer therefore
    retires the private hard link, captures the public file's final ctime, and
    leaves the validated marker in place for :func:`_finalize_bound_capture`.
    """
    target = Path(path).absolute()
    generated_revision = hashlib.sha256(data).hexdigest()
    temporary_name = ""
    temporary_identity: tuple[int, int] | None = None
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=True) as binding,
            contextlib.ExitStack() as cleanup,
        ):
            directory_fd = binding.descriptor

            # Never settle an earlier capture merely because a callback was
            # invoked again.  Until its journal receipt exists, the marker is
            # the only durable proof that the lexical public name is ours.
            if _has_active_cas_marker(binding, target.name):
                raise DataError(
                    f"An incomplete bound write already exists for {target}"
                )
            try:
                before = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                before = None
            if before is not None:
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                    raise DataError(
                        f"Refusing to replace non-regular target {target}"
                    )
                raise DataError(f"Bound target changed before replace: {target}")

            current_umask = os.umask(0)
            os.umask(current_umask)
            mode = 0o666 & ~current_umask
            descriptor, temporary_name, temporary_identity = (
                _allocate_bound_temporary(binding, target.name, mode)
            )

            def cleanup_temporary() -> None:
                if temporary_name and not _has_active_cas_marker(
                    binding, target.name
                ):
                    _retire_created_entry(
                        binding, temporary_name, temporary_identity
                    )

            cleanup.callback(cleanup_temporary)
            marker = ""
            marker_state: _EntryState | None = None
            marker_revision = ""
            with os.fdopen(descriptor, "wb") as handle:
                marker, marker_state, marker_revision = _create_cas_marker(
                    binding,
                    target.name,
                    temporary_name,
                    temporary_identity=temporary_identity,
                    expected_absent=True,
                    expected_state=None,
                    expected_revision=None,
                    generated_state=None,
                    generated_revision=generated_revision,
                    retirement_snapshot_required=True,
                )
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            generated_state, observed_revision = _read_bound_entry(
                directory_fd, temporary_name
            )
            if (
                temporary_identity is None
                or generated_state[:2] != temporary_identity
                or observed_revision != generated_revision
            ):
                raise DataError(
                    f"Could not verify operation-bound capture for {target}"
                )
            if _bound_snapshot(directory_fd, marker) != (
                marker_state,
                marker_revision,
            ):
                raise DataError(
                    f"Bound-write marker changed before publication: {marker}"
                )
            try:
                _link_entry_exclusive(directory_fd, temporary_name, target.name)
            except FileExistsError:
                raise DataError(
                    f"Bound target changed before replace: {target}"
                ) from None
            os.fsync(directory_fd)
            _validate_bound_directory(binding)
            if _bound_snapshot(directory_fd, target.name) != (
                generated_state,
                generated_revision,
            ):
                raise DataError(
                    f"Could not verify write-once publication for {target}"
                )
            temporary_retirement_snapshot = _cleanup_bound_snapshot(
                directory_fd, temporary_name
            )
            public_retirement_snapshot = _cleanup_bound_snapshot(
                directory_fd, target.name
            )
            if (
                temporary_retirement_snapshot is None
                or temporary_retirement_snapshot != public_retirement_snapshot
            ):
                raise DataError(
                    f"Could not bind private retirement for captured answer "
                    f"{target}"
                )
            marker_state, marker_revision = _append_cas_retirement_snapshot(
                binding,
                marker,
                target.name,
                expected_state=marker_state,
                expected_revision=marker_revision,
                temporary_snapshot=temporary_retirement_snapshot,
            )
            marker = _move_cas_marker(
                binding,
                marker,
                _CAS_VALIDATED_SUFFIX,
                expected_state=marker_state,
                expected_revision=marker_revision,
            )

            def public_still_holds_answer() -> bool:
                current = _bound_snapshot(directory_fd, target.name)
                _validate_bound_directory(binding)
                return current == (generated_state, generated_revision)

            if not _retire_exact_entry(
                binding,
                temporary_name,
                temporary_retirement_snapshot[0][:4],
                temporary_retirement_snapshot[1],
                expected_ctime_ns=temporary_retirement_snapshot[0][4],
                confirm_while_reversible=public_still_holds_answer,
            ):
                raise DataError(
                    f"Committed write-once bound write for {target} cannot "
                    "retire its private answer while the public name is changing"
                )
            temporary_name = ""
            entry_state, public_revision = _read_cleanup_bound_entry(
                directory_fd, target.name
            )
            if (
                entry_state[:2] != generated_state[:2]
                or public_revision != generated_revision
            ):
                raise DataError(
                    f"Committed write-once bound write for {target} lost its "
                    "public answer before receipt"
                )
            # The marker must still be terminal and exact when the receipt is
            # handed to the journal.  Its retirement is deliberately later.
            marker_entry_state, final_marker_revision = (
                _read_cleanup_bound_entry(directory_fd, marker)
            )
            if (
                not marker.endswith(_CAS_VALIDATED_SUFFIX)
                or marker_entry_state[:4] != marker_state
                or final_marker_revision != marker_revision
            ):
                raise DataError(
                    f"Bound-write marker changed before receipt: {marker}"
                )
            parent = os.fstat(directory_fd)
            _validate_bound_directory(binding)
            return _BoundFileReceipt(
                target=target,
                directory_identity=(parent.st_dev, parent.st_ino),
                entry_state=entry_state,
                content_sha256=public_revision,
                marker_name=marker,
                marker_state=marker_entry_state,
                marker_revision=final_marker_revision,
            )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not write {target}: {exc.strerror or exc}") from exc


def _finalize_bound_capture(expected: _BoundFileReceipt) -> None:
    """Retire a capture's terminal marker after its journal receipt is durable."""
    target = expected.target

    def retire_detached_marker() -> None:
        _retire_detached_entry(
            expected.directory_identity,
            expected.marker_name,
            expected.marker_state[:4],
            expected.marker_revision,
            expected_ctime_ns=expected.marker_state[4],
        )

    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_cleanup_directory(
                target.parent, expected.directory_identity
            ) as binding,
        ):
            if binding is None:
                retire_detached_marker()
                return
            _retire_receipted_marker(
                binding,
                directory_identity=expected.directory_identity,
                name=expected.marker_name,
                expected_state=expected.marker_state,
                expected_revision=expected.marker_revision,
            )
            os.fsync(binding.descriptor)
            _validate_bound_directory(binding)
    except DataError:
        raise
    except OSError as exc:
        raise DataError(
            f"Could not finalize captured answer {target}: {exc.strerror or exc}"
        ) from exc


def atomic_write_bytes_bound(
    path: Path,
    data: bytes,
    *,
    expected_revision: str | None = None,
    expected_identity: tuple[int, int] | None = None,
    expected_absent: bool = False,
    expected_directory_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically replace one regular directory entry without following links.

    Paid media staging and canonical promotion deliberately do *not* inherit
    the ordinary writer's symlink-through behavior: a malicious or accidental
    symlink would otherwise overwrite a different live clip. The directory
    descriptor also binds the temp and replace operations to the same real
    directory entry. ``expected_revision`` and ``expected_identity`` bind an
    existing target selected by an earlier plan. ``expected_absent`` makes a
    recovery artifact write-once:
    a reply already captured under that operation ID is evidence, never a
    target for a later callback to replace. ``expected_directory_identity``
    refuses before recovery or temporary allocation unless the opened parent
    is the directory an earlier plan bound.
    """
    if expected_absent and (
        expected_revision is not None or expected_identity is not None
    ):
        raise DataError("A bound write cannot expect content and absence together")
    target = Path(path).absolute()
    temporary_name = ""
    temporary_identity: tuple[int, int] | None = None
    prepared_marker: tuple[str, _EntryState, str] | None = None
    generated_revision = hashlib.sha256(data).hexdigest()
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(
                target.parent,
                create=expected_directory_identity is None,
            ) as binding,
            contextlib.ExitStack() as cleanup,
        ):
            directory_fd = binding.descriptor
            parent = os.fstat(directory_fd)
            if expected_directory_identity is not None and (
                _directory_identity(parent) != expected_directory_identity
            ):
                raise DataError(
                    f"Bound target directory changed: {target.parent}"
                )
            _recover_bound_target_under_lock(binding, target)

            def cleanup_temporary() -> None:
                if temporary_name and not _has_active_cas_marker(
                    binding, target.name
                ):
                    _retire_created_entry(
                        binding,
                        temporary_name,
                        temporary_identity,
                    )

            cleanup.callback(cleanup_temporary)
            try:
                before = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                bound_state: _EntryState | None = None
                current_umask = os.umask(0)
                os.umask(current_umask)
                mode = 0o666 & ~current_umask
            else:
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                    raise DataError(
                        f"Refusing to replace non-regular target {target}"
                    )
                if expected_absent:
                    raise DataError(
                        f"Bound target changed before replace: {target}"
                    )
                bound_state = _entry_state(before)
                if (
                    expected_identity is not None
                    and bound_state[:2] != expected_identity
                ):
                    raise DataError(f"Bound target changed identity: {target}")
                if expected_revision is not None:
                    current_state, current_revision, _payload = _read_bound_bytes(
                        directory_fd, target.name
                    )
                    if (
                        current_state != bound_state
                        or current_revision != expected_revision
                    ):
                        raise DataError(f"Bound target changed content: {target}")
                mode = stat.S_IMODE(before.st_mode)
            descriptor, temporary_name, temporary_identity = (
                _allocate_bound_temporary(binding, target.name, mode)
            )
            with os.fdopen(descriptor, "wb") as handle:
                if expected_absent:
                    prepared_marker = _create_cas_marker(
                        binding,
                        target.name,
                        temporary_name,
                        temporary_identity=temporary_identity,
                        expected_absent=True,
                        expected_state=None,
                        expected_revision=None,
                        generated_state=None,
                        generated_revision=generated_revision,
                    )
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            commit_name = temporary_name
            temporary_name = ""
            _commit_bound_target(
                binding,
                commit_name,
                target,
                bound_state=bound_state,
                expected_revision=expected_revision,
                expected_identity=expected_identity,
                expected_absent=expected_absent,
                prepared_marker=prepared_marker,
            )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not write {target}: {exc.strerror or exc}") from exc


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
    temporary_name = ""
    temporary_identity: tuple[int, int] | None = None
    prepared_marker: tuple[str, _EntryState, str] | None = None
    generated_revision = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        with (
            exclusive_path_lock(_cas_lock_path(target)),
            _open_bound_directory(target.parent, create=True) as binding,
            contextlib.ExitStack() as cleanup,
        ):
            directory_fd = binding.descriptor
            _recover_bound_target_under_lock(binding, target)

            def cleanup_temporary() -> None:
                if temporary_name and not _has_active_cas_marker(
                    binding, target.name
                ):
                    _retire_created_entry(
                        binding,
                        temporary_name,
                        temporary_identity,
                    )

            cleanup.callback(cleanup_temporary)
            try:
                details = os.stat(
                    target.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                current_umask = os.umask(0)
                os.umask(current_umask)
                mode = 0o666 & ~current_umask
            else:
                if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(
                    details.st_mode
                ):
                    raise DataError(
                        f"Refusing to replace non-regular target {target}"
                    )
                if expected_absent:
                    raise DataError(
                        f"Bound target changed before replace: {target}"
                    )
                mode = stat.S_IMODE(details.st_mode)
            descriptor, temporary_name, temporary_identity = (
                _allocate_bound_temporary(binding, target.name, mode)
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                if expected_absent:
                    prepared_marker = _create_cas_marker(
                        binding,
                        target.name,
                        temporary_name,
                        temporary_identity=temporary_identity,
                        expected_absent=True,
                        expected_state=None,
                        expected_revision=None,
                        generated_state=None,
                        generated_revision=generated_revision,
                    )
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            bound_state: _EntryState | None = None
            try:
                current_fd = os.open(
                    target.name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
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
                    bound_state = _entry_state(current_details)
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
                    if _entry_state(after_hash) != bound_state:
                        raise DataError(f"Bound target changed content: {target}")
                finally:
                    os.close(current_fd)
            commit_name = temporary_name
            temporary_name = ""
            _commit_bound_target(
                binding,
                commit_name,
                target,
                bound_state=bound_state,
                expected_revision=expected_revision,
                expected_identity=expected_identity,
                expected_absent=expected_absent,
                prepared_marker=prepared_marker,
            )
    except DataError:
        raise
    except OSError as exc:
        raise DataError(f"Could not write {target}: {exc.strerror or exc}") from exc


def _parse_structured_text(
    path: Path,
    text: str,
    *,
    yaml_loader: type[yaml.SafeLoader] = yaml.SafeLoader,
) -> Any:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            return yaml.load(text, Loader=yaml_loader)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DataError(f"Could not parse {path}: {exc}") from exc
    raise DataError(f"Unsupported file type for {path}; expected JSON or YAML")


def load_structured(
    path: Path,
    *,
    yaml_loader: type[yaml.SafeLoader] = yaml.SafeLoader,
) -> Any:
    try:
        text = read_text_bound(path)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataError(f"Could not parse {path}: {exc}") from exc
    return _parse_structured_text(path, text, yaml_loader=yaml_loader)


def _records_from_text(path: Path, text: str) -> list[VocabularyRecord]:
    """Parse vocabulary records from bytes already bound to one read."""
    data = _parse_structured_text(path, text)
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


def load_records(path: Path) -> list[VocabularyRecord]:
    try:
        text = read_text_bound(path)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    return _records_from_text(path, text)


def load_records_snapshot(
    path: Path,
) -> tuple[list[VocabularyRecord], RecordsRevision]:
    """Read records and their later-write token from the exact same text.

    A separate ``records_revision`` followed by ``load_records`` can observe
    revision A and records B. If the file returns to A before the guarded
    write, the compare-and-swap accepts a merge built over transient B. The
    revision already owns the exact text; parse that text rather than opening
    the path a second time.

    Absence is a real revision and an empty collection, matching first-write
    callers that previously paired ``records_revision`` with an existence
    check.
    """
    revision = records_revision(path)
    if revision.text is None:
        return [], revision
    return _records_from_text(revision.path, revision.text), revision


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
    with exclusive_path_lock(target):
        save_records_json_locked(target, records, expected=expected)


def save_records_json_locked(
    path: Path,
    records: list[VocabularyRecord],
    *,
    expected: RecordsRevision | None = None,
) -> None:
    """Save records while the caller holds ``exclusive_path_lock(path)``.

    This narrow seam lets a multi-file transaction keep the normalized-record
    lock across its CAS, media publication, and ledger commit. Ordinary callers
    must use :func:`save_records_json`, which acquires the lock itself.
    """
    target = Path(os.path.realpath(path))
    if expected is not None and target != expected.path:
        raise DataError(
            f"Records revision for {expected.path} cannot guard a write to {target}."
        )
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
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
    annotations: dict[str, str] = {}
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
        # user's own data — curated by arrival — but the
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
    *,
    prefer_incoming_by_id: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[VocabularyRecord], dict[str, MergeOutcome]]:
    """Merge an import into stored records without destroying curation.

    Existing wins: an incoming value lands only where the existing field is
    empty (``""``/``[]``/``{}``/``None`` — zero is a value, not a hole). Where
    both sides are non-empty and differ, the existing value is kept and the
    disagreement is reported as a conflict. ``tags`` is a sorted union;
    ``source`` is the existing record's, unconditionally — the first sighting
    sticks and later ones belong in the ledger. Fields named in
    ``prefer_incoming`` falls back to incoming-wins for deliberate refreshes.
    ``prefer_incoming_by_id`` is the narrower form used by reviewed staging:
    only those fields on that one record replace a non-empty value.  It carries
    no authority by itself; :func:`japanese_anki.staging.authorized_field_replacements`
    verifies the old-value fingerprints before a caller passes the result here.

    Returns the merged records sorted by id, plus an outcome map covering
    **exactly the incoming record ids** — existing records the import never
    mentioned are carried through untouched and do not appear. An id the
    import carries twice gets one folded outcome, so nothing it reported on
    the first row is lost.
    """
    prefer = frozenset(validate_prefer_incoming(prefer_incoming))
    per_record = {
        str(record_id): frozenset(validate_prefer_incoming(fields))
        for record_id, fields in (prefer_incoming_by_id or {}).items()
    }
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
        merged, outcome = _merge_one(
            old, new, prefer | per_record.get(new.id, frozenset())
        )
        by_id[new.id] = merged
        seen = outcomes.get(new.id)
        outcomes[new.id] = _combine_outcomes(seen, outcome) if seen else outcome

    return sorted(by_id.values(), key=lambda item: item.id), outcomes
