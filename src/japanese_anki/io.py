from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.errors import JankiError
from japanese_anki.models import ModelError, VocabularyRecord


class DataError(JankiError):
    pass


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


def save_records_json(path: Path, records: list[VocabularyRecord]) -> None:
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


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


def _merge_one(
    old: VocabularyRecord, new: VocabularyRecord, prefer_incoming: frozenset[str]
) -> tuple[VocabularyRecord, MergeOutcome]:
    changes: dict[str, Any] = {}
    filled: list[str] = []
    conflicts: list[tuple[str, Any, Any]] = []

    for name in _CONTENT_FIELDS:
        old_value = getattr(old, name)
        new_value = getattr(new, name)
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
