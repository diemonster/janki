"""Move a deck's inline notes into the normalized file, once and for all.

Inline notes are a dead zone: every writer janki grows from here (enrich, audio,
promote) writes to ``vocabulary.json``, so a record that lives inside a deck
YAML can never gain pitch accent, audio or examples. ``janki migrate-inline``
lifts those notes out — merged into the normalized file under the *same ids*, so
Anki's GUIDs and the review history behind them survive — and leaves the deck
file as what it should have been all along: a filter over shared records.

Three invariants shape this module. All three are about not surprising the
person who typed the command:

* **Every deck resolves to the same notes afterwards.** Not just the deck being
  migrated: the moment its records land in the normalized file, *any other deck
  reading that file* starts resolving them too, and two ``.apkg`` files carrying
  one GUID is a genuine Anki hazard. Such a deck gets ``exclude_ids`` for the
  records it did not have before, and the command says so out loud.
* **Nothing is written until the outcome is known to be right.** The record the
  deck will resolve from the normalized file is compared field by field against
  the record it resolves today, before any file is touched; a disagreement the
  merge cannot resolve (the normalized copy already holds a different
  expression, say) aborts with the fields named.
* **Running it twice is safe.** A deck with no inline notes left has nothing to
  migrate and is not rewritten.

Deck files are edited in place as *text* wherever the change is an addition —
a new key, or a new item on an existing block sequence: YAML comments are
documentation, and re-serializing a file to add one id deletes them. The edited
text is parsed back and compared against the mapping the edit was supposed to
produce; anything short of an exact match falls back to a full re-serialization,
so the shortcut can never produce a file that means something else. That
fallback is reported as a warning rather than taken quietly, because what it
costs is hand-written documentation in a curated file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.io import (
    MERGEABLE_FIELDS,
    DataError,
    MergeOutcome,
    atomic_write_text,
    is_empty,
    load_records,
    load_structured,
    merge_records,
    save_records_json,
)
from japanese_anki.ledger import Ledger
from japanese_anki.models import VocabularyRecord


class MigrateError(JankiError):
    pass


# What the migrated record must still say once it comes from the normalized
# file. Everything except ``tags``, which the merge unions rather than replaces:
# a grown tag list changes a note's tags but not one field value, so it is
# reported rather than refused.
_COMPARED_FIELDS: tuple[str, ...] = tuple(
    item.name for item in fields(VocabularyRecord) if item.name != "tags"
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeckGuard:
    """Another deck that reads the normalized file, and what was done for it."""

    path: Path
    excluded_ids: list[str] = field(default_factory=list)
    # Migrated ids this deck's own ``include_ids`` names: the user asked for
    # them by hand, so they are honored and only reported.
    requested_ids: list[str] = field(default_factory=list)
    # Ids it already resolved whose content the merge changed.
    changed_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MigrationResult:
    deck_path: Path
    normalized_file: Path
    migrated: list[VocabularyRecord] = field(default_factory=list)
    outcomes: dict[str, MergeOutcome] = field(default_factory=dict)
    include_ids: list[str] = field(default_factory=list)
    source_value: str = ""
    added_source: bool = False
    guards: list[DeckGuard] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    # What the ledger actually gained, not how many notes moved. The mutators
    # return a bool precisely so a command can report the truth: a migration
    # after a `status --rebuild` registers nothing new, and saying otherwise
    # would have the transcript claim work that did not happen.
    ledger_added: int = 0
    ledger_sources: int = 0

    @property
    def migrated_ids(self) -> list[str]:
        return [record.id for record in self.migrated]

    @property
    def changed(self) -> bool:
        return bool(self.migrated)


# ---------------------------------------------------------------------------
# Deck YAML rewriting
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}") from exc


def _dump_mapping(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )


def _top_level_block(lines: list[str], key: str) -> tuple[int, int] | None:
    """Half-open line range of a top-level ``key:`` block, or ``None``.

    The block runs from the key's own line to the next line that starts in
    column zero with something other than whitespace — a sibling key, or a
    comment introducing one. A column-zero ``- `` is *not* a sibling: YAML lets
    a block sequence sit at its key's own indent, so ``notes:`` followed by
    unindented items is one block, and treating the first item as the next key
    would leave the items behind when the block is deleted.
    """
    pattern = re.compile(rf"^{re.escape(key)}\s*:")
    item = re.compile(r"^-(\s|$)")
    start: int | None = None
    for index, line in enumerate(lines):
        if start is None:
            if pattern.match(line):
                start = index
            continue
        if line.strip() and not line[:1].isspace() and not item.match(line):
            return start, index
    return (start, len(lines)) if start is not None else None


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _add_key(
    lines: list[str], block: tuple[int, int], indent: int, key: str, value: Any
) -> list[str]:
    """Insert a brand-new ``key:`` at the end of the ``deck:`` mapping."""
    start, end = block
    # After the mapping's last populated line: a dedent to the mapping's own
    # indent closes whatever nested block that line belonged to.
    insert_at = start + 1
    for index in range(start + 1, end):
        if lines[index].strip():
            insert_at = index + 1
    addition = [" " * indent + line for line in _dump_mapping({key: value}).splitlines()]
    edited = list(lines)
    edited[insert_at:insert_at] = addition
    return edited


def _append_to_sequence(
    lines: list[str],
    block: tuple[int, int],
    indent: int,
    key: str,
    current: Any,
    value: Any,
) -> list[str] | None:
    """Append ``value``'s new items to an existing block sequence.

    Adding an id to an ``exclude_ids:`` list *is* an addition, so it belongs on
    the text path like any other: a deck guarded a second time must not pay for
    it with its comments. Returns ``None`` for anything that is not a plain
    block sequence gaining items at the end — a flow list, an empty one, a
    replacement, a reordering — leaving the caller to re-serialize and say so.
    """
    if not isinstance(current, list) or not isinstance(value, list):
        return None
    if value[: len(current)] != current:
        return None  # a rewrite, not an append
    new_items = value[len(current) :]
    if not new_items:
        return None
    start, end = block
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}\s*:")
    index = next((probe for probe in range(start + 1, end) if pattern.match(lines[probe])), None)
    if index is None:
        return None
    # The value ends at the next sibling key. A block sequence may sit at the
    # mapping's own indent (``exclude_ids:`` then ``  - id`` under two-space
    # keys), so indent alone does not close the value — an item line does not.
    stop = end
    for probe in range(index + 1, end):
        line = lines[probe]
        if not line.strip():
            continue
        if _indent_of(line) <= indent and not line.lstrip(" ").startswith("- "):
            stop = probe
            break
    items = [
        probe for probe in range(index + 1, stop) if lines[probe].lstrip(" ").startswith("- ")
    ]
    if not items or len(items) != len(current):
        # An empty or flow-style list, or items spanning more than a line each:
        # nothing here can say where the last one ends.
        return None
    item_indent = " " * _indent_of(lines[items[-1]])
    addition: list[str] = []
    for item in new_items:
        dumped = _dump_mapping({key: [item]}).splitlines()
        if len(dumped) != 2:  # an item the dumper wrapped or nested
            return None
        addition.append(item_indent + dumped[1])
    edited = list(lines)
    edited[items[-1] + 1 : items[-1] + 1] = addition
    return edited


def _edit_deck_text(
    text: str,
    deck_config: dict[str, Any],
    updates: dict[str, Any],
    drop_keys: tuple[str, ...],
) -> str | None:
    """Apply ``updates`` to the ``deck:`` mapping and delete ``drop_keys``.

    A line-level edit, so every other byte — comments, quoting, blank lines —
    survives. Returns ``None`` when the file's shape is not the plain one this
    handles, leaving the caller to re-serialize instead.
    """
    lines = text.splitlines()
    for key, value in updates.items():
        block = _top_level_block(lines, "deck")
        if block is None:
            return None
        start, end = block
        body = [line for line in lines[start + 1 : end] if line.strip()]
        if not body or any(not line[:1].isspace() for line in body):
            return None
        indent = min(_indent_of(line) for line in body)
        edited = (
            _append_to_sequence(lines, block, indent, key, deck_config[key], value)
            if key in deck_config
            else _add_key(lines, block, indent, key, value)
        )
        if edited is None:
            return None
        lines = edited

    for key in drop_keys:
        dropped = _top_level_block(lines, key)
        if dropped is not None:
            del lines[dropped[0] : dropped[1]]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def rewrite_deck_file(
    path: Path,
    raw: dict[str, Any],
    updates: dict[str, Any],
    drop_keys: tuple[str, ...] = (),
) -> bool:
    """Write ``path`` with ``updates`` in its deck mapping and ``drop_keys`` gone.

    The in-place text edit is used only when it can be *proved* right: the
    result is parsed and must equal, key for key, the mapping this was meant to
    produce. Anything else falls back to re-serializing the whole document,
    which is correct but loses every comment, quote style and blank line in it.

    Returns whether that fallback fired, so the caller can say so out loud: the
    file is curated by hand, and losing its documentation silently is how a
    second migration quietly deletes work nobody asked it to touch.
    """
    deck_config = raw.get("deck") or {}
    expected = {key: value for key, value in raw.items() if key not in drop_keys}
    expected["deck"] = {**deck_config, **updates}

    edited = _edit_deck_text(_read_text(path), deck_config, updates, drop_keys)
    if edited is not None:
        try:
            if yaml.safe_load(edited) != expected:
                edited = None
        except yaml.YAMLError:
            edited = None
    atomic_write_text(path, edited if edited is not None else _dump_mapping(expected))
    return edited is None


# ---------------------------------------------------------------------------
# The migration
# ---------------------------------------------------------------------------


def _repo_path(path: Path, root: Path) -> str:
    """``path`` as a human standing in the project root would type it."""
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:  # pragma: no cover - different drives, Windows only
        return str(path)


def _deck_files(config: ProjectConfig) -> list[Path]:
    return sorted([*config.deck_dir.glob("*.yaml"), *config.deck_dir.glob("*.yml")])


def _inline_ids(raw: dict[str, Any]) -> list[str]:
    """The record id of every inline note, in file order, without duplicates.

    A note with no ``id:`` is not skipped: the deck already builds it under the
    id minted from its expression and reading, and that id — not the absence of
    one — is what its GUID was derived from. Leaving such a note behind would
    also leave the deck with a ``notes:`` section this command promises to
    remove.
    """
    notes = raw.get("notes") or []
    if not isinstance(notes, list):  # pragma: no cover - resolve_deck_records refuses first
        raise DataError("The notes section must be a list")
    ordered: list[str] = []
    for item in notes:
        if not isinstance(item, dict):  # pragma: no cover - likewise
            raise DataError("Each note must be a mapping")
        record_id = str(item.get("id", "")).strip() or VocabularyRecord.from_dict(item).id
        if record_id not in ordered:
            ordered.append(record_id)
    return ordered


def _deck_source_path(deck_path: Path, deck_config: dict[str, Any]) -> Path | None:
    source_value = deck_config.get("source")
    if not source_value:
        return None
    return (deck_path.parent / str(source_value)).resolve()


def _check_merged_faithfully(
    merged: dict[str, VocabularyRecord],
    migrated: list[VocabularyRecord],
    deck_path: Path,
    normalized_file: Path,
) -> list[str]:
    """Refuse a migration that would overrule the note the deck exports today.

    Where the normalized file already holds a migrated id, the merge takes the
    note's value for every content field the note actually fills — but never for
    ``expression``, ``reading`` or ``source``, which M1.1 protects. A
    disagreement there is two copies of one record that do not agree on what
    the record *is*, and migrating would quietly change the deck's cards; only a
    human can say which copy is right.

    Fields the note left empty are a different story: the normalized copy fills
    them, which is a fill, not an overrule — the whole reason for migrating.
    Those are returned as notes to print, along with tags gained the same way.
    """
    gains: list[str] = []
    for record in migrated:
        after = merged[record.id]
        overruled: list[str] = []
        filled: list[str] = []
        for name in _COMPARED_FIELDS:
            note_value = getattr(record, name)
            if getattr(after, name) == note_value:
                continue
            (filled if is_empty(note_value) else overruled).append(name)
        if overruled:
            raise MigrateError(
                f"{record.id} would not survive the migration unchanged: "
                f"{normalized_file} already holds it and the merge keeps its "
                f"{', '.join(overruled)} rather than the note's. Migrating now "
                f"would change what {deck_path} exports. Reconcile the two copies "
                "by hand (they are the same record under one id), then re-run."
            )
        filled.extend(f"tag {tag}" for tag in sorted(set(after.tags) - set(record.tags)))
        if filled:
            gains.append(
                f"{record.id} gains {', '.join(filled)} from the copy already in "
                f"{normalized_file}, so its card says more than it did"
            )
    return gains


def _guard_other_decks(
    config: ProjectConfig,
    deck_path: Path,
    normalized_file: Path,
    migrated_ids: list[str],
    merged: dict[str, VocabularyRecord],
) -> tuple[list[DeckGuard], list[str]]:
    """Work out what each *other* deck reading the normalized file needs.

    A deck that reads the file the migrated records are about to land in would
    start resolving them — silently gaining notes whose GUIDs the migrated deck
    also emits. Every migrated id it does not already resolve is added to its
    ``exclude_ids``.

    A deck with a non-empty ``include_ids`` gets none: ``resolve_deck_records``
    applies ``include_ids`` first, so its membership is already closed and every
    id excluded from it would be dead config — one line per record migrated from
    anywhere else, forever, in a file a human reads. What such a deck does still
    get is the report: an id its own ``include_ids`` names is one it asked for
    by name and is about to start exporting.
    """
    guards: list[DeckGuard] = []
    warnings: list[str] = []
    for other in _deck_files(config):
        if other.resolve() == deck_path:
            continue
        try:
            other_config, other_records = resolve_deck_records(other)
        except JankiError as exc:
            warnings.append(
                f"could not read {other}, so this migration cannot promise its "
                f"contents are unchanged: {exc}"
            )
            continue
        if _deck_source_path(other, other_config) != normalized_file:
            continue
        have = {record.id: record for record in other_records}
        requested = {str(value) for value in other_config.get("include_ids") or []}
        guard = DeckGuard(
            path=other,
            excluded_ids=[]
            if requested
            else [record_id for record_id in migrated_ids if record_id not in have],
            requested_ids=[
                record_id
                for record_id in migrated_ids
                if record_id not in have and record_id in requested
            ],
            changed_ids=[
                record_id
                for record_id in migrated_ids
                if record_id in have and merged[record_id] != have[record_id]
            ],
        )
        if guard.excluded_ids or guard.requested_ids or guard.changed_ids:
            guards.append(guard)
    return guards, warnings


def migrate_inline(deck_path: Path, config: ProjectConfig, book: Ledger) -> MigrationResult:
    """Move ``deck_path``'s inline notes into the normalized file.

    Registers each migrated record in ``book`` (in memory — the caller saves the
    ledger once) and returns what happened. A deck with no inline notes is a
    no-op: nothing is written and the result reports nothing migrated.
    """
    deck_path = Path(deck_path).resolve()
    normalized_file = config.normalized_file.resolve()

    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    # Resolves and validates the deck exactly as a build does, so the records
    # below are the ones this deck exports today, filters included.
    deck_config, before = resolve_deck_records(deck_path)

    inline_ids = _inline_ids(raw)
    if not inline_ids:
        return MigrationResult(deck_path=deck_path, normalized_file=normalized_file)

    wanted = set(inline_ids)
    by_id = {record.id: record for record in before}
    filtered_out = [record_id for record_id in inline_ids if record_id not in by_id]
    if filtered_out:
        raise MigrateError(
            f"{deck_path} has inline note(s) its own filters exclude: "
            f"{', '.join(filtered_out)}. Migrating them would put records into "
            f"{normalized_file} that this deck deliberately does not export. "
            "Delete those notes or widen the filter, then re-run."
        )
    migrated = [record for record in before if record.id in wanted]

    source_path = _deck_source_path(deck_path, deck_config)
    if source_path is not None and source_path != normalized_file:
        raise MigrateError(
            f"{deck_path} reads its records from {source_path}, but migration "
            f"writes to {normalized_file}; the migrated notes would land in a "
            "file this deck never reads. Point the deck at the normalized file "
            "first, or migrate a deck that already does."
        )
    added_source = source_path is None
    source_value = (
        _repo_path(normalized_file, deck_path.parent)
        if added_source
        else str(deck_config.get("source"))
    )

    existing = load_records(normalized_file) if normalized_file.exists() else []
    # Inline notes are the authoritative content: they are what the deck exports
    # today, so every field they actually carry wins over the normalized copy.
    # Empty inline fields are not listed — the merge ignores an empty incoming
    # value before it ever consults this list, and naming them would suggest
    # otherwise.
    prefer_incoming = tuple(
        name
        for name in MERGEABLE_FIELDS
        if any(not is_empty(getattr(record, name)) for record in migrated)
    )
    records, outcomes = merge_records(existing, migrated, prefer_incoming)
    merged = {record.id: record for record in records}
    warnings = _check_merged_faithfully(merged, migrated, deck_path, normalized_file)

    guards, guard_warnings = _guard_other_decks(
        config, deck_path, normalized_file, [record.id for record in migrated], merged
    )
    warnings.extend(guard_warnings)
    for guard in guards:
        for record_id in guard.requested_ids:
            warnings.append(
                f"{guard.path} names {record_id} in its own include_ids, so it will "
                "start exporting that record — its include_ids asked for it by name"
            )
        for record_id in guard.changed_ids:
            warnings.append(
                f"{guard.path} already exports {record_id}; the note migrated out of "
                f"{deck_path.name} changes what that record says there too"
            )

    # A deck that had no source is about to gain one, which would otherwise turn
    # "exactly these notes" into "everything in the normalized file, forever".
    # A deck that already read the file keeps whatever membership rule its owner
    # wrote: its inline ids were already part of that membership and are simply
    # coming from the file now, so pinning it would freeze out future imports it
    # is meant to receive.
    include_ids = [record.id for record in before] if added_source else []

    # Guards first: they only ever *remove* records that do not exist yet, so a
    # failure part-way through leaves decks with a harmless no-op filter rather
    # than notes they never had.
    written: list[Path] = []
    reserialized: list[Path] = []
    for guard in guards:
        if not guard.excluded_ids:
            continue
        other_raw = load_structured(guard.path)
        if not isinstance(other_raw, dict):  # pragma: no cover - resolved fine moments ago
            raise DataError(f"Deck file must contain a mapping: {guard.path}")
        merged_excludes = [
            str(value) for value in (other_raw.get("deck") or {}).get("exclude_ids") or []
        ]
        merged_excludes.extend(
            record_id for record_id in guard.excluded_ids if record_id not in merged_excludes
        )
        if rewrite_deck_file(guard.path, other_raw, {"exclude_ids": merged_excludes}):
            reserialized.append(guard.path)
        written.append(guard.path)

    save_records_json(normalized_file, records)
    written.append(normalized_file)

    updates: dict[str, Any] = {}
    if added_source:
        updates["source"] = source_value
        updates["include_ids"] = include_ids
    if rewrite_deck_file(deck_path, raw, updates, drop_keys=("notes",)):
        reserialized.append(deck_path)
    written.append(deck_path)

    for path in reserialized:
        warnings.append(
            f"{path} could not be updated as a text edit, so the whole file was "
            "re-serialized: its YAML comments, quote styles and blank lines are "
            "gone. The deck's meaning is unchanged — check `git diff` and put back "
            "anything you want to keep."
        )

    added = sources = 0
    for record in migrated:
        # The reference is the one `status --rebuild` reconstructs from the
        # record that now lives in the normalized file — its own `source.type`
        # and `source.imported_from`. A reference's identity is every key but
        # the date, so any other shape (the deck path, say) would have the
        # documented recovery command append a second, near-duplicate entry to
        # every migrated record. The deck the note came from is reported on
        # stdout instead: it is where the note lived, not where it came from.
        stored = merged[record.id]
        added += book.record_added(record.id)
        sources += book.record_source_seen(
            record.id, stored.source.type or "manual", stored.source.imported_from
        )

    return MigrationResult(
        deck_path=deck_path,
        normalized_file=normalized_file,
        migrated=migrated,
        outcomes=outcomes,
        include_ids=include_ids,
        source_value=source_value,
        added_source=added_source,
        guards=guards,
        warnings=warnings,
        written=written,
        ledger_added=added,
        ledger_sources=sources,
    )


def format_details(result: MigrationResult, root: Path) -> list[str]:
    """The lines that follow the header and the merge summary."""
    deck = _repo_path(result.deck_path, root)
    lines = [f"Rewrote {deck}: inline notes removed."]
    if result.added_source:
        lines.append(
            f"  source: {result.source_value}, include_ids pins the "
            f"{len(result.include_ids)} record(s) it exported before."
        )
    else:
        lines.append(
            "  It already read the normalized file, so no include_ids was added: "
            "its membership rule is unchanged."
        )
    for guard in result.guards:
        if not guard.excluded_ids:
            continue
        lines.append(
            f"{_repo_path(guard.path, root)} reads the same normalized file and would "
            f"have started exporting {len(guard.excluded_ids)} of these record(s) — "
            f"notes sharing GUIDs with {deck}, which Anki cannot hold twice."
        )
        lines.append(
            "  Added them to its exclude_ids so its contents are exactly what they "
            "were. Delete those ids if you do want that deck to have them."
        )
    return lines
