"""Exact destructive Assistant actions over canonical cards and deck files.

The provider supplies only opaque catalogue resources and canonical record ids.
This module resolves those targets locally, renders the complete structural
consequences, and binds them to one fingerprint.  Execution takes the same
repository locks, re-plans, and delegates to janki's bound collection writer or
exact no-follow unlink primitive.  It never interprets Japanese.

Deleting a configured deck removes its source-of-truth YAML only.  Generated
packages are deliberately retained: ``dist`` is disposable build output, and a
single-file canonical transition can be made atomic while a two-file unlink
cannot.  The confirmation says so explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import ledger, status
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    assistant_record_value,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import (
    deck_kind,
    project_deck_records,
    resolve_deck_records,
)
from japanese_anki.io import (
    RecordsRevision,
    atomic_unlink_bound,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records,
    load_records_snapshot,
    load_structured,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "AssistantCanonicalDeletionExecution",
    "AssistantCanonicalDeletionPlan",
    "AssistantDeckDeletionExecution",
    "AssistantDeckDeletionPlan",
    "AssistantDeletionError",
    "execute_canonical_deletion",
    "execute_deck_deletion",
    "plan_canonical_deletion",
    "plan_deck_deletion",
]


class AssistantDeletionError(JankiError):
    """A destructive Assistant action is incomplete, unsafe, or stale."""


@dataclass(frozen=True, slots=True)
class AssistantCanonicalDeletionPlan:
    """One exact canonical collection replacement behind owner authority."""

    repository_root: Path
    card_resource_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    instruction: str
    collection_path: Path
    collection_revision: RecordsRevision
    remaining_records: tuple[VocabularyRecord, ...]
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_plan_common(
            self.repository_root,
            self.instruction,
            self.projection_wire,
            self.fingerprint,
        )
        if not self.card_resource_ids or not self.record_ids:
            raise ValueError("Canonical deletion needs exact card targets")
        if len(self.card_resource_ids) != len(set(self.card_resource_ids)):
            raise ValueError("Canonical deletion card resources must be unique")
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("Canonical deletion record ids must be unique")
        if self.collection_revision.path != self.collection_path:
            raise ValueError("Canonical deletion revision targets another collection")
        _contained(self.repository_root, self.collection_path, "canonical collection")

    @property
    def projection(self) -> Mapping[str, Any]:
        return _parsed_projection(self.projection_wire)


@dataclass(frozen=True, slots=True)
class AssistantCanonicalDeletionExecution:
    """The freshly matched collection deletion result."""

    plan: AssistantCanonicalDeletionPlan
    removed_record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssistantDeckDeletionPlan:
    """One exact configured-deck YAML unlink behind owner authority."""

    repository_root: Path
    deck_resource_id: str
    instruction: str
    deck_path: Path
    deck_bytes: bytes
    deck_sha256: str
    projection_wire: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_plan_common(
            self.repository_root,
            self.instruction,
            self.projection_wire,
            self.fingerprint,
        )
        if not self.deck_resource_id:
            raise ValueError("Deck deletion needs an opaque deck resource")
        _contained(self.repository_root, self.deck_path, "configured deck")
        if not self.deck_bytes:
            raise ValueError("Deck deletion needs exact configured bytes")
        if _sha(self.deck_bytes) != self.deck_sha256:
            raise ValueError("Deck deletion bytes do not match their fingerprint")

    @property
    def projection(self) -> Mapping[str, Any]:
        return _parsed_projection(self.projection_wire)


@dataclass(frozen=True, slots=True)
class AssistantDeckDeletionExecution:
    """The freshly matched exact configured-deck unlink result."""

    plan: AssistantDeckDeletionPlan
    removed_deck_path: Path


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Assistant deletion cannot fingerprint its exact plan: {exc}"
        ) from exc


def _parsed_projection(wire: str) -> Mapping[str, Any]:
    value = json.loads(wire)
    assert isinstance(value, Mapping)
    return value


def _validate_plan_common(
    repository_root: Path,
    instruction: str,
    projection_wire: str,
    fingerprint: str,
) -> None:
    if not repository_root.is_absolute() or repository_root != repository_root.resolve():
        raise ValueError("Assistant deletion repository root must be canonical")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Assistant deletion instruction must be nonblank")
    try:
        projection = json.loads(projection_wire)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Assistant deletion projection must be JSON") from exc
    if not isinstance(projection, Mapping) or _canonical_json(projection) != projection_wire:
        raise ValueError("Assistant deletion projection must be a canonical JSON object")
    if _sha(projection_wire.encode("utf-8")) != fingerprint:
        raise ValueError("Assistant deletion fingerprint does not bind its projection")


def _contained(root: Path, path: Path, label: str) -> Path:
    target = Path(os.path.abspath(path))
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AssistantDeletionError(
            f"Assistant deletion {label} escapes the configured repository."
        ) from exc
    return target


def _contained_without_symlinks(root: Path, path: Path, label: str) -> Path:
    """Validate one configured lexical path without following any alias."""

    target = _contained(root, path, label)
    current = root
    for component in target.relative_to(root).parts:
        current /= component
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise AssistantDeletionError(
                f"Could not inspect configured {label} {current}: "
                f"{exc.strerror or exc}"
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise AssistantDeletionError(
                f"Assistant deletion {label} traverses a symlink: {current}"
            )
    return target


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    return _contained(config.root.resolve(), path, label).relative_to(
        config.root.resolve()
    ).as_posix()


def _sequence(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise AssistantDeletionError(f"Assistant deletion {label} must be a list of text.")
    selected = tuple(value)
    if not selected or any(not isinstance(item, str) or not item.strip() for item in selected):
        raise AssistantDeletionError(
            f"Assistant deletion needs one or more nonblank {label}."
        )
    repeated = [item for item, count in Counter(selected).items() if count > 1]
    if repeated:
        raise AssistantDeletionError(
            f"Assistant deletion {label[:-1]} {repeated[0]!r} was supplied twice."
        )
    return selected


def _deck_paths(config: ProjectConfig) -> tuple[Path, ...]:
    try:
        return tuple(Path(os.path.abspath(path)) for path in status.deck_files(config))
    except (JankiError, OSError) as exc:
        raise AssistantDeletionError(
            f"Could not enumerate configured decks for deletion: {exc}"
        ) from exc


def _deck_source(config: ProjectConfig, deck_path: Path) -> Path | None:
    """Return the structural records source used by an existing deck reader."""

    kind = deck_kind(deck_path)
    if kind == "conjugation":
        try:
            text = read_bytes_bound(deck_path).decode("utf-8", errors="strict")
            section = pattern_cards.conjugation_deck_section(deck_path, text)
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantDeletionError(
                f"Could not resolve conjugation deck source {deck_path}: {exc}"
            ) from exc
    else:
        raw = load_structured(deck_path)
        section = raw.get("deck") if isinstance(raw, Mapping) else None
        if not isinstance(section, Mapping):
            raise AssistantDeletionError(
                f"The deck section must be a mapping: {deck_path}"
            )
    source = section.get("source")
    if source is None or source == "":
        if kind == "conjugation":
            target = Path(os.path.abspath(config.normalized_file))
        elif kind == "kanji":
            # A character deck naming no store reads the project's curated one,
            # exactly as `resolve_kanji_deck_notes` does. Bound here so the
            # bytes a deletion was planned against are locked and fingerprinted
            # whether or not the file spells the path out.
            target = Path(os.path.abspath(config.kanji_notes_file))
        else:
            return None
    else:
        if not isinstance(source, str):
            raise AssistantDeletionError(f"Deck source must be text: {deck_path}")
        target = Path(os.path.abspath(deck_path.parent / source))
    target = _contained(config.root.resolve(), target, "deck source")
    try:
        read_bytes_bound(target)
    except (JankiError, OSError) as exc:
        raise AssistantDeletionError(
            f"Could not safely bind deck source {target}: {exc}"
        ) from exc
    return target


def _dependency_paths(
    config: ProjectConfig,
    deck_paths: Sequence[Path],
) -> tuple[Path, ...]:
    paths = {
        Path(os.path.abspath(config.normalized_file)),
        Path(os.path.abspath(config.ledger_file)),
        Path(os.path.abspath(config.patterns_file)),
        *(Path(os.path.abspath(path)) for path in deck_paths),
    }
    for deck_path in deck_paths:
        source = _deck_source(config, deck_path)
        if source is not None:
            paths.add(source)
    return tuple(sorted(paths, key=os.fspath))


def _validate_configured_containment(config: ProjectConfig) -> None:
    """Refuse an Assistant action before any configured external path is read."""

    root = config.root.resolve()
    configured = (
        (config.raw_dir, "raw source directory"),
        (config.normalized_file, "canonical collection"),
        (config.deck_dir, "configured deck directory"),
        (config.template_dir, "template directory"),
        (config.dist_dir, "generated package directory"),
        (config.ledger_file, "ledger"),
        (config.staging_dir, "staging directory"),
        (config.media_dir, "media directory"),
        (config.assistant_dir, "Assistant turn directory"),
        (config.kanji_file, "kanji store"),
        (config.operations_file, "operation journal"),
        (config.patterns_file, "pattern store"),
        (config.scan_inbox, "source inbox"),
    )
    for path, label in configured:
        _contained_without_symlinks(root, path, label)


@contextmanager
def _locked_repository_inputs(config: ProjectConfig) -> Iterator[tuple[Path, ...]]:
    """Freeze every card/deck input across destructive re-plan and commit."""

    _validate_configured_containment(config)
    operation_lock = config.root / ".janki-audio-operation"
    with exclusive_path_lock(operation_lock), exclusive_path_lock(config.deck_dir):
        try:
            first_decks = _deck_paths(config)
            first_dependencies = _dependency_paths(config, first_decks)
        except AssistantDeletionError:
            raise
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantDeletionError(
                f"Could not bind deletion dependencies safely: {exc}"
            ) from exc
        with ExitStack() as stack:
            for path in first_dependencies:
                stack.enter_context(exclusive_path_lock(path))
            try:
                second_decks = _deck_paths(config)
                second_dependencies = _dependency_paths(config, second_decks)
            except AssistantDeletionError:
                raise
            except (JankiError, OSError, UnicodeError, ValueError) as exc:
                raise AssistantDeletionError(
                    f"Could not re-bind deletion dependencies safely: {exc}"
                ) from exc
            if second_decks != first_decks or second_dependencies != first_dependencies:
                raise AssistantDeletionError(
                    "The configured deck or card dependency set changed while Janki "
                    "was locking it; refresh and review the deletion again."
                )
            yield first_decks


def _ledger_snapshot(config: ProjectConfig) -> tuple[ledger.Ledger, str | None]:
    try:
        if config.ledger_file.exists() or config.ledger_file.is_symlink():
            wire = read_bytes_bound(config.ledger_file)
        else:
            wire = None
        return ledger.load_snapshot(config.ledger_file, wire), (
            _sha(wire) if wire is not None else None
        )
    except (JankiError, OSError) as exc:
        raise AssistantDeletionError(f"Could not bind the deletion ledger: {exc}") from exc


def _repository_file_inputs(
    config: ProjectConfig,
    paths: Sequence[Path],
) -> list[dict[str, object]]:
    """Fingerprint every repository file whose bytes can shape the plan."""

    values: list[dict[str, object]] = []
    for path in paths:
        try:
            wire = read_bytes_bound(path)
        except FileNotFoundError:
            wire = None
        except (JankiError, OSError) as exc:
            raise AssistantDeletionError(
                f"Could not bind deletion input {path}: {exc}"
            ) from exc
        values.append(
            {
                "repository_file": _relative(config, path, label="deletion input"),
                "exists": wire is not None,
                "sha256": _sha(wire) if wire is not None else None,
            }
        )
    return values


def _package_path(
    config: ProjectConfig,
    deck_path: Path,
    section: Mapping[str, Any],
) -> Path:
    value = section.get("output", f"{deck_path.stem}.apkg")
    if not isinstance(value, str) or not value.strip():
        raise AssistantDeletionError(f"Deck output must be nonblank text: {deck_path}")
    target = Path(os.path.abspath(config.dist_dir / value))
    try:
        target.relative_to(config.dist_dir)
    except ValueError as exc:
        raise AssistantDeletionError(
            f"Configured deck output escapes the build directory: {deck_path}"
        ) from exc
    _contained(config.root.resolve(), target, "generated package")
    return target


def _deck_records(
    config: ProjectConfig,
    deck_path: Path,
    *,
    projected_canonical: Sequence[VocabularyRecord] | None = None,
) -> tuple[Mapping[str, Any], tuple[VocabularyRecord, ...]]:
    kind = deck_kind(deck_path)
    # Neither kind ships a vocabulary record: a pattern deck's cards come from
    # the pattern store and a character deck's from the curated character
    # store. Resolving either through the word-deck reader would read that
    # store as a vocabulary collection and refuse, which is how one character
    # deck used to break every canonical deletion in the project.
    if kind in {"pattern", "kanji"}:
        raw = load_structured(deck_path)
        section = raw.get("deck") if isinstance(raw, Mapping) else None
        if not isinstance(section, Mapping):
            raise AssistantDeletionError(f"The deck section must be a mapping: {deck_path}")
        return section, ()
    if kind == "conjugation":
        text = read_bytes_bound(deck_path).decode("utf-8", errors="strict")
        section = pattern_cards.conjugation_deck_section(deck_path, text)
        source = _deck_source(config, deck_path)
        assert source is not None
        records = (
            list(projected_canonical)
            if projected_canonical is not None
            and source == Path(os.path.abspath(config.normalized_file))
            else load_records(source)
        )
        shipping = pattern_cards.shipping_records_for_section(deck_path, section, records)
        return section, tuple(shipping)
    section, current = resolve_deck_records(deck_path)
    source = _deck_source(config, deck_path)
    if projected_canonical is not None and source == Path(
        os.path.abspath(config.normalized_file)
    ):
        section, current = project_deck_records(
            deck_path,
            config.normalized_file,
            projected_canonical,
        )
    return section, tuple(current)


def _media_references(records: Sequence[VocabularyRecord]) -> list[str]:
    names: set[str] = set()
    for record in records:
        for value in (record.audio, record.image, *(item.audio for item in record.examples)):
            name = Path(str(value).replace("\\", "/")).name
            if name not in {"", ".", ".."}:
                names.add(name)
    return sorted(names)


def _resolved_card_ids(
    broker: AssistantContextBroker,
    resource_ids: Sequence[str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for resource_id in resource_ids:
        try:
            disclosure = broker.snapshot(resource_id)
            value = json.loads(disclosure.wire)
        except (AssistantContextError, JankiError, OSError, ValueError) as exc:
            raise AssistantDeletionError(
                f"Could not resolve canonical card resource {resource_id!r}: {exc}"
            ) from exc
        data = value.get("data") if isinstance(value, Mapping) else None
        card = data.get("card") if isinstance(data, Mapping) else None
        record_id = card.get("id") if isinstance(card, Mapping) else None
        if disclosure.kind != "card" or not isinstance(record_id, str) or not record_id:
            raise AssistantDeletionError(
                "Canonical deletion accepts exact card resources only."
            )
        resolved.append(record_id)
    return tuple(resolved)


def _canonical_bytes(records: Sequence[VocabularyRecord]) -> bytes:
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _prepare_canonical_locked(
    config: ProjectConfig,
    *,
    deck_paths: Sequence[Path],
    card_resource_ids: Sequence[str],
    record_ids: Sequence[str],
    instruction: str,
) -> AssistantCanonicalDeletionPlan:
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantDeletionError(
            "Canonical deletion needs the owner's nonblank instruction."
        )
    resources = _sequence(card_resource_ids, label="card resource ids")
    selected = _sequence(record_ids, label="record ids")
    try:
        broker = AssistantContextBroker(config)
        resolved_ids = _resolved_card_ids(broker, resources)
    except AssistantDeletionError:
        raise
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Could not resolve exact canonical deletion targets: {exc}"
        ) from exc
    if resolved_ids != selected:
        raise AssistantDeletionError(
            "Canonical deletion card resources and record ids must identify the same "
            "cards in the same order."
        )

    try:
        current, revision = load_records_snapshot(config.normalized_file)
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Could not read the canonical card collection: {exc}"
        ) from exc
    if revision.text is None:
        raise AssistantDeletionError("The canonical card collection does not exist.")
    positions: dict[str, list[VocabularyRecord]] = {}
    for record in current:
        positions.setdefault(record.id, []).append(record)
    deleted: list[VocabularyRecord] = []
    for record_id in selected:
        matches = positions.get(record_id, [])
        if len(matches) != 1:
            detail = "is absent" if not matches else "appears more than once"
            raise AssistantDeletionError(
                f"Canonical card {record_id!r} {detail}; refresh before deleting."
            )
        deleted.append(matches[0])
    selected_set = set(selected)
    remaining = tuple(record for record in current if record.id not in selected_set)

    book, ledger_sha = _ledger_snapshot(config)
    pending = sorted(
        key
        for key, entry in book.pending_audio.items()
        if isinstance(entry, Mapping) and entry.get("record_id") in selected_set
    )
    if pending:
        raise AssistantDeletionError(
            "The selected card has pending audio recovery. Finish or explicitly "
            "settle that exact recovery transaction before deleting the card."
        )

    affected: list[dict[str, object]] = []
    packages: set[str] = set()
    for deck_path in deck_paths:
        try:
            section, before = _deck_records(config, deck_path)
            _after_section, after = _deck_records(
                config,
                deck_path,
                projected_canonical=remaining,
            )
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantDeletionError(
                f"Could not project canonical deletion through {deck_path}: {exc}"
            ) from exc
        before_ids = tuple(record.id for record in before)
        after_ids = tuple(record.id for record in after)
        selected_before = [record_id for record_id in selected if record_id in before_ids]
        selected_after = [record_id for record_id in selected if record_id in after_ids]
        if before_ids == after_ids and not selected_before and not selected_after:
            continue
        package = _relative(
            config,
            _package_path(config, deck_path, section),
            label="generated package",
        )
        packages.add(package)
        affected.append(
            {
                "name": str(section.get("name") or deck_path.stem),
                "configured_file": _relative(
                    config, deck_path, label="configured deck"
                ),
                "selected_before": selected_before,
                "selected_after_rebuild": selected_after,
                "notes_before": len(before_ids),
                "notes_after_rebuild": len(after_ids),
                "declared_package": package,
            }
        )

    before_bytes = revision.text.encode("utf-8")
    after_bytes = _canonical_bytes(remaining)
    cards = []
    for resource_id, record_id, record in zip(
        resources, resolved_ids, deleted, strict=True
    ):
        record_wire = _canonical_json(record.to_dict())
        cards.append(
            {
                "resource_id": resource_id,
                "record_id": record_id,
                "canonical_sha256": _sha(record_wire.encode("utf-8")),
                "record": assistant_record_value(record),
            }
        )
    repository_inputs = _repository_file_inputs(
        config,
        _dependency_paths(config, deck_paths),
    )
    projection = {
        "schema_version": 1,
        "kind": "delete_canonical_cards",
        "instruction": instruction,
        "selection": {"record_ids": list(selected), "cards": cards},
        "canonical_collection": {
            "configured_file": _relative(
                config, config.normalized_file, label="canonical collection"
            ),
            "rows_before": len(current),
            "rows_removed": len(deleted),
            "rows_after": len(remaining),
            "before_sha256": _sha(before_bytes),
            "after_sha256": _sha(after_bytes),
        },
        "affected_decks": affected,
        "retained": {
            "configured_deck_definitions": True,
            "generated_packages": sorted(packages),
            "ledger_history_for_record_ids": [
                record_id for record_id in selected if record_id in book.records
            ],
            "media_references": _media_references(deleted),
            "staging_proposals": True,
        },
        "inputs": {
            "repository_files": repository_inputs,
            "ledger_sha256": ledger_sha,
        },
        "provider": None,
        "billing_class": "local",
        "writes": {
            "replace_canonical_collection": _relative(
                config, config.normalized_file, label="canonical collection"
            ),
            "remove_record_ids": list(selected),
        },
    }
    wire = _canonical_json(projection)
    return AssistantCanonicalDeletionPlan(
        repository_root=config.root.resolve(),
        card_resource_ids=resources,
        record_ids=selected,
        instruction=instruction,
        collection_path=Path(os.path.abspath(config.normalized_file)),
        collection_revision=revision,
        remaining_records=remaining,
        projection_wire=wire,
        fingerprint=_sha(wire.encode("utf-8")),
    )


def plan_canonical_deletion(
    config: ProjectConfig,
    *,
    card_resource_ids: Sequence[str],
    record_ids: Sequence[str],
    instruction: str,
) -> AssistantCanonicalDeletionPlan:
    """Render an exact no-write plan for removing canonical card records."""

    with _locked_repository_inputs(config) as deck_paths:
        return _prepare_canonical_locked(
            config,
            deck_paths=deck_paths,
            card_resource_ids=card_resource_ids,
            record_ids=record_ids,
            instruction=instruction,
        )


def execute_canonical_deletion(
    config: ProjectConfig,
    expected: AssistantCanonicalDeletionPlan,
) -> AssistantCanonicalDeletionExecution:
    """Re-plan and atomically replace one exact canonical collection."""

    if expected.repository_root != config.root.resolve():
        raise AssistantDeletionError(
            "This canonical deletion belongs to another repository."
        )
    try:
        with _locked_repository_inputs(config) as deck_paths:
            current, revision = load_records_snapshot(config.normalized_file)
            del current
            if revision.text != expected.collection_revision.text:
                raise AssistantDeletionError(
                    "The canonical deletion changed after it was displayed; "
                    "refresh and review the current cards before confirming."
                )
            fresh = _prepare_canonical_locked(
                config,
                deck_paths=deck_paths,
                card_resource_ids=expected.card_resource_ids,
                record_ids=expected.record_ids,
                instruction=expected.instruction,
            )
            if (
                fresh.fingerprint != expected.fingerprint
                or fresh.projection_wire != expected.projection_wire
            ):
                raise AssistantDeletionError(
                    "The canonical deletion changed after it was displayed; "
                    "refresh and review the current consequences before confirming."
                )
            before_sha256 = fresh.projection["canonical_collection"][
                "before_sha256"
            ]
            assert isinstance(before_sha256, str)
            atomic_write_text_bound(
                fresh.collection_path,
                _canonical_bytes(fresh.remaining_records).decode("utf-8"),
                expected_revision=before_sha256,
            )
            written = read_bytes_bound(fresh.collection_path)
            expected_after = fresh.projection["canonical_collection"]["after_sha256"]
            if _sha(written) != expected_after:
                raise AssistantDeletionError(
                    "The canonical collection did not match the confirmed deletion."
                )
    except AssistantDeletionError:
        raise
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Could not apply the exact canonical deletion: {exc}"
        ) from exc
    return AssistantCanonicalDeletionExecution(
        plan=fresh,
        removed_record_ids=fresh.record_ids,
    )


def _character_deck_ids(deck_path: Path, section: Mapping[str, Any]) -> set[str]:
    """The exact character identities one character deck names."""

    declared = section.get("include_ids")
    if isinstance(declared, str | bytes) or not isinstance(declared, Sequence):
        raise AssistantDeletionError(
            f"A character deck names the characters it holds as a list: {deck_path}"
        )
    ids: set[str] = set()
    for item in declared:
        if not isinstance(item, str) or not item.strip():
            raise AssistantDeletionError(
                f"Each character-deck identity must be nonblank text: {deck_path}"
            )
        ids.add(item)
    return ids


def _deck_identity_sets(
    config: ProjectConfig,
    deck_path: Path,
) -> tuple[set[str], set[str]]:
    """Return source identities and all identities declared by one deck."""

    raw = load_structured(deck_path)
    section = raw.get("deck") if isinstance(raw, Mapping) else None
    if not isinstance(section, Mapping):
        raise AssistantDeletionError(f"The deck section must be a mapping: {deck_path}")
    if deck_kind(deck_path) == "kanji":
        # A character deck holds no inline word notes and no vocabulary source.
        # It names ids into the curated character store, which deleting the
        # deck never touches, so nothing it declares is inline-only and the two
        # sets are equal.
        ids = _character_deck_ids(deck_path, section)
        return ids, set(ids)
    source = _deck_source(config, deck_path)
    source_ids = set() if source is None else {record.id for record in load_records(source)}
    declared_ids = set(source_ids)
    notes = raw.get("notes") or []
    if not isinstance(notes, list):
        raise AssistantDeletionError(f"The notes section must be a list: {deck_path}")
    for item in notes:
        if not isinstance(item, Mapping):
            raise AssistantDeletionError(f"Each note must be a mapping: {deck_path}")
        record_id = str(item.get("id") or "").strip()
        if not record_id:
            try:
                record_id = VocabularyRecord.from_dict(dict(item)).id
            except (JankiError, ValueError) as exc:
                raise AssistantDeletionError(
                    f"Could not resolve an inline card identity in {deck_path}: {exc}"
                ) from exc
        declared_ids.add(record_id)
    return source_ids, declared_ids


def _inline_ids(config: ProjectConfig, deck_path: Path) -> set[str]:
    source_ids, declared_ids = _deck_identity_sets(config, deck_path)
    return declared_ids - source_ids
def _resolve_deck_resource(
    config: ProjectConfig,
    broker: AssistantContextBroker,
    deck_paths: Sequence[Path],
    resource_id: str,
) -> Path:
    matches: list[Path] = []
    for path in deck_paths:
        try:
            candidate = broker.resource_id_for_deck(path)
        except (AssistantContextError, JankiError):
            continue
        if secrets.compare_digest(candidate, resource_id):
            matches.append(path)
    if len(matches) != 1:
        raise AssistantDeletionError(
            "Deck deletion needs one exact current configured-deck resource."
        )
    try:
        broker.deck_context(resource_id)
    except (AssistantContextError, JankiError, OSError, ValueError) as exc:
        raise AssistantDeletionError(
            f"The configured deck cannot be read safely for deletion: {exc}"
        ) from exc
    return matches[0]


def _prepare_deck_locked(
    config: ProjectConfig,
    *,
    deck_paths: Sequence[Path],
    deck_resource_id: str,
    instruction: str,
) -> AssistantDeckDeletionPlan:
    if not isinstance(deck_resource_id, str) or not deck_resource_id.strip():
        raise AssistantDeletionError(
            "Deck deletion needs one nonblank configured-deck resource id."
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantDeletionError(
            "Deck deletion needs the owner's nonblank instruction."
        )
    try:
        broker = AssistantContextBroker(config)
        target = _resolve_deck_resource(config, broker, deck_paths, deck_resource_id)
        deck_bytes = read_bytes_bound(target)
        raw = load_structured(target)
        section = raw.get("deck") if isinstance(raw, Mapping) else None
        if not isinstance(section, Mapping):
            raise AssistantDeletionError(f"The deck section must be a mapping: {target}")
        kind = deck_kind(target) or "vocabulary"
        context = broker.deck_context(deck_resource_id)
        inline_ids = _inline_ids(config, target)
        canonical, _revision = load_records_snapshot(config.normalized_file)
        surviving_ids = {record.id for record in canonical}
        for sibling in deck_paths:
            if sibling != target:
                _source_ids, declared_ids = _deck_identity_sets(config, sibling)
                surviving_ids.update(declared_ids)
        disappearing = sorted(inline_ids - surviving_ids)
        package = _package_path(config, target, section)
        book, ledger_sha = _ledger_snapshot(config)
    except AssistantDeletionError:
        raise
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Could not prepare configured-deck deletion: {exc}"
        ) from exc

    drill_owners: set[str] = set()
    if kind == "conjugation":
        try:
            drill_owners = set(pattern_cards.declared_drill_audio_owner_ids(target))
        except (JankiError, OSError, ValueError) as exc:
            raise AssistantDeletionError(
                f"Could not inspect deck-owned audio recovery: {exc}"
            ) from exc
    removed_audio_owners = drill_owners | set(disappearing)
    pending = sorted(
        key
        for key, entry in book.pending_audio.items()
        if isinstance(entry, Mapping)
        and entry.get("record_id") in removed_audio_owners
    )
    if pending:
        raise AssistantDeletionError(
            "This deck owns pending paid audio recovery. Finish or explicitly "
            "settle that exact recovery transaction before deleting the deck."
        )

    projection = {
        "schema_version": 1,
        "kind": "delete_configured_deck",
        "instruction": instruction,
        "target": {
            "resource_id": deck_resource_id,
            "name": str(section.get("name") or target.stem),
            "kind": kind,
            "configured_file": _relative(config, target, label="configured deck"),
            "sha256": _sha(deck_bytes),
        },
        "consequences": {
            "currently_selected_record_ids": [record.id for record in context.records],
            "canonical_cards_removed": [],
            "inline_only_record_ids_removed_from_library": disappearing,
            "configured_deck_count_before": len(deck_paths),
            "configured_deck_count_after": len(deck_paths) - 1,
        },
        "retained": {
            "canonical_collection": _relative(
                config, config.normalized_file, label="canonical collection"
            ),
            "generated_package": _relative(
                config, package, label="generated package"
            ),
            "ledger_history": True,
            "media": True,
            "staging_proposals": True,
        },
        "inputs": {
            "repository_files": _repository_file_inputs(
                config,
                _dependency_paths(config, deck_paths),
            ),
            "ledger_sha256": ledger_sha,
        },
        "provider": None,
        "billing_class": "local",
        "removals": [_relative(config, target, label="configured deck")],
    }
    wire = _canonical_json(projection)
    return AssistantDeckDeletionPlan(
        repository_root=config.root.resolve(),
        deck_resource_id=deck_resource_id,
        instruction=instruction,
        deck_path=target,
        deck_bytes=deck_bytes,
        deck_sha256=_sha(deck_bytes),
        projection_wire=wire,
        fingerprint=_sha(wire.encode("utf-8")),
    )


def plan_deck_deletion(
    config: ProjectConfig,
    *,
    deck_resource_id: str,
    instruction: str,
) -> AssistantDeckDeletionPlan:
    """Render an exact no-write plan for removing one configured deck YAML."""

    with _locked_repository_inputs(config) as deck_paths:
        return _prepare_deck_locked(
            config,
            deck_paths=deck_paths,
            deck_resource_id=deck_resource_id,
            instruction=instruction,
        )


def execute_deck_deletion(
    config: ProjectConfig,
    expected: AssistantDeckDeletionPlan,
) -> AssistantDeckDeletionExecution:
    """Re-plan and safely unlink only the exact confirmed deck definition."""

    if expected.repository_root != config.root.resolve():
        raise AssistantDeletionError("This deck deletion belongs to another repository.")
    try:
        with _locked_repository_inputs(config) as deck_paths:
            try:
                current = read_bytes_bound(expected.deck_path)
            except (JankiError, OSError) as exc:
                raise AssistantDeletionError(
                    "The configured-deck deletion changed after it was displayed; "
                    "refresh and review it again."
                ) from exc
            if current != expected.deck_bytes:
                raise AssistantDeletionError(
                    "The configured-deck deletion changed after it was displayed; "
                    "refresh and review it again."
                )
            fresh = _prepare_deck_locked(
                config,
                deck_paths=deck_paths,
                deck_resource_id=expected.deck_resource_id,
                instruction=expected.instruction,
            )
            if (
                fresh.fingerprint != expected.fingerprint
                or fresh.projection_wire != expected.projection_wire
            ):
                raise AssistantDeletionError(
                    "The configured-deck deletion changed after it was displayed; "
                    "refresh and review the current consequences before confirming."
                )
            atomic_unlink_bound(fresh.deck_path, expected_revision=fresh.deck_sha256)
            try:
                os.lstat(fresh.deck_path)
            except FileNotFoundError:
                pass
            else:
                raise AssistantDeletionError(
                    "The exact configured deck remained after deletion."
                )
    except AssistantDeletionError:
        raise
    except (JankiError, OSError, UnicodeError, ValueError) as exc:
        raise AssistantDeletionError(
            f"Could not apply the exact configured-deck deletion: {exc}"
        ) from exc
    return AssistantDeckDeletionExecution(plan=fresh, removed_deck_path=fresh.deck_path)
