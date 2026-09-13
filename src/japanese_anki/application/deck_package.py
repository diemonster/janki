"""Plan-bound publication of any one configured Anki package.

The exporters remain the only code that interprets a deck file and writes an
``.apkg``.  This layer gives browser/Assistant callers the missing application
contract around those exporters: enumerate and fingerprint every filesystem
input, display the complete package consequence, then re-plan under the same
repository locks before writing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Literal

from japanese_anki import ledger, patterns, status
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_build, deck_capabilities
from japanese_anki.application.build import _record_gaps
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import anki as anki_exporter
from japanese_anki.exporters import kanji_cards, pattern_cards
from japanese_anki.exporters.anki import (
    CARD_FILES,
    FIELD_NAMES,
    RenderedDeck,
    build_deck,
    deck_kind,
    project_deck_media_paths,
    project_deck_records,
    render_deck,
    resolve_card_types,
    resolve_deck_media_paths,
    resolve_deck_output_path,
    resolve_deck_records,
)
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    atomic_write_bytes_bound,
    exclusive_path_lock,
    load_structured,
    prepare_bound_directory,
    read_bytes_bound,
    read_bytes_bound_snapshot,
    records_json_text,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import has_errors, refusal_text, validate_records

__all__ = [
    "PREPARED_PACKAGE_DIR_NAME",
    "PREPARED_PACKAGE_SCHEMA",
    "DeckPackageError",
    "DeckPackageExport",
    "DeckPackageInput",
    "DeckPackageInventory",
    "DeckPackageNote",
    "DeckPackagePlan",
    "DeckPackagePreparation",
    "DeckPackageResult",
    "assert_projection_realized",
    "execute_deck_package",
    "execute_deck_package_locked",
    "plan_deck_package",
    "plan_vocabulary_deck_package_revision",
    "prepare_deck_package",
    "publish_prepared_deck_package",
    "recover_prepared_deck_package",
]

#: The private directory a prepared artifact is staged in, under ``dist``.
PREPARED_PACKAGE_DIR_NAME = ".janki-prepared"

PREPARED_PACKAGE_SCHEMA = "janki-prepared-deck-package-v1"

#: genanki's package layout. There is no second supported one.
_ARCHIVE_COLLECTION = "collection.anki2"
_ARCHIVE_MANIFEST = "media"


PackageKind = Literal["vocabulary", "pattern", "conjugation", "kanji"]


class DeckPackageError(JankiError):
    """A configured package could not consume the exact confirmed plan."""


@dataclass(frozen=True, slots=True)
class DeckPackageInput:
    """One exact repository input consumed by a package build."""

    label: str
    path: Path
    sha256: str | None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("A deck package input needs a nonblank label")
        if not self.path.is_absolute() or self.path != _lexical(self.path):
            raise ValueError("Deck package input paths must be lexical absolute paths")
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise ValueError("Deck package input hashes must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DeckPackagePlan:
    """Every exact input and consequence of one configured package build."""

    repository_root: Path
    deck_path: Path
    output_path: Path
    kind: PackageKind
    deck_name: str
    variant: str
    card_types: tuple[str, ...]
    note_count: int
    card_count: int
    record_ids: tuple[str, ...]
    deck_input: DeckPackageInput
    source_inputs: tuple[DeckPackageInput, ...]
    template_inputs: tuple[DeckPackageInput, ...]
    media_inputs: tuple[DeckPackageInput, ...]
    output_revision: str | None
    output_identity: tuple[int, int] | None
    configuration_fingerprint: str
    fingerprint: str
    conjugation_plan: deck_build.ConjugationDeckBuildPlan | None = None

    def __post_init__(self) -> None:
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Deck package repository root must be canonical")
        if self.deck_path != _lexical(self.deck_path):
            raise ValueError("Deck package path must be lexical absolute")
        if self.output_path != _lexical(self.output_path):
            raise ValueError("Deck package output path must be lexical absolute")
        for path in (self.deck_path, self.output_path):
            try:
                path.relative_to(self.repository_root)
            except ValueError as exc:
                raise ValueError("Deck package paths must stay in the repository") from exc
        if self.deck_input.path != self.deck_path or self.deck_input.sha256 is None:
            raise ValueError("Deck package plan must bind its configured deck bytes")
        if not self.deck_name.strip() or not self.card_types:
            raise ValueError("Deck package plan needs a name and card types")
        if len(set(self.card_types)) != len(self.card_types):
            raise ValueError("Deck package card types must be unique")
        if self.note_count < 0 or self.card_count < 0:
            raise ValueError("Deck package counts cannot be negative")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("Deck package record ids must be unique")
        inputs = self.source_inputs + self.template_inputs + self.media_inputs
        for item in inputs:
            try:
                item.path.relative_to(self.repository_root)
            except ValueError as exc:
                raise ValueError("Deck package inputs must stay in the repository") from exc
        if len({item.path for item in self.template_inputs}) != len(self.template_inputs):
            raise ValueError("Deck package template inputs must be unique")
        if len({item.path for item in self.media_inputs}) != len(self.media_inputs):
            raise ValueError("Deck package media inputs must be unique")
        if (self.output_revision is None) != (self.output_identity is None):
            raise ValueError("Deck package output content and identity must be bound together")
        if self.output_revision is not None and not _is_sha256(self.output_revision):
            raise ValueError("Deck package output revision must be lowercase SHA-256")
        if self.output_identity is not None and (
            len(self.output_identity) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.output_identity
            )
        ):
            raise ValueError("Deck package output identity must be a device/inode pair")
        if not _is_sha256(self.fingerprint):
            raise ValueError("Deck package fingerprint must be lowercase SHA-256")
        if not _is_sha256(self.configuration_fingerprint):
            raise ValueError("Deck package configuration fingerprint must be lowercase SHA-256")
        if (self.kind == "conjugation") != (self.conjugation_plan is not None):
            raise ValueError("Only a conjugation package may carry a conjugation service plan")

    @property
    def all_inputs(self) -> tuple[DeckPackageInput, ...]:
        """The complete exact filesystem input set, in display order."""

        return (self.deck_input,) + self.source_inputs + self.template_inputs + self.media_inputs


@dataclass(frozen=True, slots=True)
class DeckPackageResult:
    """The exact configured package proven to have landed."""

    output_path: Path
    package_sha256: str
    note_count: int
    card_count: int
    card_types: tuple[str, ...]
    media_count: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.output_path.is_absolute():
            raise ValueError("A deck package result path must be absolute")
        if not _is_sha256(self.package_sha256):
            raise ValueError("A deck package result needs a lowercase SHA-256")
        if min(self.note_count, self.card_count, self.media_count) < 0:
            raise ValueError("Deck package result counts cannot be negative")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical(path: Path | str) -> Path:
    """Return one normalized absolute name without following symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(config: ProjectConfig, path: Path) -> str:
    target = _lexical(path)
    try:
        return target.relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise DeckPackageError(
            f"Deck package input escapes the configured repository: {target}"
        ) from exc


def _input(
    config: ProjectConfig,
    label: str,
    path: Path,
    *,
    optional: bool = False,
) -> DeckPackageInput:
    target = _lexical(path)
    _relative(config, target)
    try:
        value = read_bytes_bound(target)
    except FileNotFoundError:
        if optional:
            return DeckPackageInput(label=label, path=target, sha256=None)
        raise DeckPackageError(f"Deck package input is missing: {target}") from None
    except (DataError, OSError) as exc:
        raise DeckPackageError(f"Could not read deck package input {target}: {exc}") from exc
    return DeckPackageInput(label=label, path=target, sha256=_sha(value))


def _text_input(config: ProjectConfig, label: str, path: Path) -> DeckPackageInput:
    target = _lexical(path)
    _relative(config, target)
    try:
        value = read_bytes_bound(target)
        value.decode("utf-8", errors="strict")
    except FileNotFoundError:
        raise DeckPackageError(f"Deck package input is missing: {target}") from None
    except (DataError, OSError, UnicodeError) as exc:
        raise DeckPackageError(f"Could not read text deck package input {target}: {exc}") from exc
    return DeckPackageInput(label=label, path=target, sha256=_sha(value))


def _known_deck(config: ProjectConfig, deck_path: Path | str) -> Path:
    target = _lexical(deck_path)
    matches = [
        candidate
        for path in status.deck_files(config)
        if (candidate := _lexical(path)) == target
    ]
    if len(matches) != 1:
        raise DeckPackageError(f"Deck package target {target} is not exactly one configured deck.")
    try:
        read_bytes_bound(target)
    except FileNotFoundError:
        raise DeckPackageError(f"Configured deck no longer exists: {target}") from None
    except (DataError, OSError) as exc:
        raise DeckPackageError(
            f"Configured deck must remain a direct regular non-symlink file: {target}: {exc}"
        ) from exc
    return target


def _safe_output(config: ProjectConfig, raw: Path) -> Path:
    target = _lexical(raw)
    dist = _lexical(config.dist_dir)
    if (
        target.parent != dist
        or PurePath(target.name).name != target.name
        or target.suffix.lower() != ".apkg"
    ):
        raise DeckPackageError(
            "An Assistant deck build requires one direct .apkg filename under "
            "the configured dist directory."
        )
    _relative(config, target)
    return target


def _source_path(deck_path: Path, section: Mapping[str, object]) -> Path | None:
    raw = section.get("source")
    if not raw:
        return None
    if not isinstance(raw, str):
        raise DeckPackageError(f"deck.source must be a path, got {type(raw).__name__}: {deck_path}")
    return _lexical(deck_path.parent / raw)


def _vocabulary_templates(
    config: ProjectConfig, card_types: Sequence[str]
) -> tuple[DeckPackageInput, ...]:
    paths: list[tuple[str, Path]] = []
    for card_type in card_types:
        front, back, _name = CARD_FILES[card_type]
        paths.extend(
            (
                (f"{card_type} front template", config.template_dir / front),
                (f"{card_type} back template", config.template_dir / back),
            )
        )
    paths.append(("card stylesheet", config.template_dir / "style.css"))
    return tuple(_text_input(config, label, path) for label, path in paths)


def _fingerprint(config: ProjectConfig, plan: DeckPackagePlan) -> str:
    def inputs(values: Sequence[DeckPackageInput]) -> list[dict[str, object]]:
        return [
            {
                "label": item.label,
                "path": _relative(config, item.path),
                "sha256": item.sha256,
            }
            for item in values
        ]

    payload = {
        "version": 1,
        "deck": inputs((plan.deck_input,))[0],
        "kind": plan.kind,
        "name": plan.deck_name,
        "variant": plan.variant,
        "output": {
            "path": _relative(config, plan.output_path),
            "sha256": plan.output_revision,
            "identity": (
                None
                if plan.output_identity is None
                else list(plan.output_identity)
            ),
        },
        "card_types": list(plan.card_types),
        "note_count": plan.note_count,
        "card_count": plan.card_count,
        "record_ids": list(plan.record_ids),
        "configuration_fingerprint": plan.configuration_fingerprint,
        "sources": inputs(plan.source_inputs),
        "templates": inputs(plan.template_inputs),
        "media": inputs(plan.media_inputs),
        "conjugation_service_fingerprint": (
            None if plan.conjugation_plan is None else plan.conjugation_plan.fingerprint
        ),
    }
    return _sha(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _finish_plan(config: ProjectConfig, **values: object) -> DeckPackagePlan:
    output = values.get("output_path")
    if not isinstance(output, Path):
        raise DeckPackageError("Deck package plan needs one output path.")
    # Every planner ends here, so this is where the capability table's media
    # column is checked against what the planner actually bound. A `pattern`
    # or `kanji` package is a legitimate package with no media at all; a
    # media input under one of those kinds means a planner and the table
    # disagree about what the artifact contains.
    capability = deck_capabilities.capability(str(values.get("kind")))
    if not capability.packages_media and values.get("media_inputs"):
        raise DeckPackageError(
            f"A {capability.kind} deck packages no media, but this plan binds "
            f"media inputs for {output}."
        )
    output_revision, output_identity = _output_binding(output)
    draft = DeckPackagePlan(  # type: ignore[arg-type]
        output_revision=output_revision,
        output_identity=output_identity,
        configuration_fingerprint=_configuration_fingerprint(config),
        fingerprint="0" * 64,
        **values,
    )
    return replace(draft, fingerprint=_fingerprint(config, draft))


def _output_binding(path: Path) -> tuple[str | None, tuple[int, int] | None]:
    """Bind either exact existing package bytes and inode, or exact absence."""

    try:
        state, revision, _payload = read_bytes_bound_snapshot(path)
    except FileNotFoundError:
        return None, None
    except (DataError, OSError) as exc:
        raise DeckPackageError(
            f"Deck package output must be absent or a direct regular file: {path}: {exc}"
        ) from exc
    return revision, state[:2]


def _configuration_fingerprint(config: ProjectConfig) -> str:
    """Bind every project setting that can alter an exporter consequence."""

    payload = {
        "paths": {
            "normalized": _relative(config, config.normalized_file),
            "decks": _relative(config, config.deck_dir),
            "templates": _relative(config, config.template_dir),
            "dist": _relative(config, config.dist_dir),
            "ledger": _relative(config, config.ledger_file),
            "media": _relative(config, config.media_dir),
            "kanji": _relative(config, config.kanji_file),
            "kanji_notes": _relative(config, config.kanji_notes_file),
            "jpdb_readings": _relative(config, config.jpdb_readings_file),
            "patterns": _relative(config, config.patterns_file),
        },
        "defaults": {
            "deck_name": config.default_deck_name,
            "deck_id": config.default_deck_id,
            "model_id_base": config.model_id_base,
            "cards": dict(config.default_cards),
            "max_meanings": config.max_meanings,
        },
    }
    return _sha(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _plan_vocabulary(config: ProjectConfig, target: Path) -> DeckPackagePlan:
    deck_config, records = resolve_deck_records(target)
    issues = validate_records(records, target)
    if has_errors(issues):
        raise DeckPackageError(refusal_text(target.name, issues))
    card_types = tuple(resolve_card_types(deck_config, config))
    source = _source_path(target, deck_config)
    sources = tuple(
        [
            *([_input(config, "record source", source)] if source is not None else []),
            _input(
                config,
                "kanji reference store",
                config.kanji_file,
                optional=True,
            ),
            _input(
                config,
                "jpdb reading facts",
                config.jpdb_readings_file,
                optional=True,
            ),
        ]
    )
    media = tuple(
        _input(config, "packaged media", path) for path in resolve_deck_media_paths(target, config)
    )
    name = str(deck_config.get("name", config.default_deck_name))
    return _finish_plan(
        config,
        repository_root=config.root.resolve(),
        deck_path=target,
        output_path=_safe_output(config, resolve_deck_output_path(target, deck_config, config)),
        kind="vocabulary",
        deck_name=name,
        variant="",
        card_types=card_types,
        note_count=len(records),
        card_count=len(records) * len(card_types),
        record_ids=tuple(record.id for record in records),
        deck_input=_input(config, "configured deck", target),
        source_inputs=sources,
        template_inputs=_vocabulary_templates(config, card_types),
        media_inputs=media,
        conjugation_plan=None,
    )


def _plan_pattern(config: ProjectConfig, target: Path) -> DeckPackagePlan:
    store = patterns.load_store(config.patterns_file)
    problems = pattern_cards.deck_problems(target, store, config)
    if problems:
        raise DeckPackageError(f"{target}: " + "; ".join(problems))
    raw = load_structured(target)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("deck"), Mapping):
        raise DeckPackageError(f"Deck file must contain a deck mapping: {target}")
    section = raw["deck"]
    assert isinstance(section, Mapping)
    document = str(section.get("document") or "").strip()
    cards = pattern_cards.cards_for(store[document])
    filename = str(section.get("output", f"{target.stem}.apkg"))
    templates = tuple(
        _text_input(config, "pattern template", path)
        for path in pattern_cards.pattern_template_paths(config.template_dir)
    )
    return _finish_plan(
        config,
        repository_root=config.root.resolve(),
        deck_path=target,
        output_path=_safe_output(config, config.dist_dir / filename),
        kind="pattern",
        deck_name=str(section.get("name") or target.stem),
        variant=document,
        card_types=("rule",),
        note_count=len(cards),
        card_count=len(cards),
        record_ids=(),
        deck_input=_input(config, "configured deck", target),
        source_inputs=(
            _input(
                config,
                "reviewed pattern store",
                config.patterns_file,
                optional=True,
            ),
        ),
        template_inputs=templates,
        media_inputs=(),
        conjugation_plan=None,
    )


def _wrap_conjugation(
    config: ProjectConfig,
    service: deck_build.ConjugationDeckBuildPlan,
) -> DeckPackagePlan:
    deck_input = _input(config, "configured deck", service.deck_path)
    source_input = _input(config, "record source", service.source_path)
    if deck_input.sha256 != service.deck_sha256 or source_input.sha256 != service.source_sha256:
        raise DeckPackageError(
            "Conjugation package inputs changed while its service plan was being "
            "wrapped; retry from a fresh deck list."
        )
    if len(service.record_ids) != service.card_count:
        raise DeckPackageError(
            "Conjugation package service returned a card scope that does not "
            "match its card count."
        )
    return _finish_plan(
        config,
        repository_root=config.root.resolve(),
        deck_path=service.deck_path,
        output_path=_safe_output(config, service.output_path),
        kind="conjugation",
        deck_name=service.deck_name,
        variant=service.form,
        card_types=("drill",),
        note_count=service.card_count,
        card_count=service.card_count,
        record_ids=service.record_ids,
        deck_input=deck_input,
        source_inputs=(source_input,),
        template_inputs=tuple(
            DeckPackageInput("pattern template", item.path, item.sha256)
            for item in service.template_inputs
        ),
        media_inputs=tuple(
            DeckPackageInput("packaged media", item.path, item.sha256)
            for item in service.media_inputs
        ),
        conjugation_plan=service,
    )


def _plan_kanji(config: ProjectConfig, target: Path) -> DeckPackagePlan:
    """Bind a character package: its curated notes, its templates, nothing else.

    The provider facts file is deliberately not an input. Every figure a
    character card shows was copied onto its note when the note was written,
    so a build reads the curated store and the templates — and a refresh of
    the facts does not silently restate what a built package claims.
    """
    deck = kanji_cards.resolve_kanji_deck_notes(target, config)
    filename = str(deck.section.get("output", f"{target.stem}.apkg"))
    return _finish_plan(
        config,
        repository_root=config.root.resolve(),
        deck_path=target,
        output_path=_safe_output(config, config.dist_dir / filename),
        kind="kanji",
        deck_name=deck.deck_name,
        variant="+".join(deck.directions),
        card_types=deck.directions,
        note_count=len(deck.notes),
        card_count=len(deck.notes) * len(deck.directions),
        record_ids=tuple(note.id for note in deck.notes),
        deck_input=_input(config, "configured deck", target),
        source_inputs=(_input(config, "character note store", deck.source_path),),
        template_inputs=tuple(
            _text_input(config, "character template", path)
            for path in kanji_cards.kanji_template_paths(
                config.template_dir, deck.directions
            )
        ),
        media_inputs=(),
        conjugation_plan=None,
    )


def _package_capability(target: Path) -> deck_capabilities.DeckMediaCapability:
    """Resolve a configured deck's media contract before any planner runs.

    Refused here rather than inside a planner: a kind with no capability row
    has no stated media contract, so nothing may package, publish or prune for
    it — and a planner that ran first would already have read its inputs.
    """
    return deck_capabilities.capability(deck_kind(target))


def _plan_unlocked(config: ProjectConfig, target: Path) -> DeckPackagePlan:
    kind = _package_capability(target).kind
    if kind in {"", "vocabulary"}:
        return _plan_vocabulary(config, target)
    if kind == "pattern":
        return _plan_pattern(config, target)
    if kind == "kanji":
        return _plan_kanji(config, target)
    if kind == "conjugation":
        return _wrap_conjugation(
            config,
            deck_build.plan_conjugation_deck_build(config, target),
        )
    raise DeckPackageError(f"Unsupported configured deck kind {kind!r}: {target}")


def _locked_paths(plan: DeckPackagePlan) -> set[Path]:
    return {
        plan.deck_path,
        plan.output_path,
        *(item.path for item in plan.all_inputs),
    }


def _assert_whole_deck(record_ids: Sequence[str]) -> None:
    if isinstance(record_ids, str | bytes):
        raise DeckPackageError("Deck package record ids must be a list of text.")
    if tuple(record_ids):
        raise DeckPackageError(
            "A deck build always packages the complete configured deck; record "
            "ids cannot narrow it."
        )


def plan_deck_package(
    config: ProjectConfig,
    deck_path: Path | str,
    *,
    record_ids: Sequence[str] = (),
) -> DeckPackagePlan:
    """Prepare one display-only exact package plan for any configured deck."""

    _assert_whole_deck(record_ids)
    target = _known_deck(config, deck_path)
    try:
        kind = _package_capability(target).kind
    except DeckPackageError:
        raise
    except (JankiError, OSError, ValueError) as exc:
        raise DeckPackageError(f"Could not plan deck package: {exc}") from exc
    if kind == "conjugation":
        try:
            return _wrap_conjugation(
                config,
                deck_build.plan_conjugation_deck_build(config, target),
            )
        except DeckPackageError:
            raise
        except (JankiError, OSError, ValueError) as exc:
            raise DeckPackageError(f"Could not plan deck package: {exc}") from exc

    try:
        with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
            first = _plan_unlocked(config, target)
            paths = _locked_paths(first) - {target}
            with contextlib.ExitStack() as locks:
                for path in sorted(paths, key=str):
                    locks.enter_context(exclusive_path_lock(path))
                fresh = _plan_unlocked(config, target)
                if _locked_paths(fresh) != _locked_paths(first):
                    raise DeckPackageError(
                        "Deck package input paths changed while the plan was being "
                        "prepared; retry from a fresh deck list."
                    )
                return fresh
    except DeckPackageError:
        raise
    except (JankiError, OSError, ValueError) as exc:
        raise DeckPackageError(f"Could not plan deck package: {exc}") from exc


def _canonical_records_text(records: Sequence[VocabularyRecord]) -> str:
    """The prospective collection's exact canonical bytes.

    Delegates to :func:`io.records_json_text` so a projected package plans over
    the same bytes a save would write; the refusal is re-raised in this
    module's own error so a caller still learns which planning step refused.
    """

    try:
        return records_json_text(records)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DeckPackageError(
            f"Prospective package records cannot be serialized exactly: {exc}"
        ) from exc


def _reference_paths(config: ProjectConfig) -> tuple[tuple[str, Path], ...]:
    """The two optional reference stores a vocabulary package reads, in order."""

    return (
        ("kanji reference store", _lexical(config.kanji_file)),
        ("jpdb reading facts", _lexical(config.jpdb_readings_file)),
    )


def _normalized_reference_sha256(
    config: ProjectConfig,
    reference_sha256: Mapping[Path, str | None] | None,
) -> dict[Path, str | None]:
    """Accept prepared after-hashes for exactly the two reference stores.

    This is not an input override. A projection is planned before the finish's
    own reference writes land, so those two files — and only those two — may be
    described by the payload that is about to write them. Everything else the
    package reads is bound from the bytes on disk, and a ``None`` here is the
    *absent* state the fresh plan will have to reproduce, never a wildcard.
    """

    supplied = dict(reference_sha256 or {})
    if not supplied:
        return {}
    allowed = {path: label for label, path in _reference_paths(config)}
    if len(allowed) != 2:
        raise DeckPackageError(
            "Prospective package reference hashes need two distinct reference "
            "stores; this configuration points both at one file."
        )
    normalized: dict[Path, str | None] = {}
    for raw_path, sha256 in supplied.items():
        path = _lexical(raw_path)
        if path not in allowed:
            raise DeckPackageError(
                "Prospective package reference hashes cover the kanji reference "
                f"store and the jpdb reading facts only; {path} is another input."
            )
        if path in normalized:
            raise DeckPackageError(f"Prospective package reference repeats {path}.")
        if sha256 is not None and not _is_sha256(sha256):
            raise DeckPackageError(
                f"Prospective package reference hash is malformed for {path}."
            )
        normalized[path] = sha256
    return normalized


def _reference_input(
    config: ProjectConfig,
    label: str,
    path: Path,
    references: Mapping[Path, str | None],
) -> DeckPackageInput:
    target = _lexical(path)
    if target not in references:
        return _input(config, label, target, optional=True)
    _relative(config, target)
    return DeckPackageInput(label=label, path=target, sha256=references[target])


def _plan_vocabulary_revision_unlocked(
    config: ProjectConfig,
    target: Path,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    media_sha256: Mapping[Path, str | None],
    reference_sha256: Mapping[Path, str | None] | None = None,
) -> DeckPackagePlan:
    canonical = config.normalized_file.resolve()
    if revision.path.resolve() != canonical or revision.text is None:
        raise DeckPackageError(
            "A prospective vocabulary package needs a present canonical revision."
        )
    if revision.text != _canonical_records_text(records):
        raise DeckPackageError(
            "Prospective package records do not reproduce their exact canonical bytes."
        )
    if (deck_kind(target) or "vocabulary") != "vocabulary":
        raise DeckPackageError(
            "Prospective canonical package planning supports vocabulary decks only."
        )
    references = _normalized_reference_sha256(config, reference_sha256)
    try:
        deck_config, projected = project_deck_records(target, canonical, records)
        issues = validate_records(projected, target)
        if has_errors(issues):
            raise DeckPackageError(refusal_text(target.name, issues))
        card_types = tuple(resolve_card_types(deck_config, config))
        source = _source_path(target, deck_config)
        if source != canonical:
            raise DeckPackageError(
                f"Vocabulary deck {target.name} does not read the canonical collection."
            )
        normalized_media: dict[Path, str | None] = {}
        for raw_path, sha256 in media_sha256.items():
            path = _lexical(raw_path)
            _relative(config, path)
            if sha256 is not None and not _is_sha256(sha256):
                raise DeckPackageError(
                    f"Prospective package media hash is malformed for {path}."
                )
            if path in normalized_media:
                raise DeckPackageError(
                    f"Prospective package media repeats {path}."
                )
            normalized_media[path] = sha256
        paths = project_deck_media_paths(
            target,
            config,
            canonical,
            records,
            allowed_missing_media=frozenset(normalized_media),
        )
        if set(paths) != set(normalized_media):
            raise DeckPackageError(
                "Prospective package media does not exactly match the projected deck."
            )
    except DeckPackageError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise DeckPackageError(f"Could not project vocabulary package: {exc}") from exc
    return _finish_plan(
        config,
        repository_root=config.root.resolve(),
        deck_path=target,
        output_path=_safe_output(
            config,
            resolve_deck_output_path(target, deck_config, config),
        ),
        kind="vocabulary",
        deck_name=str(deck_config.get("name", config.default_deck_name)),
        variant="",
        card_types=card_types,
        note_count=len(projected),
        card_count=len(projected) * len(card_types),
        record_ids=tuple(record.id for record in projected),
        deck_input=_input(config, "configured deck", target),
        source_inputs=(
            DeckPackageInput("record source", canonical, _sha(revision.text.encode("utf-8"))),
            *(
                _reference_input(config, label, path, references)
                for label, path in _reference_paths(config)
            ),
        ),
        template_inputs=_vocabulary_templates(config, card_types),
        media_inputs=tuple(
            DeckPackageInput("packaged media", path, normalized_media[path])
            for path in sorted(paths, key=str)
        ),
        conjugation_plan=None,
    )


def plan_vocabulary_deck_package_revision(
    config: ProjectConfig,
    deck_path: Path | str,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    *,
    media_sha256: Mapping[Path, str | None],
    reference_sha256: Mapping[Path, str | None] | None = None,
) -> DeckPackagePlan:
    """Bind a vocabulary package as it will exist after one exact revision.

    The supplied paths and their future hashes belong to a larger receipt;
    they may therefore be absent at projection time.  This value is not
    directly executable until those bytes exist and the caller revalidates
    every projected input against the ordinary package plan.

    ``reference_sha256`` is consulted only where the two optional reference
    stores are read, and carries the exact prepared after-hashes of the
    finish's own reference write.  Those bytes reach the card's character
    block rather than its media, so the projected record, media and counts are
    the same either way; what the payload changes is the input the realized
    plan will have to equal.
    """

    target = _known_deck(config, deck_path)
    try:
        with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
            first = _plan_vocabulary_revision_unlocked(
                config, target, records, revision, media_sha256, reference_sha256
            )
            canonical = revision.path.resolve()
            paths = _locked_paths(first) - {target, canonical}
            with contextlib.ExitStack() as locks:
                locks.enter_context(exclusive_path_lock(canonical))
                for path in sorted(paths, key=str):
                    locks.enter_context(exclusive_path_lock(path))
                fresh = _plan_vocabulary_revision_unlocked(
                    config, target, records, revision, media_sha256, reference_sha256
                )
                _assert_same(first, fresh)
                return fresh
    except DeckPackageError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise DeckPackageError(f"Could not plan prospective vocabulary package: {exc}") from exc


def _assert_same(expected: DeckPackagePlan, fresh: DeckPackagePlan) -> None:
    if fresh != expected or fresh.fingerprint != expected.fingerprint:
        raise DeckPackageError(
            "The deck package plan changed after confirmation; reload and review the fresh plan."
        )


def _assert_same_after_output(expected: DeckPackagePlan, fresh: DeckPackagePlan) -> None:
    """Compare every input after build while allowing the planned output effect."""

    normalized = replace(
        fresh,
        output_revision=expected.output_revision,
        output_identity=expected.output_identity,
        fingerprint=expected.fingerprint,
    )
    if normalized != expected:
        raise DeckPackageError(
            "The deck package plan changed during generation; the generated "
            "package was not accepted."
        )


def _assert_pending_audio_clear(config: ProjectConfig) -> ledger.Ledger:
    try:
        book = ledger.load(config.ledger_file)
    except (JankiError, OSError) as exc:
        raise DeckPackageError(f"Could not verify pending audio: {exc}") from exc
    if book.pending_audio:
        raise DeckPackageError(
            "The repository has pending audio recovery. Finish that exact "
            "transaction before building any package."
        )
    return book


def _verify_package(path: Path, expected: Path) -> str:
    if _lexical(path) != expected:
        raise DeckPackageError("The deck builder returned an output outside the confirmed plan.")
    try:
        return _sha(read_bytes_bound(expected))
    except (DataError, OSError) as exc:
        raise DeckPackageError(f"Could not verify the generated package {expected}: {exc}") from exc


def _execute_nonconjugation_locked(
    config: ProjectConfig,
    expected: DeckPackagePlan,
) -> DeckPackageResult:
    with exclusive_path_lock(config.deck_dir), exclusive_path_lock(expected.deck_path):
        first = _plan_unlocked(config, expected.deck_path)
        paths = _locked_paths(first) - {expected.deck_path}
        with contextlib.ExitStack() as locks:
            for path in sorted(paths, key=str):
                locks.enter_context(exclusive_path_lock(path))
            fresh = _plan_unlocked(config, expected.deck_path)
            if _locked_paths(fresh) != _locked_paths(first):
                raise DeckPackageError(
                    "Deck package input paths changed while confirmation was being "
                    "checked; reload and review the fresh plan."
                )
            _assert_same(expected, fresh)
            book = _assert_pending_audio_clear(config)
            if fresh.kind == "vocabulary":
                result = build_deck(
                    fresh.deck_path,
                    config,
                    output_path=fresh.output_path,
                    output_expected_revision=fresh.output_revision,
                    output_expected_identity=fresh.output_identity,
                    output_expected_absent=fresh.output_revision is None,
                )
                package_sha = _verify_package(result.output_path, fresh.output_path)
                if (
                    result.note_count != fresh.note_count
                    or result.card_types != fresh.card_types
                    or result.record_ids != fresh.record_ids
                    or result.media_count != len(fresh.media_inputs)
                ):
                    raise DeckPackageError(
                        "The vocabulary builder reported a result outside the "
                        "confirmed package plan."
                    )
                warnings = result.warnings
                media_count = result.media_count
            elif fresh.kind == "kanji":
                result = kanji_cards.build_kanji_deck(
                    fresh.deck_path,
                    config,
                    output_path=fresh.output_path,
                    output_expected_revision=fresh.output_revision,
                    output_expected_identity=fresh.output_identity,
                    output_expected_absent=fresh.output_revision is None,
                )
                package_sha = _verify_package(result.output_path, fresh.output_path)
                if (
                    result.note_count != fresh.note_count
                    or result.card_types != fresh.card_types
                    or result.record_ids != fresh.record_ids
                ):
                    raise DeckPackageError(
                        "The character builder reported a result outside the "
                        "confirmed package plan."
                    )
                warnings = result.warnings
                media_count = result.media_count
            elif fresh.kind == "pattern":
                target, count = pattern_cards.build_pattern_deck(
                    fresh.deck_path,
                    config,
                    patterns.load_store(config.patterns_file),
                    fresh.output_path,
                    output_expected_revision=fresh.output_revision,
                    output_expected_identity=fresh.output_identity,
                    output_expected_absent=fresh.output_revision is None,
                )
                package_sha = _verify_package(target, fresh.output_path)
                if count != fresh.card_count:
                    raise DeckPackageError(
                        "The pattern builder reported a result outside the confirmed package plan."
                    )
                warnings = ()
                media_count = 0
            else:  # pragma: no cover - dispatch invariant
                raise DeckPackageError("A conjugation plan reached the non-conjugation executor.")

            after = _plan_unlocked(config, fresh.deck_path)
            _assert_same_after_output(expected, after)
            if fresh.kind == "vocabulary":
                try:
                    by_id = {
                        record.id: record
                        for record in resolve_deck_records(fresh.deck_path)[1]
                    }
                    for record_id in fresh.record_ids:
                        book.record_export(
                            record_id,
                            fresh.deck_path.stem,
                            gaps=_record_gaps(by_id[record_id]),
                        )
                    book.save()
                except (JankiError, OSError, KeyError) as exc:
                    raise DeckPackageError(
                        f"Package {fresh.output_path} landed with SHA-256 "
                        f"{package_sha}, but its export history could not be "
                        f"saved: {exc}"
                    ) from exc
            return DeckPackageResult(
                output_path=fresh.output_path,
                package_sha256=package_sha,
                note_count=fresh.note_count,
                card_count=fresh.card_count,
                card_types=fresh.card_types,
                media_count=media_count,
                warnings=tuple(warnings),
            )


def execute_deck_package(
    config: ProjectConfig,
    expected: DeckPackagePlan,
) -> DeckPackageResult:
    """Re-plan under repository locks and publish only an exact match."""

    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        return execute_deck_package_locked(config, expected)


def execute_deck_package_locked(
    config: ProjectConfig,
    expected: DeckPackagePlan,
) -> DeckPackageResult:
    """Publish one exact package while the caller owns the audio-operation lock."""

    if expected.repository_root != config.root.resolve():
        raise DeckPackageError("Deck package plan belongs to another repository.")
    target = _known_deck(config, expected.deck_path)
    if target != expected.deck_path:
        raise DeckPackageError("The configured deck changed after confirmation.")
    current_kind = _package_capability(target).kind or "vocabulary"
    if current_kind != expected.kind:
        raise DeckPackageError(
            "The deck package kind changed after confirmation; reload and "
            "review the fresh plan."
        )
    if expected.kind == "conjugation":
        service = expected.conjugation_plan
        if service is None:  # pragma: no cover - dataclass invariant
            raise DeckPackageError("Conjugation package plan is incomplete.")
        fresh = _wrap_conjugation(
            config,
            deck_build.plan_conjugation_deck_build(config, target),
        )
        _assert_same(expected, fresh)
        result = deck_build.execute_conjugation_deck_build_locked(config, service)
        return DeckPackageResult(
            output_path=result.output_path,
            package_sha256=result.package_sha256,
            note_count=fresh.note_count,
            card_count=result.card_count,
            card_types=fresh.card_types,
            media_count=len(fresh.media_inputs),
        )
    return _execute_nonconjugation_locked(config, expected)


# --- realizing one projected package ----------------------------------------
#
# A finish binds the whole-deck package *before* it spends anything, so its
# projection necessarily carries `None` where a paid clip's bytes will be. The
# comparator below is what lets that projection be realized without becoming a
# blank cheque: every other input must be equal, and the only value allowed to
# appear is a media hash the audio phase's immutable proof already accounts for
# (contracts §7.11). A whole-fingerprint compare cannot express that, and the
# permissive build comparator in the finish modules is explicitly too weak.


def _plan_field_names() -> tuple[str, ...]:
    return (
        "repository_root",
        "deck_path",
        "output_path",
        "kind",
        "deck_name",
        "variant",
        "card_types",
        "note_count",
        "card_count",
        "record_ids",
        "deck_input",
        "source_inputs",
        "template_inputs",
        "media_inputs",
        "output_revision",
        "output_identity",
        "configuration_fingerprint",
        "fingerprint",
        "conjugation_plan",
    )


def _named_inputs(values: Sequence[DeckPackageInput]) -> dict[Path, DeckPackageInput]:
    return {item.path: item for item in values}


def _assert_input_group_equal(
    label: str,
    projected: Sequence[DeckPackageInput],
    fresh: Sequence[DeckPackageInput],
) -> None:
    expected = _named_inputs(projected)
    observed = _named_inputs(fresh)
    for path in sorted(set(expected) | set(observed), key=str):
        if path not in observed:
            raise DeckPackageError(
                f"The package {label} {path} vanished after the projection was bound."
            )
        if path not in expected:
            raise DeckPackageError(
                f"The package {label} {path} appeared after the projection was bound."
            )
        if expected[path] != observed[path]:
            raise DeckPackageError(
                f"The package {label} {path} changed after the projection was bound."
            )
    if tuple(item.path for item in projected) != tuple(item.path for item in fresh):
        raise DeckPackageError(
            f"The package {label} order changed after the projection was bound."
        )


def _record_source_sha256(plan: DeckPackagePlan) -> str:
    for item in plan.source_inputs:
        if item.label == "record source" and item.sha256 is not None:
            return item.sha256
    raise DeckPackageError(
        "A prospective vocabulary package must bind its canonical record source."
    )


def assert_projection_realized(
    projection: DeckPackagePlan,
    fresh: DeckPackagePlan,
    *,
    audio_completion: audio_application.AudioCompletionProof,
) -> None:
    """Refuse unless ``fresh`` is exactly ``projection`` plus its proven audio.

    Pure: two plans and one already-revalidated proof, no disk read. Revalidating
    the proof against current bytes, the ledger and the journal is the caller's
    separate step, under the audio-operation lock — comparison and revalidation
    are deliberately not one function.

    Every field is compared for equality except the constructed ``fingerprint``,
    which differs by construction. The one allowance is a media path the
    projection bound to ``None``: it must now carry a real hash, and that hash
    must be the one this finish's own completion proof accounts for. A path the
    projection bound to a hash stays fixed; the path set cannot change.

    The proof's repository root and canonical digest must match the plan's. That
    is a scope check on the artifacts, not a proof that two jobs are different:
    two jobs can legitimately share a repository, a collection and a media
    directory, and binding this proof to *this* job's authority is the
    coordinator's own obligation.
    """

    if projection.kind != "vocabulary" or fresh.kind != "vocabulary":
        raise DeckPackageError(
            "Projection realization covers vocabulary packages only."
        )
    names = _plan_field_names()
    if {item.name for item in projection.__dataclass_fields__.values()} != set(names) or (
        {item.name for item in fresh.__dataclass_fields__.values()} != set(names)
    ):  # pragma: no cover - dataclass invariant, kept so a new field is noticed
        raise DeckPackageError(
            "The package plan shape changed; projection realization must be updated."
        )
    for name in names:
        if name in {"fingerprint", "media_inputs", "source_inputs", "template_inputs"}:
            continue
        if getattr(projection, name) != getattr(fresh, name):
            raise DeckPackageError(
                f"The package {name.replace('_', ' ')} changed after the "
                "projection was bound."
            )
    _assert_input_group_equal("source input", projection.source_inputs, fresh.source_inputs)
    _assert_input_group_equal(
        "template input", projection.template_inputs, fresh.template_inputs
    )
    if audio_completion.repository_root != projection.repository_root:
        raise DeckPackageError(
            "The audio completion proof belongs to another repository."
        )
    if audio_completion.canonical_sha256 != _record_source_sha256(projection):
        raise DeckPackageError(
            "The audio completion proof does not cover the exact canonical "
            "records this package projection was planned over."
        )
    realized = audio_completion.realized_media_sha256
    origins = audio_completion.origin_by_media_path
    expected_media = _named_inputs(projection.media_inputs)
    observed_media = _named_inputs(fresh.media_inputs)
    for path in sorted(set(expected_media) | set(observed_media), key=str):
        if path not in observed_media:
            raise DeckPackageError(
                f"The packaged media {path} vanished after the projection was bound."
            )
        if path not in expected_media:
            raise DeckPackageError(
                f"The packaged media {path} appeared after the projection was bound."
            )
        projected_sha = expected_media[path].sha256
        observed_sha = observed_media[path].sha256
        if expected_media[path].label != observed_media[path].label:
            raise DeckPackageError(
                f"The packaged media {path} changed after the projection was bound."
            )
        if projected_sha is not None:
            if projected_sha != observed_sha:
                raise DeckPackageError(
                    f"The packaged media {path} changed after the projection was bound."
                )
            continue
        if not _is_sha256(observed_sha):
            raise DeckPackageError(
                f"The packaged media {path} was enumerated for this finish's audio "
                "but still has no bytes."
            )
        proven = realized.get(path)
        if proven is None:
            raise DeckPackageError(
                f"The packaged media {path} was enumerated for this finish's audio "
                "but the completion proof accounts for no clip there."
            )
        if proven != observed_sha:
            raise DeckPackageError(
                f"The packaged media {path} does not hold the bytes this finish's "
                "audio completion proved."
            )
        if origins.get(path) == "reused":
            raise DeckPackageError(
                f"The packaged media {path} was enumerated as new work but its "
                "proof reports a reused clip, whose hash the projection fixes."
            )
    if tuple(item.path for item in projection.media_inputs) != tuple(
        item.path for item in fresh.media_inputs
    ):  # pragma: no cover - both are sorted by path, so equal sets are equal orders
        raise DeckPackageError(
            "The packaged media order changed after the projection was bound."
        )


# --- what the built archive actually contains -------------------------------


@dataclass(frozen=True, slots=True)
class DeckPackageNote:
    """One note as the archive stores it, matched to the record that made it."""

    record_id: str
    guid: str
    #: Positional, exactly the ``\x1f``-joined bytes in ``notes.flds``.
    fields: tuple[str, ...]
    tags: tuple[str, ...]
    #: The enabled template ordinals genanki expanded this note into.
    card_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeckPackageInventory:
    """The whole deck, read back out of the artifact it was written into.

    Whole-deck throughout, with one named exception: the three sentence numbers
    below describe the audio proof's own scope, which is the job's selection.

    Content only. genanki mints note and card ids, ``mod``, ``usn`` and
    ``col.crt`` from the build clock and the zip carries per-member timestamps,
    so APKG bytes are not reproducible across builds. The artifact SHA proves
    *this file*; this inventory proves *what it contains*, which is what a free
    rebuild has to reproduce.
    """

    deck_id: int
    deck_name: str
    deck_description: str
    model_id: int
    model_name: str
    field_names: tuple[str, ...]
    card_types: tuple[str, ...]
    #: ``(display name, qfmt, afmt)`` per enabled direction, in card order.
    templates: tuple[tuple[str, str, str], ...]
    stylesheet_sha256: str
    notes: tuple[DeckPackageNote, ...]
    #: ``(packaged filename, sha256)``, sorted by name.
    media: tuple[tuple[str, str], ...]
    deck_sha256: str
    configuration_fingerprint: str
    output_path: str
    #: Stored sentence slots, exported sentence slots and unique clips: three
    #: numbers, never summed and never substituted for one another. Unlike every
    #: other field here they describe the **audio proof's own scope** — the job's
    #: selection — because they are counted from its slots, so a deck record the
    #: audio phase never covered contributes nothing to them. A consumer that
    #: prints them beside the whole-deck totals has to say which scope each is;
    #: `packaged_receipt` groups them under `audio_selection` with the exact
    #: record ids for that reason.
    stored_sentence_slots: int
    exported_sentence_slots: int
    sentence_clips: int
    fingerprint: str

    @property
    def card_count(self) -> int:
        """The whole deck's **validated** card total, from this archive.

        A sum of the per-note expansions each note was refused for not carrying,
        never a count of whatever rows the `cards` table happened to hold. It is
        a different number from `DeckPackagePlan.card_count`, which is
        ``len(records) × len(card_types)`` computed before the build: that is the
        capacity the deck's configuration allows, and genanki legitimately writes
        fewer when a template's question has no content for a note.
        """

        return sum(len(note.card_ordinals) for note in self.notes)

    def to_wire(self) -> dict[str, Any]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def _payload(self) -> dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "deck_name": self.deck_name,
            "deck_description": self.deck_description,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "field_names": list(self.field_names),
            "card_types": list(self.card_types),
            "templates": [list(item) for item in self.templates],
            "stylesheet_sha256": self.stylesheet_sha256,
            "notes": [
                {
                    "record_id": note.record_id,
                    "guid": note.guid,
                    "fields": list(note.fields),
                    "tags": list(note.tags),
                    "card_ordinals": list(note.card_ordinals),
                }
                for note in self.notes
            ],
            "media": [list(item) for item in self.media],
            "deck_sha256": self.deck_sha256,
            "configuration_fingerprint": self.configuration_fingerprint,
            "output_path": self.output_path,
            "stored_sentence_slots": self.stored_sentence_slots,
            "exported_sentence_slots": self.exported_sentence_slots,
            "sentence_clips": self.sentence_clips,
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> DeckPackageInventory:
        try:
            notes = tuple(
                DeckPackageNote(
                    record_id=str(item["record_id"]),
                    guid=str(item["guid"]),
                    fields=tuple(str(value) for value in item["fields"]),
                    tags=tuple(str(value) for value in item["tags"]),
                    card_ordinals=tuple(int(value) for value in item["card_ordinals"]),
                )
                for item in raw["notes"]
            )
            inventory = cls(
                deck_id=int(raw["deck_id"]),
                deck_name=str(raw["deck_name"]),
                deck_description=str(raw["deck_description"]),
                model_id=int(raw["model_id"]),
                model_name=str(raw["model_name"]),
                field_names=tuple(str(value) for value in raw["field_names"]),
                card_types=tuple(str(value) for value in raw["card_types"]),
                templates=tuple(
                    (str(item[0]), str(item[1]), str(item[2]))
                    for item in raw["templates"]
                ),
                stylesheet_sha256=str(raw["stylesheet_sha256"]),
                notes=notes,
                media=tuple(
                    (str(item[0]), str(item[1])) for item in raw["media"]
                ),
                deck_sha256=str(raw["deck_sha256"]),
                configuration_fingerprint=str(raw["configuration_fingerprint"]),
                output_path=str(raw["output_path"]),
                stored_sentence_slots=int(raw["stored_sentence_slots"]),
                exported_sentence_slots=int(raw["exported_sentence_slots"]),
                sentence_clips=int(raw["sentence_clips"]),
                fingerprint="0" * 64,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DeckPackageError(
                f"A package inventory is malformed: {exc}"
            ) from exc
        fingerprint = raw.get("fingerprint")
        if not _is_sha256(fingerprint):
            raise DeckPackageError("A package inventory needs a lowercase SHA-256.")
        restored = replace(inventory, fingerprint=str(fingerprint))
        if _inventory_fingerprint(restored) != restored.fingerprint:
            raise DeckPackageError(
                "A package inventory does not match its own fingerprint."
            )
        return restored


def _inventory_fingerprint(inventory: DeckPackageInventory) -> str:
    return _sha(_canonical_bytes(inventory._payload()))


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeckPackageError(
            f"Package evidence is not finite JSON: {exc}"
        ) from exc


def _unsupported_archive(path: Path, detail: str) -> DeckPackageError:
    return DeckPackageError(
        f"Unsupported package archive layout in {path}: {detail}. janki reads "
        "the genanki layout — collection.anki2, a media manifest and its "
        "numbered members — and guesses at no second one."
    )


def _archive_members(path: Path) -> tuple[bytes, dict[str, str]]:
    """The collection database and ``packaged name -> sha256``, or a refusal."""

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if any(info.is_dir() for info in infos):
                raise _unsupported_archive(path, "it carries directory entries")
            if len(set(names)) != len(names):
                raise _unsupported_archive(path, "it repeats a member name")
            if _ARCHIVE_COLLECTION not in names:
                raise _unsupported_archive(
                    path, f"there is no {_ARCHIVE_COLLECTION} member"
                )
            if _ARCHIVE_MANIFEST not in names:
                raise _unsupported_archive(path, "there is no media manifest")
            try:
                manifest = json.loads(archive.read(_ARCHIVE_MANIFEST).decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise _unsupported_archive(
                    path, f"its media manifest is not JSON ({exc})"
                ) from exc
            if not isinstance(manifest, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in manifest.items()
            ):
                raise _unsupported_archive(
                    path, "its media manifest is not a name mapping"
                )
            expected = {_ARCHIVE_COLLECTION, _ARCHIVE_MANIFEST} | set(manifest)
            if set(names) != expected:
                raise _unsupported_archive(
                    path,
                    "its members and its media manifest disagree "
                    f"({sorted(set(names) ^ expected)})",
                )
            media: dict[str, str] = {}
            for index, name in sorted(manifest.items()):
                if not index.isdigit():
                    raise _unsupported_archive(
                        path, f"media member {index!r} is not a numbered entry"
                    )
                if not name or PurePosixPath(name).name != name:
                    raise _unsupported_archive(
                        path, f"media entry {name!r} is not one filename"
                    )
                if name in media:
                    raise _unsupported_archive(
                        path, f"media entry {name!r} appears twice"
                    )
                media[name] = _sha(archive.read(index))
            collection = archive.read(_ARCHIVE_COLLECTION)
    except DeckPackageError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise _unsupported_archive(path, str(exc)) from exc
    return collection, media


_ArchiveCollection = tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[tuple[str, int, str, str]],
    list[tuple[str, int, int]],
]


def _archive_collection(path: Path, collection: bytes) -> _ArchiveCollection:
    """Read the content-only columns of the packaged collection database."""

    with tempfile.TemporaryDirectory(prefix="janki-inventory-") as temporary:
        database = Path(temporary).resolve() / _ARCHIVE_COLLECTION
        database.write_bytes(collection)
        connection = sqlite3.connect(database)
        try:
            row = connection.execute("select models, decks from col").fetchone()
            notes = connection.execute(
                "select guid, mid, flds, tags from notes"
            ).fetchall()
            cards = connection.execute("select nid, ord, did from cards").fetchall()
            note_ids = connection.execute("select id, guid from notes").fetchall()
        except sqlite3.DatabaseError as exc:
            raise _unsupported_archive(
                path, f"its collection database is not readable ({exc})"
            ) from exc
        finally:
            connection.close()
    if row is None:
        raise _unsupported_archive(path, "its collection has no configuration row")
    try:
        models = json.loads(row[0])
        decks = json.loads(row[1])
    except (TypeError, ValueError) as exc:
        raise _unsupported_archive(
            path, f"its notetype or deck configuration is not JSON ({exc})"
        ) from exc
    if not isinstance(models, Mapping) or not isinstance(decks, Mapping):
        raise _unsupported_archive(
            path, "its notetype or deck configuration is not a mapping"
        )
    by_note_id = {int(identifier): str(guid) for identifier, guid in note_ids}
    expanded = [
        (by_note_id.get(int(nid), ""), int(ordinal), int(deck_id))
        for nid, ordinal, deck_id in cards
    ]
    if any(not guid for guid, _ordinal, _deck in expanded):
        raise _unsupported_archive(path, "a packaged card names no note")
    return models, decks, notes, expanded


def _exported_example_fields(
    record: VocabularyRecord,
) -> tuple[tuple[str, int], ...]:
    """The note fields a card draws a sentence clip into, with their positions.

    The exporter decides which stored examples reach fields; this reads that
    decision off the same two selectors it uses, so a record with more stored
    examples than slots keeps them voiced and linked without inventing a third
    field or distorting the exported count.
    """

    chosen = (
        ("ExampleAudio", record.main_example()),
        ("CasualAudio", record.example_in("casual")),
    )
    exported: list[tuple[str, int]] = []
    for field_name, example in chosen:
        for position, candidate in enumerate(record.examples):
            if candidate is example:
                exported.append((field_name, position))
                break
    return tuple(exported)


def _assert_packaged_audio_matches_proof(
    records: Sequence[VocabularyRecord],
    notes: Mapping[str, tuple[str, ...]],
    media: Mapping[str, str],
    audio_completion: audio_application.AudioCompletionProof,
) -> tuple[int, int, int]:
    """Prove each exported slot's stored sound field and packaged bytes.

    Sentence coverage is checked slot by slot against the proof, never from an
    aggregate media count: a word clip is in the same manifest and cannot stand
    in for a sentence. Returns the three counts that stay three numbers —
    stored sentence slots, exported sentence slots, unique sentence clips.
    """

    guid_for = anki_exporter.genanki.guid_for
    index = {name: position for position, name in enumerate(FIELD_NAMES)}
    exported = 0
    for record in records:
        fields = notes.get(guid_for(record.id))
        if fields is None or len(fields) != len(FIELD_NAMES):
            continue
        word = audio_completion.slots_for(record.id, "word", None)
        if word is not None:
            if fields[index["Audio"]] != f"[sound:{word.target}]":
                raise DeckPackageError(
                    f"The packaged note for {record.id} does not draw its proven "
                    f"word clip {word.target!r}."
                )
            if media.get(word.target) != word.media_sha256:
                raise DeckPackageError(
                    f"The packaged word clip {word.target!r} for {record.id} is "
                    "missing or does not hold its proven bytes."
                )
        for field_name, position in _exported_example_fields(record):
            slot = audio_completion.slots_for(record.id, "example", position)
            if slot is None:
                continue
            exported += 1
            if fields[index[field_name]] != f"[sound:{slot.target}]":
                raise DeckPackageError(
                    f"The packaged note for {record.id} does not draw its proven "
                    f"sentence clip {slot.target!r} in {field_name}."
                )
            if media.get(slot.target) != slot.media_sha256:
                raise DeckPackageError(
                    f"The packaged sentence clip {slot.target!r} for {record.id} "
                    "is missing or does not hold its proven bytes."
                )
    sentence_slots = [slot for slot in audio_completion.slots if slot.kind == "example"]
    return (
        len(sentence_slots),
        exported,
        len({slot.target for slot in sentence_slots}),
    )


def _stylesheet_sha256(plan: DeckPackagePlan) -> str:
    for item in plan.template_inputs:
        if item.label == "card stylesheet" and item.sha256 is not None:
            return item.sha256
    raise DeckPackageError("A vocabulary package must bind its card stylesheet.")


def _card_ordinals_text(ordinals: Sequence[int]) -> str:
    """``no card``, ``card 0`` or ``cards 0, 1``, for one readable refusal.

    An empty expansion is the case worth naming in words: ``[]`` beside
    ``[0, 1]`` in a message reads as a formatting artefact, and "no card" is
    what actually went wrong.
    """

    if not ordinals:
        return "no card"
    listed = ", ".join(str(ordinal) for ordinal in ordinals)
    return f"card {listed}" if len(ordinals) == 1 else f"cards {listed}"


def _read_package_inventory(
    config: ProjectConfig,
    plan: DeckPackagePlan,
    rendering: RenderedDeck,
    records: Sequence[VocabularyRecord],
    audio_completion: audio_application.AudioCompletionProof,
    archive_path: Path,
) -> DeckPackageInventory:
    """Read the built archive back and validate it note by note.

    Expected values come from the owning exporter over the same bytes the build
    consumed, under the same locks, so this proves the archive faithfully
    carries what the renderer rendered from the reviewed inputs. It cannot
    catch a mutation *inside* that renderer — a second independent renderer
    would, and would drift, which is why the seam forbids one; renderer
    correctness stays with the exporter's own tests.
    """

    if anki_exporter.genanki is None:  # pragma: no cover - dev install carries it
        raise DeckPackageError(
            "genanki is not installed, so a package inventory cannot be read."
        )
    collection, media = _archive_members(archive_path)
    models, decks, raw_notes, cards = _archive_collection(archive_path, collection)
    model = next(
        (
            value
            for value in models.values()
            if isinstance(value, Mapping) and int(value.get("id", -1)) == rendering.model_id
        ),
        None,
    )
    if model is None:
        raise DeckPackageError(
            f"The packaged notetype {rendering.model_id} is not in {archive_path}."
        )
    deck = next(
        (
            value
            for value in decks.values()
            if isinstance(value, Mapping) and int(value.get("id", -1)) == rendering.deck_id
        ),
        None,
    )
    if deck is None:
        raise DeckPackageError(
            f"The packaged deck {rendering.deck_id} is not in {archive_path}."
        )
    if str(model.get("name", "")) != rendering.model_name:
        raise DeckPackageError("The packaged notetype name is not the one built.")
    if str(deck.get("name", "")) != rendering.deck_name:
        raise DeckPackageError("The packaged deck name is not the one built.")
    if str(deck.get("desc", "")) != rendering.deck_description:
        raise DeckPackageError("The packaged deck description is not the one built.")
    archived_fields = tuple(
        str(item.get("name", "")) for item in model.get("flds", []) or ()
    )
    if archived_fields != rendering.field_names:
        raise DeckPackageError(
            "The packaged notetype fields are not the exporter's field list."
        )
    archived_templates = tuple(
        (str(item.get("name", "")), str(item.get("qfmt", "")), str(item.get("afmt", "")))
        for item in model.get("tmpls", []) or ()
    )
    if archived_templates != rendering.templates:
        raise DeckPackageError(
            "The packaged card templates are not the ones this build rendered."
        )
    stylesheet_sha256 = _stylesheet_sha256(plan)
    if _sha(str(model.get("css", "")).encode("utf-8")) != stylesheet_sha256:
        raise DeckPackageError(
            "The packaged stylesheet is not the bound card stylesheet."
        )
    # The GUIDs and the card expansion both come from the notes the builder
    # packaged, through the owning exporter: `card_types` says what the deck
    # offers, and genanki's required-field evaluation says which of those
    # templates *this* note's rendered fields produce a card for. Asserting one
    # card per enabled direction would refuse a note genanki legitimately
    # expands into fewer, and counting whatever the archive holds is what let a
    # note with no cards at all through. `css=""` deliberately: the stylesheet
    # rides on the notetype, not on the expansion, and it is compared just
    # above against the hash the plan bound rather than against a file read
    # again here.
    expansion = anki_exporter.expand_deck(rendering, css="")
    expected_notes = {
        item.guid: rendered
        for rendered, item in zip(rendering.notes, expansion, strict=True)
    }
    expected_cards = {item.guid: item.card_ordinals for item in expansion}
    if len(expected_notes) != len(rendering.notes):  # pragma: no cover - ids are unique
        raise DeckPackageError("Two packaged notes share one GUID.")
    if len(raw_notes) != len(expected_notes):
        raise DeckPackageError(
            f"The archive holds {len(raw_notes)} notes; the build rendered "
            f"{len(expected_notes)}."
        )
    observed_fields = {
        str(guid): tuple(str(flds).split("\x1f")) for guid, _mid, flds, _tags in raw_notes
    }
    # The proof-to-artifact check runs first, and deliberately: it is the only
    # one that reads the audio completion rather than the plan, so a package
    # whose cards draw a clip nobody proved is refused on those terms instead
    # of being caught incidentally by the field comparison below.
    stored, exported, clips = _assert_packaged_audio_matches_proof(
        records, observed_fields, media, audio_completion
    )
    ordinals: dict[str, list[int]] = {}
    for guid, ordinal, deck_id in cards:
        if deck_id != rendering.deck_id:
            raise DeckPackageError(
                f"A packaged card for {guid} belongs to deck {deck_id}, not "
                f"{rendering.deck_id}."
            )
        ordinals.setdefault(guid, []).append(ordinal)
    notes: list[DeckPackageNote] = []
    for guid, model_id, flds, tags in raw_notes:
        rendered = expected_notes.get(guid)
        if rendered is None:
            raise DeckPackageError(
                f"The archive holds note {guid}, which this build did not render."
            )
        if int(model_id) != rendering.model_id:
            raise DeckPackageError(
                f"Packaged note {guid} uses notetype {model_id}, not "
                f"{rendering.model_id}."
            )
        fields = tuple(str(flds).split("\x1f"))
        if fields != rendered.fields:
            differing = next(
                (
                    FIELD_NAMES[position]
                    for position, (left, right) in enumerate(
                        zip(fields, rendered.fields, strict=False)
                    )
                    if left != right
                ),
                "field count",
            )
            raise DeckPackageError(
                f"Packaged note {rendered.record_id} does not carry the exact "
                f"{differing} the build rendered."
            )
        packaged_tags = tuple(str(tags).split())
        if packaged_tags != rendered.tags:
            raise DeckPackageError(
                f"Packaged note {rendered.record_id} does not carry the exact tags "
                "the build rendered."
            )
        # Sorted, and a multiset rather than a set: a deck whose two rows both
        # say template 0 has two recognition cards and no production card, and
        # its card *count* still matches the plan. An ordinal no enabled
        # template has is a difference from this expansion like any other, so
        # there is no separate bounds check to keep in step with it.
        archived_cards = tuple(sorted(ordinals.get(guid, ())))
        if archived_cards != expected_cards[guid]:
            raise DeckPackageError(
                f"Packaged note {rendered.record_id} does not carry the exact "
                "cards genanki expands its rendered fields into: the archive "
                f"has {_card_ordinals_text(archived_cards)}, the build renders "
                f"{_card_ordinals_text(expected_cards[guid])}."
            )
        notes.append(
            DeckPackageNote(
                record_id=rendered.record_id,
                guid=guid,
                fields=fields,
                tags=packaged_tags,
                card_ordinals=archived_cards,
            )
        )
    expected_media = {item.path.name: item.sha256 for item in plan.media_inputs}
    if len(expected_media) != len(plan.media_inputs):  # pragma: no cover - claimed names
        raise DeckPackageError("Two packaged media inputs share one filename.")
    for name in sorted(set(expected_media) | set(media)):
        if name not in media:
            raise DeckPackageError(
                f"The packaged media manifest is missing {name}, which the build "
                "referenced."
            )
        if name not in expected_media:
            raise DeckPackageError(
                f"The archive packages {name}, which this build did not reference."
            )
        if expected_media[name] != media[name]:
            raise DeckPackageError(
                f"The packaged media {name} does not hold the bound bytes."
            )
    draft = DeckPackageInventory(
        deck_id=rendering.deck_id,
        deck_name=rendering.deck_name,
        deck_description=rendering.deck_description,
        model_id=rendering.model_id,
        model_name=rendering.model_name,
        field_names=rendering.field_names,
        card_types=rendering.card_types,
        templates=rendering.templates,
        stylesheet_sha256=stylesheet_sha256,
        notes=tuple(sorted(notes, key=lambda note: note.record_id)),
        media=tuple(sorted(media.items())),
        deck_sha256=str(plan.deck_input.sha256),
        configuration_fingerprint=plan.configuration_fingerprint,
        output_path=_relative(config, plan.output_path),
        stored_sentence_slots=stored,
        exported_sentence_slots=exported,
        sentence_clips=clips,
        fingerprint="0" * 64,
    )
    return replace(draft, fingerprint=_inventory_fingerprint(draft))


# --- the durable build intent, its publication and its recovery --------------
#
# An APKG SHA cannot be predicted, so the order here is private build -> actual
# SHA -> intent -> publish (§7.1's "artifact first"). Everything a later step
# needs is bound in the preparation before the confirmed target is touched at
# all, and every date-bearing value is frozen now so a resume on the next day
# replays these exact bytes instead of recomputing a day.


@dataclass(frozen=True, slots=True)
class DeckPackageExport:
    """One frozen ``record_export`` row this publication owes the ledger."""

    record_id: str
    deck_stem: str
    gaps: tuple[str, ...]
    at: str


@dataclass(frozen=True, slots=True)
class DeckPackagePreparation:
    """One built-but-unpublished package and everything publishing it needs.

    ``packaged`` is a build proof, not completion: this value carries the
    evidence a later preview receipt is minted from and deliberately offers no
    download token and no finish action.
    """

    schema: str
    preparation_id: str
    #: The authority's whole-deck projection, with its enumerated audio slots.
    projection: DeckPackagePlan
    #: The fresh concrete plan every later strict comparison uses.
    plan: DeckPackagePlan
    audio_completion: audio_application.AudioCompletionProof
    staged_path: Path
    #: Also the published ``package_sha256``: publication copies these bytes.
    staged_sha256: str
    staged_identity: tuple[int, int]
    #: The **private** staging directory's identity, which the artifact is read
    #: back from.
    expected_directory_identity: tuple[int, int]
    #: The directory the confirmed target lives in, which publication writes
    #: into. A separate binding on purpose: replacing `dist/` and replacing
    #: `dist/.janki-prepared/` are different events, and the writer would
    #: otherwise create a missing output parent rather than refuse it.
    output_directory_identity: tuple[int, int]
    inventory: DeckPackageInventory
    output_before_revision: str | None
    output_before_identity: tuple[int, int] | None
    ledger_before_sha256: str
    export_delta: tuple[DeckPackageExport, ...]
    ledger_after_sha256: str
    warnings: tuple[str, ...]
    #: The preparation this one immutably supersedes, or ``""``.
    supersedes: str
    fingerprint: str

    @property
    def package_sha256(self) -> str:
        return self.staged_sha256

    def packaged_receipt(self, config: ProjectConfig) -> dict[str, Any]:
        """The exact evidence a ``packaged`` receipt carries — and no more.

        No download offer and no completion claim: those are minted from a
        verified preview receipt, which is a later phase's business.

        Two scopes, each named. ``note_count``, ``card_count``, ``card_types``
        and ``media_count`` are the **whole deck**, and ``card_count`` is the
        archive's validated expansion rather than the plan's configured capacity:
        this is a completed build proof, so it reports what the package was
        proven to contain. ``audio_selection`` is the **job's selection** — the
        three sentence numbers come from the audio proof's slots — and carries
        the record ids they describe, because six numbers at two scopes side by
        side otherwise read as one (§7.12).
        """

        return {
            "preparation_id": self.preparation_id,
            "output_path": _relative(config, self.plan.output_path),
            "package_sha256": self.staged_sha256,
            "inventory_fingerprint": self.inventory.fingerprint,
            "note_count": self.plan.note_count,
            "card_count": self.inventory.card_count,
            "card_types": list(self.plan.card_types),
            "media_count": len(self.plan.media_inputs),
            "audio_selection": {
                "record_ids": list(self.audio_completion.record_ids),
                "stored_sentence_slots": self.inventory.stored_sentence_slots,
                "exported_sentence_slots": self.inventory.exported_sentence_slots,
                "sentence_clips": self.inventory.sentence_clips,
            },
            "audio_completion_fingerprint": self.audio_completion.fingerprint,
            "ledger_after_sha256": self.ledger_after_sha256,
            "warnings": list(self.warnings),
            "supersedes": self.supersedes,
        }

    def to_wire(self, config: ProjectConfig) -> dict[str, Any]:
        payload = self._payload(config)
        payload["fingerprint"] = self.fingerprint
        return payload

    def _payload(self, config: ProjectConfig) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "preparation_id": self.preparation_id,
            "projection": _plan_wire(config, self.projection),
            "plan": _plan_wire(config, self.plan),
            "audio_completion": self.audio_completion.to_wire(),
            "staged_path": _relative(config, self.staged_path),
            "staged_sha256": self.staged_sha256,
            "staged_identity": list(self.staged_identity),
            "expected_directory_identity": list(self.expected_directory_identity),
            "output_directory_identity": list(self.output_directory_identity),
            "inventory": self.inventory.to_wire(),
            "output_before_revision": self.output_before_revision,
            "output_before_identity": (
                None
                if self.output_before_identity is None
                else list(self.output_before_identity)
            ),
            "ledger_before_sha256": self.ledger_before_sha256,
            "export_delta": [
                {
                    "record_id": entry.record_id,
                    "deck_stem": entry.deck_stem,
                    "gaps": list(entry.gaps),
                    "at": entry.at,
                }
                for entry in self.export_delta
            ],
            "ledger_after_sha256": self.ledger_after_sha256,
            "warnings": list(self.warnings),
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_wire(
        cls, config: ProjectConfig, raw: Mapping[str, Any]
    ) -> DeckPackagePreparation:
        if raw.get("schema") != PREPARED_PACKAGE_SCHEMA:
            raise DeckPackageError(
                f"Unknown prepared package schema {raw.get('schema')!r}."
            )
        for key in ("audio_completion", "inventory", "projection", "plan"):
            if not isinstance(raw.get(key), Mapping):
                raise DeckPackageError(f"A prepared package {key} is malformed.")
        delta = raw.get("export_delta")
        if not isinstance(delta, list) or any(
            not isinstance(item, Mapping) for item in delta
        ):
            raise DeckPackageError("A prepared package export delta is malformed.")
        try:
            proof = audio_application.AudioCompletionProof.from_wire(
                config, raw["audio_completion"]
            )
        except audio_application.AudioProofError as exc:
            raise DeckPackageError(
                f"A prepared package audio completion proof is malformed: {exc}"
            ) from exc
        preparation = cls(
            schema=PREPARED_PACKAGE_SCHEMA,
            preparation_id=_wire_identifier(raw.get("preparation_id")),
            projection=_plan_from_wire(config, raw["projection"]),
            plan=_plan_from_wire(config, raw["plan"]),
            audio_completion=proof,
            staged_path=_wire_path(config, raw.get("staged_path"), "staged artifact"),
            staged_sha256=_wire_digest(raw.get("staged_sha256"), "staged artifact"),
            staged_identity=_wire_identity(raw.get("staged_identity"), "staged artifact"),
            expected_directory_identity=_wire_identity(
                raw.get("expected_directory_identity"), "staging directory"
            ),
            output_directory_identity=_wire_identity(
                raw.get("output_directory_identity"), "output directory"
            ),
            inventory=DeckPackageInventory.from_wire(raw["inventory"]),
            output_before_revision=(
                None
                if raw.get("output_before_revision") is None
                else _wire_digest(raw.get("output_before_revision"), "bound output")
            ),
            output_before_identity=(
                None
                if raw.get("output_before_identity") is None
                else _wire_identity(raw.get("output_before_identity"), "bound output")
            ),
            ledger_before_sha256=_wire_digest(
                raw.get("ledger_before_sha256"), "ledger before state"
            ),
            export_delta=tuple(
                DeckPackageExport(
                    record_id=_wire_identifier(item.get("record_id")),
                    deck_stem=_wire_identifier(item.get("deck_stem")),
                    gaps=tuple(_wire_text_list(item.get("gaps"), "export gaps")),
                    at=_wire_identifier(item.get("at")),
                )
                for item in delta
            ),
            ledger_after_sha256=_wire_digest(
                raw.get("ledger_after_sha256"), "ledger after state"
            ),
            warnings=tuple(_wire_text_list(raw.get("warnings"), "build warnings")),
            supersedes=str(raw.get("supersedes") or ""),
            fingerprint=_wire_digest(raw.get("fingerprint"), "prepared package"),
        )
        if (preparation.output_before_revision is None) != (
            preparation.output_before_identity is None
        ):
            raise DeckPackageError(
                "A prepared package binds its output content and identity together."
            )
        if _preparation_fingerprint(config, preparation) != preparation.fingerprint:
            raise DeckPackageError(
                "A prepared package does not match its own fingerprint."
            )
        return preparation


def _preparation_fingerprint(
    config: ProjectConfig, preparation: DeckPackagePreparation
) -> str:
    return _sha(_canonical_bytes(preparation._payload(config)))


def _wire_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeckPackageError("A prepared package identifier is malformed.")
    return value


def _wire_text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DeckPackageError(f"A prepared package {label} list is malformed.")
    return [str(item) for item in value]


def _wire_digest(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise DeckPackageError(f"A prepared package {label} digest is malformed.")
    return str(value)


def _wire_identity(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise DeckPackageError(f"A prepared package {label} identity is malformed.")
    return int(value[0]), int(value[1])


def _wire_path(config: ProjectConfig, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DeckPackageError(f"A prepared package {label} path is malformed.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise DeckPackageError(
            f"A prepared package {label} path is not repository-relative."
        )
    return _lexical(config.root.resolve() / Path(*pure.parts))


def _input_wire(config: ProjectConfig, item: DeckPackageInput) -> dict[str, Any]:
    return {
        "label": item.label,
        "path": _relative(config, item.path),
        "sha256": item.sha256,
    }


def _input_from_wire(config: ProjectConfig, raw: object) -> DeckPackageInput:
    if not isinstance(raw, Mapping) or set(raw) != {"label", "path", "sha256"}:
        raise DeckPackageError("A prepared package input is malformed.")
    sha256 = raw.get("sha256")
    if sha256 is not None and not _is_sha256(sha256):
        raise DeckPackageError("A prepared package input hash is malformed.")
    try:
        return DeckPackageInput(
            label=str(raw.get("label")),
            path=_wire_path(config, raw.get("path"), "input"),
            sha256=None if sha256 is None else str(sha256),
        )
    except ValueError as exc:
        raise DeckPackageError(f"A prepared package input is malformed: {exc}") from exc


def _plan_wire(config: ProjectConfig, plan: DeckPackagePlan) -> dict[str, Any]:
    if plan.conjugation_plan is not None:
        raise DeckPackageError(
            "A prepared package covers vocabulary decks, which carry no "
            "conjugation service plan."
        )
    return {
        "repository_root": _relative(config, plan.repository_root)
        if plan.repository_root != config.root.resolve()
        else ".",
        "deck_path": _relative(config, plan.deck_path),
        "output_path": _relative(config, plan.output_path),
        "kind": plan.kind,
        "deck_name": plan.deck_name,
        "variant": plan.variant,
        "card_types": list(plan.card_types),
        "note_count": plan.note_count,
        "card_count": plan.card_count,
        "record_ids": list(plan.record_ids),
        "deck_input": _input_wire(config, plan.deck_input),
        "source_inputs": [_input_wire(config, item) for item in plan.source_inputs],
        "template_inputs": [_input_wire(config, item) for item in plan.template_inputs],
        "media_inputs": [_input_wire(config, item) for item in plan.media_inputs],
        "output_revision": plan.output_revision,
        "output_identity": (
            None if plan.output_identity is None else list(plan.output_identity)
        ),
        "configuration_fingerprint": plan.configuration_fingerprint,
        "plan_fingerprint": plan.fingerprint,
    }


def _plan_from_wire(config: ProjectConfig, raw: Mapping[str, Any]) -> DeckPackagePlan:
    for key in ("source_inputs", "template_inputs", "media_inputs"):
        if not isinstance(raw.get(key), list):
            raise DeckPackageError(f"A prepared package plan {key} is malformed.")
    if raw.get("kind") != "vocabulary":
        raise DeckPackageError("A prepared package plan must be a vocabulary package.")
    if raw.get("repository_root") != ".":
        raise DeckPackageError("A prepared package plan belongs to another repository.")
    identity = raw.get("output_identity")
    revision = raw.get("output_revision")
    if (revision is None) != (identity is None):
        raise DeckPackageError(
            "A prepared package plan binds its output content and identity together."
        )
    try:
        draft = DeckPackagePlan(
            repository_root=config.root.resolve(),
            deck_path=_wire_path(config, raw.get("deck_path"), "deck"),
            output_path=_wire_path(config, raw.get("output_path"), "output"),
            kind="vocabulary",
            deck_name=str(raw.get("deck_name")),
            variant=str(raw.get("variant")),
            card_types=tuple(_wire_text_list(raw.get("card_types"), "card types")),
            note_count=_wire_count(raw.get("note_count"), "note count"),
            card_count=_wire_count(raw.get("card_count"), "card count"),
            record_ids=tuple(_wire_text_list(raw.get("record_ids"), "record ids")),
            deck_input=_input_from_wire(config, raw.get("deck_input")),
            source_inputs=tuple(
                _input_from_wire(config, item) for item in raw["source_inputs"]
            ),
            template_inputs=tuple(
                _input_from_wire(config, item) for item in raw["template_inputs"]
            ),
            media_inputs=tuple(
                _input_from_wire(config, item) for item in raw["media_inputs"]
            ),
            output_revision=(
                None if revision is None else _wire_digest(revision, "bound output")
            ),
            output_identity=(
                None if identity is None else _wire_identity(identity, "bound output")
            ),
            configuration_fingerprint=_wire_digest(
                raw.get("configuration_fingerprint"), "configuration"
            ),
            fingerprint="0" * 64,
            conjugation_plan=None,
        )
    except ValueError as exc:
        raise DeckPackageError(f"A prepared package plan is malformed: {exc}") from exc
    restored = replace(draft, fingerprint=_fingerprint(config, draft))
    if restored.fingerprint != raw.get("plan_fingerprint"):
        raise DeckPackageError(
            "A prepared package plan does not match its own fingerprint."
        )
    return restored


def _wire_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeckPackageError(f"A prepared package {label} is malformed.")
    return value


#: The two directories a preparation binds, for the diagnostics below. They are
#: different things: one holds the private artifact, the other is where the
#: confirmed package is published.
_STAGING_DIRECTORY = "private package staging directory"
_OUTPUT_DIRECTORY = "package output directory"


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    """One bound directory's ``(dev, ino)``, or a refusal naming it.

    ``FileNotFoundError`` passes through deliberately rather than becoming a
    `DeckPackageError`: a private staging directory that is simply **not there**
    is a vanished stage, which §7.11's recovery table answers with a free
    rebuild, while a directory that *is* there under another identity is the
    refusal below. The two outcomes differ, so the absent case must stay
    distinguishable by type instead of by message text.
    """

    try:
        details = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise DeckPackageError(
            f"Could not bind the {label} {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(details.st_mode):
        raise DeckPackageError(f"The {label} {path} is not a directory.")
    return details.st_dev, details.st_ino


def _ledger_digest(book: ledger.Ledger) -> str:
    return _sha(book.serialized_text().encode("utf-8"))


def _load_ledger(config: ProjectConfig) -> ledger.Ledger:
    try:
        return ledger.load(config.ledger_file)
    except (JankiError, OSError) as exc:
        raise DeckPackageError(f"Could not read the export ledger: {exc}") from exc


def _staged_bytes(preparation: DeckPackagePreparation) -> bytes:
    """Re-verify the staged artifact at its bound inode inside its bound directory.

    A missing private directory raises `FileNotFoundError`, exactly as a missing
    artifact inside a surviving one does, because they are the same situation:
    the disposable stage is gone. `dist/` is not content authority, so the
    demonstrated case is `dist/.janki-prepared/` removed outright in this
    checkout — the directory absent, not merely emptied. A directory that is
    present under a different identity is a refusal, not a rebuild: something
    replaced the thing this intent bound.

    The absence stays typed here because recovery classifies it into §7.11's
    free-rebuild row; `_publish_locked` normalizes it into a named refusal for
    the public publication path.
    """

    if _directory_identity(preparation.staged_path.parent, _STAGING_DIRECTORY) != (
        preparation.expected_directory_identity
    ):
        raise DeckPackageError(
            "The private package staging directory was replaced since the "
            f"artifact {preparation.staged_path} was prepared."
        )
    state, revision, payload = read_bytes_bound_snapshot(preparation.staged_path)
    if state[:2] != preparation.staged_identity:
        raise DeckPackageError(
            f"The staged package {preparation.staged_path} was replaced by another "
            "file since it was prepared."
        )
    if revision != preparation.staged_sha256:
        raise DeckPackageError(
            f"The staged package {preparation.staged_path} no longer holds the "
            f"prepared bytes {preparation.staged_sha256}."
        )
    return payload


def _prepared_directory(config: ProjectConfig) -> Path:
    return _lexical(config.dist_dir / PREPARED_PACKAGE_DIR_NAME)


@contextlib.contextmanager
def _package_locks(config: ProjectConfig, deck_path: Path | str):
    """Take the package's own lock set, exactly as ordinary execution does."""

    target = _known_deck(config, deck_path)
    with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
        first = _plan_unlocked(config, target)
        paths = _locked_paths(first) - {target}
        with contextlib.ExitStack() as locks:
            for path in sorted(paths, key=str):
                locks.enter_context(exclusive_path_lock(path))
            fresh = _plan_unlocked(config, target)
            if _locked_paths(fresh) != _locked_paths(first):
                raise DeckPackageError(
                    "Deck package input paths changed while the prepared package "
                    "was being checked; reload and review the fresh plan."
                )
            yield target, fresh


def prepare_deck_package(
    config: ProjectConfig,
    projection: DeckPackagePlan,
    *,
    audio_completion: audio_application.AudioCompletionProof,
    supersedes: DeckPackagePreparation | None = None,
) -> DeckPackagePreparation:
    """Build the confirmed package privately and bind everything publishing needs.

    The caller **must** already hold
    ``exclusive_path_lock(config.root / ".janki-audio-operation")``; the audio
    completion proof is revalidated under it before anything is compared, and
    this function takes only the package's own lock set on top. The confirmed
    target is not touched: the artifact is built into a private
    ``dist/.janki-prepared/<preparation_id>.apkg`` and the canonical export
    ledger is not written.

    ``supersedes`` mints the free-rebuild attempt §7.11's recovery table calls
    for: a **new** preparation over the same projection, the same proof and the
    same bound input inventory, linked to the attempt it replaces. The old
    intent is never edited.
    """

    try:
        audio_application.revalidate_audio_completion(config, audio_completion)
    except audio_application.AudioProofError as exc:
        raise DeckPackageError(
            f"The audio completion proof is no longer valid: {exc}"
        ) from exc
    if projection.repository_root != config.root.resolve():
        raise DeckPackageError("The package projection belongs to another repository.")
    if supersedes is not None:
        if supersedes.projection != projection:
            raise DeckPackageError(
                "A superseding package preparation must rebuild the same bound "
                "projection."
            )
        if supersedes.audio_completion.fingerprint != audio_completion.fingerprint:
            raise DeckPackageError(
                "A superseding package preparation must carry the same proven audio."
            )
    with _package_locks(config, projection.deck_path) as (target, fresh):
        assert_projection_realized(projection, fresh, audio_completion=audio_completion)
        book = _assert_pending_audio_clear(config)
        ledger_before_sha256 = _ledger_digest(book)
        deck_config, records = resolve_deck_records(target)
        rendering = render_deck(target, config, deck_config, records)
        if rendering.record_ids != fresh.record_ids:
            raise DeckPackageError(
                "The deck rendered a record set outside the confirmed package plan."
            )
        staging_dir = _prepared_directory(config)
        staging_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            prepare_bound_directory(staging_dir)
        except (DataError, OSError) as exc:
            raise DeckPackageError(
                f"Could not prepare the private package directory {staging_dir}: {exc}"
            ) from exc
        # Both directories this preparation binds, captured here: the private one
        # it stages into, and the one the confirmed package is published to, which
        # §7.11's publication call names and `_publish_locked` has to pass. They
        # are not the same directory and the staging identity is not a stand-in
        # for the output one. `_directory_identity` lets absence through for
        # recovery's sake; both of these exist as of the statements above, so
        # absence here is a race, and prepare still speaks one error type.
        try:
            directory_identity = _directory_identity(staging_dir, _STAGING_DIRECTORY)
            output_directory_identity = _directory_identity(
                fresh.output_path.parent, _OUTPUT_DIRECTORY
            )
        except FileNotFoundError as exc:  # pragma: no cover - just prepared above
            raise DeckPackageError(
                f"A bound package directory under {config.dist_dir} vanished while "
                "the package was being prepared."
            ) from exc
        preparation_id = secrets.token_hex(16)
        staged_path = staging_dir / f"{preparation_id}.apkg"
        result = build_deck(
            target,
            config,
            output_path=staged_path,
            output_expected_absent=True,
        )
        if _lexical(result.output_path) != staged_path:
            raise DeckPackageError(
                "The deck builder returned an output outside the private package "
                "staging directory."
            )
        if (
            result.note_count != fresh.note_count
            or result.card_types != fresh.card_types
            or result.record_ids != fresh.record_ids
            or result.media_count != len(fresh.media_inputs)
        ):
            raise DeckPackageError(
                "The vocabulary builder reported a result outside the confirmed "
                "package plan."
            )
        state, staged_sha256, _payload = read_bytes_bound_snapshot(staged_path)
        after = _plan_unlocked(config, target)
        _assert_same_after_output(fresh, after)
        inventory = _read_package_inventory(
            config, fresh, rendering, records, audio_completion, staged_path
        )
        if supersedes is not None and inventory.fingerprint != (
            supersedes.inventory.fingerprint
        ):
            raise DeckPackageError(
                "A superseding package preparation must reproduce the same bound "
                "input inventory."
            )
        frozen_date = date.today().isoformat()
        by_id = {record.id: record for record in records}
        delta: list[DeckPackageExport] = []
        replay = _load_ledger(config)
        if _ledger_digest(replay) != ledger_before_sha256:
            raise DeckPackageError(
                "The export ledger changed while the package was being prepared."
            )
        try:
            for record_id in fresh.record_ids:
                entry = DeckPackageExport(
                    record_id=record_id,
                    deck_stem=target.stem,
                    gaps=_record_gaps(by_id[record_id]),
                    at=frozen_date,
                )
                replay.record_export(
                    entry.record_id,
                    entry.deck_stem,
                    gaps=entry.gaps,
                    at=entry.at,
                )
                delta.append(entry)
        except (JankiError, KeyError) as exc:
            raise DeckPackageError(
                f"Could not freeze the package export history: {exc}"
            ) from exc
        draft = DeckPackagePreparation(
            schema=PREPARED_PACKAGE_SCHEMA,
            preparation_id=preparation_id,
            projection=projection,
            plan=fresh,
            audio_completion=audio_completion,
            staged_path=staged_path,
            staged_sha256=staged_sha256,
            staged_identity=state[:2],
            expected_directory_identity=directory_identity,
            output_directory_identity=output_directory_identity,
            inventory=inventory,
            output_before_revision=fresh.output_revision,
            output_before_identity=fresh.output_identity,
            ledger_before_sha256=ledger_before_sha256,
            export_delta=tuple(delta),
            ledger_after_sha256=_ledger_digest(replay),
            warnings=tuple(result.warnings),
            supersedes="" if supersedes is None else supersedes.preparation_id,
            fingerprint="0" * 64,
        )
        return replace(draft, fingerprint=_preparation_fingerprint(config, draft))


def _prepared_result(
    preparation: DeckPackagePreparation,
) -> DeckPackageResult:
    """The landed result, whose card total is the archive's proven expansion.

    A result reports a package that exists, so its `card_count` comes from the
    validated inventory rather than from `plan.card_count`'s notes-times-
    directions capacity. The ordinary `execute_deck_package` path still reports
    the plan's number: it reads no archive back, and this correction does not
    widen it.
    """

    return DeckPackageResult(
        output_path=preparation.plan.output_path,
        package_sha256=preparation.staged_sha256,
        note_count=preparation.plan.note_count,
        card_count=preparation.inventory.card_count,
        card_types=preparation.plan.card_types,
        media_count=len(preparation.plan.media_inputs),
        warnings=preparation.warnings,
    )


def _ledger_at_a_bound_state(
    config: ProjectConfig,
    preparation: DeckPackagePreparation,
) -> tuple[ledger.Ledger, bool]:
    """The ledger and whether it is already at this preparation's after-state.

    The one owning before / after / third classifier, used by the publication's
    precheck and by the apply itself, so the two can never disagree about which
    ledger a frozen delta may be replayed onto. A third state refuses and names
    both digests; it is never merged, because ``Ledger.save`` refuses a ledger
    that moved since it was read and merging would discard the owner's change.
    """

    book = _load_ledger(config)
    observed = _ledger_digest(book)
    if observed == preparation.ledger_after_sha256:
        return book, True
    if observed != preparation.ledger_before_sha256:
        raise DeckPackageError(
            f"The export ledger is at {observed}, which is neither the prepared "
            f"before state {preparation.ledger_before_sha256} nor the prepared "
            f"after state {preparation.ledger_after_sha256}; it is preserved."
        )
    return book, False


def _apply_frozen_export_delta(
    config: ProjectConfig,
    preparation: DeckPackagePreparation,
) -> None:
    """Move the ledger from its bound before-state to its bound after-state.

    Exactly before to after: ``Ledger.save`` refuses a ledger that moved since
    it was read, so a permissive merge onto whatever is current would both
    contradict that refusal and let this job discard an owner change made in
    between.

    Read again here rather than reusing the publication's precheck: the write
    happened in between, and ``Ledger.save``'s guard needs the baseline this
    replay was computed from.
    """

    book, already_applied = _ledger_at_a_bound_state(config, preparation)
    if already_applied:
        return
    try:
        for entry in preparation.export_delta:
            book.record_export(
                entry.record_id,
                entry.deck_stem,
                gaps=entry.gaps,
                at=entry.at,
            )
    except JankiError as exc:
        raise DeckPackageError(
            f"Could not replay the frozen package export history: {exc}"
        ) from exc
    replayed = _ledger_digest(book)
    if replayed != preparation.ledger_after_sha256:
        raise DeckPackageError(
            f"Replaying the frozen package export history reached {replayed}, not "
            f"the prepared after state {preparation.ledger_after_sha256}."
        )
    try:
        book.save()
    except (JankiError, OSError) as exc:
        raise DeckPackageError(
            f"Package {preparation.plan.output_path} landed with SHA-256 "
            f"{preparation.staged_sha256}, but its export history could not be "
            f"saved: {exc}"
        ) from exc


def _publish_locked(
    config: ProjectConfig,
    preparation: DeckPackagePreparation,
    fresh: DeckPackagePlan,
) -> DeckPackageResult:
    _assert_same(preparation.plan, fresh)
    # `_staged_bytes` speaks the absent stage as a raw `FileNotFoundError`, which
    # is what recovery's own earlier call classifies into §7.11's free-rebuild
    # row. Publication is a public entry point instead, so the same absence has
    # to leave this module inside its own error family, naming the preparation
    # and the path a caller would act on. Only this call is normalized: the
    # classification recovery depends on is untouched, and nothing else here is
    # caught more widely than before.
    try:
        data = _staged_bytes(preparation)
    except FileNotFoundError as exc:
        raise DeckPackageError(
            f"The prepared package {preparation.preparation_id} can no longer be "
            f"published: its staged artifact {preparation.staged_path} is gone. "
            "Recover this preparation and prepare a superseding one."
        ) from exc
    # §7.5: an apply prechecks its **entire** bound set under its locks before
    # any write. The ledger is the one bound component publication was checking
    # only afterwards, and it costs a read the apply was going to do anyway. A
    # third state there makes the frozen delta unappliable, so writing first
    # would spend the artifact write to reach a state `publish` can no longer
    # retry — its `_assert_same` refuses once the target has moved — leaving
    # only recovery able to report it.
    _ledger_at_a_bound_state(config, preparation)
    try:
        atomic_write_bytes_bound(
            fresh.output_path,
            data,
            expected_revision=preparation.output_before_revision,
            expected_identity=preparation.output_before_identity,
            expected_absent=preparation.output_before_revision is None,
            expected_directory_identity=preparation.output_directory_identity,
        )
    except (DataError, OSError) as exc:
        raise DeckPackageError(
            f"Could not publish the prepared package {fresh.output_path}: {exc}"
        ) from exc
    _apply_frozen_export_delta(config, preparation)
    return _prepared_result(preparation)


def publish_prepared_deck_package(
    config: ProjectConfig,
    preparation: DeckPackagePreparation,
) -> DeckPackageResult:
    """Publish one prepared artifact to the confirmed target, then the ledger.

    The caller holds ``.janki-audio-operation``. The proof is revalidated, the
    concrete plan is strictly re-compared — the projection has already been
    realized, so nothing is permissive from here — the staged bytes are
    re-verified at their bound inode inside their bound directory, and only
    then are those exact bytes published and the frozen delta applied.
    """

    try:
        audio_application.revalidate_audio_completion(
            config, preparation.audio_completion
        )
    except audio_application.AudioProofError as exc:
        raise DeckPackageError(
            f"The audio completion proof is no longer valid: {exc}"
        ) from exc
    with _package_locks(config, preparation.plan.deck_path) as (_target, fresh):
        return _publish_locked(config, preparation, fresh)


def recover_prepared_deck_package(
    config: ProjectConfig,
    preparation: DeckPackagePreparation,
) -> DeckPackageResult | None:
    """Finish one interrupted publication from the intent and nothing else.

    Recognizes its own published output **before** any strict re-plan: once the
    target lands, ``_output_binding`` returns the new revision and a blind
    comparison would refuse this job's own work.

    Returns ``None`` when the staged artifact is gone while the target is still
    the state the preparation bound. `dist/` is disposable, so rebuilding is
    free — the caller mints a **new** superseding preparation from the same
    bound inventory rather than editing this intent.
    """

    try:
        audio_application.revalidate_audio_completion(
            config, preparation.audio_completion
        )
    except audio_application.AudioProofError as exc:
        raise DeckPackageError(
            f"The audio completion proof is no longer valid: {exc}"
        ) from exc
    target_deck = _known_deck(config, preparation.plan.deck_path)
    with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target_deck):
        first = _plan_unlocked(config, target_deck)
        paths = _locked_paths(first) - {target_deck}
        with contextlib.ExitStack() as locks:
            for path in sorted(paths, key=str):
                locks.enter_context(exclusive_path_lock(path))
            output = preparation.plan.output_path
            observed_revision, observed_identity = _output_binding(output)
            if observed_revision == preparation.staged_sha256:
                _apply_frozen_export_delta(config, preparation)
                return _prepared_result(preparation)
            if (observed_revision, observed_identity) != (
                preparation.output_before_revision,
                preparation.output_before_identity,
            ):
                raise DeckPackageError(
                    f"The package target {output} is at {observed_revision}, which "
                    f"is neither this preparation's artifact "
                    f"{preparation.staged_sha256} nor the state it bound "
                    f"({preparation.output_before_revision}); it is preserved."
                )
            try:
                _staged_bytes(preparation)
            except FileNotFoundError:
                return None
            except DataError as exc:
                raise DeckPackageError(
                    f"Could not read the staged package {preparation.staged_path}: "
                    f"{exc}"
                ) from exc
            fresh = _plan_unlocked(config, target_deck)
            if _locked_paths(fresh) != _locked_paths(first):
                raise DeckPackageError(
                    "Deck package input paths changed while the prepared package "
                    "was being recovered; reload and review the fresh plan."
                )
            return _publish_locked(config, preparation, fresh)
