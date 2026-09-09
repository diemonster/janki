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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Literal

from japanese_anki import ledger, patterns, status
from japanese_anki.application import deck_build, deck_capabilities
from japanese_anki.application.build import _record_gaps
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import kanji_cards, pattern_cards
from japanese_anki.exporters.anki import (
    CARD_FILES,
    build_deck,
    deck_kind,
    project_deck_media_paths,
    project_deck_records,
    resolve_card_types,
    resolve_deck_media_paths,
    resolve_deck_output_path,
    resolve_deck_records,
)
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    exclusive_path_lock,
    load_structured,
    read_bytes_bound,
    read_bytes_bound_snapshot,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import has_errors, refusal_text, validate_records

__all__ = [
    "DeckPackageError",
    "DeckPackageInput",
    "DeckPackagePlan",
    "DeckPackageResult",
    "execute_deck_package",
    "execute_deck_package_locked",
    "plan_deck_package",
    "plan_vocabulary_deck_package_revision",
]


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
    try:
        values = [
            record.to_dict()
            for record in sorted(records, key=lambda item: item.id)
        ]
        return json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (AttributeError, TypeError, ValueError) as exc:
        raise DeckPackageError(
            f"Prospective package records cannot be serialized exactly: {exc}"
        ) from exc


def _plan_vocabulary_revision_unlocked(
    config: ProjectConfig,
    target: Path,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    media_sha256: Mapping[Path, str | None],
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
            _input(config, "kanji reference store", config.kanji_file, optional=True),
            _input(config, "jpdb reading facts", config.jpdb_readings_file, optional=True),
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
) -> DeckPackagePlan:
    """Bind a vocabulary package as it will exist after one exact revision.

    The supplied paths and their future hashes belong to a larger receipt;
    they may therefore be absent at projection time.  This value is not
    directly executable until those bytes exist and the caller revalidates
    every projected input against the ordinary package plan.
    """

    target = _known_deck(config, deck_path)
    try:
        with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
            first = _plan_vocabulary_revision_unlocked(
                config, target, records, revision, media_sha256
            )
            canonical = revision.path.resolve()
            paths = _locked_paths(first) - {target, canonical}
            with contextlib.ExitStack() as locks:
                locks.enter_context(exclusive_path_lock(canonical))
                for path in sorted(paths, key=str):
                    locks.enter_context(exclusive_path_lock(path))
                fresh = _plan_vocabulary_revision_unlocked(
                    config, target, records, revision, media_sha256
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
