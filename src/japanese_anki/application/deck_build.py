"""Plan-bound publication of one rich conjugation practice deck."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

from japanese_anki import ledger, status
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.exporters.anki import deck_kind
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    exclusive_path_lock,
    load_records_snapshot,
    read_bytes_bound,
    records_revision,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "ConjugationDeckBuildPlan",
    "ConjugationDeckBuildResult",
    "ConjugationMediaInput",
    "ConjugationTemplateInput",
    "DeckBuildError",
    "execute_conjugation_deck_build",
    "execute_conjugation_deck_build_locked",
    "plan_conjugation_deck_build",
    "plan_conjugation_deck_build_revision",
]


class DeckBuildError(JankiError):
    """A deck build cannot consume the exact plan the owner confirmed."""


@dataclass(frozen=True, slots=True)
class ConjugationTemplateInput:
    """One exact HTML/CSS file consumed by a conjugation package."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ConjugationMediaInput:
    """One exact referenced media path, or one not-yet-produced finish target."""

    path: Path
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ConjugationDeckBuildPlan:
    """Every repository input and consequence displayed before one build."""

    repository_root: Path
    deck_path: Path
    source_path: Path
    output_path: Path
    deck_sha256: str
    source_sha256: str
    template_inputs: tuple[ConjugationTemplateInput, ...]
    media_inputs: tuple[ConjugationMediaInput, ...]
    deck_name: str
    form: str
    card_count: int
    records: tuple[VocabularyRecord, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ConjugationDeckBuildResult:
    """The package proven to have landed for a confirmed build plan."""

    output_path: Path
    card_count: int
    package_sha256: str


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _template_paths(config: ProjectConfig) -> tuple[Path, ...]:
    return pattern_cards.pattern_template_paths(config.template_dir)


def _template_inputs(config: ProjectConfig) -> tuple[ConjugationTemplateInput, ...]:
    inputs: list[ConjugationTemplateInput] = []
    for path in _template_paths(config):
        try:
            wire = read_bytes_bound(path)
            wire.decode("utf-8", errors="strict")
        except (DataError, OSError, UnicodeError) as exc:
            raise DeckBuildError(
                f"Could not read the conjugation build template {path}: {exc}"
            ) from exc
        inputs.append(ConjugationTemplateInput(path=path, sha256=_sha(wire)))
    return tuple(inputs)


def _fingerprint_path(config: ProjectConfig, path: Path) -> str:
    """Use a stable repository path when possible, else the exact absolute path."""

    try:
        return path.relative_to(config.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _media_inputs(
    config: ProjectConfig,
    deck: Path,
    section: Mapping[str, object],
    records: tuple[VocabularyRecord, ...] | list[VocabularyRecord],
    form: str,
    projected_media: Mapping[Path, str | None] | None,
) -> tuple[ConjugationMediaInput, ...]:
    projected: dict[Path, str | None] | None = None
    if projected_media is not None:
        projected = {}
        for raw_path, sha256 in projected_media.items():
            path = Path(raw_path).resolve()
            if path in projected:
                raise DeckBuildError(f"Conjugation build repeats media target: {path}")
            if sha256 is not None and (
                len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise DeckBuildError(
                    f"Conjugation build has a malformed projected media hash: {path}"
                )
            projected[path] = sha256
    try:
        paths = pattern_cards.conjugation_media_paths_for_section(
            deck,
            config,
            section,
            records,
            form,
            allowed_missing_media=frozenset(projected or ()),
        )
    except (JankiError, OSError) as exc:
        raise DeckBuildError(f"Could not resolve conjugation build media: {exc}") from exc
    if projected is not None and set(paths) != set(projected):
        raise DeckBuildError(
            "The projected conjugation media target set does not match the deck."
        )
    inputs: list[ConjugationMediaInput] = []
    root = config.root.resolve()
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DeckBuildError(
                f"Conjugation build media must stay inside the repository: {path}"
            ) from exc
        expected_sha256 = projected[path] if projected is not None else None
        try:
            actual_sha256 = _sha(read_bytes_bound(path))
        except FileNotFoundError:
            if projected is None:
                raise DeckBuildError(
                    f"Conjugation build media is missing: {path}"
                ) from None
            actual_sha256 = None
        except (DataError, OSError) as exc:
            raise DeckBuildError(
                f"Could not safely read conjugation build media {path}: {exc}"
            ) from exc
        if expected_sha256 is not None and actual_sha256 not in {
            None,
            expected_sha256,
        }:
            raise DeckBuildError(
                f"Conjugation build media differs from its projected bytes: {path}"
            )
        sha256 = expected_sha256 if projected is not None else actual_sha256
        inputs.append(ConjugationMediaInput(path=path, sha256=sha256))
    return tuple(inputs)


def _known_deck(config: ProjectConfig, deck_path: Path | str) -> Path:
    target = Path(deck_path).resolve()
    matches = [path.resolve() for path in status.deck_files(config) if path.resolve() == target]
    if len(matches) != 1:
        raise DeckBuildError(
            f"Deck build target {target} is not exactly one configured deck."
        )
    if deck_kind(target) != "conjugation":
        raise DeckBuildError(f"Deck build target is not a conjugation deck: {target}")
    return target


def _output_path(config: ProjectConfig, section: Mapping[str, object], deck: Path) -> Path:
    raw = section.get("output", f"{deck.stem}.apkg")
    if not isinstance(raw, str) or not raw.strip():
        raise DeckBuildError("Conjugation deck output must be a nonblank filename.")
    name = raw.strip()
    if PurePath(name).name != name or Path(name).suffix.lower() != ".apkg":
        raise DeckBuildError(
            "A workbench conjugation build requires one direct .apkg filename "
            "under the configured dist directory."
        )
    return (config.dist_dir / name).resolve()


def _revision_inputs(
    config: ProjectConfig,
    deck: Path,
    revision: RecordsRevision,
) -> tuple[Mapping[str, object], Path]:
    if revision.path.resolve() != deck:
        raise DeckBuildError(
            f"Conjugation deck snapshot for {revision.path} cannot bind {deck}."
        )
    if revision.text is None:
        raise DeckBuildError(f"Conjugation deck snapshot is missing: {deck}")
    section = pattern_cards.conjugation_deck_section(deck, revision.text)
    source = pattern_cards.collection_for_section(deck, config, section).resolve()
    return section, source


def _plan_revision_locked(
    config: ProjectConfig,
    deck: Path,
    revision: RecordsRevision,
    section: Mapping[str, object],
    source: Path,
    configured_revision: RecordsRevision,
    projected_media: Mapping[Path, str | None] | None = None,
) -> ConjugationDeckBuildPlan:
    assert revision.text is not None  # established by `_revision_inputs`
    try:
        records, source_revision = load_records_snapshot(source)
    except (DataError, OSError) as exc:
        raise DeckBuildError(f"Could not read the conjugation build inputs: {exc}") from exc
    if source_revision.text is None:
        raise DeckBuildError(f"Conjugation deck source snapshot is missing: {source}")
    deck_wire = revision.text.encode("utf-8")
    source_wire = source_revision.text.encode("utf-8")
    form = str(section.get("form") or "te_form").strip()
    name = str(section.get("name") or deck.stem).strip()
    shipping = pattern_cards.shipping_records_for_section(
        deck,
        section,
        records,
        form,
    )
    cards = pattern_cards.drill_cards(shipping, form)
    if not cards:
        raise DeckBuildError(f"Conjugation deck would build no cards: {deck}")
    output = _output_path(config, section, deck)
    deck_sha = _sha(deck_wire)
    source_sha = _sha(source_wire)
    template_inputs = _template_inputs(config)
    media_inputs = _media_inputs(
        config,
        deck,
        section,
        records,
        form,
        projected_media,
    )
    try:
        deck_relative = deck.relative_to(config.root.resolve()).as_posix()
        source_relative = source.relative_to(config.root.resolve()).as_posix()
        output_relative = output.relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise DeckBuildError("Conjugation build inputs must stay inside the repository.") from exc
    fingerprint = _fingerprint(
        {
            "version": 3,
            "deck": {"path": deck_relative, "sha256": deck_sha},
            "source": {"path": source_relative, "sha256": source_sha},
            "output": output_relative,
            "name": name,
            "form": form,
            "card_count": len(cards),
            "templates": [
                {
                    "path": _fingerprint_path(config, item.path),
                    "sha256": item.sha256,
                }
                for item in template_inputs
            ],
            "media": [
                {
                    "path": _fingerprint_path(config, item.path),
                    "sha256": item.sha256,
                }
                for item in media_inputs
            ],
        }
    )
    if records_revision(source) != source_revision:
        raise DeckBuildError(
            "Conjugation build source changed while the plan was being prepared."
        )
    if records_revision(deck) != configured_revision:
        raise DeckBuildError(
            "The configured deck changed while the projected conjugation plan "
            "was being prepared."
        )
    return ConjugationDeckBuildPlan(
        repository_root=config.root.resolve(),
        deck_path=deck,
        source_path=source,
        output_path=output,
        deck_sha256=deck_sha,
        source_sha256=source_sha,
        template_inputs=template_inputs,
        media_inputs=media_inputs,
        deck_name=name,
        form=form,
        card_count=len(cards),
        records=tuple(records),
        fingerprint=fingerprint,
    )


def plan_conjugation_deck_build(
    config: ProjectConfig, deck_path: Path | str
) -> ConjugationDeckBuildPlan:
    """Prepare one display-only exact deck build."""
    target = _known_deck(config, deck_path)
    with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
        deck = _known_deck(config, target)
        configured_revision = records_revision(deck)
        section, source = _revision_inputs(config, deck, configured_revision)
        with contextlib.ExitStack() as locks:
            for path in sorted({source, *_template_paths(config)}, key=str):
                locks.enter_context(exclusive_path_lock(path))
            return _plan_revision_locked(
                config,
                deck,
                configured_revision,
                section,
                source,
                configured_revision,
            )


def plan_conjugation_deck_build_revision(
    config: ProjectConfig,
    deck_path: Path | str,
    revision: RecordsRevision,
    *,
    projected_media: Mapping[Path, str | None] | None = None,
) -> ConjugationDeckBuildPlan:
    """Plan the package produced after one exact deck snapshot lands."""

    target = _known_deck(config, deck_path)
    with exclusive_path_lock(config.deck_dir), exclusive_path_lock(target):
        deck = _known_deck(config, target)
        configured_revision = records_revision(deck)
        try:
            section, source = _revision_inputs(config, deck, revision)
            with contextlib.ExitStack() as locks:
                for path in sorted(
                    {source, *_template_paths(config), *(projected_media or {})},
                    key=str,
                ):
                    locks.enter_context(exclusive_path_lock(path))
                return _plan_revision_locked(
                    config,
                    deck,
                    revision,
                    section,
                    source,
                    configured_revision,
                    projected_media,
                )
        except DeckBuildError:
            raise
        except (JankiError, OSError) as exc:
            raise DeckBuildError(str(exc)) from exc


def execute_conjugation_deck_build(
    config: ProjectConfig,
    expected: ConjugationDeckBuildPlan,
) -> ConjugationDeckBuildResult:
    """Re-plan under the audio/owner locks and publish only an exact match."""
    if expected.repository_root != config.root.resolve():
        raise DeckBuildError("Conjugation build plan belongs to another repository.")
    if any(item.sha256 is None for item in expected.media_inputs):
        raise DeckBuildError(
            "Conjugation build media is not fully produced; prepare a fresh exact plan."
        )
    operation_lock = config.root / ".janki-audio-operation"
    with exclusive_path_lock(operation_lock):
        return execute_conjugation_deck_build_locked(config, expected)


def execute_conjugation_deck_build_locked(
    config: ProjectConfig,
    expected: ConjugationDeckBuildPlan,
) -> ConjugationDeckBuildResult:
    """Execute while the caller already owns ``.janki-audio-operation``.

    ``janki build`` holds that lock across a multi-deck sweep; the browser and
    assistant call :func:`execute_conjugation_deck_build`, which acquires it for
    their one deck. Keeping only the lock acquisition separate means every
    surface still consumes the same re-plan, pending-audio gate and exporter.
    """
    if expected.repository_root != config.root.resolve():
        raise DeckBuildError("Conjugation build plan belongs to another repository.")
    if any(item.sha256 is None for item in expected.media_inputs):
        raise DeckBuildError(
            "Conjugation build media is not fully produced; prepare a fresh exact plan."
        )
    configured_templates = _template_paths(config)
    if tuple(item.path for item in expected.template_inputs) != configured_templates:
        raise DeckBuildError(
            "The conjugation build template set changed after confirmation; "
            "reload and review the fresh plan."
        )
    expected.output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        exclusive_path_lock(config.deck_dir),
        contextlib.ExitStack() as locks,
    ):
        for path in sorted(
            {
                expected.deck_path,
                expected.source_path,
                expected.output_path,
                *configured_templates,
                *(item.path for item in expected.media_inputs),
            },
            key=lambda value: str(value),
        ):
            locks.enter_context(exclusive_path_lock(path))
        deck = _known_deck(config, expected.deck_path)
        configured_revision = records_revision(deck)
        section, source = _revision_inputs(
            config,
            deck,
            configured_revision,
        )
        if source != expected.source_path:
            raise DeckBuildError(
                "The conjugation deck build source changed after confirmation; "
                "reload and review the fresh plan."
            )
        fresh = _plan_revision_locked(
            config,
            deck,
            configured_revision,
            section,
            source,
            configured_revision,
        )
        if fresh != expected:
            raise DeckBuildError(
                "The conjugation deck build plan changed after confirmation; "
                "reload and review the fresh plan."
            )
        try:
            book = ledger.load(config.ledger_file)
        except (JankiError, OSError) as exc:
            raise DeckBuildError(f"Could not verify pending audio: {exc}") from exc
        if book.pending_audio:
            raise DeckBuildError(
                "The repository has pending audio recovery. Finish that exact "
                "transaction before building any package."
            )
        try:
            target, count = pattern_cards.build_conjugation_deck(
                fresh.deck_path,
                config,
                fresh.records,
                fresh.output_path,
            )
            package = read_bytes_bound(target)
        except (JankiError, OSError) as exc:
            raise DeckBuildError(f"Could not build this conjugation deck: {exc}") from exc
        if target.resolve() != fresh.output_path or count != fresh.card_count:
            raise DeckBuildError(
                "The conjugation builder returned a result outside the confirmed plan."
            )
        after = _plan_revision_locked(
            config,
            deck,
            configured_revision,
            section,
            source,
            configured_revision,
        )
        if after != expected:
            raise DeckBuildError(
                "Conjugation build inputs changed during package generation; "
                "the generated package was not accepted."
            )
        return ConjugationDeckBuildResult(
            output_path=fresh.output_path,
            card_count=count,
            package_sha256=_sha(package),
        )
