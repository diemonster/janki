"""Plan and explicitly create one thematic vocabulary deck.

The workbench asks a learner for two things: the name they will see in Anki
and the card directions they want.  Repository handles are not another naming
decision.  This service derives a stable file stem, a machine-owned intake tag,
an output name and a deck id from that display name, then exposes the exact
path and YAML bytes before anything is written.

Creation is a separate call.  It re-plans while holding the deck-directory
lock, refuses any change to the deck set rendered in the preview, and publishes
the one new file through janki's write-once bound writer.  Uploading a source or
merely rendering a plan therefore cannot invent a deck.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import (
    DeckSelection,
    deck_kind,
    deck_selection,
    resolve_deck_records,
)
from japanese_anki.io import (
    DataError,
    atomic_write_bytes_bound,
    exclusive_path_lock,
    load_structured,
    read_bytes_bound,
)

__all__ = [
    "CreatedStudyDeck",
    "StudyDeckCreationError",
    "StudyDeckCreationPlan",
    "create_study_deck",
    "plan_study_deck",
]


class StudyDeckCreationError(JankiError):
    """A study-deck request cannot be planned or safely created."""


@dataclass(frozen=True, slots=True)
class StudyDeckCreationPlan:
    """The exact new deck file shown before the explicit create action."""

    name: str
    recognition: bool
    production: bool
    reading: bool
    stem: str
    intake_tag: str
    deck_id: int
    path: Path
    output_path: Path
    yaml_bytes: bytes
    deck_set_fingerprint: str
    project_root: Path
    canonical_source: Path


@dataclass(frozen=True, slots=True)
class CreatedStudyDeck:
    """The file an explicit execution published."""

    path: Path
    yaml_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ExistingWordDeck:
    path: Path
    selection: DeckSelection


@dataclass(frozen=True, slots=True)
class _IntakeProbe:
    id: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExistingDeckOutput:
    deck_path: Path
    output_path: Path


def _name_parts(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name.strip():
        raise StudyDeckCreationError("A study deck needs a learner-facing name.")
    display_name = name.strip()
    normalized = unicodedata.normalize("NFKC", display_name)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    ascii_name = unicodedata.normalize("NFKD", normalized).encode(
        "ascii", errors="ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    readable = slug[:48].rstrip("-") or "study-deck"
    return display_name, f"{readable}-{digest[:10]}"


def _directions(
    recognition: bool, production: bool, reading: bool
) -> dict[str, bool]:
    values = {
        "recognition": recognition,
        "production": production,
        "reading": reading,
    }
    for direction, enabled in values.items():
        if not isinstance(enabled, bool):
            raise StudyDeckCreationError(
                f"The {direction} direction must be true or false."
            )
    if not any(values.values()):
        raise StudyDeckCreationError("Enable at least one card direction.")
    return values


def _source_reference(deck_dir: Path, canonical_source: Path) -> str:
    try:
        relative = os.path.relpath(canonical_source, start=deck_dir)
    except ValueError:  # Different Windows drives cannot be expressed relatively.
        return canonical_source.as_posix()
    return Path(relative).as_posix()


def _deck_paths(config: ProjectConfig) -> tuple[Path, ...]:
    try:
        return tuple(status_module.deck_files(config))
    except JankiError as exc:
        raise StudyDeckCreationError(str(exc)) from exc


def _deck_set_fingerprint(config: ProjectConfig, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            relative = path.relative_to(config.deck_dir).as_posix()
            payload = read_bytes_bound(path)
        except (DataError, OSError, ValueError) as exc:
            raise StudyDeckCreationError(f"Could not bind deck file {path}: {exc}") from exc
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _existing_deck_state(
    config: ProjectConfig,
    paths: tuple[Path, ...],
) -> tuple[
    frozenset[str],
    frozenset[int],
    tuple[_ExistingWordDeck, ...],
    tuple[_ExistingDeckOutput, ...],
]:
    intake_tags: set[str] = set()
    deck_ids: set[int] = set()
    word_decks: list[_ExistingWordDeck] = []
    outputs: list[_ExistingDeckOutput] = []
    for path in paths:
        try:
            raw = load_structured(path)
            section: Any = raw.get("deck") if isinstance(raw, dict) else None
            if not isinstance(section, dict):
                raise StudyDeckCreationError(
                    f"The deck section must be a mapping: {path}"
                )

            raw_deck_id = section.get("deck_id", config.default_deck_id)
            if isinstance(raw_deck_id, bool) or not isinstance(raw_deck_id, int):
                raise StudyDeckCreationError(
                    f"deck.deck_id must be an integer, got {raw_deck_id!r}: {path}"
                )
            deck_ids.add(raw_deck_id)
            filename = str(section.get("output", f"{path.stem}.apkg"))
            outputs.append(
                _ExistingDeckOutput(
                    deck_path=path,
                    output_path=(config.dist_dir / filename).resolve(),
                )
            )

            if deck_kind(path) not in ("", "vocabulary"):
                continue
            deck_config, _records = resolve_deck_records(path)
            selection = deck_selection(deck_config, path)
        except StudyDeckCreationError:
            raise
        except JankiError as exc:
            raise StudyDeckCreationError(str(exc)) from exc
        word_decks.append(_ExistingWordDeck(path=path, selection=selection))
        if selection.intake_tag is None:
            continue
        if selection.intake_tag in intake_tags:
            raise StudyDeckCreationError(
                f"More than one word deck declares intake tag "
                f"{selection.intake_tag!r}."
            )
        intake_tags.add(selection.intake_tag)
    return (
        frozenset(intake_tags),
        frozenset(deck_ids),
        tuple(word_decks),
        tuple(outputs),
    )


def _path_key(path: Path) -> str:
    """A portable key for output names that may collide on the build host."""
    return unicodedata.normalize("NFC", str(path)).casefold()


def _cross_selection_probe(
    word_decks: tuple[_ExistingWordDeck, ...], intake_tag: str
) -> _IntakeProbe:
    reserved_ids = {
        record_id
        for deck in word_decks
        for record_id in deck.selection.include_ids | deck.selection.exclude_ids
    }
    record_id = "word:assignment:assignment"
    while record_id in reserved_ids:
        record_id += "-probe"
    return _IntakeProbe(id=record_id, tags=(intake_tag,))


def _deck_id(stem: str, existing: frozenset[int]) -> int:
    for counter in range(10_000):
        candidate_digest = hashlib.sha256(
            f"{stem}:{counter}".encode("ascii")
        ).digest()
        candidate = (
            1_000_000_000
            + int.from_bytes(candidate_digest[:8], "big") % 1_000_000_000
        )
        if candidate not in existing:
            return candidate
    raise StudyDeckCreationError("Could not choose a unique Anki deck id.")


def _render_deck(
    *,
    name: str,
    stem: str,
    intake_tag: str,
    deck_id: int,
    directions: dict[str, bool],
    source: str,
) -> bytes:
    document = {
        "deck": {
            "kind": "vocabulary",
            "name": name,
            "deck_id": deck_id,
            "output": f"{stem}.apkg",
            "cards": directions,
            "source": source,
            "intake_tag": intake_tag,
            "include_tags": [intake_tag],
        }
    }
    return yaml.safe_dump(
        document, allow_unicode=True, sort_keys=False
    ).encode("utf-8")


def _plan_under_lock(
    config: ProjectConfig,
    *,
    name: str,
    recognition: bool,
    production: bool,
    reading: bool,
) -> StudyDeckCreationPlan:
    display_name, stem = _name_parts(name)
    directions = _directions(recognition, production, reading)
    paths = _deck_paths(config)
    colliding_paths = [
        path for path in paths if path.stem.casefold() == stem.casefold()
    ]
    if colliding_paths:
        raise StudyDeckCreationError(
            f"A deck already uses the stable file stem {stem!r}: {colliding_paths[0]}"
        )

    intake_tags, deck_ids, word_decks, outputs = _existing_deck_state(config, paths)
    intake_tag = f"janki:deck:{stem}"
    if intake_tag in intake_tags:
        raise StudyDeckCreationError(
            f"A word deck already declares intake tag {intake_tag!r}."
        )
    # This is the same structural probe used by the real-corpus W4.0 contract:
    # if an existing selector also takes a record carrying the new assignment
    # tag, the creator would mint a destination that cannot own its cards
    # uniquely. No Japanese content is inspected.
    probe = _cross_selection_probe(word_decks, intake_tag)
    if claimers := [
        deck.path for deck in word_decks if deck.selection.includes(probe)
    ]:
        raise StudyDeckCreationError(
            f"The new intake tag {intake_tag!r} would also select existing word "
            f"deck {claimers[0]}."
        )

    target = config.deck_dir / f"{stem}.yaml"
    output_path = (config.dist_dir / f"{stem}.apkg").resolve()
    if collisions := [
        output.deck_path
        for output in outputs
        if _path_key(output.output_path) == _path_key(output_path)
    ]:
        raise StudyDeckCreationError(
            f"Study deck output {output_path} is already declared by "
            f"{collisions[0]}."
        )
    canonical = config.normalized_file.resolve()
    deck_id = _deck_id(stem, deck_ids)
    yaml_bytes = _render_deck(
        name=display_name,
        stem=stem,
        intake_tag=intake_tag,
        deck_id=deck_id,
        directions=directions,
        source=_source_reference(target.parent, canonical),
    )
    return StudyDeckCreationPlan(
        name=display_name,
        recognition=directions["recognition"],
        production=directions["production"],
        reading=directions["reading"],
        stem=stem,
        intake_tag=intake_tag,
        deck_id=deck_id,
        path=target,
        output_path=output_path,
        yaml_bytes=yaml_bytes,
        deck_set_fingerprint=_deck_set_fingerprint(config, paths),
        project_root=config.root.resolve(),
        canonical_source=canonical,
    )


def plan_study_deck(
    config: ProjectConfig,
    *,
    name: str,
    recognition: bool = True,
    production: bool = False,
    reading: bool = False,
) -> StudyDeckCreationPlan:
    """Return the exact path and YAML for a new vocabulary deck, without writing."""
    with exclusive_path_lock(config.deck_dir):
        return _plan_under_lock(
            config,
            name=name,
            recognition=recognition,
            production=production,
            reading=reading,
        )


def create_study_deck(
    config: ProjectConfig, plan: StudyDeckCreationPlan
) -> CreatedStudyDeck:
    """Publish an unchanged plan once, after an explicit create action."""
    if (
        plan.project_root != config.root.resolve()
        or plan.path.parent != config.deck_dir
        or plan.canonical_source != config.normalized_file.resolve()
    ):
        raise StudyDeckCreationError(
            "This study-deck plan belongs to a different project; preview it again."
        )
    with exclusive_path_lock(config.deck_dir):
        fresh = _plan_under_lock(
            config,
            name=plan.name,
            recognition=plan.recognition,
            production=plan.production,
            reading=plan.reading,
        )
        if fresh != plan:
            raise StudyDeckCreationError(
                "The deck set changed after this preview; preview the study deck again."
            )
        try:
            atomic_write_bytes_bound(plan.path, plan.yaml_bytes, expected_absent=True)
        except (DataError, OSError) as exc:
            raise StudyDeckCreationError(
                f"Could not create study deck {plan.path}: {exc}"
            ) from exc
    return CreatedStudyDeck(path=plan.path, yaml_bytes=plan.yaml_bytes)
