"""Plan and explicitly create one thematic study deck.

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
from collections.abc import Sequence
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
from japanese_anki.identifiers import record_scope_id, stable_record_id
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
    #: The one tag an assignment surface may write to place a card here. A
    #: character deck has none: characters are named one at a time, never swept
    #: up by a tag a promotion happened to write.
    intake_tag: str
    deck_id: int
    path: Path
    output_path: Path
    yaml_bytes: bytes
    deck_set_fingerprint: str
    project_root: Path
    canonical_source: Path
    kind: str = "vocabulary"
    #: The exact character identities a kanji deck ships, in the order written.
    include_ids: tuple[str, ...] = ()
    model_id: int | None = None
    #: Whether this vocabulary deck holds its own copies of the words it takes
    #: rather than sharing the canonical ones with every other deck.
    standalone: bool = False
    #: The immutable scope a standalone deck's copies are identified under,
    #: chosen once here and never rederived from the display name again.
    scope_id: str = ""


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
    frozenset[str],
]:
    intake_tags: set[str] = set()
    deck_ids: set[int] = set()
    word_decks: list[_ExistingWordDeck] = []
    outputs: list[_ExistingDeckOutput] = []
    scope_ids: set[str] = set()
    for path in paths:
        try:
            raw = load_structured(path)
            section: Any = raw.get("deck") if isinstance(raw, dict) else None
            if not isinstance(section, dict):
                raise StudyDeckCreationError(
                    f"The deck section must be a mapping: {path}"
                )

            # Every persisted scope is held, whatever else the file says about
            # itself. A renamed deck still owns the Anki history its scope keys,
            # so this collects the value rather than re-deriving or judging it.
            held_scope = section.get("scope_id")
            if isinstance(held_scope, str) and held_scope.strip():
                scope_ids.add(held_scope.strip())

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
        frozenset(scope_ids),
    )


def _path_key(path: Path) -> str:
    """A portable key for output names that may collide on the build host."""
    return unicodedata.normalize("NFC", str(path)).casefold()


def _cross_selection_probe(
    word_decks: tuple[_ExistingWordDeck, ...], intake_tag: str, scope_id: str = ""
) -> _IntakeProbe:
    reserved_ids = {
        record_id
        for deck in word_decks
        for record_id in deck.selection.include_ids | deck.selection.exclude_ids
    }
    # The probe has to carry the scope it probes for: a shared selector must not
    # be able to claim a standalone card, and a standalone one must not answer
    # for the shared collection. The suffix varies the reading part, so a
    # reserved literal is stepped over without leaving the scope being tested.
    record_id = stable_record_id("assignment", "assignment", scope_id=scope_id)
    while record_id in reserved_ids:
        record_id += "-probe"
    return _IntakeProbe(id=record_id, tags=(intake_tag,))


#: The domain a standalone vocabulary scope is derived under. Explicit, so a
#: scope can never collide with an unrelated digest this project derives from
#: the same stem — the Anki deck id below is derived from that stem too.
SCOPE_DOMAIN = "janki:deck-scope:v1"


@dataclass(frozen=True, slots=True)
class _ScopeHolders:
    """Every scope a new standalone deck may not adopt."""

    deck_scopes: frozenset[str]
    record_ids: tuple[str, ...]

    def holds(self, scope_id: str) -> bool:
        if scope_id in self.deck_scopes:
            return True
        # The core helper is the only thing that reads a scope out of an id.
        return any(
            record_scope_id(record_id) == scope_id for record_id in self.record_ids
        )


def _canonical_record_ids(config: ProjectConfig) -> tuple[str, ...]:
    """The identities the canonical word collection currently holds.

    An absent file is an empty library, which is what creating the first deck in
    a fresh project looks like. Entries this cannot read are skipped rather than
    reported: the collection has its own validation, and a creator that refused
    on an unrelated record would make one broken row block every new deck.
    """
    path = config.normalized_file
    if not path.exists():
        return ()
    try:
        raw: Any = load_structured(path)
    except (DataError, JankiError, OSError) as exc:
        raise StudyDeckCreationError(
            f"Could not read the canonical word collection {path}: {exc}"
        ) from exc
    if isinstance(raw, dict):
        raw = raw.get("records", ())
    if not isinstance(raw, list):
        return ()
    return tuple(
        entry["id"]
        for entry in raw
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def _scope_id(stem: str, holders: _ScopeHolders) -> str:
    """A full SHA-256 scope for this deck, skipping every one already held.

    Derived like :func:`_deck_id` — the stable stem plus a deterministic counter
    — so the same creation in the same repository plans the same scope twice,
    and the counter steps past a scope some surviving deck or record still owns.
    Deleting a deck and recreating it under its old name therefore cannot adopt
    the review history the old scope still keys.
    """
    for counter in range(10_000):
        candidate = hashlib.sha256(
            f"{SCOPE_DOMAIN}:{stem}:{counter}".encode("ascii")
        ).hexdigest()
        if not holders.holds(candidate):
            return candidate
    raise StudyDeckCreationError("Could not choose a unique standalone deck scope.")


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


#: Where character notetype ids start, above the eight a word deck's enabled
#: direction mask can reach from ``model_id_base``. A character deck pins its
#: own id in its file; this only decides the first one offered.
KANJI_MODEL_OFFSET = 8


def _model_id(
    config: ProjectConfig, kind: str, directions: dict[str, bool]
) -> int | None:
    """The notetype id a new deck pins, or ``None`` to leave it derived.

    A word deck derives its id from the enabled direction mask at build time
    and has done since before this service existed; pinning one here would
    change the notetype of every deck created from now on. A character deck
    pins, because its directions are fixed once review history exists and a
    derived id would move the moment one was added.
    """
    if kind != "kanji":
        return None
    from japanese_anki.exporters.kanji_cards import KANJI_CARD_FILES

    mask = sum(
        1 << index
        for index, name in enumerate(KANJI_CARD_FILES)
        if directions[name]
    )
    return config.model_id_base + KANJI_MODEL_OFFSET + mask


def _render_deck(
    *,
    kind: str,
    name: str,
    stem: str,
    intake_tag: str,
    deck_id: int,
    model_id: int | None,
    directions: dict[str, bool],
    source: str,
    include_ids: tuple[str, ...],
    scope_id: str = "",
) -> bytes:
    """The exact bytes of a new deck file, for the kind that was asked for.

    A word deck selects by its machine-owned intake tag. A character deck
    names its characters instead and pins its notetype id, because a
    character notetype's id is what an existing collection matches its notes
    against and deriving it would move the day a direction is added.
    """
    section: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "deck_id": deck_id,
    }
    if model_id is not None:
        section["model_id"] = model_id
    section["output"] = f"{stem}.apkg"
    section["cards"] = directions
    section["source"] = source
    if kind == "kanji":
        section["include_ids"] = list(include_ids)
    else:
        section["intake_tag"] = intake_tag
        section["include_tags"] = [intake_tag]
        # Only a standalone deck writes a scope, and it writes it once. A shared
        # deck's definition is byte for byte what it has always been.
        if scope_id:
            section["scope_id"] = scope_id
    return yaml.safe_dump(
        {"deck": section}, allow_unicode=True, sort_keys=False
    ).encode("utf-8")


def _plan_under_lock(
    config: ProjectConfig,
    *,
    name: str,
    recognition: bool,
    production: bool,
    reading: bool,
    kind: str,
    include_ids: tuple[str, ...],
    standalone: bool = False,
) -> StudyDeckCreationPlan:
    # Structural, and here rather than only at the entrypoint, so the locked
    # re-plan cannot be handed a scope decision the first plan never made.
    if not isinstance(standalone, bool):
        raise StudyDeckCreationError("Standalone must be true or false.")
    if standalone and kind != "vocabulary":
        raise StudyDeckCreationError(
            "Only a vocabulary deck can hold standalone copies."
        )
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

    intake_tags, deck_ids, word_decks, outputs, deck_scopes = _existing_deck_state(
        config, paths
    )
    intake_tag = f"janki:deck:{stem}"
    scope_id = ""
    if standalone:
        # Chosen here, under the deck-directory lock this function already holds,
        # and chosen again on the locked re-plan — so a scope claimed between the
        # preview and the create stales the plan instead of being adopted twice.
        scope_id = _scope_id(
            stem,
            _ScopeHolders(
                deck_scopes=deck_scopes,
                record_ids=_canonical_record_ids(config),
            ),
        )
    if kind == "vocabulary":
        if intake_tag in intake_tags:
            raise StudyDeckCreationError(
                f"A word deck already declares intake tag {intake_tag!r}."
            )
        # This is the same structural probe used by the real-corpus W4.0
        # contract: if an existing selector also takes a record carrying the
        # new assignment tag, the creator would mint a destination that cannot
        # own its cards uniquely. No Japanese content is inspected.
        probe = _cross_selection_probe(word_decks, intake_tag, scope_id)
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
    canonical = (
        config.kanji_notes_file if kind == "kanji" else config.normalized_file
    ).resolve()
    deck_id = _deck_id(stem, deck_ids)
    model_id = _model_id(config, kind, directions)
    yaml_bytes = _render_deck(
        kind=kind,
        name=display_name,
        stem=stem,
        intake_tag=intake_tag,
        deck_id=deck_id,
        model_id=model_id,
        directions=directions,
        source=_source_reference(target.parent, canonical),
        include_ids=include_ids,
        scope_id=scope_id,
    )
    return StudyDeckCreationPlan(
        name=display_name,
        recognition=directions["recognition"],
        production=directions["production"],
        reading=directions["reading"],
        stem=stem,
        intake_tag="" if kind == "kanji" else intake_tag,
        deck_id=deck_id,
        path=target,
        output_path=output_path,
        yaml_bytes=yaml_bytes,
        deck_set_fingerprint=_deck_set_fingerprint(config, paths),
        project_root=config.root.resolve(),
        canonical_source=canonical,
        kind=kind,
        include_ids=include_ids,
        model_id=model_id,
        standalone=standalone,
        scope_id=scope_id,
    )


def plan_study_deck(
    config: ProjectConfig,
    *,
    name: str,
    recognition: bool = True,
    production: bool = False,
    reading: bool = False,
    kind: str = "vocabulary",
    include_ids: Sequence[str] = (),
    standalone: bool = False,
) -> StudyDeckCreationPlan:
    """Return the exact path and YAML for a new study deck, without writing.

    ``kind`` chooses what the deck holds. A vocabulary deck selects records by
    the intake tag this derives for it; a ``kanji`` deck names the exact
    character identities it ships and reads the curated character store
    instead of the word collection.

    ``standalone`` gives a vocabulary deck its own copies of the words it takes,
    under an immutable scope chosen here once. It reads the same canonical
    collection and declares the same unique intake tag as a shared deck; there
    is no second vocabulary file, and no existing record or identity moves.
    """
    if not isinstance(standalone, bool):
        raise StudyDeckCreationError("Standalone must be true or false.")
    if kind not in ("vocabulary", "kanji"):
        raise StudyDeckCreationError(
            f"A study deck is 'vocabulary' or 'kanji', not {kind!r}."
        )
    exact_ids = tuple(include_ids)
    if kind == "vocabulary" and exact_ids:
        raise StudyDeckCreationError(
            "A word deck selects by its intake tag, not by an exact id list."
        )
    if kind == "kanji":
        if not exact_ids:
            raise StudyDeckCreationError(
                "A character deck names the exact characters it holds."
            )
        repeated = sorted({item for item in exact_ids if exact_ids.count(item) > 1})
        if repeated:
            raise StudyDeckCreationError(
                f"Character {repeated[0]} is listed more than once."
            )
    with exclusive_path_lock(config.deck_dir):
        return _plan_under_lock(
            config,
            name=name,
            recognition=recognition,
            production=production,
            reading=reading,
            kind=kind,
            include_ids=exact_ids,
            standalone=standalone,
        )


def create_study_deck(
    config: ProjectConfig, plan: StudyDeckCreationPlan
) -> CreatedStudyDeck:
    """Publish an unchanged plan once, after an explicit create action."""
    expected_source = (
        config.kanji_notes_file if plan.kind == "kanji" else config.normalized_file
    ).resolve()
    if (
        plan.project_root != config.root.resolve()
        or plan.path.parent != config.deck_dir
        or plan.canonical_source != expected_source
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
            kind=plan.kind,
            include_ids=plan.include_ids,
            standalone=plan.standalone,
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
