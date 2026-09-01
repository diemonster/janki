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
    exclusive_path_lock,
    load_records,
    load_structured,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "ConjugationDeckBuildPlan",
    "ConjugationDeckBuildResult",
    "DeckBuildError",
    "execute_conjugation_deck_build",
    "execute_conjugation_deck_build_locked",
    "plan_conjugation_deck_build",
]


class DeckBuildError(JankiError):
    """A deck build cannot consume the exact plan the owner confirmed."""


@dataclass(frozen=True, slots=True)
class ConjugationDeckBuildPlan:
    """Every repository input and consequence displayed before one build."""

    repository_root: Path
    deck_path: Path
    source_path: Path
    output_path: Path
    deck_sha256: str
    source_sha256: str
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


def _plan_locked(config: ProjectConfig, deck_path: Path) -> ConjugationDeckBuildPlan:
    deck = _known_deck(config, deck_path)
    source = pattern_cards.collection_for(deck, config).resolve()
    try:
        deck_wire = read_bytes_bound(deck)
        source_wire = read_bytes_bound(source)
        raw = load_structured(deck)
        records = load_records(source)
    except (DataError, OSError) as exc:
        raise DeckBuildError(f"Could not read the conjugation build inputs: {exc}") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("deck"), Mapping):
        raise DeckBuildError(f"Conjugation deck needs one deck mapping: {deck}")
    section = raw["deck"]
    assert isinstance(section, Mapping)
    form = str(section.get("form") or "te_form").strip()
    name = str(section.get("name") or deck.stem).strip()
    shipping = pattern_cards.shipping_records(deck, records, form)
    cards = pattern_cards.drill_cards(shipping, form)
    if not cards:
        raise DeckBuildError(f"Conjugation deck would build no cards: {deck}")
    output = _output_path(config, section, deck)
    deck_sha = _sha(deck_wire)
    source_sha = _sha(source_wire)
    try:
        deck_relative = deck.relative_to(config.root.resolve()).as_posix()
        source_relative = source.relative_to(config.root.resolve()).as_posix()
        output_relative = output.relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise DeckBuildError("Conjugation build inputs must stay inside the repository.") from exc
    fingerprint = _fingerprint(
        {
            "version": 1,
            "deck": {"path": deck_relative, "sha256": deck_sha},
            "source": {"path": source_relative, "sha256": source_sha},
            "output": output_relative,
            "name": name,
            "form": form,
            "card_count": len(cards),
        }
    )
    if read_bytes_bound(deck) != deck_wire or read_bytes_bound(source) != source_wire:
        raise DeckBuildError(
            "Conjugation build inputs changed while the plan was being prepared."
        )
    return ConjugationDeckBuildPlan(
        repository_root=config.root.resolve(),
        deck_path=deck,
        source_path=source,
        output_path=output,
        deck_sha256=deck_sha,
        source_sha256=source_sha,
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
        source = pattern_cards.collection_for(target, config).resolve()
        with exclusive_path_lock(source):
            return _plan_locked(config, target)


def execute_conjugation_deck_build(
    config: ProjectConfig,
    expected: ConjugationDeckBuildPlan,
) -> ConjugationDeckBuildResult:
    """Re-plan under the audio/owner locks and publish only an exact match."""
    if expected.repository_root != config.root.resolve():
        raise DeckBuildError("Conjugation build plan belongs to another repository.")
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
    expected.output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        exclusive_path_lock(config.deck_dir),
        contextlib.ExitStack() as locks,
    ):
        for path in sorted(
            {expected.deck_path, expected.source_path, expected.output_path},
            key=lambda value: str(value),
        ):
            locks.enter_context(exclusive_path_lock(path))
        fresh = _plan_locked(config, expected.deck_path)
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
        return ConjugationDeckBuildResult(
            output_path=fresh.output_path,
            card_count=count,
            package_sha256=_sha(package),
        )
