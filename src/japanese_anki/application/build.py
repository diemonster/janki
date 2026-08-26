"""Plan and execute the exact deck builds at the end of one promotion.

The promotion receipt decides *which decks* may be built.  It does not narrow
those decks to the rows in the receipt: an Anki package is the current complete
study deck, while the finish preview is deliberately limited to the cards the
person just added.  This service keeps those two scopes separate and binds both
to one fingerprint before anything is written.

Builds share the audio operation lock so a package cannot capture the interval
between an audio record write and media publication.  The owner deck files and
their record sources are then held unchanged across the fresh plan and every
package write.  Export history is saved once when the build pass stops.  A
later-deck or ledger failure is returned as an explicit partial result:
packages are build artifacts and are not rolled back or described as absent.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from japanese_anki import ledger
from japanese_anki import status as status_module
from japanese_anki.application.finish import (
    FinishScope,
    records_revision_fingerprint,
    resolve_finish_scope,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import (
    BuildResult,
    build_deck,
    resolve_card_types,
    resolve_deck_output_path,
    resolve_deck_records,
)
from japanese_anki.io import (
    DataError,
    exclusive_path_lock,
    load_structured,
    read_bytes_bound,
    records_revision,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.preview import resolve_preview_records

__all__ = [
    "FinishBuildError",
    "FinishBuildExecution",
    "FinishBuildPlan",
    "FinishDeckBuildPlan",
    "execute_finish_build",
    "plan_finish_build",
]


class FinishBuildError(JankiError):
    """One finish build cannot be planned or safely dispatched."""


@dataclass(frozen=True, slots=True)
class FinishDeckBuildPlan:
    """The distinct preview and package scopes for one receipted owner deck."""

    stem: str
    deck_path: Path
    source_path: Path | None
    name: str
    output_path: Path
    card_types: tuple[str, ...]
    receipt_record_ids: tuple[str, ...]
    preview_records: tuple[VocabularyRecord, ...]
    records: tuple[VocabularyRecord, ...]
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not self.stem.strip():
            raise ValueError("A finish deck plan needs a nonblank stem")
        if not self.deck_path.is_absolute() or not self.output_path.is_absolute():
            raise ValueError("Finish deck plan paths must be absolute")
        if self.source_path is not None and not self.source_path.is_absolute():
            raise ValueError("A finish deck source path must be absolute")
        if not self.card_types or len(set(self.card_types)) != len(self.card_types):
            raise ValueError("A finish deck plan needs unique enabled card types")
        preview_ids = tuple(record.id for record in self.preview_records)
        if preview_ids != self.receipt_record_ids:
            raise ValueError("Finish preview records must exactly follow receipt IDs")
        if not self.records or len({record.id for record in self.records}) != len(
            self.records
        ):
            raise ValueError("A finish deck build needs unique resolved records")
        by_id = {record.id: record for record in self.records}
        if any(by_id.get(record.id) != record for record in self.preview_records):
            raise ValueError("Finish previews must use the deck's resolved versions")
        if not _is_sha256(self.configuration_fingerprint):
            raise ValueError("A deck configuration fingerprint must be SHA-256")


@dataclass(frozen=True, slots=True)
class FinishBuildPlan:
    """One read-only, fingerprinted build plan for a promotion receipt."""

    receipt_id: str
    scope_fingerprint: str
    canonical_path: Path
    canonical_revision: str
    deck_configuration_revision: str
    decks: tuple[FinishDeckBuildPlan, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        for value in (
            self.receipt_id,
            self.scope_fingerprint,
            self.canonical_revision,
            self.deck_configuration_revision,
            self.fingerprint,
        ):
            if not _is_sha256(value):
                raise ValueError("Finish build identities must be lowercase SHA-256")
        if not self.canonical_path.is_absolute():
            raise ValueError("A finish build canonical path must be absolute")
        if not self.decks:
            raise ValueError("A finish build needs at least one owner deck")
        if len({deck.stem for deck in self.decks}) != len(self.decks):
            raise ValueError("A finish build cannot repeat an owner deck stem")
        if len({deck.output_path for deck in self.decks}) != len(self.decks):
            raise ValueError("Finish owner decks must not overwrite one package path")


FinishBuildState = Literal[
    "complete",
    "build_incomplete",
    "ledger_incomplete",
    "build_and_ledger_incomplete",
]


@dataclass(frozen=True, slots=True)
class FinishBuildExecution:
    """Packages that landed, plus whether their export history also committed."""

    state: FinishBuildState
    plan: FinishBuildPlan
    results: tuple[BuildResult, ...]
    build_error: str | None = None
    ledger_error: str | None = None

    def __post_init__(self) -> None:
        if not self.results or len(self.results) > len(self.plan.decks):
            raise ValueError("A finish execution must identify every landed package")
        build_incomplete = self.state in {
            "build_incomplete",
            "build_and_ledger_incomplete",
        }
        ledger_incomplete = self.state in {
            "ledger_incomplete",
            "build_and_ledger_incomplete",
        }
        if build_incomplete != bool(self.build_error):
            raise ValueError("A partial build result must exactly explain its error")
        if ledger_incomplete != bool(self.ledger_error):
            raise ValueError("A partial ledger result must exactly explain its error")
        if not build_incomplete and len(self.results) != len(self.plan.decks):
            raise ValueError("A completed build needs every owner result")

    @property
    def package_paths(self) -> tuple[Path, ...]:
        """Every package path known to have reached disk."""
        return tuple(result.output_path for result in self.results)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_fingerprint(value: object, *, description: str) -> str:
    try:
        wire = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinishBuildError(
            f"[finish-build-plan-invalid] {description} cannot be bound to "
            f"the build plan: {exc}"
        ) from exc
    return hashlib.sha256(wire).hexdigest()


def _deck_source_path(deck_path: Path) -> Path | None:
    raw = load_structured(deck_path)
    if not isinstance(raw, Mapping):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    deck_config = raw.get("deck") or {}
    if not isinstance(deck_config, Mapping):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    source = deck_config.get("source")
    if not source:
        return None
    return (deck_path.parent / str(source)).resolve()


def _deck_configuration_snapshot(
    config: ProjectConfig,
) -> tuple[tuple[Path, ...], str]:
    """Reproduce the exact configured-deck binding carried by FinishScope."""
    try:
        paths = tuple(path.resolve() for path in status_module.deck_files(config))
        digest = hashlib.sha256()
        for path in paths:
            wire = read_bytes_bound(path)
            encoded_path = str(path).encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(len(wire).to_bytes(8, "big"))
            digest.update(wire)
        current = tuple(path.resolve() for path in status_module.deck_files(config))
    except (JankiError, OSError) as exc:
        raise FinishBuildError(
            f"[finish-build-decks-unreadable] could not bind configured deck "
            f"files: {exc}"
        ) from exc
    if current != paths:
        raise FinishBuildError(
            "[finish-build-decks-stale] the configured deck-file set changed "
            "while it was being read; reload the finish page"
        )
    return paths, digest.hexdigest()


def _assert_scope_inputs_current(config: ProjectConfig, scope: FinishScope) -> None:
    canonical = config.normalized_file.resolve()
    if scope.canonical_path != canonical:
        raise FinishBuildError(
            "[finish-build-canonical-changed] the receipt names a different "
            "canonical collection; reload the finish page"
        )
    current_canonical = records_revision_fingerprint(records_revision(canonical))
    if current_canonical != scope.canonical_revision:
        raise FinishBuildError(
            "[finish-build-canonical-stale] the vocabulary collection changed "
            "after the finish scope was read; reload it"
        )

    paths, revision = _deck_configuration_snapshot(config)
    if revision != scope.deck_configuration_revision:
        raise FinishBuildError(
            "[finish-build-decks-stale] deck configuration changed after the "
            "finish scope was read; reload it"
        )
    configured = set(paths)
    for group in scope.owner_groups:
        if group.deck_path.resolve() not in configured:
            raise FinishBuildError(
                f"[finish-build-owner-missing] owner deck {group.stem!r} is no "
                "longer configured; reload the finish page"
            )


@contextmanager
def _owner_inputs_locked(
    config: ProjectConfig, scope: FinishScope
) -> Iterator[None]:
    """Hold every repository input that the owner builds resolve from."""
    with exclusive_path_lock(config.deck_dir):
        initial_sources = {
            source
            for group in scope.owner_groups
            if (source := _deck_source_path(group.deck_path)) is not None
        }
        paths = {
            config.normalized_file.resolve(),
            *(group.deck_path.resolve() for group in scope.owner_groups),
            *initial_sources,
        }
        with ExitStack() as stack:
            for path in sorted(paths, key=os.fspath):
                stack.enter_context(exclusive_path_lock(path))
            current_sources = {
                source
                for group in scope.owner_groups
                if (source := _deck_source_path(group.deck_path)) is not None
            }
            if current_sources != initial_sources:
                raise FinishBuildError(
                    "[finish-build-source-stale] an owner deck changed its record "
                    "source while the build was being locked; reload it"
                )
            _assert_scope_inputs_current(config, scope)
            yield


def _plan_finish_build_locked(
    config: ProjectConfig, scope: FinishScope
) -> FinishBuildPlan:
    canonical_revision = records_revision_fingerprint(
        records_revision(config.normalized_file.resolve())
    )
    _paths, deck_revision = _deck_configuration_snapshot(config)
    decks: list[FinishDeckBuildPlan] = []
    configurations: list[dict[str, object]] = []

    for group in scope.owner_groups:
        deck_path = group.deck_path.resolve()
        deck_config, resolved = resolve_deck_records(deck_path)
        preview = tuple(resolve_preview_records(deck_path, group.record_ids))
        configuration_fingerprint = _json_fingerprint(
            deck_config,
            description=f"deck configuration {deck_path}",
        )
        planned = FinishDeckBuildPlan(
            stem=group.stem,
            deck_path=deck_path,
            source_path=_deck_source_path(deck_path),
            name=str(deck_config.get("name", config.default_deck_name)),
            output_path=resolve_deck_output_path(deck_path, deck_config, config),
            card_types=tuple(resolve_card_types(deck_config, config)),
            receipt_record_ids=group.record_ids,
            preview_records=preview,
            records=tuple(resolved),
            configuration_fingerprint=configuration_fingerprint,
        )
        decks.append(planned)
        configurations.append(dict(deck_config))

    outputs = [deck.output_path for deck in decks]
    if len(set(outputs)) != len(outputs):
        repeated = next(path for path in outputs if outputs.count(path) > 1)
        raise FinishBuildError(
            f"[finish-build-output-collision] more than one owner deck would "
            f"write {repeated}; give each deck a distinct output"
        )

    draft = FinishBuildPlan(
        receipt_id=scope.receipt_id,
        scope_fingerprint=scope.fingerprint,
        canonical_path=scope.canonical_path,
        canonical_revision=canonical_revision,
        deck_configuration_revision=deck_revision,
        decks=tuple(decks),
        fingerprint="0" * 64,
    )
    payload = {
        "version": 1,
        "receipt_id": draft.receipt_id,
        "scope_fingerprint": draft.scope_fingerprint,
        "canonical": {
            "path": str(draft.canonical_path),
            "revision": draft.canonical_revision,
        },
        "deck_configuration_revision": draft.deck_configuration_revision,
        "decks": [
            {
                "stem": deck.stem,
                "path": str(deck.deck_path),
                "source_path": (
                    None if deck.source_path is None else str(deck.source_path)
                ),
                "name": deck.name,
                "output_path": str(deck.output_path),
                "card_types": list(deck.card_types),
                "receipt_record_ids": list(deck.receipt_record_ids),
                "preview_records": [
                    record.to_dict() for record in deck.preview_records
                ],
                "records": [record.to_dict() for record in deck.records],
                "configuration": configuration,
                "configuration_fingerprint": deck.configuration_fingerprint,
            }
            for deck, configuration in zip(
                draft.decks, configurations, strict=True
            )
        ],
    }
    return replace(
        draft,
        fingerprint=_json_fingerprint(payload, description="finish build plan"),
    )


def plan_finish_build(config: ProjectConfig, scope: FinishScope) -> FinishBuildPlan:
    """Plan receipted previews and complete owner packages without writing."""
    with _owner_inputs_locked(config, scope):
        return _plan_finish_build_locked(config, scope)


def _record_gaps(record: VocabularyRecord) -> tuple[str, ...]:
    """The exact incomplete-card facts stored by the CLI build ledger."""
    gaps: list[str] = []
    if not record.audio:
        gaps.append("audio")
    if not any(example.japanese.strip() for example in record.examples):
        gaps.append("examples")
    if not record.pitch_accent and not record.audio_accent.strip():
        gaps.append("accent")
    return tuple(gaps)


def _require_result_matches_plan(
    result: BuildResult, planned: FinishDeckBuildPlan
) -> None:
    expected_ids = tuple(record.id for record in planned.records)
    if result.output_path.resolve() != planned.output_path:
        raise FinishBuildError(
            f"[finish-build-result-invalid] {planned.stem} wrote "
            f"{result.output_path}, not its configured output {planned.output_path}"
        )
    if result.record_ids != expected_ids or result.note_count != len(expected_ids):
        raise FinishBuildError(
            f"[finish-build-result-invalid] {planned.stem} reported a different "
            "record scope after writing its package"
        )
    if result.card_types != planned.card_types:
        raise FinishBuildError(
            f"[finish-build-result-invalid] {planned.stem} reported different "
            "card directions after writing its package"
        )
    if result.deck_name != planned.name:
        raise FinishBuildError(
            f"[finish-build-result-invalid] {planned.stem} reported a different "
            "deck name after writing its package"
        )


def execute_finish_build(
    config: ProjectConfig,
    receipt_id: str,
    *,
    expected_scope_fingerprint: str,
    expected_plan_fingerprint: str,
) -> FinishBuildExecution:
    """Freshly re-plan, build every receipted owner deck, and record exports."""
    operation_lock = config.root / ".janki-audio-operation"
    with exclusive_path_lock(operation_lock):
        scope = resolve_finish_scope(config, receipt_id)
        if scope.fingerprint != expected_scope_fingerprint:
            raise FinishBuildError(
                "[finish-build-scope-stale] the promoted-card scope changed "
                "after this page was rendered; reload it"
            )

        with _owner_inputs_locked(config, scope):
            plan = _plan_finish_build_locked(config, scope)
            if plan.fingerprint != expected_plan_fingerprint:
                raise FinishBuildError(
                    "[finish-build-plan-stale] the owner deck build changed "
                    "after this page was rendered; reload it"
                )

            book = ledger.load(config.ledger_file)
            if book.pending_audio:
                raise FinishBuildError(
                    f"[finish-build-audio-pending] refusing to build while "
                    f"{len(book.pending_audio)} pending audio transaction(s) "
                    "still separate saved record references from canonical media; "
                    "finish the indicated audio recovery first"
                )

            results: list[BuildResult] = []
            build_error: str | None = None
            ledger_errors: list[str] = []
            for planned in plan.decks:
                try:
                    result = build_deck(
                        planned.deck_path,
                        config,
                        output_path=planned.output_path,
                    )
                except Exception as exc:
                    if not results:
                        raise FinishBuildError(
                            f"[finish-build-failed] {planned.stem} did not return "
                            f"a completed package: {exc}"
                        ) from exc
                    build_error = (
                        f"{planned.stem} did not return a completed package: {exc}"
                    )
                    break
                results.append(result)

                try:
                    _require_result_matches_plan(result, planned)
                except Exception as exc:
                    build_error = (
                        f"{planned.stem}'s package landed, but its result could "
                        f"not be verified: {exc}"
                    )
                    break

                by_id = {record.id: record for record in planned.records}
                try:
                    for record_id in result.record_ids:
                        book.record_export(
                            record_id,
                            planned.stem,
                            gaps=_record_gaps(by_id[record_id]),
                        )
                except Exception as exc:
                    ledger_errors.append(
                        f"{planned.stem}'s package landed, but its in-memory "
                        f"export history could not be completed: {exc}"
                    )

            try:
                book.save()
            except ledger.LedgerError as exc:
                ledger_errors.append(str(exc))
            ledger_error = "; ".join(ledger_errors) or None
            if build_error is not None and ledger_error is not None:
                state: FinishBuildState = "build_and_ledger_incomplete"
            elif build_error is not None:
                state = "build_incomplete"
            elif ledger_error is not None:
                state = "ledger_incomplete"
            else:
                state = "complete"
            return FinishBuildExecution(
                state=state,
                plan=plan,
                results=tuple(results),
                build_error=build_error,
                ledger_error=ledger_error,
            )
