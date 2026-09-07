"""One owner-authorized apply-and-finish for prepared character notes.

The dictionary preparation already happened, and the owner confirmed the exact
notes it produced. This service binds that one serialized plan together with
its deck and package consequences behind a durable receipt, then applies and
builds under it.

It writes no study content itself: :mod:`character_notes` remains the sole
writer of character notes and any new deck definition, and
:mod:`deck_package` remains the only ``.apkg`` publisher. The receipt is the
only authority an interruption may resume — a resume re-reads it, never the
dictionaries, and never asks the owner again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from japanese_anki.application import assistant_kanji_notes, character_notes, deck_package
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    prepare_bound_directory,
    read_bytes_bound,
)

__all__ = [
    "KanjiFinishError",
    "KanjiFinishPlan",
    "KanjiFinishResult",
    "execute_kanji_finish",
    "inspect_kanji_finish",
    "list_kanji_finishes",
    "plan_kanji_finish",
    "resume_kanji_finish",
]


KanjiFinishState = Literal["authorized", "applied", "complete"]
KanjiFinishPhase = Literal[
    "Preparing finish",
    "Writing character notes",
    "Building Anki package",
    "Saving finish receipt",
]
KanjiFinishProgress = Callable[[KanjiFinishPhase], None]

_STATES: tuple[KanjiFinishState, ...] = ("authorized", "applied", "complete")
_STATE_INDEX = {state: position for position, state in enumerate(_STATES)}
_RECORD_KEYS = {
    "schema_version",
    "kind",
    "receipt_id",
    "state",
    "authority",
    "authorized_at",
    "updated_at",
    "apply_receipt",
    "build_receipt",
}
_AUTHORITY_KEYS = {"version", "instruction", "notes", "plan", "build"}
#: A recovery list, not an archive browser: enough to find the interrupted
#: batch without turning the receipt directory into a paginated surface. It
#: caps what is *shown*, never what is read, so no number of finished batches
#: can hide unfinished work.
_LIST_LIMIT = 20


class KanjiFinishError(JankiError):
    """The exact character-note finish authority cannot safely continue."""


@dataclass(frozen=True, slots=True)
class KanjiFinishPlan:
    """Display-only consequences of one confirmed character-note batch."""

    repository_root: Path
    instruction: str
    deck_path: Path
    output_path: Path
    deck_name: str
    deck_state: Literal["new", "existing"]
    characters: tuple[str, ...]
    directions: tuple[str, ...]
    #: What this batch selects, and what the deck holds once it lands. They
    #: differ whenever the destination already had notes.
    note_count: int
    card_count: int
    deck_note_count: int
    deck_card_count: int
    projection: Mapping[str, Any]
    finish_directory: Path
    record_path: Path
    authority: Mapping[str, Any]
    fingerprint: str

    @property
    def paid_provider_calls(self) -> int:
        """Character notes are dictionary-only: this batch never pays."""

        return 0


@dataclass(frozen=True, slots=True)
class KanjiFinishResult:
    """Truthful durable state after an initial or resumed finish execution."""

    receipt_id: str
    state: KanjiFinishState
    record_path: Path
    deck_path: Path
    output_path: Path
    deck_name: str = ""
    characters: tuple[str, ...] = ()
    note_count: int | None = None
    card_count: int | None = None
    package_sha256: str | None = None
    record_ids: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.state == "complete"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise KanjiFinishError(
            f"Character-note finish authority is not finite JSON: {exc}"
        ) from exc


def _relative(config: ProjectConfig, path: Path, *, label: str) -> str:
    root = config.root.resolve()
    target = path.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise KanjiFinishError(
            f"Character-note finish {label} escapes the repository: {target}"
        ) from exc
    value = relative.as_posix()
    if not value or ".." in PurePosixPath(value).parts:
        raise KanjiFinishError(
            f"Character-note finish {label} is not repository-relative."
        )
    return value


def _authority_path(config: ProjectConfig, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KanjiFinishError(f"Character-note finish {label} path is malformed.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise KanjiFinishError(
            f"Character-note finish {label} path is not repository-relative."
        )
    return (config.root.resolve() / Path(*pure.parts)).absolute()


def _finish_directory(config: ProjectConfig) -> Path:
    return (config.staging_dir / "done" / "kanji").absolute()


def plan_kanji_finish(
    config: ProjectConfig,
    notes_plan: assistant_kanji_notes.AssistantKanjiNotesPlan,
) -> KanjiFinishPlan:
    """Bind one prepared character-note plan to its apply and build receipt."""

    if notes_plan.repository_root != config.root.resolve():
        raise KanjiFinishError(
            "The prepared character-note plan belongs to another repository."
        )
    projection = notes_plan.projection
    target = projection.get("target")
    if not isinstance(target, Mapping):
        raise KanjiFinishError("The prepared character-note projection is incomplete.")
    service = notes_plan.service_plan
    deck_path = service.deck_path.absolute()
    output_path = service.output_path.absolute()
    deck_state = target.get("deck_state")
    if deck_state not in {"new", "existing"}:
        raise KanjiFinishError(
            "The prepared character-note projection has no exact deck state."
        )
    authority = {
        "version": 1,
        "instruction": notes_plan.request.instruction,
        "notes": json.loads(notes_plan.projection_wire),
        "plan": {
            "sha256": _sha(notes_plan.plan_wire.encode("utf-8")),
            "fingerprint": service.fingerprint,
            "wire": notes_plan.plan_wire,
        },
        "build": {
            "deck_path": _relative(config, deck_path, label="deck"),
            "output_path": _relative(config, output_path, label="package"),
            "deck_state": deck_state,
            # What the package will hold, which is the whole deck rather than
            # this batch's selection. Checking a build of an existing deck
            # against the selection would refuse every ordinary addition.
            "deck_note_count": service.deck_note_count,
            "deck_card_count": service.deck_card_count,
        },
    }
    fingerprint = _sha(_canonical(authority).encode("utf-8"))
    directory = _finish_directory(config)
    return KanjiFinishPlan(
        repository_root=config.root.resolve(),
        instruction=notes_plan.request.instruction,
        deck_path=deck_path,
        output_path=output_path,
        deck_name=service.deck_name,
        deck_state=deck_state,
        characters=tuple(service.characters),
        directions=tuple(service.directions),
        note_count=service.note_count,
        card_count=service.card_count,
        deck_note_count=service.deck_note_count,
        deck_card_count=service.deck_card_count,
        projection=projection,
        finish_directory=directory,
        record_path=directory / f"kanji-finish-{fingerprint}.json",
        authority=authority,
        fingerprint=fingerprint,
    )


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise KanjiFinishError(f"Character-note finish JSON repeats key {key!r}.")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise KanjiFinishError(
        f"Character-note finish JSON contains non-finite number {value}."
    )


def _record_text(record: Mapping[str, Any]) -> str:
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def _strict_record(wire: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            wire.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except KanjiFinishError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise KanjiFinishError(
            f"Could not parse character-note finish {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise KanjiFinishError("Character-note finish has invalid top-level fields.")
    receipt_id = value.get("receipt_id")
    authority = value.get("authority")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "kanji_finish"
        or not _is_sha(receipt_id)
        or path.name != f"kanji-finish-{receipt_id}.json"
        or not isinstance(authority, Mapping)
        or set(authority) != _AUTHORITY_KEYS
        or authority.get("version") != 1
        or _sha(_canonical(authority).encode("utf-8")) != receipt_id
    ):
        raise KanjiFinishError(
            "Character-note finish identity or authority is corrupt."
        )
    state = value.get("state")
    if state not in _STATE_INDEX:
        raise KanjiFinishError("Character-note finish state is invalid.")
    for key in ("authorized_at", "updated_at"):
        timestamp = value.get(key)
        if not isinstance(timestamp, str):
            raise KanjiFinishError(f"Character-note finish {key} is malformed.")
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise KanjiFinishError(
                f"Character-note finish {key} is malformed."
            ) from exc
    required = _STATE_INDEX[state]
    for index, key in enumerate(("apply_receipt", "build_receipt")):
        present = isinstance(value.get(key), Mapping)
        if (index < required) != present:
            raise KanjiFinishError(
                "Character-note finish receipts do not match its durable state."
            )
    return value


def _read_record(path: Path) -> tuple[dict[str, Any], str]:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError as exc:
        raise KanjiFinishError(
            f"Character-note finish no longer exists: {path}"
        ) from exc
    except (DataError, OSError) as exc:
        raise KanjiFinishError(
            f"Could not safely read character-note finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _read_record_optional(path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise KanjiFinishError(
            f"Could not safely read character-note finish {path}: {exc}"
        ) from exc
    return _strict_record(wire, path), _sha(wire)


def _new_record(plan: KanjiFinishPlan) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "kind": "kanji_finish",
        "receipt_id": plan.fingerprint,
        "state": "authorized",
        "authority": dict(plan.authority),
        "authorized_at": now,
        "updated_at": now,
        "apply_receipt": None,
        "build_receipt": None,
    }


def _write_new(path: Path, record: Mapping[str, Any]) -> str:
    text = _record_text(record)
    _strict_record(text.encode("utf-8"), path)
    try:
        atomic_write_text_bound(path, text, expected_absent=True)
    except (DataError, OSError) as exc:
        raise KanjiFinishError(
            f"Could not record finish authority before writing character notes: {exc}"
        ) from exc
    return _sha(text.encode("utf-8"))


def _advance(
    path: Path,
    record: Mapping[str, Any],
    revision: str,
    *,
    to_state: KanjiFinishState,
    receipt_key: str,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if _STATE_INDEX[to_state] != _STATE_INDEX[str(record.get("state"))] + 1:
        raise KanjiFinishError(
            f"Character-note finish cannot advance from {record.get('state')!r} "
            f"to {to_state!r}."
        )
    updated = dict(record)
    updated["state"] = to_state
    updated[receipt_key] = dict(receipt)
    updated["updated_at"] = datetime.now(UTC).isoformat()
    text = _record_text(updated)
    _strict_record(text.encode("utf-8"), path)
    with exclusive_path_lock(path):
        current, current_revision = _read_record(path)
        if current != dict(record) or current_revision != revision:
            raise KanjiFinishError(
                "Character-note finish changed while a completed phase was recorded."
            )
        try:
            atomic_write_text_bound(path, text, expected_revision=revision)
        except (DataError, OSError) as exc:
            raise KanjiFinishError(
                f"Could not record a completed character-note finish phase: {exc}"
            ) from exc
    return updated, _sha(text.encode("utf-8"))


def _authority_section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    authority = record.get("authority")
    section = authority.get(key) if isinstance(authority, Mapping) else None
    if not isinstance(section, Mapping):
        raise KanjiFinishError(f"Character-note finish {key} authority is malformed.")
    return section


def _emit(progress: KanjiFinishProgress | None, phase: KanjiFinishPhase) -> None:
    if progress is None:
        return
    try:
        progress(phase)
    except Exception:
        return


def _result(
    config: ProjectConfig,
    path: Path,
    record: Mapping[str, Any],
) -> KanjiFinishResult:
    build = _authority_section(record, "build")
    target = _authority_section(record, "notes").get("target")
    if not isinstance(target, Mapping):
        raise KanjiFinishError("Character-note finish target authority is malformed.")
    characters = target.get("characters")
    apply_receipt = record.get("apply_receipt")
    build_receipt = record.get("build_receipt")
    counts: Mapping[str, Any] = (
        build_receipt if isinstance(build_receipt, Mapping) else {}
    )
    record_ids = (
        apply_receipt.get("record_ids") if isinstance(apply_receipt, Mapping) else None
    )
    return KanjiFinishResult(
        receipt_id=str(record["receipt_id"]),
        state=str(record["state"]),  # type: ignore[arg-type]
        record_path=path,
        deck_path=_authority_path(config, build.get("deck_path"), label="deck"),
        output_path=_authority_path(config, build.get("output_path"), label="package"),
        deck_name=str(target.get("deck_name") or ""),
        characters=(
            tuple(str(item) for item in characters)
            if isinstance(characters, list)
            else ()
        ),
        note_count=(
            int(counts["note_count"]) if isinstance(counts.get("note_count"), int) else None
        ),
        card_count=(
            int(counts["card_count"]) if isinstance(counts.get("card_count"), int) else None
        ),
        package_sha256=(
            str(counts["package_sha256"])
            if isinstance(counts.get("package_sha256"), str)
            else None
        ),
        record_ids=tuple(record_ids) if isinstance(record_ids, list) else (),
    )


def _saved_plan(record: Mapping[str, Any]) -> character_notes.CharacterNotesPlan:
    """The exact plan the owner confirmed, rebuilt from the receipt alone."""

    section = _authority_section(record, "plan")
    wire = section.get("wire")
    fingerprint = section.get("fingerprint")
    if (
        not isinstance(wire, str)
        or _sha(wire.encode("utf-8")) != section.get("sha256")
        or not isinstance(fingerprint, str)
    ):
        raise KanjiFinishError(
            "The saved character-note plan is not bound to this finish authority."
        )
    saved = assistant_kanji_notes.restore_service_plan(wire)
    if saved.fingerprint != fingerprint:
        raise KanjiFinishError(
            "The saved character-note plan no longer matches its authorized "
            "fingerprint."
        )
    return saved


def _apply(
    config: ProjectConfig,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exact saved plan; never prepare, look up, or fetch again."""

    saved = _saved_plan(record)
    try:
        applied = character_notes.execute_character_notes(
            config,
            saved,
            expected_fingerprint=saved.fingerprint,
        )
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise KanjiFinishError(
            f"Could not write the confirmed character notes: {exc}"
        ) from exc
    return {
        "deck_path": _relative(config, applied.deck_path.absolute(), label="deck"),
        "output_path": _relative(
            config, applied.output_path.absolute(), label="package"
        ),
        "record_ids": list(applied.record_ids),
        "note_count": int(applied.note_count),
        "card_count": int(applied.card_count),
        "changed": bool(applied.changed),
    }


def _build(config: ProjectConfig, record: Mapping[str, Any]) -> dict[str, Any]:
    """Package the deck the receipt bound, once its applied state still holds.

    The order matters and is the whole point. Planning the package captures
    the exact deck and store bytes it will read; the applied state is then
    verified against the same saved plan the owner confirmed; and the build
    consumes that already-captured plan, whose executor re-checks its own
    inputs under locks. A resume that arrives after somebody edited the notes
    therefore refuses before publishing, without this module inventing a
    second lock or restating the core's fingerprint rules.
    """

    authority = _authority_section(record, "build")
    deck_path = _authority_path(config, authority.get("deck_path"), label="deck")
    output_path = _authority_path(config, authority.get("output_path"), label="package")
    deck_note_count = authority.get("deck_note_count")
    deck_card_count = authority.get("deck_card_count")
    if not isinstance(deck_note_count, int) or not isinstance(deck_card_count, int):
        raise KanjiFinishError(
            "Character-note finish build authority has no whole-deck counts."
        )
    try:
        plan = deck_package.plan_deck_package(config, deck_path)
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise KanjiFinishError(
            f"Could not plan the confirmed character-deck package: {exc}"
        ) from exc
    if plan.deck_path != deck_path.absolute() or plan.output_path != output_path:
        raise KanjiFinishError(
            "The character deck now builds to a different package than the "
            "confirmed batch bound."
        )
    saved = _saved_plan(record)
    try:
        character_notes.verify_character_notes_applied(config, saved)
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise KanjiFinishError(
            f"The confirmed character notes are no longer the applied state: {exc}"
        ) from exc
    try:
        built = deck_package.execute_deck_package(config, plan)
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise KanjiFinishError(
            f"Could not build the confirmed character deck: {exc}"
        ) from exc
    if built.note_count != deck_note_count or built.card_count != deck_card_count:
        raise KanjiFinishError(
            f"The built character deck holds {built.note_count} note(s) and "
            f"{built.card_count} card(s); the confirmed batch bound "
            f"{deck_note_count} and {deck_card_count}."
        )
    return {
        "output_path": _relative(config, built.output_path, label="package"),
        "package_sha256": built.package_sha256,
        "note_count": int(built.note_count),
        "card_count": int(built.card_count),
        "plan_fingerprint": plan.fingerprint,
    }


def _run(
    config: ProjectConfig,
    path: Path,
    record: dict[str, Any],
    revision: str,
    *,
    progress: KanjiFinishProgress | None,
) -> KanjiFinishResult:
    if record.get("state") == "authorized":
        _emit(progress, "Writing character notes")
        receipt = _apply(config, record)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="applied",
            receipt_key="apply_receipt",
            receipt=receipt,
        )
    if record.get("state") == "applied":
        _emit(progress, "Building Anki package")
        receipt = _build(config, record)
        _emit(progress, "Saving finish receipt")
        record, revision = _advance(
            path,
            record,
            revision,
            to_state="complete",
            receipt_key="build_receipt",
            receipt=receipt,
        )
    return _result(config, path, record)


def execute_kanji_finish(
    config: ProjectConfig,
    plan: KanjiFinishPlan,
    *,
    progress: KanjiFinishProgress | None = None,
) -> KanjiFinishResult:
    """Record the owner's authority, then apply and build under it exactly once."""

    if plan.repository_root != config.root.resolve():
        raise KanjiFinishError(
            "The character-note finish plan belongs to another repository."
        )
    _emit(progress, "Preparing finish")
    path = plan.record_path
    try:
        prepare_bound_directory(plan.finish_directory)
    except (DataError, OSError) as exc:
        raise KanjiFinishError(
            f"Could not prepare the character-note finish directory: {exc}"
        ) from exc
    existing = _read_record_optional(path)
    if existing is None:
        record = _new_record(plan)
        revision = _write_new(path, record)
    else:
        # The same exact batch was already authorized and interrupted. Continue
        # that receipt instead of re-authorizing identical work.
        record, revision = existing
    return _run(config, path, record, revision, progress=progress)


def resume_kanji_finish(
    config: ProjectConfig,
    receipt_id: str,
    *,
    progress: KanjiFinishProgress | None = None,
) -> KanjiFinishResult:
    """Continue one durable receipt without a new owner decision or lookup."""

    path = _receipt_path(config, receipt_id)
    record, revision = _read_record(path)
    _emit(progress, "Preparing finish")
    return _run(config, path, record, revision, progress=progress)


def inspect_kanji_finish(config: ProjectConfig, receipt_id: str) -> KanjiFinishResult:
    """Report one durable receipt's truthful state without changing it."""

    path = _receipt_path(config, receipt_id)
    record, _revision = _read_record(path)
    return _result(config, path, record)


def list_kanji_finishes(
    config: ProjectConfig,
    *,
    limit: int = _LIST_LIMIT,
) -> tuple[KanjiFinishResult, ...]:
    """Return the durable receipts a fresh process can still act on.

    This is how an interrupted batch is found again after a restart: the
    receipt on disk is the owner's original authority, so nothing here plans,
    prepares, or asks. An unreadable sibling is skipped rather than hiding the
    receipts beside it.

    Every receipt is read before the cap is applied, and the cap is applied to
    unfinished work first. A receipt is named for the SHA-256 of its authority,
    so ordering the *filenames* orders them arbitrarily; capping that order
    dropped whichever receipt happened to sort late, and past twenty completed
    batches it dropped it permanently — leaving the one authority an owner may
    resume unreachable from the surface built to offer it.
    """

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise KanjiFinishError("A character-note finish list limit must be an integer.")
    if limit <= 0:
        return ()
    directory = _finish_directory(config)
    try:
        names = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.name.startswith("kanji-finish-") and entry.name.endswith(".json")
        )
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise KanjiFinishError(
            f"Could not list character-note finish receipts: {exc}"
        ) from exc
    found: list[tuple[str, KanjiFinishResult]] = []
    for name in names:
        path = directory / name
        try:
            record, _revision = _read_record(path)
            found.append((str(record["updated_at"]), _result(config, path, record)))
        except KanjiFinishError:
            continue
    # Two stable passes, so the second keeps the order the first established.
    # `updated_at` is compared as text: every receipt janki writes carries an
    # aware UTC `datetime.isoformat()`, whose lexical order is chronological,
    # and comparing parsed values would raise the moment a hand-written naive
    # timestamp met an aware one.
    found.sort(key=lambda item: (item[0], item[1].receipt_id), reverse=True)
    found.sort(key=lambda item: item[1].succeeded)
    return tuple(result for _updated_at, result in found[:limit])


def _receipt_path(config: ProjectConfig, receipt_id: str) -> Path:
    if not _is_sha(receipt_id):
        raise KanjiFinishError("A character-note finish receipt id is malformed.")
    return _finish_directory(config) / f"kanji-finish-{receipt_id}.json"
