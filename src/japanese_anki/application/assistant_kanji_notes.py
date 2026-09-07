"""Plan-bound Assistant preparation of dedicated kanji character notes.

An explicit character target is its own content type.  This broker validates
the closed Assistant intent, asks :mod:`character_notes` for one exact local
preparation, and exposes a canonical path-free projection carrying the actual
note sides the owner confirms — not a prose summary of them.

Preparation is where the bounded dictionary read happens: the owner's message
authorized looking up exactly these characters, and the projection names them.
Confirmation must therefore never re-prepare or re-fetch. It hands the already
serialized plan back to the application service, which owns the locks and is
the sole writer of character notes, deck definitions, and reference facts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import jpdb_kanji, kanji_notes
from japanese_anki.application import character_notes
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

__all__ = [
    "SUPPORTED_DIRECTIONS",
    "AssistantKanjiNotesError",
    "AssistantKanjiNotesExecution",
    "AssistantKanjiNotesPlan",
    "AssistantKanjiNotesRequest",
    "characters_display",
    "execute_kanji_notes",
    "plan_kanji_notes",
    "restore_service_plan",
]

#: The card directions a character note can offer, in display order.  Which of
#: them one prepared note actually supports is the application service's gate;
#: this surface only refuses a direction that is not a direction at all.
SUPPORTED_DIRECTIONS = ("recognition", "production", "reading")

_DEFAULT_DIRECTIONS = ("recognition",)


class AssistantKanjiNotesError(JankiError):
    """An Assistant character-note request cannot be planned or executed."""


@dataclass(frozen=True, slots=True)
class AssistantKanjiNotesRequest:
    """Exact owner-supplied characters, destination, and card directions."""

    characters: tuple[str, ...]
    deck_path: Path | None
    deck_name: str | None
    directions: tuple[str, ...]
    refresh_readings: bool
    production_cues: tuple[tuple[str, str], ...]
    instruction: str

    def __post_init__(self) -> None:
        _validate_request_values(
            characters=self.characters,
            deck_path=self.deck_path,
            deck_name=self.deck_name,
            directions=self.directions,
            refresh_readings=self.refresh_readings,
            production_cues=self.production_cues,
            instruction=self.instruction,
        )


@dataclass(frozen=True, slots=True)
class AssistantKanjiNotesPlan:
    """One prepared service plan plus the exact confirmation projection."""

    repository_root: Path
    request: AssistantKanjiNotesRequest
    projection_wire: str
    plan_wire: str
    fingerprint: str
    service_plan: character_notes.CharacterNotesPlan

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("Assistant character-note root must be absolute")
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Assistant character-note root must be canonical")
        for label, wire in (
            ("projection", self.projection_wire),
            ("serialized plan", self.plan_wire),
        ):
            try:
                parsed = json.loads(wire)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Assistant character-note {label} must be JSON"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(
                    f"Assistant character-note {label} must be a JSON object"
                )
            if _canonical_json(parsed) != wire:
                raise ValueError(
                    f"Assistant character-note {label} must use canonical JSON"
                )
        if _sha256(self.projection_wire.encode("utf-8")) != self.fingerprint:
            raise ValueError(
                "Assistant character-note fingerprint does not bind its projection"
            )

    @property
    def projection(self) -> Mapping[str, Any]:
        """Return the parsed value safe for a browser confirmation card."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value

    @property
    def note_count(self) -> int:
        return len(self.request.characters)

    @property
    def card_count(self) -> int:
        return self.note_count * len(self.request.directions)


@dataclass(frozen=True, slots=True)
class AssistantKanjiNotesExecution:
    """The confirmed plan and the application service's applied result."""

    plan: AssistantKanjiNotesPlan
    result: Any


def _sha256(value: bytes) -> str:
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
        raise AssistantKanjiNotesError(
            f"Assistant character-note plan cannot be fingerprinted: {exc}"
        ) from exc


def _validate_request_values(
    *,
    characters: object,
    deck_path: object,
    deck_name: object,
    directions: object,
    refresh_readings: object,
    production_cues: object,
    instruction: object,
) -> None:
    if (
        not isinstance(characters, tuple)
        or not characters
        or any(not isinstance(item, str) or len(item) != 1 for item in characters)
    ):
        raise AssistantKanjiNotesError(
            "Character notes need one or more explicit single-character targets."
        )
    if len(set(characters)) != len(characters):
        raise AssistantKanjiNotesError(
            "Character notes name each requested character exactly once."
        )
    if (deck_path is None) == (deck_name is None):
        raise AssistantKanjiNotesError(
            "Character notes need either one existing character deck or one new "
            "deck name, not both and not neither."
        )
    if deck_path is not None and not isinstance(deck_path, Path):
        raise AssistantKanjiNotesError(
            "The character-note destination deck is not one resolved deck file."
        )
    if deck_name is not None and (
        not isinstance(deck_name, str)
        or not deck_name.strip()
        or deck_name != deck_name.strip()
    ):
        raise AssistantKanjiNotesError(
            "A new character deck needs an exact nonblank learner-facing name."
        )
    if (
        not isinstance(directions, tuple)
        or not directions
        or len(set(directions)) != len(directions)
        or any(direction not in SUPPORTED_DIRECTIONS for direction in directions)
    ):
        raise AssistantKanjiNotesError(
            "Character-note directions must be unique values chosen from "
            + ", ".join(SUPPORTED_DIRECTIONS)
            + "."
        )
    if not isinstance(refresh_readings, bool):
        raise AssistantKanjiNotesError(
            "Reading refresh must be an explicit true-or-false owner decision."
        )
    if not isinstance(production_cues, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], str)
        or item[0] not in characters
        or not item[1].strip()
        for item in production_cues
    ):
        raise AssistantKanjiNotesError(
            "Every production cue must name one requested character and carry the "
            "owner's own nonblank text."
        )
    if len({character for character, _cue in production_cues}) != len(production_cues):
        raise AssistantKanjiNotesError(
            "Each character carries at most one production cue."
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantKanjiNotesError(
            "Character notes need the owner's nonblank instruction."
        )


def _relative_path(config: ProjectConfig, path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantKanjiNotesError(
            f"Assistant character-note {label} escapes the configured repository."
        ) from exc


def _text(value: object) -> str:
    return str(value or "").strip()


def _note_value(note: kanji_notes.CharacterNote) -> Mapping[str, Any]:
    try:
        value = note.to_dict()  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssistantKanjiNotesError(
            f"A prepared character note cannot be serialized exactly: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise AssistantKanjiNotesError(
            "A prepared character note did not serialize to a JSON object."
        )
    return value


def _example_lines(examples: Sequence[jpdb_kanji.BoundExample]) -> list[str]:
    """The words the provider itself filed under one reading, in its order."""

    lines: list[str] = []
    for example in examples:
        written = _text(example.written)
        if not written:
            continue
        pronounced = _text(example.pronounced)
        gloss = _text(example.gloss)
        line = " ".join(part for part in (written, pronounced) if part)
        lines.append(f"{line} — {gloss}" if gloss else line)
    return lines


def _evidence_lines(evidence: kanji_notes.ReadingEvidence | None) -> list[str]:
    """Render the provider's own reading labels and figures, group by group.

    The snapshot's groups and readings are walked in the order the provider
    printed them, and each reading's own bound examples follow it, because the
    provider is the only thing that binds a word to a reading.

    A percentage the provider published stands on its own: a reading without a
    bound example still shows, because "no example yet" is not "no reported
    usage". A reading the provider printed *no* figure beside shows as the
    label it supplied and nothing more — the reported-usage wording belongs to
    a reported figure, and janki invents neither the number nor the claim. The
    printed text is carried through as printed, so an upper bound stays the
    bound it is. Nothing here reads the Japanese, classifies a reading, or
    ranks one against another.
    """

    if evidence is None:
        return []
    source = _text(evidence.source)
    lines: list[str] = []
    for group in evidence.readings.groups:
        for usage in group.readings:
            label = _text(usage.label)
            if not label:
                continue
            percent = _text(usage.percent_text)
            lines.append(
                f"{source} reported usage · {label} {percent}"
                if percent
                else f"{source} reading · {label}"
            )
            lines.extend(_example_lines(usage.examples))
    return lines


def _recognition_sides(
    note: kanji_notes.CharacterNote,
    meanings: Sequence[str],
) -> tuple[str, list[str]]:
    """The character, answered by everything the reference and facts hold."""

    back: list[str] = [f"Meanings: {', '.join(meanings)}"]
    if note.stroke_count > 0:
        back.append(f"Strokes: {note.stroke_count}")
    back.extend(_evidence_lines(note.reading_evidence))
    # The on/kun inventory is a separate lookup from the provider's figures and
    # stays its own line: neither store restates the other.
    inventory = [
        text for item in note.kanjidic_readings if (text := _text(item.reading))
    ]
    if inventory:
        back.append("Additional readings: " + ", ".join(inventory))
    return _text(note.character), back


def _reading_sides(
    note: kanji_notes.CharacterNote,
    meanings: Sequence[str],
) -> tuple[str, list[str]]:
    """The one example the provider bound, asked and then answered.

    The prompt is the note's fixed ``reading_example`` and nothing else: the
    card asks a word the source printed under a reading, and choosing a
    different one would be reading Japanese.
    """

    example = note.reading_example
    if example is None:
        raise AssistantKanjiNotesError(
            f"{note.character} has no provider-bound example, so a reading card "
            "has nothing to ask."
        )
    front = _text(example.written)
    if not front:
        raise AssistantKanjiNotesError(
            f"{note.character}'s bound reading example carries no written form."
        )
    answer = _text(example.furigana) or _text(example.pronounced)
    back = [line for line in (answer, _text(example.gloss)) if line]
    back.append(f"{note.character} · {', '.join(meanings)}")
    return front, back


def _production_sides(
    note: kanji_notes.CharacterNote,
    meanings: Sequence[str],
) -> tuple[str, list[str]]:
    """The owner's cue, answered by the character it disambiguates."""

    cue = _text(note.production_cue)
    if not cue:
        raise AssistantKanjiNotesError(
            f"{note.character} has no owner-written cue, so a production card "
            "would have a blank front."
        )
    return cue, [_text(note.character), f"Meanings: {', '.join(meanings)}"]


_CARD_SIDES = {
    "recognition": _recognition_sides,
    "reading": _reading_sides,
    "production": _production_sides,
}


def _note_projection(
    note: kanji_notes.CharacterNote,
    directions: Sequence[str],
) -> dict[str, Any]:
    """Project one prepared note as the exact card sides shown before apply.

    One entry per *selected* direction, in card order, each read off the note
    field its own template renders. A preview that showed the recognition
    front for a reading-only batch would be previewing a card the owner is
    not making.

    Its serialized form rides along unchanged as the exact snapshot the owner
    confirms, but a note states its character and identity as attributes —
    ``to_dict`` files the identity under ``id`` and carries no character key
    at all, because the store is keyed on it.
    """

    value = _note_value(note)
    character = _text(note.character)
    meanings = [text for item in note.meanings if (text := _text(item))]
    if not character or not meanings:
        raise AssistantKanjiNotesError(
            "A prepared character note has no exact source-backed character and "
            "meanings; Janki will not show an empty card side."
        )
    cards: list[dict[str, Any]] = []
    for direction in directions:
        sides = _CARD_SIDES.get(direction)
        if sides is None:
            raise AssistantKanjiNotesError(
                f"A character note has no {direction!r} card to preview."
            )
        front, back = sides(note, meanings)
        cards.append({"direction": direction, "front": front, "back": back})
    return {
        "character": character,
        "record_id": _text(note.id),
        "cards": cards,
        "note": dict(value),
    }


def _projection(
    config: ProjectConfig,
    request: AssistantKanjiNotesRequest,
    service: character_notes.CharacterNotesPlan,
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    notes = [
        _note_projection(note, service.directions) for note in service.notes
    ]
    if [note["character"] for note in notes] != list(service.characters):
        raise AssistantKanjiNotesError(
            "The prepared character notes do not match the requested characters."
        )
    if tuple(service.characters) != request.characters:
        raise AssistantKanjiNotesError(
            "The prepared plan changed the exact requested character targets."
        )
    if tuple(service.directions) != request.directions:
        raise AssistantKanjiNotesError(
            "The prepared plan changed the exact requested card directions."
        )
    if service.note_count != len(notes) or service.card_count != len(notes) * len(
        request.directions
    ):
        raise AssistantKanjiNotesError(
            "The prepared plan's note and card counts are inconsistent."
        )
    deck_record_ids = list(service.deck_record_ids)
    if service.deck_note_count != len(deck_record_ids) or (
        service.deck_card_count != len(deck_record_ids) * len(request.directions)
    ):
        raise AssistantKanjiNotesError(
            "The prepared plan's whole-deck note and card counts are inconsistent."
        )
    selected_ids = [note["record_id"] for note in notes]
    outside = [item for item in selected_ids if item not in deck_record_ids]
    if outside:
        raise AssistantKanjiNotesError(
            f"The prepared plan adds {outside[0]} without listing it among the "
            "identities the deck will hold."
        )
    writes = {
        "character_notes": _relative_path(
            config, config.kanji_notes_file, label="character note store"
        ),
        "package": _relative_path(config, service.output_path, label="package"),
    }
    if request.deck_name is not None:
        writes["deck_definition"] = _relative_path(
            config, service.deck_path, label="deck definition"
        )
    return {
        "schema_version": 1,
        "kind": "add_kanji_notes",
        "instruction": request.instruction,
        "target": {
            "deck_name": service.deck_name,
            "deck_state": "new" if request.deck_name is not None else "existing",
            "configured_file": _relative_path(
                config, service.deck_path, label="deck"
            ),
            "future_package": _relative_path(
                config, service.output_path, label="package"
            ),
            "characters": list(service.characters),
            "directions": list(service.directions),
            # What this batch selects, and separately what the package built
            # afterwards will hold. The two differ whenever the destination
            # already had notes, and a preview that reported only the first
            # would understate the deck the owner ends up with.
            "note_count": service.note_count,
            "card_count": service.card_count,
            "record_ids": selected_ids,
            "deck_note_count": service.deck_note_count,
            "deck_card_count": service.deck_card_count,
            "deck_record_ids": deck_record_ids,
            "notes": notes,
        },
        "inputs": {
            "queried_characters": list(service.characters),
            "dictionary_sources": ["KANJIDIC", "KanjiVG", "JPDB kanji pages"],
            "refresh_readings": request.refresh_readings,
            "plan_sha256": plan_sha256,
            "service_fingerprint": service.fingerprint,
        },
        "provider": None,
        "billing_class": "local",
        "writes": writes,
    }


def _serialized_plan(service: character_notes.CharacterNotesPlan) -> str:
    try:
        value = service.to_dict()
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssistantKanjiNotesError(
            f"The prepared character-note plan cannot be serialized: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise AssistantKanjiNotesError(
            "The prepared character-note plan did not serialize to a JSON object."
        )
    return _canonical_json(value)


def plan_kanji_notes(
    config: ProjectConfig,
    request: AssistantKanjiNotesRequest,
) -> AssistantKanjiNotesPlan:
    """Prepare one exact character-note plan for the owner's confirmation."""

    if not isinstance(request, AssistantKanjiNotesRequest):
        raise AssistantKanjiNotesError(
            "Assistant character notes need one validated typed request."
        )
    try:
        service = character_notes.prepare_character_notes(
            config,
            request.characters,
            deck_path=request.deck_path,
            deck_name=request.deck_name,
            directions=request.directions,
            refresh_readings=request.refresh_readings,
            production_cues=dict(request.production_cues) or None,
        )
        plan_wire = _serialized_plan(service)
        projection = _projection(
            config,
            request,
            service,
            plan_sha256=_sha256(plan_wire.encode("utf-8")),
        )
    except AssistantKanjiNotesError:
        raise
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise AssistantKanjiNotesError(
            "Could not prepare character notes for "
            f"{''.join(request.characters)}: {exc}"
        ) from exc
    wire = _canonical_json(projection)
    return AssistantKanjiNotesPlan(
        repository_root=config.root.resolve(),
        request=request,
        projection_wire=wire,
        plan_wire=plan_wire,
        fingerprint=_sha256(wire.encode("utf-8")),
        service_plan=service,
    )


def execute_kanji_notes(
    config: ProjectConfig,
    expected: AssistantKanjiNotesPlan,
) -> AssistantKanjiNotesExecution:
    """Apply the exact serialized plan the owner confirmed, without re-preparing.

    Re-planning here would repeat the bounded dictionary read the preparation
    already performed and could silently substitute different facts for the
    ones the owner saw. The application service holds the locks and refuses
    anything that no longer matches its bound preconditions.
    """

    if expected.repository_root != config.root.resolve():
        raise AssistantKanjiNotesError(
            "The Assistant character-note plan belongs to another repository."
        )
    try:
        result = character_notes.execute_character_notes(
            config,
            expected.service_plan,
            expected_fingerprint=expected.service_plan.fingerprint,
        )
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise AssistantKanjiNotesError(
            f"Could not apply the confirmed character notes: {exc}"
        ) from exc
    return AssistantKanjiNotesExecution(plan=expected, result=result)


def restore_service_plan(plan_wire: str) -> character_notes.CharacterNotesPlan:
    """Rebuild one exact saved plan for a durable resume, without preparing."""

    try:
        value = json.loads(plan_wire)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AssistantKanjiNotesError(
            f"The saved character-note plan is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise AssistantKanjiNotesError(
            "The saved character-note plan is not a JSON object."
        )
    try:
        return character_notes.CharacterNotesPlan.from_dict(value)
    except (JankiError, OSError, TypeError, ValueError) as exc:
        raise AssistantKanjiNotesError(
            f"The saved character-note plan could not be restored: {exc}"
        ) from exc


def characters_display(characters: Sequence[str]) -> str:
    """Join requested characters for one owner-facing line."""

    return "、".join(characters)
