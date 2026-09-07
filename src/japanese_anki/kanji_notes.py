"""The curated character notes a kanji deck is built from.

Three stores sit beside each other and never merge. ``data/kanji.json`` is a
refreshable KANJIDIC/KanjiVG *reference cache*; ``data/jpdb_readings.json``
holds published reading percentages as *facts* fetched from a provider; this
one holds the **notes** — what a card actually is, curated, hand-editable, and
never replaced by a refresh of either cache.

A note's identity is ``kanji:理``: the character, not its readings. A note is
therefore stable while its reference data is refetched and its evidence
refreshed, which is what keeps Anki's review history attached across rebuilds.
Compounds appear inside a note as provider-bound examples and mint no
vocabulary record.

The evidence a note carries is a *copy* taken when the note was written, with
its source, retrieval time and response hash. That is deliberate: a card must
be able to say where its figures came from and when, and a later refresh of
the facts store must not silently restate what a shipped card claims.

Nothing here reads Japanese. Every question this module answers — is there an
example bound to a reading, is there a cue, is this one character — is about
presence and shape.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import jpdb_kanji
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import IdentityError, character_record_id
from japanese_anki.io import atomic_write_text, read_text_bound

__all__ = [
    "CharacterNote",
    "KanjiNoteError",
    "KanjidicReading",
    "ReadingEvidence",
    "SCHEMA_VERSION",
    "bound_examples",
    "evidence_from_readings",
    "first_bound_example",
    "load_notes",
    "render_notes",
    "save_notes",
    "supported_directions",
]

SCHEMA_VERSION = 1

#: Every direction a character note can offer, in card order. Appended-only
#: for the same reason the field list is: a template's ordinal is its identity
#: in a collection that has review history.
DIRECTIONS: tuple[str, ...] = ("recognition", "reading", "production")


class KanjiNoteError(JankiError):
    """A character note store is not shaped the way this module writes it."""


@dataclass(frozen=True, slots=True)
class ReadingEvidence:
    """The provider snapshot a note was written from, and what its figures are.

    The facts themselves are :class:`jpdb_kanji.CharacterReadings` — the same
    value the facts store holds, copied verbatim rather than re-expressed, so
    a note cannot quietly lose a label, a bound example or the response hash
    it was taken from. ``source`` and ``metric`` ride along because they live
    in the facts file's header: a card has to be able to say whose figure it
    is showing without opening that file.
    """

    source: str
    metric: str
    readings: jpdb_kanji.CharacterReadings

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metric": self.metric,
            "readings": self.readings.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReadingEvidence:
        values = _mapping(raw, "reading evidence")
        try:
            readings = jpdb_kanji.CharacterReadings.from_dict(
                values.get("readings")
            )
        except JankiError as exc:
            raise KanjiNoteError(f"reading_evidence: {exc}") from exc
        return cls(
            source=_text(values.get("source"), "evidence.source"),
            metric=_text(values.get("metric"), "evidence.metric"),
            readings=readings,
        )


def evidence_from_readings(readings: jpdb_kanji.CharacterReadings) -> ReadingEvidence:
    """Copy one facts entry onto a note, labelled with what its numbers are."""
    return ReadingEvidence(
        source=jpdb_kanji.SOURCE,
        metric=jpdb_kanji.METRIC,
        readings=readings,
    )


@dataclass(frozen=True, slots=True)
class KanjidicReading:
    """One entry of the character's on/kun inventory, as looked up."""

    kind: str
    reading: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reading": self.reading}

    @classmethod
    def from_dict(cls, raw: Any) -> KanjidicReading:
        values = _mapping(raw, "a reference reading")
        return cls(
            kind=_text(values.get("kind"), "kanjidic_readings[].kind"),
            reading=_text(values.get("reading"), "kanjidic_readings[].reading"),
        )


@dataclass(frozen=True, slots=True)
class CharacterNote:
    """One curated character note: the thing a kanji card is built from."""

    character: str
    id: str
    meanings: tuple[str, ...] = ()
    stroke_count: int = 0
    #: SVG path data, one per stroke, in writing order — copied from the
    #: reference cache so a build needs nothing but this file.
    strokes: tuple[str, ...] = ()
    kanjidic_readings: tuple[KanjidicReading, ...] = ()
    reading_evidence: ReadingEvidence | None = None
    #: The one example a reading card asks. Fixed when the note is written and
    #: preserved by every later refresh: a prompt that moved would change the
    #: question a card with review history is asking.
    reading_example: jpdb_kanji.BoundExample | None = None
    #: The owner's disambiguating cue. janki never writes one — a production
    #: card asks for a character from a hint, and authoring that hint is
    #: writing study content.
    production_cue: str = ""
    tags: tuple[str, ...] = ()
    created_at: str = ""
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "meanings": list(self.meanings),
            "stroke_count": self.stroke_count,
            "strokes": list(self.strokes),
            "kanjidic_readings": [
                reading.to_dict() for reading in self.kanjidic_readings
            ],
            "reading_evidence": (
                None if self.reading_evidence is None else self.reading_evidence.to_dict()
            ),
            "reading_example": (
                None if self.reading_example is None else self.reading_example.to_dict()
            ),
            "production_cue": self.production_cue,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, character: str, raw: Any) -> CharacterNote:
        values = _mapping(raw, f"the note for {character!r}")
        try:
            record_id = character_record_id(character)
        except IdentityError as exc:
            raise KanjiNoteError(str(exc)) from exc
        stored_id = _text(values.get("id"), "id") or record_id
        if stored_id != record_id:
            raise KanjiNoteError(
                f"The note filed under {character!r} carries id {stored_id!r}; "
                f"its identity is {record_id!r}. The key is the identity."
            )
        evidence = values.get("reading_evidence")
        example = values.get("reading_example")
        return cls(
            character=character,
            id=record_id,
            meanings=tuple(
                _text(item, "meanings[]")
                for item in _sequence(values.get("meanings"), "meanings")
            ),
            stroke_count=_int(values.get("stroke_count"), "stroke_count"),
            strokes=tuple(
                _text(item, "strokes[]")
                for item in _sequence(values.get("strokes"), "strokes")
            ),
            kanjidic_readings=tuple(
                KanjidicReading.from_dict(item)
                for item in _sequence(
                    values.get("kanjidic_readings"), "kanjidic_readings"
                )
            ),
            reading_evidence=(
                None if evidence is None else ReadingEvidence.from_dict(evidence)
            ),
            reading_example=(
                None if example is None else jpdb_kanji.BoundExample.from_dict(example)
            ),
            production_cue=_text(values.get("production_cue"), "production_cue"),
            tags=tuple(
                _text(item, "tags[]") for item in _sequence(values.get("tags"), "tags")
            ),
            created_at=_text(values.get("created_at"), "created_at"),
            sources=tuple(
                _text(item, "sources[]")
                for item in _sequence(values.get("sources"), "sources")
            ),
        )


def bound_examples(
    evidence: ReadingEvidence | None,
) -> tuple[jpdb_kanji.BoundExample, ...]:
    """Every provider-bound example on one snapshot, in source order."""
    if evidence is None:
        return ()
    return tuple(
        example
        for group in evidence.readings.groups
        for reading in group.readings
        for example in reading.examples
    )


def first_bound_example(
    evidence: ReadingEvidence | None,
) -> jpdb_kanji.BoundExample | None:
    """The example a new note fixes as its reading prompt, or ``None``.

    The *first* one the source printed, and nothing cleverer. Any other choice
    — commonest, shortest, most familiar — is a judgement about Japanese, and
    the provider's own order is the one fact available without making it.
    """
    examples = bound_examples(evidence)
    return examples[0] if examples else None


def supported_directions(note: CharacterNote) -> frozenset[str]:
    """Which card directions this note can carry, by presence alone.

    Recognition always: a character and its meanings are the note. Reading
    only with a fixed provider-bound example, because otherwise the card has
    nothing to ask. Production only with a cue the owner wrote, because
    otherwise the front is blank and janki will not fill it.
    """
    supported = {"recognition"}
    if note.reading_example is not None:
        supported.add("reading")
    if note.production_cue.strip():
        supported.add("production")
    return frozenset(supported)


def render_notes(notes: Mapping[str, CharacterNote]) -> str:
    """The exact file contents for one store state, sorted by character.

    Separate from :func:`save_notes` because a plan shows and binds the bytes
    it proposes before anything is written, and the two must be the same bytes.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "notes": {
            character: notes[character].to_dict() for character in sorted(notes)
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def load_notes(path: Path | str) -> dict[str, CharacterNote]:
    """Read the curated store. A missing file is an empty store, not an error."""
    file = Path(path)
    try:
        raw = json.loads(read_text_bound(file))
    except FileNotFoundError:
        return {}
    except (JankiError, OSError, ValueError) as exc:
        raise KanjiNoteError(f"Could not read {file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise KanjiNoteError(f"{file} must hold a JSON object")
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise KanjiNoteError(
            f"{file}: schema_version is {version!r}, not {SCHEMA_VERSION}"
        )
    entries = raw.get("notes")
    if not isinstance(entries, dict):
        raise KanjiNoteError(f"{file}: 'notes' must be an object keyed by character")
    notes: dict[str, CharacterNote] = {}
    key_by_id: dict[str, str] = {}
    for character, value in entries.items():
        key = str(character)
        try:
            note = CharacterNote.from_dict(key, value)
        except KanjiNoteError as exc:
            raise KanjiNoteError(f"{file}: {exc}") from exc
        first = key_by_id.get(note.id)
        if first is not None:
            raise KanjiNoteError(
                f"{file}: {first!r} ({_codepoints(first)}) and "
                f"{key!r} ({_codepoints(key)}) are two keys for one note, "
                f"{note.id!r} — NFKC folds them together. Building this store "
                "would ship one of them and silently drop the other's curated "
                "content. Keep the entry you want and delete the other by "
                "hand; janki will not choose, rename a key, or merge them."
            )
        key_by_id[note.id] = key
        notes[key] = note
    return notes


def save_notes(path: Path | str, notes: Mapping[str, CharacterNote]) -> None:
    """Write the curated store through the ordinary atomic writer."""
    atomic_write_text(Path(path), render_notes(notes))


def _codepoints(text: str) -> str:
    """``U+FA10`` for one key, so two keys that print alike can be told apart.

    A compatibility ideograph and the character it folds to are the same glyph
    on screen and in ``repr``; without this the refusal names two keys a reader
    cannot distinguish in their own file.
    """
    return " ".join(f"U+{ord(character):04X}" for character in text)


def _mapping(raw: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise KanjiNoteError(f"{what} must be an object, got {type(raw).__name__}")
    return raw


def _sequence(raw: Any, what: str) -> Iterable[Any]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, list | tuple):
        raise KanjiNoteError(f"{what} must be a list, got {type(raw).__name__}")
    return raw


def _text(raw: Any, what: str) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise KanjiNoteError(f"{what} must be text, got {type(raw).__name__}")
    return raw


def _optional_text(raw: Any, what: str) -> str | None:
    return None if raw is None else _text(raw, what)


def _int(raw: Any, what: str) -> int:
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise KanjiNoteError(f"{what} must be a whole number, got {raw!r}")
    return raw


def _optional_int(raw: Any, what: str) -> int | None:
    return None if raw is None else _int(raw, what)


def _optional_bool(raw: Any, what: str) -> bool | None:
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise KanjiNoteError(f"{what} must be true, false or null, got {raw!r}")
    return raw
