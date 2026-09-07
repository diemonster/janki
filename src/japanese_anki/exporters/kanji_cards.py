"""Character decks: one note per character, built from the curated store.

A kanji deck is not a word deck with a different template. Its notes come from
``data/kanji_notes.json`` rather than the vocabulary collection, its identities
are ``kanji:理`` rather than ``word:料理:りょうり``, and it selects by explicit
id rather than by intake tag — characters are chosen one at a time, not swept
up by a tag a promotion happened to write.

The answer side is the paper kanji card: meanings, the stroke strip, and the
readings the provider reports with the examples that provider itself bound to
them, all visible at once. Everything additional — the provider's later
reading groups, KANJIDIC's on/kun inventory — sits behind one disclosure,
because an answer split across three of them is an answer nobody opens.

Nothing here reads Japanese. Which reading a card shows, which example proves
it, and where a word breaks into kana are all decided by the source and copied
onto the note; this module decides only how they are laid out and escaped.
"""

from __future__ import annotations

import html
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import jpdb_kanji, kanji, kanji_notes
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import (
    BuildResult,
    _read_text,
    _write_package_atomic,
    resolve_deck_output_path,
)
from japanese_anki.io import DataError, load_structured

# One renderer for both card kinds. A word card's character block and a
# character card's answer show the same reported readings and the same
# dictionary inventory; two copies of that markup would drift, and the pair
# would then disagree about what a source said.
from japanese_anki.kanji import (
    _further_disclosure,
    _inventory_html,
    _labels_html,
    _reported_usage_html,
    _split_reported,
)
from japanese_anki.kanji_notes import CharacterNote
from japanese_anki.validation import field_separator_fault

try:  # pragma: no cover - exercised by the import guard in `build_kanji_deck`
    import genanki
except ImportError:  # pragma: no cover
    genanki = None

__all__ = [
    "KANJI_CARD_FILES",
    "KANJI_FIELDS",
    "KanjiDeckError",
    "build_kanji_deck",
    "deck_directions",
    "deck_problems",
    "kanji_template_paths",
    "note_values",
    "resolve_kanji_deck_notes",
]


class KanjiDeckError(JankiError):
    """A character deck cannot be read, or asks for a card its notes lack."""


#: Field order. Appended-only, like the vocabulary and pattern notetypes': a
#: note's values are positional, so an insertion shifts every value after it
#: on every note already in a collection.
KANJI_FIELDS: tuple[str, ...] = (
    "RecordID",
    "Character",
    "Meanings",
    "StrokeCount",
    "StrokeOrder",
    "CommonReadings",
    "OtherReadings",
    "ReadingInventory",
    "ReadingPrompt",
    "ReadingFurigana",
    "ReadingPronunciation",
    "ReadingGloss",
    "ProductionCue",
    "Sources",
)

#: Direction -> (front, back, Anki template name), in card order.
KANJI_CARD_FILES: dict[str, tuple[str, str, str]] = {
    "recognition": (
        "kanji-recognition-front.html",
        "kanji-recognition-back.html",
        "Kanji Recognition",
    ),
    "reading": (
        "kanji-reading-front.html",
        "kanji-reading-back.html",
        "Kanji Reading",
    ),
    "production": (
        "kanji-production-front.html",
        "kanji-production-back.html",
        "Kanji Production",
    ),
}

#: What a character deck builds when it says nothing. Deliberately *not*
#: ``[cards]`` from janki.toml: that default enables production for words,
#: where the prompt is a meaning the record already carries. A character's
#: production prompt is a cue only the owner can write, so a project-wide
#: word default must not silently mint a card with a blank front.
DEFAULT_DIRECTIONS: dict[str, bool] = {
    "recognition": True,
    "reading": False,
    "production": False,
}

DEFAULT_MODEL_NAME = "Japanese Kanji"


@dataclass(frozen=True, slots=True)
class KanjiDeck:
    """One character deck resolved into exactly what it will ship."""

    path: Path
    section: dict[str, Any]
    source_path: Path
    notes: tuple[CharacterNote, ...]
    directions: tuple[str, ...]
    deck_id: int
    model_id: int
    model_name: str
    deck_name: str


def kanji_template_paths(
    template_dir: Path, directions: tuple[str, ...]
) -> tuple[Path, ...]:
    """Every exact template file the character notetype consumes."""
    paths: list[Path] = []
    for direction in directions:
        front, back, _name = KANJI_CARD_FILES[direction]
        paths.extend([template_dir / front, template_dir / back])
    paths.append(template_dir / "style.css")
    return tuple(path.resolve() for path in paths)


def _section(deck_path: Path) -> dict[str, Any]:
    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    section = raw.get("deck") or {}
    if not isinstance(section, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    if str(section.get("kind") or "").strip().lower() != "kanji":
        raise KanjiDeckError(f"{deck_path} is not a character deck.")
    if raw.get("notes") is not None:
        raise KanjiDeckError(
            f"{deck_path}: a character deck holds no inline word notes; its "
            "cards come from the curated character store."
        )
    return section


def _identifier(section: dict[str, Any], key: str, deck_path: Path) -> int:
    """A deck or model id, refusing anything that only looks like one.

    Pinned in the file rather than derived, and required: the notetype id is
    what an existing collection matches a character note against, so deriving
    it from the enabled directions would move it the day a direction is added.
    """
    value = section.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"deck.{key} must be an integer, got {value!r}: {deck_path}")
    return value


def deck_directions(section: Mapping[str, Any], deck_path: Path) -> tuple[str, ...]:
    """Which card directions a character deck file enables, in card order."""
    cards = section.get("cards")
    if cards is None:
        cards = {}
    if not isinstance(cards, dict):
        raise DataError(
            f"deck.cards must be a mapping of card type to true/false, got "
            f"{type(cards).__name__}: {deck_path}"
        )
    unknown = sorted(set(cards) - set(KANJI_CARD_FILES))
    if unknown:
        raise KanjiDeckError(
            f"{deck_path}: a character deck has no {unknown[0]!r} card. Valid "
            f"directions: {', '.join(KANJI_CARD_FILES)}"
        )
    enabled = {**DEFAULT_DIRECTIONS, **cards}
    for direction, value in enabled.items():
        if not isinstance(value, bool):
            raise DataError(
                f"deck.cards.{direction} must be true or false, got {value!r}: "
                f"{deck_path}"
            )
    directions = tuple(name for name in KANJI_CARD_FILES if enabled[name])
    if not directions:
        raise KanjiDeckError(f"{deck_path}: enable at least one card direction.")
    return directions


def _declared_ids(section: dict[str, Any], deck_path: Path) -> tuple[str, ...]:
    value = section.get("include_ids")
    if not isinstance(value, list | tuple):
        raise DataError(
            f"deck.include_ids must be a list of character ids, got "
            f"{type(value).__name__}: {deck_path}"
        )
    ids = tuple(str(item) for item in value)
    if not ids:
        raise KanjiDeckError(
            f"{deck_path}: a character deck names the exact characters it "
            "holds, and this one names none."
        )
    repeated = sorted({item for item in ids if ids.count(item) > 1})
    if repeated:
        raise KanjiDeckError(f"{deck_path}: {repeated[0]} is listed more than once.")
    return ids


def resolve_kanji_deck_notes(
    deck_path: Path, config: ProjectConfig | None = None
) -> KanjiDeck:
    """Read one character deck and the exact notes it ships.

    Notes are ordered by character, which is the order the store is written
    in, so a rebuild after an addition moves nothing that was already there.
    """
    target = Path(deck_path).resolve()
    section = _section(target)
    source_value = section.get("source")
    if source_value:
        source_path = (target.parent / str(source_value)).resolve()
    elif config is not None:
        source_path = config.kanji_notes_file.resolve()
    else:
        raise KanjiDeckError(
            f"{target}: a character deck needs 'source:' naming its character store."
        )
    try:
        store = kanji_notes.load_notes(source_path)
    except JankiError as exc:
        raise KanjiDeckError(f"{target}: {exc}") from exc
    by_id = {note.id: note for note in store.values()}
    declared = _declared_ids(section, target)
    missing = [record_id for record_id in declared if record_id not in by_id]
    if missing:
        raise KanjiDeckError(
            f"{target}: {missing[0]} is not in {source_path}. Prepare the "
            "character note before building the deck that ships it."
        )
    notes = tuple(
        sorted((by_id[record_id] for record_id in declared), key=lambda n: n.character)
    )
    directions = deck_directions(section, target)
    for note in notes:
        supported = kanji_notes.supported_directions(note)
        for direction in directions:
            if direction in supported:
                continue
            raise KanjiDeckError(
                f"{target}: {note.character} supports no {direction} card. "
                + (
                    "A reading card asks one example the source bound to a "
                    "reading, and this note has none."
                    if direction == "reading"
                    else "A production card needs the disambiguating cue only "
                    "the owner can write."
                )
            )
    return KanjiDeck(
        path=target,
        section=section,
        source_path=source_path,
        notes=notes,
        directions=directions,
        deck_id=_identifier(section, "deck_id", target),
        model_id=_identifier(section, "model_id", target),
        model_name=str(section.get("model_name") or DEFAULT_MODEL_NAME),
        deck_name=str(section.get("name") or target.stem),
    )


def deck_problems(
    deck_path: Path, config: ProjectConfig | None = None
) -> list[str]:
    """Everything wrong with a character deck, for `janki validate`.

    The build refuses all of these; `validate` is the command whose job is
    catching a broken deck *before* one.
    """
    try:
        resolve_kanji_deck_notes(Path(deck_path), config)
    except JankiError as exc:
        return [str(exc)]
    return []


def _reported(
    note: CharacterNote,
) -> tuple[list[jpdb_kanji.ReadingUsage], list[jpdb_kanji.ReadingUsage]]:
    """The note's reported readings, split the way a word card splits them."""
    evidence = note.reading_evidence
    return _split_reported(None if evidence is None else evidence.readings)


def _common_readings(note: CharacterNote) -> str:
    """The readings the provider printed a figure beside, on the answer's face.

    The split is the provider's own doing rather than a judgement made here,
    and a quantified reading with no bound example still shows its figure.
    """
    quantified, _unquantified = _reported(note)
    return _reported_usage_html(quantified)


def _other_readings(note: CharacterNote) -> str:
    _quantified, unquantified = _reported(note)
    return _further_disclosure(
        "Other JPDB readings", "kanji-other", _labels_html(unquantified)
    )


def _reading_inventory(note: CharacterNote) -> str:
    """KANJIDIC's on/kun list, kept apart from the provider's groups.

    A reported reading group is not an on/kun inventory entry, and rendering
    them together would state a relationship neither source claims.
    """
    return _further_disclosure(
        "KANJIDIC readings",
        "kanji-inventory",
        _inventory_html(
            tuple(
                kanji.Reading(kind=reading.kind, reading=reading.reading)
                for reading in note.kanjidic_readings
            )
        ),
    )


def _strokes_html(note: CharacterNote, index: int) -> str:
    if not note.strokes:
        return ""
    info = kanji.KanjiInfo(
        character=note.character,
        stroke_count=note.stroke_count,
        strokes=note.strokes,
    )
    return kanji.render_stroke_strip(info, prefix=f"kanji-note-{index}")


def _sources_html(note: CharacterNote) -> str:
    parts = [html.escape(source) for source in note.sources]
    evidence = note.reading_evidence
    if evidence is not None and evidence.readings.source_url:
        url = html.escape(evidence.readings.source_url, quote=True)
        label = html.escape(evidence.source or evidence.readings.source_url)
        retrieved = (
            f" {html.escape(evidence.readings.fetched_at_utc)}"
            if evidence.readings.fetched_at_utc
            else ""
        )
        parts.append(f'<a href="{url}">{label}</a>{retrieved}')
    return " · ".join(part for part in parts if part)


def note_values(note: CharacterNote, index: int) -> list[str]:
    """One character note's field values, in :data:`KANJI_FIELDS` order.

    ``index`` scopes the stroke diagram's element ids so several characters on
    one card cannot define the same id twice.
    """
    example = note.reading_example
    return [
        note.id,
        html.escape(note.character),
        html.escape(", ".join(note.meanings)),
        str(note.stroke_count) if note.stroke_count else "",
        _strokes_html(note, index),
        _common_readings(note),
        _other_readings(note),
        _reading_inventory(note),
        html.escape(example.written) if example else "",
        html.escape(example.furigana) if example else "",
        html.escape(example.pronounced) if example else "",
        html.escape(example.gloss) if example else "",
        html.escape(note.production_cue),
        _sources_html(note),
    ]


def _notetype(deck: KanjiDeck, template_dir: Path) -> Any:
    templates = []
    for direction in deck.directions:
        front, back, name = KANJI_CARD_FILES[direction]
        templates.append(
            {
                "name": name,
                "qfmt": _read_text(template_dir / front),
                "afmt": _read_text(template_dir / back),
            }
        )
    return genanki.Model(
        deck.model_id,
        deck.model_name,
        fields=[{"name": name} for name in KANJI_FIELDS],
        templates=templates,
        css=_read_text(template_dir / "style.css"),
        # The character, so a collection sorts and dedups on the thing the
        # note is about rather than on its id string.
        sort_field_index=1,
    )


def build_kanji_deck(
    deck_path: Path,
    project_config: ProjectConfig,
    output_path: Path | None = None,
    *,
    output_expected_revision: str | None = None,
    output_expected_identity: tuple[int, int] | None = None,
    output_expected_absent: bool = False,
) -> BuildResult:
    """Build one character deck package from the curated notes it names.

    Offline by construction: every fact a card shows was copied onto its note
    when the note was written, so a build reads this repository and nothing
    else.
    """
    if genanki is None:
        raise KanjiDeckError(
            "genanki is not installed. Run: python -m pip install -e '.[dev]'"
        )
    deck_definition = resolve_kanji_deck_notes(Path(deck_path), project_config)
    model = _notetype(deck_definition, project_config.template_dir)
    deck = genanki.Deck(deck_definition.deck_id, deck_definition.deck_name)
    deck.description = str(deck_definition.section.get("description", ""))

    for index, note in enumerate(deck_definition.notes):
        values = note_values(note, index)
        fault = field_separator_fault(list(KANJI_FIELDS), values)
        if fault is not None:
            raise KanjiDeckError(
                f"{note.id}: the {fault} field contains U+001F, the separator "
                "Anki joins a note's fields with. Writing it would shift every "
                "later field out of place."
            )
        deck.add_note(
            genanki.Note(
                model=model,
                fields=values,
                tags=[tag.replace(" ", "_") for tag in note.tags if tag.strip()],
                guid=genanki.guid_for(note.id),
            )
        )

    target = output_path or resolve_deck_output_path(
        deck_definition.path, deck_definition.section, project_config
    )
    target = Path(os.path.abspath(os.fspath(target)))
    _write_package_atomic(
        genanki.Package(deck),
        target,
        expected_revision=output_expected_revision,
        expected_identity=output_expected_identity,
        expected_absent=output_expected_absent,
    )
    return BuildResult(
        output_path=target,
        deck_name=deck_definition.deck_name,
        note_count=len(deck_definition.notes),
        card_types=deck_definition.directions,
        media_count=0,
        record_ids=tuple(note.id for note in deck_definition.notes),
    )
