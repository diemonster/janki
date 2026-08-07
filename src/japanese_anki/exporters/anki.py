from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import genanki
except ImportError:  # pragma: no cover - exercised by the bootstrap environment
    genanki = None

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError, load_records, load_structured
from japanese_anki.models import ModelError, VocabularyRecord
from japanese_anki.validation import has_errors, validate_records


class AnkiBuildError(JankiError):
    pass


FIELD_NAMES = [
    "RecordID",
    "Expression",
    "Reading",
    "Furigana",
    "Romaji",
    "Meanings",
    "PartOfSpeech",
    "VerbGroup",
    "Transitivity",
    "ExampleJapanese",
    "ExampleFurigana",
    "ExampleRomaji",
    "ExampleEnglish",
    "Conjugations",
    "UsageNotes",
    "Audio",
    "Image",
    "ShirabeQuery",
    "Source",
]

CARD_FILES = {
    "recognition": ("recognition-front.html", "recognition-back.html", "Recognition"),
    "production": ("production-front.html", "production-back.html", "Production"),
    "reading": ("reading-front.html", "reading-back.html", "Reading"),
}


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_path: Path
    deck_name: str
    note_count: int
    card_types: tuple[str, ...]
    media_count: int


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AnkiBuildError(f"Missing template file: {path}") from exc


def _clean_tag(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def _html_lines(values: list[str]) -> str:
    return "<br>".join(html.escape(value) for value in values if value)


def _conjugation_html(values: dict[str, str]) -> str:
    if not values:
        return ""
    items = []
    for label, value in values.items():
        items.append(
            "<div class=\"conjugation-row\">"
            f"<span class=\"conjugation-label\">{html.escape(label)}</span>"
            f"<span class=\"conjugation-value\">{html.escape(value)}</span>"
            "</div>"
        )
    return "".join(items)


def _source_text(record: VocabularyRecord) -> str:
    source = record.source
    parts = [source.type]
    if source.imported_from:
        parts.append(source.imported_from)
    if source.row is not None:
        parts.append(f"row {source.row}")
    return " · ".join(part for part in parts if part)


def _field_values(
    record: VocabularyRecord, deck_path: Path, media_files: list[str]
) -> list[str]:
    example = record.first_example
    audio_field = ""
    if record.audio:
        if record.audio.startswith("[sound:"):
            audio_field = record.audio
        else:
            audio_path = (deck_path.parent / record.audio).resolve()
            if not audio_path.exists():
                raise AnkiBuildError(
                    f"Audio file for {record.id} does not exist: {audio_path}"
                )
            media_files.append(str(audio_path))
            audio_field = f"[sound:{audio_path.name}]"

    image_field = ""
    if record.image:
        image_path = (deck_path.parent / record.image).resolve()
        if not image_path.exists():
            raise AnkiBuildError(
                f"Image file for {record.id} does not exist: {image_path}"
            )
        media_files.append(str(image_path))
        image_field = f'<img src="{html.escape(image_path.name)}">'

    return [
        record.id,
        html.escape(record.expression),
        html.escape(record.reading),
        html.escape(record.furigana),
        html.escape(record.romaji),
        _html_lines(record.meanings),
        html.escape(record.part_of_speech),
        html.escape(record.verb_group),
        html.escape(record.transitivity),
        html.escape(example.japanese),
        html.escape(example.furigana),
        html.escape(example.romaji),
        html.escape(example.english),
        _conjugation_html(record.conjugations),
        html.escape(record.usage_notes).replace("\n", "<br>"),
        audio_field,
        image_field,
        html.escape(record.expression),
        html.escape(_source_text(record)),
    ]


def _merge_inline_record(base: VocabularyRecord | None, raw: dict[str, Any]) -> VocabularyRecord:
    inline = VocabularyRecord.from_dict(raw)
    if base is None:
        return inline
    data = base.to_dict()
    for key, value in raw.items():
        if key == "source" and isinstance(value, dict):
            merged_source = dict(data.get("source") or {})
            merged_source.update(value)
            data["source"] = merged_source
        else:
            data[key] = value
    return VocabularyRecord.from_dict(data)


# The ``deck:`` keys whose shape this module iterates or unpacks, and the noun
# each one holds. A scalar written where a list belongs used to escape as
# ``TypeError: 'int' object is not iterable`` — not a JankiError, so `janki
# status` (whose contract is to warn and skip one broken deck) died on it, and
# `build`/`validate` printed a traceback instead of naming the file.
_DECK_LIST_KEYS: dict[str, str] = {
    "include_ids": "ids",
    "exclude_ids": "ids",
    "include_tags": "tags",
    "exclude_tags": "tags",
}


def _deck_string_set(deck_config: dict[str, Any], key: str, deck_path: Path) -> set[str]:
    """One deck filter as the set of strings it names, or a clean error.

    Only an absent key (``exclude_ids:`` with nothing after it) reads as "no
    filter"; ``0`` and ``""`` are values written where a list belongs, and the
    old ``or []`` swallowed them along with the scalars.
    """
    value = deck_config.get(key)
    if value is None:
        return set()
    if not isinstance(value, list | tuple):
        raise DataError(
            f"deck.{key} must be a list of {_DECK_LIST_KEYS[key]}, got "
            f"{type(value).__name__}: {deck_path}"
        )
    return {str(item) for item in value}


def resolve_deck_records(deck_path: Path) -> tuple[dict[str, Any], list[VocabularyRecord]]:
    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    deck_config = raw.get("deck") or {}
    if not isinstance(deck_config, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    # Checked here rather than in `_resolve_card_types`, which only `build`
    # reaches: every path into a deck file comes through this function, so this
    # is where a deck's shape is refused once for all three commands.
    cards = deck_config.get("cards")
    if cards is not None and not isinstance(cards, dict):
        raise DataError(
            f"deck.cards must be a mapping of card type to true/false, got "
            f"{type(cards).__name__}: {deck_path}"
        )

    by_id: dict[str, VocabularyRecord] = {}
    source_value = deck_config.get("source")
    if source_value:
        source_path = (deck_path.parent / str(source_value)).resolve()
        for record in load_records(source_path):
            by_id[record.id] = record

    inline_notes = raw.get("notes") or []
    if not isinstance(inline_notes, list):
        raise DataError(f"The notes section must be a list: {deck_path}")
    for item in inline_notes:
        if not isinstance(item, dict):
            raise DataError(f"Each note must be a mapping: {deck_path}")
        record_id = str(item.get("id", "")).strip()
        base = by_id.get(record_id) if record_id else None
        try:
            merged = _merge_inline_record(base, item)
        except ModelError as exc:
            # The constructor knows the field; only this frame knows the deck.
            raise DataError(f"Could not read a note in {deck_path}: {exc}") from exc
        by_id[merged.id] = merged

    records = list(by_id.values())
    include_ids = _deck_string_set(deck_config, "include_ids", deck_path)
    exclude_ids = _deck_string_set(deck_config, "exclude_ids", deck_path)
    include_tags = _deck_string_set(deck_config, "include_tags", deck_path)
    exclude_tags = _deck_string_set(deck_config, "exclude_tags", deck_path)

    if include_ids:
        records = [record for record in records if record.id in include_ids]
    if include_tags:
        records = [record for record in records if include_tags & set(record.tags)]
    if exclude_ids:
        records = [record for record in records if record.id not in exclude_ids]
    if exclude_tags:
        records = [record for record in records if not (exclude_tags & set(record.tags))]

    records.sort(key=lambda record: (record.expression, record.reading, record.id))
    return deck_config, records


def _card_mask(card_types: list[str]) -> int:
    mask = 0
    if "recognition" in card_types:
        mask |= 1
    if "production" in card_types:
        mask |= 2
    if "reading" in card_types:
        mask |= 4
    return mask


def _resolve_card_types(
    deck_config: dict[str, Any], project_config: ProjectConfig
) -> list[str]:
    """Which card types to build. ``deck_config`` comes from
    :func:`resolve_deck_records`, which has already refused a ``cards`` that is
    not a mapping — that check names the deck file, which this frame cannot."""
    card_config = dict(project_config.default_cards)
    card_config.update(deck_config.get("cards") or {})
    card_types = [name for name in CARD_FILES if bool(card_config.get(name, False))]
    if not card_types:
        raise AnkiBuildError("At least one card type must be enabled")
    return card_types


def build_deck(
    deck_path: Path,
    project_config: ProjectConfig,
    output_path: Path | None = None,
) -> BuildResult:
    if genanki is None:
        raise AnkiBuildError(
            "genanki is not installed. Run: python -m pip install -e '.[dev]'"
        )
    deck_path = deck_path.resolve()
    deck_config, records = resolve_deck_records(deck_path)
    issues = validate_records(records, deck_path)
    if has_errors(issues):
        formatted = "\n".join(issue.format() for issue in issues)
        raise AnkiBuildError(f"Deck validation failed:\n{formatted}")

    card_types = _resolve_card_types(deck_config, project_config)
    template_dir = project_config.template_dir
    templates = []
    for card_type in card_types:
        front_file, back_file, display_name = CARD_FILES[card_type]
        templates.append(
            {
                "name": display_name,
                "qfmt": _read_text(template_dir / front_file),
                "afmt": _read_text(template_dir / back_file),
            }
        )

    model_id = int(
        deck_config.get(
            "model_id",
            project_config.model_id_base + _card_mask(card_types),
        )
    )
    deck_id = int(deck_config.get("deck_id", project_config.default_deck_id))
    deck_name = str(deck_config.get("name", project_config.default_deck_name))
    model_name = str(
        deck_config.get(
            "model_name",
            f"Japanese Study ({'+'.join(card_types)})",
        )
    )

    model = genanki.Model(
        model_id,
        model_name,
        fields=[{"name": name} for name in FIELD_NAMES],
        templates=templates,
        css=_read_text(template_dir / "style.css"),
        sort_field_index=1,
    )
    deck = genanki.Deck(deck_id, deck_name)
    deck.description = str(deck_config.get("description", ""))

    media_files: list[str] = []
    for record in records:
        note = genanki.Note(
            model=model,
            fields=_field_values(record, deck_path, media_files),
            tags=[_clean_tag(tag) for tag in record.tags if _clean_tag(tag)],
            guid=genanki.guid_for(record.id),
        )
        deck.add_note(note)

    if output_path is None:
        filename = str(deck_config.get("output", f"{deck_path.stem}.apkg"))
        output_path = project_config.dist_dir / filename
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    package = genanki.Package(deck)
    package.media_files = sorted(set(media_files))
    package.write_to_file(str(output_path))
    return BuildResult(
        output_path=output_path,
        deck_name=deck_name,
        note_count=len(records),
        card_types=tuple(card_types),
        media_count=len(set(media_files)),
    )
