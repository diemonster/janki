from __future__ import annotations

import html
import os
import re
import unicodedata
from collections.abc import Container
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
from japanese_anki.kanji import load_store as load_kanji_store
from japanese_anki.kanji import render_kanji_html
from japanese_anki.models import ModelError, VocabularyRecord
from japanese_anki.pitch import PitchError, render_pitch_html
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
    # M5.4, appended in one release. **Never insert into this list** — a note's
    # values are positional, so an insertion shifts every value after it on
    # every existing note, and an append is one-way besides: the field is in
    # every collection that has imported the deck. See docs/NOTETYPE_UPGRADE.md
    # for what an append does to a live notetype, and for the "Merge Notetypes"
    # box that has to be ticked for it to happen at all.
    "PitchAccent",
    "FrequencyRank",
    "ExampleAudio",
    "KanjiInfo",
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
    #: Media janki could not package — a hand-written ``[sound:]`` tag. The
    #: build succeeds; the caller decides how loudly to say so.
    warnings: tuple[str, ...] = ()
    #: The records that reached the package, in note order. What the caller
    #: writes export entries from: ``--only-new`` narrows the set here, so a
    #: caller re-deriving it from the deck file would record records the
    #: package does not contain and make them invisible to the next run.
    record_ids: tuple[str, ...] = ()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AnkiBuildError(f"Missing template file: {path}") from exc


def _clean_tag(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def _html_lines(values: list[str]) -> str:
    return "<br>".join(html.escape(value) for value in values if value)


def _meanings_html(values: list[str], limit: int) -> str:
    """The glosses a card shows, and an honest note when it shows fewer.

    Capped at *display* rather than trimmed at import: jpdb's sense list is
    input, and this project does not discard input — `janki status`, a later
    search, and a human deciding which sense matters all still see all of them
    in `vocabulary.json`. The card is the thing with a size.

    The order is jpdb's own, which is roughly commonest-first, so the first few
    are the ones a learner meets. The count is kept rather than dropped: that
    する has 13 more senses is a fact about する, and a silent cut would teach
    that it has four.
    """
    shown = [value for value in values if value]
    if limit <= 0 or len(shown) <= limit:
        return _html_lines(shown)
    hidden = len(shown) - limit
    return _html_lines(shown[:limit]) + (
        f'<br><span class="more-senses">+{hidden} more sense'
        f'{"" if hidden == 1 else "s"}</span>'
    )


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


def _resolve_media(
    value: str,
    *,
    media_dir: Path,
    deck_dir: Path,
    record_id: str,
    label: str,
    media_files: list[str],
    warnings: list[str],
    claimed: dict[str, tuple[str, str]],
    sound_tags: bool = True,
) -> Path | None:
    """Find a media file, preferring ``media_dir`` and falling back to the deck.

    ``media_dir`` is where janki writes and what the config names, so it is
    tried first. The deck-relative reading is kept because it is what the
    exporter did before `media_dir` existed, and a hand-written note may still
    carry a path meant that way — dropping it would break those decks for no
    gain. Note the two agree for a `../media/...`-style path, since the
    directories are siblings; only a media-dir-relative path distinguishes them.

    Returns ``None`` when the value is a verbatim ``[sound:]`` tag, which the
    caller passes through untouched — audio fields only. An ``[sound:]`` in the
    *image* field is a mis-mapped record, not a hand-written tag: passing it
    through warned about silence for something that was never going to make a
    sound, and did it by writing the value into the card unescaped, which is
    the one field value here that would not have been.

    Anki flattens media into one folder by basename, and genanki packages each
    file under `os.path.basename`, so two different files whose names collide
    become one on import — the second overwrites the first and a card plays
    another word's audio. Content-addressed `janki-<fp>.wav` names make that
    unreachable for generated clips; a hand-written path can still do it, so a
    basename already claimed by a *different* file is refused here.
    """
    if value.startswith("[sound:") and sound_tags:
        # No file is packaged for a tag written by hand, so if the media is not
        # already in the collection the card is silently mute. Worth a word,
        # since this pipeline generates its own audio and would not produce one.
        warnings.append(
            f"{record_id}: {label} is a verbatim {value!r} — janki packages no "
            "file for it, so the card is silent unless that media is already in "
            "your collection."
        )
        return None
    for base in (media_dir, deck_dir):
        candidate = (base / value).resolve()
        if not candidate.exists():
            continue
        resolved = str(candidate)
        # Keyed the way Anki's media folder collides, not byte-for-byte: it
        # normalizes names to NFC, and this project's platform is case
        # insensitive, so `Hand.wav` and `hand.wav` are one file there and a
        # verbatim comparison would wave them through. The message keeps both
        # names as written, since those are what exist on disk.
        key = unicodedata.normalize("NFC", candidate.name).casefold()
        owner = claimed.setdefault(key, (resolved, record_id))
        if owner[0] != resolved:
            # `resolve()` rebuilds the path from the components as written,
            # fixing neither case nor composition, so two spellings of *one*
            # file compare unequal on exactly the filesystem the key above
            # accounts for. Ask the filesystem: `samefile` says no on a
            # case-sensitive volume, where they really are two files and
            # refusing them is right, and yes on macOS, where they are one.
            if not _same_file(owner[0], candidate):
                raise AnkiBuildError(
                    f"Two different files would be packaged under one media "
                    f"name: {owner[0]} for {owner[1]} and {resolved} for "
                    f"{record_id}. Anki stores media by basename, so one would "
                    "overwrite the other and a card would play the wrong clip. "
                    "Rename one."
                )
            # One file, two spellings: package it under the name the first
            # record claimed. Appending this spelling too would ship the same
            # bytes twice under two names Anki then collides anyway, and count
            # two in `media_count`.
            candidate = Path(owner[0])
            resolved = owner[0]
        media_files.append(resolved)
        return candidate
    raise AnkiBuildError(
        f"{label} for {record_id} does not exist: tried "
        f"{(media_dir / value).resolve()} and {(deck_dir / value).resolve()}"
    )


def _same_file(existing: str, candidate: Path) -> bool:
    """Are these two paths the same file on disk? ``False`` if it cannot tell."""
    try:
        return Path(existing).samefile(candidate)
    except OSError:
        # One of them vanished mid-build, or the volume refused the stat. The
        # honest answer is "cannot prove they are the same", which routes to
        # the collision error rather than to silently packaging one over the
        # other.
        return False


def _accent_patterns(record: VocabularyRecord) -> list[str]:
    """Every accepted accent this record carries, primary first, de-duplicated.

    Follows `pitch.select_pattern` exactly, and that agreement is the point
    rather than tidiness: `select_pattern` decides which accent `janki audio`
    *forces into the clip*, so a diagram built from a different list draws one
    accent onto a card that plays another.

    `audio_accent` therefore **replaces** the list rather than joining it.
    `select_pattern` never falls through to `pitch_accent` once it is set, so
    drawing both puts a diagram on the card that no clip says and that the
    curator overrode on purpose — asserting 橋 has two accepted accents on the
    card whose job is telling 橋 from 端. `models.py` and DESIGN_V2 both
    describe the field as a replacement.

    Upper-cased for the reason `select_pattern` upper-cases — `_levels` reads
    H and L.
    """
    override = record.audio_accent.strip().upper()
    if override:
        return [override]
    patterns: list[str] = []
    for candidate in record.pitch_accent:
        cleaned = candidate.strip().upper()
        if cleaned and cleaned not in patterns:
            patterns.append(cleaned)
    return patterns


def _pitch_field(record: VocabularyRecord, warnings: list[str]) -> str:
    """The accent diagram, or nothing.

    Every pattern the record carries, primary first — a word with two accepted
    accents has two, and showing one would teach that the other is wrong. A
    pattern that does not fit its reading is *left out* rather than drawn
    wrong: `render_pitch_html` refuses it, and a card is the last place to
    start guessing at an alignment janki declined to guess at everywhere else.

    Rendered one pattern at a time, because `render_pitch_html` refuses a whole
    list when any single member does not fit — so one malformed second entry
    discarded a perfectly good primary and the card got no diagram at all.
    Validation rates that mismatch a *warning*, so the build proceeds and the
    loss was silent; each drop is now reported, the way `janki audio` reports
    the same refusal instead of guessing past it.
    """
    if not record.reading:
        if _accent_patterns(record):
            # The docstring above promises every drop is reported, and this
            # return is a drop. Reachable from a hand-written inline note:
            # validation's own length check is gated on `reading` too, so
            # nothing else says the pattern went nowhere.
            warnings.append(
                f"{record.id}: has a pitch pattern but no reading to draw it "
                "over, so no diagram was drawn"
            )
        return ""
    rendered: list[str] = []
    for pattern in _accent_patterns(record):
        try:
            rendered.append(render_pitch_html(record.reading, [pattern]))
        except PitchError as exc:
            warnings.append(
                f"{record.id}: pitch pattern {pattern!r} does not fit reading "
                f"{record.reading!r}, so no diagram was drawn for it ({exc})"
            )
    return "".join(rendered)


def _field_values(
    record: VocabularyRecord,
    media_dir: Path,
    deck_dir: Path,
    media_files: list[str],
    warnings: list[str],
    claimed: dict[str, tuple[str, str]],
    max_meanings: int = 0,
    kanji_html: str = "",
) -> list[str]:
    """One note's fields, with media resolved against ``media_dir``.

    Media paths are stored relative to the project's ``media_dir`` — that is
    what ``janki audio`` writes and what the config names — rather than to the
    deck file. Resolving them against the deck's own directory looked equivalent
    while nothing had audio, and stopped being equivalent the moment something
    did: the first real clip sent the exporter looking in ``data/decks/audio/``
    for a file that lives in ``data/media/audio/``.
    """
    example = record.first_example
    audio_field = ""
    if record.audio:
        found = _resolve_media(
            record.audio, media_dir=media_dir, deck_dir=deck_dir,
            record_id=record.id, label="Audio",
            media_files=media_files, warnings=warnings, claimed=claimed,
        )
        audio_field = f"[sound:{found.name}]" if found else record.audio

    example_audio_field = ""
    if example.audio:
        found = _resolve_media(
            example.audio, media_dir=media_dir, deck_dir=deck_dir,
            record_id=record.id, label="Example audio",
            media_files=media_files, warnings=warnings, claimed=claimed,
        )
        example_audio_field = f"[sound:{found.name}]" if found else example.audio

    image_field = ""
    if record.image:
        found = _resolve_media(
            record.image, media_dir=media_dir, deck_dir=deck_dir,
            record_id=record.id, label="Image",
            media_files=media_files, warnings=warnings, claimed=claimed,
            sound_tags=False,
        )
        image_field = (
            f'<img src="{html.escape(found.name)}">' if found else record.image
        )

    return [
        record.id,
        html.escape(record.expression),
        html.escape(record.reading),
        html.escape(record.furigana),
        html.escape(record.romaji),
        _meanings_html(record.meanings, max_meanings),
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
        _pitch_field(record, warnings),
        str(record.frequency_rank) if record.frequency_rank is not None else "",
        example_audio_field,
        kanji_html,
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


def deck_declared_ids(deck_path: Path) -> set[str]:
    """Every record id a deck *declares*, before any include/exclude filter.

    Different question from :func:`resolve_deck_records`, which answers "what
    does this deck build". A caller asking "does this id exist anywhere in the
    collection" — the re-mint gate, the ``--replace`` ledger prune — must not
    lose an inline note a filter happens to drop: the note is still in the file,
    still carries hand-written content, and its GUID may already be in Anki.
    """
    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise DataError(f"Deck file must contain a mapping: {deck_path}")
    ids: set[str] = set()
    deck_config = raw.get("deck") or {}
    # Refused here exactly as `resolve_deck_records` refuses it. A deck this
    # function reads happily while every other reader calls it broken is worse
    # than either answer: its source file is never opened, its ids vanish from
    # the set, and callers act on a set they believe is complete.
    if not isinstance(deck_config, dict):
        raise DataError(f"The deck section must be a mapping: {deck_path}")
    if source_value := deck_config.get("source"):
        source_path = (deck_path.parent / str(source_value)).resolve()
        ids.update(record.id for record in load_records(source_path))
    inline_notes = raw.get("notes") or []
    if not isinstance(inline_notes, list):
        raise DataError(f"The notes section must be a list: {deck_path}")
    for item in inline_notes:
        if not isinstance(item, dict):
            raise DataError(f"Each note must be a mapping: {deck_path}")
        # A note with no `id:` is not skipped: the deck already builds it under
        # the id minted from its expression and reading, and that id — not the
        # absence of one — is what its GUID came from. Same rule as
        # `migrate._inline_ids`.
        try:
            record_id = (
                str(item.get("id", "")).strip() or VocabularyRecord.from_dict(item).id
            )
        except ModelError as exc:
            raise DataError(f"Could not read a note in {deck_path}: {exc}") from exc
        ids.add(record_id)
    return ids


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


def deck_notetype(deck_path: Path, project_config: ProjectConfig) -> tuple[int, str, int]:
    """``(model_id, model_name, field count)`` a build of this deck would use.

    Exported so the collection check asks the *same* question a build answers.
    Both keys default to values derived from the enabled card set, and a deck may
    pin either — a detector that recomputed them would misreport any deck that
    does, which is the one case the M5.5 spike singled out.
    """
    deck_config, _ = resolve_deck_records(deck_path)
    card_types = _resolve_card_types(deck_config, project_config)
    raw = deck_config.get("model_id", project_config.model_id_base + _card_mask(card_types))
    try:
        model_id = int(raw)
    except (TypeError, ValueError) as exc:
        # `janki status` reads this and is documented as the command that keeps
        # working, so a hand-edited `model_id: auto` has to arrive as a clean
        # refusal naming the file rather than as a traceback out of `int()`.
        raise AnkiBuildError(
            f"{deck_path}: deck model_id must be an integer, got {raw!r}"
        ) from exc
    name = deck_config.get("model_name", f"Japanese Study ({'+'.join(card_types)})")
    if not isinstance(name, str):
        raise AnkiBuildError(f"{deck_path}: deck model_name must be a string, got {name!r}")
    return model_id, name, len(FIELD_NAMES)


def build_deck(
    deck_path: Path,
    project_config: ProjectConfig,
    output_path: Path | None = None,
    include_ids: Container[str] | None = None,
) -> BuildResult:
    """Build one deck package.

    ``include_ids`` narrows the notes to those ids — ``build --only-new``'s
    hook. Filtering happens *after* validation, deliberately: a deck whose
    excluded records are broken is a broken deck, and letting an incremental
    build pass while a full one fails would hide that until the next full
    build, which is the least convenient moment to find out.
    """
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

    if include_ids is not None:
        records = [record for record in records if record.id in include_ids]

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

    media_dir = project_config.media_dir.resolve()
    # Looked up once for the whole build: 前 is the same 前 in every word.
    kanji_store = load_kanji_store(project_config.kanji_file)
    media_files: list[str] = []
    # Packaged basename -> (absolute path, the record that claimed it first).
    claimed: dict[str, tuple[str, str]] = {}
    media_warnings: list[str] = []
    for record in records:
        note = genanki.Note(
            model=model,
            fields=_field_values(
                record, media_dir, deck_path.parent, media_files, media_warnings,
                claimed,
                # A deck may say its own number; most take the project's.
                int(deck_config.get("max_meanings", project_config.max_meanings)),
                render_kanji_html(kanji_store.for_text(record.expression)),
            ),
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
    # Written beside the target and renamed into place. genanki needs a real
    # path and writes the zip incrementally, so an interrupted build (a ^C, a
    # full disk, a media file that vanishes mid-write) otherwise leaves a
    # truncated `.apkg` where a good one was — and a truncated package does not
    # look broken until Anki refuses it. `os.replace` is atomic within a
    # directory, so the previous package survives intact until the new one is
    # whole.
    scratch = output_path.with_name(f".{output_path.name}.partial")
    try:
        package.write_to_file(str(scratch))
        os.replace(scratch, output_path)
    finally:
        scratch.unlink(missing_ok=True)
    return BuildResult(
        output_path=output_path,
        deck_name=deck_name,
        note_count=len(records),
        card_types=tuple(card_types),
        media_count=len(set(media_files)),
        warnings=tuple(media_warnings),
        record_ids=tuple(record.id for record in records),
    )
