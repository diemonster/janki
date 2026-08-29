"""End-to-end smoke tests: build the repository's own deck and read it back.

Everything else in the suite tests a stage. These build the real deck from the
real records and open the resulting package, because the failures that reach a
learner are the ones that survive every stage individually: a clip referenced
by a note but never packaged plays as silence in Anki, a GUID that moves
re-imports as a new card and drops the learner's scheduling history, and a
notetype id that moves imports the deck beside the learner's existing cards
instead of into them.

Deliberately assertion-light on *content* and strict on *structure*. Counting
notes against the deck's own membership rather than a pinned number is the same
choice `test_anki_build.py` documents: a magic number fails whenever a word is
added, which says nothing about the builder.

Nothing here re-asks a question a source-level test already answers: template
`{{Field}}` names are held to `FIELD_NAMES` in `test_card_templates.py`, and the
notetype's own field list is pinned in `test_anki_build.py`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import (
    FIELD_NAMES,
    BuildResult,
    build_deck,
    deck_notetype,
    resolve_deck_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECK = PROJECT_ROOT / "data/decks/verbs.yaml"
VOCABULARY = PROJECT_ROOT / "data/normalized/vocabulary.json"
_SOUND = re.compile(r"\[sound:([^\]]+)\]")
_IMAGE = re.compile(r'<img src="([^"]+)">')


def _notes(package: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Every note's ``(guid, fields)`` and the package's media map."""
    rows, media, _models = _package(package)
    return rows, media


def _package(
    package: Path,
) -> tuple[list[tuple[str, str]], dict[str, str], dict[str, dict]]:
    """``(notes, media map, notetypes)`` as the package actually ships them."""
    with ZipFile(package) as archive:
        media = json.loads(archive.read("media"))
        with tempfile.TemporaryDirectory() as scratch:
            archive.extract("collection.anki2", scratch)
            connection = sqlite3.connect(Path(scratch) / "collection.anki2")
            try:
                rows = connection.execute("select guid, flds from notes").fetchall()
                models = json.loads(
                    connection.execute("select models from col").fetchone()[0]
                )
            finally:
                connection.close()
    return rows, media, models


def _built(tmp_path: Path, name: str = "verbs.apkg") -> tuple[Path, BuildResult]:
    output = tmp_path / name
    result = build_deck(DECK, ProjectConfig.load(PROJECT_ROOT), output)
    return output, result


def _referenced_media(rows: list[tuple[str, str]]) -> set[str]:
    """Every file name a note asks Anki for, audio and images alike."""
    return {
        name
        for _guid, fields in rows
        for name in (*_SOUND.findall(fields), *_IMAGE.findall(fields))
    }


def test_every_clip_a_note_asks_for_is_in_the_package(tmp_path: Path) -> None:
    """A missing clip is silent in Anki rather than an error: the note renders,
    the play button does nothing, and nothing upstream reports it.

    Over images too: an `<img src>` Anki cannot resolve renders as a broken
    image on the card, and `Image` has its own resolver, so a sound-only check
    would not see it.

    A hand-written `[sound:...]` tag is the one supported way to reference media
    janki does not package, and the build says so in a warning rather than
    failing. So the invariant is that nothing goes missing *quietly* — an
    unpackaged name has to be one the build named."""
    package, result = _built(tmp_path)
    rows, media = _notes(package)

    referenced = _referenced_media(rows)
    packaged = set(media.values())

    assert referenced, "no note referenced media; this test would prove nothing"
    # Matched on the whole tag rather than the bare name, so a warning excuses
    # only the file it is about. Inert against this deck, which writes no
    # verbatim tags — unlike an unreachable *guard*, which was left out of the
    # drill exporter for that reason, a test tolerance for supported content
    # that the repository's data does not happen to contain is what keeps
    # adding that content later from turning a red suite into the report. The
    # warning's own wording is covered by
    # `test_anki_build.py::test_a_verbatim_sound_tag_warns_rather_than_going_quietly`.
    reported = "\n".join(result.warnings)
    unpackaged = sorted(
        name for name in referenced - packaged if f"[sound:{name}]" not in reported
    )
    assert not unpackaged, unpackaged


def test_the_package_carries_no_clip_no_note_asks_for(tmp_path: Path) -> None:
    """The other direction. Dead weight in the package is not a learner-visible
    failure, but it is the signature of a media map built from the wrong set —
    the same defect as a missing clip, seen from the other side.

    Referenced media means audio *and* images. A sound-only reading of it would
    call a correctly referenced picture dead weight and fail on data the
    pipeline supports — `Image` is a first-class field with its own resolver."""
    package, _result = _built(tmp_path)
    rows, media = _notes(package)

    referenced = _referenced_media(rows)

    assert set(media.values()) <= referenced, sorted(set(media.values()) - referenced)


def test_note_identity_is_derived_from_the_record_id_and_holds(
    tmp_path: Path,
) -> None:
    """Anki keys scheduling to the GUID. A GUID that moves between builds
    re-imports as a new card and silently discards the learner's review
    history, which no amount of correct content makes up for.

    Two builds agreeing only proves the build is deterministic — it says
    nothing about *what* the GUID is derived from, and a switch from the record
    id to any other field is exactly the change that moves every GUID at once.
    So the derivation is pinned as well as the determinism."""
    import genanki

    first, _ = _notes(_built(tmp_path, "one.apkg")[0])
    second, _ = _notes(_built(tmp_path, "two.apkg")[0])
    _deck, records = resolve_deck_records(DECK)

    assert [guid for guid, _ in first] == [guid for guid, _ in second]
    assert {guid for guid, _ in first} == {
        genanki.guid_for(record.id) for record in records
    }


def test_every_note_is_distinct_and_populated(tmp_path: Path) -> None:
    rows, _ = _notes(_built(tmp_path)[0])
    _deck, expected = resolve_deck_records(DECK)

    guids = [guid for guid, _ in rows]
    assert len(rows) == len(expected)
    assert len(set(guids)) == len(guids), "two notes share a GUID"
    # `FIELD_NAMES[1]`, the expression — `[0]` is the record id, which the
    # builder always fills. A card whose front is empty is unreviewable and
    # unsearchable, and nothing else in the build refuses it.
    front = FIELD_NAMES.index("Expression")
    assert all(fields.split("\x1f")[front].strip() for _guid, fields in rows)


def test_the_shipped_field_order_is_the_one_existing_collections_were_built_on(
    tmp_path: Path,
) -> None:
    """A note's values are positional. Reordering the field list reassigns every
    value on every note already imported — `Reading` under `Expression`, and so
    on down — and Anki reports nothing, because the notetype it is merging into
    has the same field count and the same names.

    Written as a literal rather than as `== FIELD_NAMES`, which is what the
    exporter builds the notetype from: comparing it to itself moves both sides
    of the assertion together and catches nothing. Appending is the one change
    `docs/NOTETYPE_UPGRADE.md` sanctions, so this list is meant to be extended
    at the end, deliberately, by whoever appends — and never rearranged."""
    package, _result = _built(tmp_path)
    _rows, _media, models = _package(package)

    assert [field["name"] for field in next(iter(models.values()))["flds"]] == [
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
        "PitchAccent",
        "FrequencyRank",
        "ExampleAudio",
        "KanjiInfo",
        "CasualJapanese",
        "CasualFurigana",
        "CasualEnglish",
        "CasualAudio",
    ]


def test_the_build_refuses_a_separator_that_reached_a_field_off_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`validate_records` refuses a record carrying one, which is the earlier
    and better error. It cannot refuse this: the kanji block is rendered from
    `data/kanji.json` at build time, so the value never passes through a record
    at all. Refused at the exporter rather than reported, because the note it
    would write is silently wrong rather than visibly missing."""
    from japanese_anki.exporters import anki

    monkeypatch.setattr(
        anki,
        "render_kanji_html",
        lambda _info, **_context: "<div>stroke\x1forder</div>",
    )

    with pytest.raises(anki.AnkiBuildError) as excinfo:
        _built(tmp_path)

    assert "KanjiInfo" in str(excinfo.value)
    assert "U+001F" in str(excinfo.value)


def test_the_notetype_id_is_derived_from_the_enabled_card_set(
    tmp_path: Path,
) -> None:
    """The other half of note identity. Anki keys a note's fields to its
    notetype id; a deck that pins `model_id` keeps it by construction, but one
    that does not derives it from the enabled card types, and a change in that
    derivation re-identifies the notetype. Anki then imports the deck as a
    second, unrelated notetype rather than updating the first — the cards
    arrive, the learner's existing ones stay behind, and nothing reports it.

    This is the *common* path, not a corner: every vocabulary deck in the
    repository derives its id, and the only two that pin one are the rule decks,
    which return before this code (see `deck_notetype`). Asserted through
    `deck_notetype` and against the built package, because a check on
    `_card_mask` alone leaves the two call sites that combine it with the base
    free to change."""
    config = ProjectConfig.load(PROJECT_ROOT)
    base = config.model_id_base

    # `verbs.yaml` enables recognition and production. Written as a literal
    # rather than as `_card_mask(...)`, which would restate the implementation
    # and agree with it however it changed.
    package, _result = _built(tmp_path)
    _rows, _media, models = _package(package)

    assert deck_notetype(DECK, config) == (
        base + 3,
        "Japanese Study (recognition+production)",
        len(FIELD_NAMES),
    )
    assert [int(key) for key in models] == [base + 3]

    # A different card set has to land somewhere else. Reading alone is bit 4,
    # so a derivation that counted enabled types instead of masking them would
    # answer `base + 1` here and collide with a recognition-only deck.
    reading_only = tmp_path / "reading-only.yaml"
    reading_only.write_text(
        "deck:\n"
        '  name: "Smoke::Reading Only"\n'
        "  deck_id: 2059400999\n"
        '  output: "smoke-reading.apkg"\n'
        "  cards:\n"
        "    recognition: false\n"
        "    production: false\n"
        "    reading: true\n"
        f"  source: {VOCABULARY}\n",
        encoding="utf-8",
    )

    assert deck_notetype(reading_only, config)[:2] == (
        base + 4,
        "Japanese Study (reading)",
    )
