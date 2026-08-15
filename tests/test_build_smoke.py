"""End-to-end smoke tests: build the repository's own deck and read it back.

Everything else in the suite tests a stage. These build the real deck from the
real records and open the resulting package, because the failures that reach a
learner are the ones that survive every stage individually: a clip referenced
by a note but never packaged plays as silence in Anki, a GUID that moves
re-imports as a new card and drops the learner's scheduling history, and a
template that stops substituting ships a literal `{{Expression}}`.

Deliberately assertion-light on *content* and strict on *structure*. Counting
notes against the deck's own membership rather than a pinned number is the same
choice `test_anki_build.py` documents: a magic number fails whenever a word is
added, which says nothing about the builder.
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
from japanese_anki.exporters.anki import build_deck, resolve_deck_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECK = PROJECT_ROOT / "data/decks/verbs.yaml"
_SOUND = re.compile(r"\[sound:([^\]]+)\]")


def _notes(package: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Every note's ``(guid, fields)`` and the package's media map."""
    with ZipFile(package) as archive:
        media = json.loads(archive.read("media"))
        with tempfile.TemporaryDirectory() as scratch:
            archive.extract("collection.anki2", scratch)
            connection = sqlite3.connect(Path(scratch) / "collection.anki2")
            try:
                rows = connection.execute("select guid, flds from notes").fetchall()
            finally:
                connection.close()
    return rows, media


def _built(tmp_path: Path, name: str = "verbs.apkg") -> Path:
    output = tmp_path / name
    build_deck(DECK, ProjectConfig.load(PROJECT_ROOT), output)
    return output


def test_every_clip_a_note_asks_for_is_in_the_package(tmp_path: Path) -> None:
    """A missing clip is silent in Anki rather than an error: the note renders,
    the play button does nothing, and nothing upstream reports it."""
    rows, media = _notes(_built(tmp_path))

    referenced = {name for _guid, fields in rows for name in _SOUND.findall(fields)}
    packaged = set(media.values())

    assert referenced, "no note referenced audio; this test would prove nothing"
    assert referenced <= packaged, sorted(referenced - packaged)


def test_the_package_carries_no_clip_no_note_asks_for(tmp_path: Path) -> None:
    """The other direction. Dead weight in the package is not a learner-visible
    failure, but it is the signature of a media map built from the wrong set —
    the same defect as a missing clip, seen from the other side."""
    rows, media = _notes(_built(tmp_path))

    referenced = {name for _guid, fields in rows for name in _SOUND.findall(fields)}

    assert set(media.values()) <= referenced, sorted(set(media.values()) - referenced)


def test_note_identity_is_stable_across_builds(tmp_path: Path) -> None:
    """Anki keys scheduling to the GUID. A GUID that moves between builds
    re-imports as a new card and silently discards the learner's review
    history, which no amount of correct content makes up for."""
    first, _ = _notes(_built(tmp_path, "one.apkg"))
    second, _ = _notes(_built(tmp_path, "two.apkg"))

    assert [guid for guid, _ in first] == [guid for guid, _ in second]


def test_every_note_is_distinct_and_populated(tmp_path: Path) -> None:
    rows, _ = _notes(_built(tmp_path))
    _deck, expected = resolve_deck_records(DECK)

    guids = [guid for guid, _ in rows]
    assert len(rows) == len(expected)
    assert len(set(guids)) == len(guids), "two notes share a GUID"
    # The first field is the expression; a card whose front is empty is
    # unreviewable and unsearchable, and nothing else in the build refuses it.
    assert all(fields.split("\x1f")[0].strip() for _guid, fields in rows)


def test_no_note_ships_an_unsubstituted_template_placeholder(tmp_path: Path) -> None:
    """A renamed field leaves `{{Old Name}}` in the rendered card rather than
    failing the build."""
    rows, _ = _notes(_built(tmp_path))

    leaked = [
        fields for _guid, fields in rows if "{{" in fields or "}}" in fields
    ]

    assert not leaked, leaked[:1]
