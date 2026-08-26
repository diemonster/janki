from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki.models import VocabularyRecord
from japanese_anki.preview import PreviewError, resolve_preview_records


def _write_deck(root: Path) -> Path:
    records = [
        VocabularyRecord(
            id="word:first:first",
            expression="first",
            reading="first",
            meanings=["source first"],
            tags=["lesson"],
        ),
        VocabularyRecord(
            id="word:second:second",
            expression="second",
            reading="second",
            meanings=["source second"],
            tags=["lesson"],
        ),
        VocabularyRecord(
            id="word:filtered:filtered",
            expression="filtered",
            reading="filtered",
            meanings=["not in this deck"],
            tags=["elsewhere"],
        ),
        VocabularyRecord(
            id="word:unrequested:unrequested",
            expression="unrequested",
            reading="unrequested",
            meanings=["in the deck but outside the exact preview"],
            tags=["lesson"],
        ),
    ]
    source = root / "vocabulary.json"
    source.write_text(
        json.dumps([record.to_dict() for record in records]), encoding="utf-8"
    )
    deck = root / "decks" / "lesson.yaml"
    deck.parent.mkdir()
    deck.write_text(
        """deck:
  name: Lesson
  source: ../vocabulary.json
  include_tags: [lesson]
notes:
  - id: word:second:second
    expression: second
    reading: second
    meanings: [inline second]
    tags: [lesson]
""",
        encoding="utf-8",
    )
    return deck


def test_exact_preview_returns_deck_versions_in_requested_order_without_writes(
    tmp_path: Path,
) -> None:
    deck = _write_deck(tmp_path)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    records = resolve_preview_records(
        deck, ("word:second:second", "word:first:first")
    )

    assert tuple(record.id for record in records) == (
        "word:second:second",
        "word:first:first",
    )
    assert records[0].meanings == ["inline second"]
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    ("record_ids", "message"),
    [
        ((), "at least one card ID"),
        ("word:first:first", "sequence of IDs, not text"),
        (b"word:first:first", "sequence of IDs, not text"),
        (("",), "nonblank strings"),
        (("   ",), "nonblank strings"),
        (
            ("word:first:first", "word:first:first"),
            "Preview card ID 'word:first:first' is repeated",
        ),
    ],
    ids=["empty", "text", "bytes", "blank", "whitespace", "duplicate"],
)
def test_exact_preview_refuses_a_scope_that_is_not_nonempty_unique_ids(
    tmp_path: Path, record_ids: object, message: str
) -> None:
    deck = _write_deck(tmp_path)

    with pytest.raises(PreviewError, match=message):
        resolve_preview_records(deck, record_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "record_id", ["word:missing:missing", "word:filtered:filtered"]
)
def test_exact_preview_refuses_ids_missing_from_the_resolved_deck(
    tmp_path: Path, record_id: str
) -> None:
    deck = _write_deck(tmp_path)

    with pytest.raises(PreviewError, match=record_id):
        resolve_preview_records(deck, (record_id,))
