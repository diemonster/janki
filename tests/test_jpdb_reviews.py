"""jpdb's review export, matched against records janki already holds.

``import-jpdb-reviews`` creates nothing — it marks what jpdb is already
drilling so a deck can exclude it — so every test here is about *matching*:
which records an entry finds, what lands on them, and what is reported when an
entry finds nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli
from japanese_anki.importers.jpdb_reviews import (
    KNOWN_TAG,
    REVIEW_COUNT_FIELD,
    JpdbReviewsError,
    apply_reviews,
    read_reviews,
)
from japanese_anki.models import SourceReference, VocabularyRecord


def review(spelling: str, reading: str, vid: int | None, count: int) -> dict[str, Any]:
    """One card in the export's shape: a word plus its review history."""
    return {
        "vid": vid,
        "spelling": spelling,
        "reading": reading,
        "reviews": [
            {"timestamp": 1700000000 + index, "grade": "okay", "from_anki": False}
            for index in range(count)
        ],
    }


def export(*cards: dict[str, Any], **sections: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"cards_vocabulary_jp_en": list(cards)}
    payload.update(sections)
    return payload


def write_export(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def record(
    expression: str, reading: str, *, vid: str = "", tags: list[str] | None = None
) -> VocabularyRecord:
    raw_fields = {"vid": vid} if vid else {}
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        tags=list(tags or []),
        source=SourceReference(type="shirabe", imported_from="e.csv", raw_fields=raw_fields),
    )


HANASU = record("話す", "はなす")
TABERU = record("食べる", "たべる")


# --- reading the export -----------------------------------------------------


def test_every_vocabulary_card_list_is_read_not_just_the_first(tmp_path: Path) -> None:
    # jpdb exports one list per card type; jp_en and en_jp are the same words
    # drilled in two directions.
    path = write_export(
        tmp_path / "reviews.json",
        export(
            review("話す", "はなす", 1562350, 3),
            cards_vocabulary_en_jp=[review("話す", "はなす", 1562350, 2)],
        ),
    )

    entries, skipped = read_reviews(path)

    # One word, its counts summed — not the same word reported twice.
    assert [(entry.spelling, entry.reviews) for entry in entries] == [("話す", 5)]
    assert skipped == []


def test_a_non_vocabulary_section_is_reported_rather_than_ignored(tmp_path: Path) -> None:
    path = write_export(
        tmp_path / "reviews.json",
        export(review("話す", "はなす", 1562350, 1), cards_kanji_keyword_char=[{"kanji": "話"}]),
    )

    entries, skipped = read_reviews(path)

    assert len(entries) == 1
    assert skipped == ["cards_kanji_keyword_char"]


def test_an_export_with_no_vocabulary_list_says_what_it_found(tmp_path: Path) -> None:
    path = write_export(tmp_path / "reviews.json", {"cards_kanji_keyword_char": []})

    with pytest.raises(JpdbReviewsError) as excinfo:
        read_reviews(path)

    assert "cards_kanji_keyword_char" in str(excinfo.value)


def test_a_file_that_is_not_json_names_the_command_that_reads_it(tmp_path: Path) -> None:
    path = tmp_path / "reviews.json"
    path.write_text("not json at all", encoding="utf-8")

    with pytest.raises(JpdbReviewsError) as excinfo:
        read_reviews(path)

    assert "Export vocabulary reviews" in str(excinfo.value)


def test_a_card_with_neither_vid_nor_spelling_is_an_error_not_a_silent_skip(
    tmp_path: Path,
) -> None:
    path = write_export(tmp_path / "reviews.json", export({"reviews": []}))

    with pytest.raises(JpdbReviewsError) as excinfo:
        read_reviews(path)

    assert "cannot be matched" in str(excinfo.value)


def test_a_card_with_no_reviews_key_counts_zero(tmp_path: Path) -> None:
    path = write_export(
        tmp_path / "reviews.json", export({"vid": 1, "spelling": "話す", "reading": "はなす"})
    )

    entries, _ = read_reviews(path)

    assert entries[0].reviews == 0


# --- matching ---------------------------------------------------------------


def test_a_record_is_matched_by_vid_before_anything_else() -> None:
    # An import-jpdb record carries its vid, and the vid is the identity jpdb
    # itself uses — a reading that drifted must not lose the match.
    stored = record("話す", "はなした", vid="1562350")

    result = apply_reviews([stored], read_entries(review("話す", "はなす", 1562350, 4)))

    assert result.matched == {"word:話す:はなした": 4}
    assert result.unmatched == []


def test_a_record_with_no_vid_is_matched_on_expression_and_reading() -> None:
    result = apply_reviews([HANASU], read_entries(review("話す", "はなす", 1562350, 4)))

    assert result.matched == {"word:話す:はなす": 4}


def test_a_matched_record_is_tagged_and_carries_its_review_count() -> None:
    result = apply_reviews([HANASU], read_entries(review("話す", "はなす", 1562350, 7)))

    updated = result.records[0]
    assert KNOWN_TAG in updated.tags
    assert updated.source.raw_fields[REVIEW_COUNT_FIELD] == "7"
    assert result.changed == ["word:話す:はなす"]


def test_matching_leaves_every_other_field_and_record_untouched() -> None:
    result = apply_reviews([HANASU, TABERU], read_entries(review("話す", "はなす", 1, 1)))

    assert result.records[1] == TABERU
    assert result.records[0].expression == "話す"
    assert result.records[0].reading == "はなす"
    assert result.records[0].id == HANASU.id
    # The record's own tags are kept, not replaced.
    assert result.records[0].source.imported_from == "e.csv"


def test_an_existing_tag_is_kept_alongside_the_new_one() -> None:
    tagged = record("話す", "はなす", tags=["genki-1", "verb"])

    result = apply_reviews([tagged], read_entries(review("話す", "はなす", 1, 1)))

    assert result.records[0].tags == ["genki-1", KNOWN_TAG, "verb"]


def test_re_running_with_unchanged_counts_rewrites_nothing() -> None:
    entries = read_entries(review("話す", "はなす", 1562350, 4))
    first = apply_reviews([HANASU], entries)

    second = apply_reviews(first.records, entries)

    assert second.changed == []
    assert second.matched == {"word:話す:はなす": 4}
    assert second.records[0] == first.records[0]


def test_a_count_that_moved_is_rewritten() -> None:
    first = apply_reviews([HANASU], read_entries(review("話す", "はなす", 1562350, 4)))

    second = apply_reviews(first.records, read_entries(review("話す", "はなす", 1562350, 9)))

    assert second.changed == ["word:話す:はなす"]
    assert second.records[0].source.raw_fields[REVIEW_COUNT_FIELD] == "9"


def test_an_entry_matching_nothing_is_reported_and_creates_no_record() -> None:
    result = apply_reviews([HANASU], read_entries(review("難しい", "むずかしい", 999, 2)))

    assert [entry.label for entry in result.unmatched] == ["難しい [むずかしい]"]
    assert len(result.records) == 1
    assert result.changed == []


def read_entries(*cards: dict[str, Any]) -> list[Any]:
    """The entries a one-section export would produce, without touching disk."""
    from japanese_anki.importers.jpdb_reviews import ReviewEntry

    return [
        ReviewEntry(
            vid="" if card.get("vid") is None else str(card["vid"]),
            spelling=str(card.get("spelling", "")),
            reading=str(card.get("reading", "")),
            reviews=len(card.get("reviews", [])),
        )
        for card in cards
    ]


# --- the CLI ----------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def test_the_command_tags_records_and_reports_what_matched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [HANASU, TABERU])
    path = write_export(
        root / "reviews.json",
        export(review("話す", "はなす", 1562350, 12), review("難しい", "むずかしい", 999, 3)),
    )

    assert cli.main(["--root", str(root), "import-jpdb-reviews", str(path)]) == 0

    out = capsys.readouterr().out
    assert "Matched 1 of 2 jpdb entries" in out
    assert "1 entry matched no record: 難しい [むずかしい]" in out
    marked = stored(root)["word:話す:はなす"]
    assert marked["tags"] == [KNOWN_TAG]
    assert marked["source"]["raw_fields"][REVIEW_COUNT_FIELD] == "12"
    assert stored(root)["word:食べる:たべる"]["tags"] == []


def test_the_ledger_gets_one_detail_free_sighting_per_matched_record(
    tmp_path: Path,
) -> None:
    # The count is deliberately absent here: record_source_seen identifies a
    # reference by every key but seen_at, so a count would append a new line
    # every week.
    root = project(tmp_path, [HANASU])
    path = write_export(root / "reviews.json", export(review("話す", "はなす", 1562350, 12)))

    cli.main(["--root", str(root), "import-jpdb-reviews", str(path)])

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    sources = book["records"]["word:話す:はなす"]["sources"]
    assert [
        {key: value for key, value in source.items() if key != "seen_at"}
        for source in sources
    ] == [{"type": "jpdb-reviews", "ref": "reviews.json"}]


def test_re_running_adds_no_second_ledger_reference(tmp_path: Path) -> None:
    root = project(tmp_path, [HANASU])
    path = write_export(root / "reviews.json", export(review("話す", "はなす", 1562350, 12)))
    cli.main(["--root", str(root), "import-jpdb-reviews", str(path)])

    # A week later, with more reviews on the same word.
    write_export(root / "reviews.json", export(review("話す", "はなす", 1562350, 40)))
    cli.main(["--root", str(root), "import-jpdb-reviews", str(path)])

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert len(book["records"]["word:話す:はなす"]["sources"]) == 1
    assert stored(root)["word:話す:はなす"]["source"]["raw_fields"][REVIEW_COUNT_FIELD] == "40"


def test_a_non_vocabulary_section_warns_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [HANASU])
    path = write_export(
        root / "reviews.json",
        export(review("話す", "はなす", 1, 1), cards_kanji_keyword_char=[{"kanji": "話"}]),
    )

    cli.main(["--root", str(root), "import-jpdb-reviews", str(path)])

    assert "cards_kanji_keyword_char" in capsys.readouterr().err


def test_an_empty_collection_is_said_plainly_rather_than_matched_against(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = write_export(root / "reviews.json", export(review("話す", "はなす", 1, 1)))

    assert cli.main(["--root", str(root), "import-jpdb-reviews", str(path)]) == 0

    assert "No records to match against" in capsys.readouterr().out


def test_the_tag_is_the_one_the_deck_filter_recipe_names() -> None:
    # README documents `exclude_tags: [jpdb-known]`; changing this string
    # silently stops every existing deck from excluding anything.
    assert KNOWN_TAG == "jpdb-known"


def test_the_documented_exclude_tags_recipe_actually_excludes(tmp_path: Path) -> None:
    # The README tells people to write `exclude_tags: [jpdb-known]` and expect
    # those words to stop appearing. Nothing else pins the two halves together:
    # this command writes the tag, and the deck filter reads it.
    from japanese_anki.exporters.anki import resolve_deck_records

    root = project(tmp_path, [HANASU, TABERU])
    path = write_export(root / "reviews.json", export(review("話す", "はなす", 1562350, 12)))
    cli.main(["--root", str(root), "import-jpdb-reviews", str(path)])

    deck = root / "deck.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Test\n"
        "  source: vocabulary.json\n"
        f"  exclude_tags: [{KNOWN_TAG}]\n",
        encoding="utf-8",
    )
    _config, records = resolve_deck_records(deck)

    assert [item.expression for item in records] == ["食べる"]


# --- foreign files and drifted shapes ---------------------------------------


def test_an_export_with_a_utf8_bom_is_read_not_refused(tmp_path: Path) -> None:
    # A file round-tripped through a Windows editor carries a BOM, and
    # json.loads rejects one as a syntax error on line 1.
    path = tmp_path / "reviews.json"
    path.write_text(
        json.dumps(export(review("話す", "はなす", 1562350, 3)), ensure_ascii=False),
        encoding="utf-8-sig",
    )

    entries, _ = read_reviews(path)

    assert [(entry.spelling, entry.reviews) for entry in entries] == [("話す", 3)]


def test_a_reviews_value_that_is_not_a_list_is_an_error_not_a_zero(
    tmp_path: Path,
) -> None:
    # Counting a drifted shape as zero would write jpdb_reviews: "0" into the
    # records as though it were the truth.
    path = write_export(
        tmp_path / "reviews.json",
        export({"vid": 1, "spelling": "話す", "reading": "はなす", "reviews": 12}),
    )

    with pytest.raises(JpdbReviewsError) as excinfo:
        read_reviews(path)

    assert "not a list of reviews" in str(excinfo.value)


def test_two_entries_landing_on_one_record_add_up(tmp_path: Path) -> None:
    # The same word can reach apply_reviews as two entries — one carrying a
    # vid, one not — and their counts have to add, the way two card types for
    # one word do.
    entries = read_entries(
        review("話す", "はなす", 1562350, 3), review("話す", "はなす", None, 4)
    )
    stored = record("話す", "はなす", vid="1562350")

    result = apply_reviews([stored], entries)

    assert result.matched == {"word:話す:はなす": 7}
    assert result.records[0].source.raw_fields[REVIEW_COUNT_FIELD] == "7"
    # And the record is reported changed once, not once per entry.
    assert result.changed == ["word:話す:はなす"]
