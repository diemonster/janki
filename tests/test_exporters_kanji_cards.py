"""The character-note exporter: counts, identity, and what a back may show.

Five explicit targets are five notes and — by default — five cards. Nothing in
here reaches the network or the vocabulary store; a kanji deck is built from
the curated character notes and the templates, and from nothing else.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest
import yaml

from japanese_anki import kanji, kanji_notes
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import kanji_cards
from japanese_anki.exporters.anki import deck_notetype
from japanese_anki.identifiers import character_record_id
from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)
from japanese_anki.kanji_notes import CharacterNote, KanjidicReading

TEMPLATES = Path(__file__).resolve().parents[1] / "templates" / "japanese-study"

#: The owner's explicit targets for this milestone.
TARGETS = ("物", "特", "鳥", "料", "理")


def usage(
    label: str,
    percent_text: str | None,
    percent: int | None,
    examples: tuple[BoundExample, ...] = (),
) -> ReadingUsage:
    """One reading as the provider printed it. ``None`` is "no figure at all",
    which is a different fact from a figure of zero."""
    return ReadingUsage(
        label=label,
        href=f"/kanji/x#{label}",
        percent_text=percent_text,
        percent=percent,
        percent_less_than=(
            None if percent_text is None else percent_text.strip("()").startswith("<")
        ),
        examples=examples,
    )


def evidence(*groups: ReadingGroup) -> kanji_notes.ReadingEvidence:
    return kanji_notes.evidence_from_readings(
        CharacterReadings(
            character="理",
            source_url="https://jpdb.io/kanji/%E7%90%86",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256="b" * 64,
            groups=groups,
        )
    )


def note(character: str, **overrides: object) -> CharacterNote:
    values: dict[str, object] = {
        "character": character,
        "id": character_record_id(character),
        "meanings": ("logic", "reason"),
        "stroke_count": 11,
        "kanjidic_readings": (
            KanjidicReading(kind="on", reading="リ"),
            KanjidicReading(kind="kun", reading="ことわり"),
        ),
        "sources": ("kanjiapi.dev (KANJIDIC2)",),
    }
    values.update(overrides)
    return CharacterNote(**values)  # type: ignore[arg-type]


def project(
    tmp_path: Path,
    notes: dict[str, CharacterNote],
    *,
    cards: dict[str, bool] | None = None,
    include_ids: list[str] | None = None,
) -> tuple[ProjectConfig, Path]:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        f'template_dir = "{TEMPLATES}"\n'
        'kanji_notes_file = "kanji_notes.json"\n',
        encoding="utf-8",
    )
    kanji_notes.save_notes(tmp_path / "kanji_notes.json", notes)
    deck_dir = tmp_path / "data" / "decks"
    deck_dir.mkdir(parents=True)
    section: dict[str, object] = {
        "kind": "kanji",
        "name": "Genki II Kanji",
        "deck_id": 1234567890,
        "model_id": 1607392319,
        "output": "kanji.apkg",
        "source": "../../kanji_notes.json",
        "include_ids": (
            include_ids
            if include_ids is not None
            else [character_record_id(character) for character in sorted(notes)]
        ),
    }
    if cards is not None:
        section["cards"] = cards
    deck_path = deck_dir / "kanji.yaml"
    deck_path.write_text(
        yaml.safe_dump({"deck": section}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path), deck_path


def notes_and_cards(package: Path) -> tuple[int, int]:
    """Read the built package's own note and card counts."""
    import sqlite3
    import tempfile

    with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as into:
        name = "collection.anki21" if "collection.anki21" in archive.namelist() else (
            "collection.anki2"
        )
        archive.extract(name, into)
        connection = sqlite3.connect(Path(into) / name)
        try:
            notes = connection.execute("select count(*) from notes").fetchone()[0]
            cards = connection.execute("select count(*) from cards").fetchone()[0]
        finally:
            connection.close()
    return notes, cards


def test_five_explicit_targets_build_five_notes_and_five_cards(tmp_path: Path) -> None:
    """Recognition is the default direction and the only one a bare note has,
    so the default deck is one card per character — not three."""
    config, deck_path = project(
        tmp_path, {character: note(character) for character in TARGETS}
    )

    result = kanji_cards.build_kanji_deck(deck_path, config)

    assert result.note_count == 5
    assert result.card_types == ("recognition",)
    assert result.record_ids == tuple(
        character_record_id(character) for character in sorted(TARGETS)
    )
    assert result.media_count == 0
    assert notes_and_cards(result.output_path) == (5, 5)


def test_a_rebuild_updates_rather_than_duplicates(tmp_path: Path) -> None:
    """The GUID is derived from the note's own identity, so the second build
    of the same character is the same note to Anki."""
    config, deck_path = project(tmp_path, {"理": note("理")})

    first = kanji_cards.build_kanji_deck(deck_path, config)
    guid_before = _guids(first.output_path)
    second = kanji_cards.build_kanji_deck(deck_path, config)

    assert guid_before == _guids(second.output_path)
    assert notes_and_cards(second.output_path) == (1, 1)


def test_the_guid_is_the_notes_identity_and_not_its_content(tmp_path: Path) -> None:
    """Refreshing meanings must not strand a card's review history."""
    import genanki

    config, deck_path = project(tmp_path, {"理": note("理")})
    kanji_cards.build_kanji_deck(deck_path, config)
    original = _guids(deck_output(config))

    kanji_notes.save_notes(
        tmp_path / "kanji_notes.json", {"理": note("理", meanings=("reason",))}
    )
    kanji_cards.build_kanji_deck(deck_path, config)

    assert original == _guids(deck_output(config))
    assert original == {genanki.guid_for("kanji:理")}


def test_the_answer_shows_meanings_readings_and_bound_examples(tmp_path: Path) -> None:
    """Show Answer reveals what the provider reports, examples included, and
    the figure stays visible for a reading that came with no example."""
    example = BoundExample(
        written="料理",
        pronounced="りょうり",
        gloss="cooking",
        furigana="料理[りょうり]",
        source_url="https://jpdb.io/kanji/%E7%90%86",
    )
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(
                usage("り", "(84%)", 84, (example,)),
                usage("ことわり", "(<1%)", 1),
            ),
        )
    )
    values = kanji_cards.note_values(note("理", reading_evidence=facts), 0)
    common = dict(zip(kanji_cards.KANJI_FIELDS, values, strict=True))["CommonReadings"]

    assert "り" in common and "(84%)" in common
    assert "料理" in common and "りょうり" in common and "cooking" in common
    assert "(&lt;1%)" in common or "(<1%)" in common
    assert "jpdb reported usage" in common.lower()
    assert "<details" not in common


def test_extra_readings_and_the_dictionary_inventory_are_disclosed(
    tmp_path: Path,
) -> None:
    """Only the readings the provider quantified are on the face of the
    answer. The ones it printed no figure for and KANJIDIC's on/kun inventory
    are separate blocks behind a disclosure, and never presented as each
    other — a reported reading group is not an on/kun reading."""
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(usage("り", "(84%)", 84),),
        ),
        ReadingGroup(
            source_class="kanji-reading-list",
            readings=(usage("ことわり", None, None),),
        ),
    )
    fields = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("理", reading_evidence=facts), 0),
            strict=True,
        )
    )

    assert "ことわり" not in fields["CommonReadings"]
    assert fields["OtherReadings"].startswith("<details")
    assert "ことわり" in fields["OtherReadings"]
    assert fields["ReadingInventory"].startswith("<details")
    assert "リ" in fields["ReadingInventory"]
    assert "84%" not in fields["ReadingInventory"]


def test_a_note_with_no_evidence_renders_no_reading_block(tmp_path: Path) -> None:
    fields = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("理"), 0),
            strict=True,
        )
    )

    assert fields["CommonReadings"] == ""
    assert fields["OtherReadings"] == ""


def test_content_is_escaped_rather_than_written_into_the_card(
    tmp_path: Path,
) -> None:
    """A gloss is text from outside; it reaches a card as text."""
    example = BoundExample(
        written="<b>x</b>", pronounced="x", gloss="a & b", furigana="", source_url=""
    )
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(usage("<i>", "(1%)", 1, (example,)),),
        )
    )
    fields = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(
                note("理", reading_evidence=facts, production_cue="a < b"), 0
            ),
            strict=True,
        )
    )

    assert "<b>x</b>" not in fields["CommonReadings"]
    assert "&lt;b&gt;x&lt;/b&gt;" in fields["CommonReadings"]
    assert "a &amp; b" in fields["CommonReadings"]
    assert fields["ProductionCue"] == "a &lt; b"


def test_an_enabled_direction_the_note_cannot_carry_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A reading card needs a fixed provider-bound example; a production card
    needs a cue the owner wrote. Neither is invented at build time."""
    config, deck_path = project(
        tmp_path, {"理": note("理")}, cards={"recognition": True, "reading": True}
    )

    with pytest.raises(kanji_cards.KanjiDeckError, match="理.*reading"):
        kanji_cards.build_kanji_deck(deck_path, config)


def test_the_reading_card_asks_the_one_fixed_example(tmp_path: Path) -> None:
    example = BoundExample(
        written="料理",
        pronounced="りょうり",
        gloss="cooking",
        furigana="料理[りょうり]",
        source_url="https://jpdb.io/kanji/%E7%90%86",
    )
    config, deck_path = project(
        tmp_path,
        {"理": note("理", reading_example=example)},
        cards={"recognition": True, "reading": True},
    )

    result = kanji_cards.build_kanji_deck(deck_path, config)
    fields = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("理", reading_example=example), 0),
            strict=True,
        )
    )

    assert result.card_types == ("recognition", "reading")
    assert notes_and_cards(result.output_path) == (1, 2)
    assert fields["ReadingPrompt"] == "料理"
    assert fields["ReadingFurigana"] == "料理[りょうり]"
    assert fields["ReadingPronunciation"] == "りょうり"


def test_a_deck_id_the_store_does_not_hold_refuses(tmp_path: Path) -> None:
    config, deck_path = project(
        tmp_path, {"理": note("理")}, include_ids=["kanji:理", "kanji:曜"]
    )

    with pytest.raises(kanji_cards.KanjiDeckError, match="kanji:曜"):
        kanji_cards.build_kanji_deck(deck_path, config)


def test_the_notetype_the_status_check_reports_is_the_one_a_build_writes(
    tmp_path: Path,
) -> None:
    """`janki status` compares a collection's notetype against this answer, so
    a disagreement here is a false drift warning on every run."""
    config, deck_path = project(tmp_path, {"理": note("理")})

    assert deck_notetype(deck_path, config) == (
        1607392319,
        "Japanese Kanji",
        len(kanji_cards.KANJI_FIELDS),
    )


def deck_output(config: ProjectConfig) -> Path:
    return config.dist_dir / "kanji.apkg"


def _guids(package: Path) -> set[str]:
    import sqlite3
    import tempfile

    with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as into:
        name = "collection.anki21" if "collection.anki21" in archive.namelist() else (
            "collection.anki2"
        )
        archive.extract(name, into)
        connection = sqlite3.connect(Path(into) / name)
        try:
            return {row[0] for row in connection.execute("select guid from notes")}
        finally:
            connection.close()


def test_the_stroke_strip_is_the_shared_renderer_scoped_per_note() -> None:
    """The same graph-paper strip a word card draws, from one renderer. Its
    element ids are document-wide, so each note's strip carries its own prefix
    — two notes on one screen would otherwise define the same id twice."""
    strokes = ("M1,1L9,9", "M2,2L8,8")
    first = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("理", strokes=strokes), 0),
            strict=True,
        )
    )["StrokeOrder"]
    second = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("物", strokes=strokes), 1),
            strict=True,
        )
    )["StrokeOrder"]

    assert first.count('class="stroke-cell"') == 2, "one cell per stroke"
    assert 'id="kanji-note-0-stroke-0"' in first
    assert 'id="kanji-note-1-stroke-0"' in second


def test_a_character_with_no_stroke_data_simply_has_no_strip() -> None:
    """KanjiVG does not cover everything, and the rest of the answer stands."""
    fields = dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(note("理"), 0),
            strict=True,
        )
    )

    assert fields["StrokeOrder"] == ""
    assert fields["Meanings"] == "logic, reason"


# --- the readings block on the answer ----------------------------------------


def card_fields(character_note: CharacterNote, index: int = 0) -> dict[str, str]:
    """One note's field values, by field name."""
    return dict(
        zip(
            kanji_cards.KANJI_FIELDS,
            kanji_cards.note_values(character_note, index),
            strict=True,
        )
    )


def ruby(*runs: tuple[str, str]) -> str:
    """The markup a rendered annotated run is, base and reading in that order."""
    return "".join(
        f"<ruby><rb>{base}</rb><rt>{reading}</rt></ruby>" for base, reading in runs
    )


def bound_words(common_readings: str) -> list[str]:
    """Each bound word as the field draws it, in the order the field lists them."""
    return re.findall(r'<span class="kanji-word">(.*?)</span>', common_readings)


def test_the_common_readings_field_is_the_shared_renderers_own_markup() -> None:
    """One renderer, two card kinds. The block a word card discloses for 理 and
    the face of 理's own answer state the same source's readings, so they come
    from the same code: a second copy of that markup here is how the two start
    disagreeing about what the provider said."""
    example = BoundExample(
        written="料理",
        pronounced="りょうり",
        gloss="cooking",
        furigana="料[りょう] 理[り]",
        source_url="https://jpdb.io/vocabulary/1550140/a#b",
    )
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(usage("り", "(84%)", 84, (example,)),),
        )
    )

    common = card_fields(note("理", reading_evidence=facts))["CommonReadings"]
    word_card = kanji.render_kanji_html(
        [kanji.KanjiInfo(character="理", meanings=("logic",))],
        reading_evidence={"理": facts.readings},
    )

    assert common, "a character with reported readings has a readings block"
    assert common in word_card


def test_a_bound_word_is_drawn_with_the_ruby_its_page_supplied() -> None:
    """Anki does not apply its ``furigana:`` filter to notation stored inside
    another field's HTML, so a character card that shipped the bracket form
    would show it to the learner as text. What the provider left unannotated
    stays unannotated and a repeated annotation is drawn twice: the segmentation
    is the source's statement, and janki neither merges nor tidies it."""
    supplied = BoundExample(
        written="無理やり",
        pronounced="むりやり",
        gloss="forcibly",
        furigana="無[む] 理[り]やり",
        source_url="https://jpdb.io/vocabulary/1531030/a#b",
    )
    repeated = BoundExample(
        written="理論物理学",
        pronounced="りろんぶつりがく",
        gloss="theoretical physics",
        furigana="理[り] 論[ろん] 物[ぶつ] 理[り] 学[がく]",
        source_url="https://jpdb.io/vocabulary/1550210/c#d",
    )
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(
                usage("り", "(84%)", 84, (supplied, repeated)),
                usage("ことわり", "(<1%)", 1),
            ),
        )
    )

    common = card_fields(note("理", reading_evidence=facts))["CommonReadings"]

    assert bound_words(common) == [
        ruby(("無", "む"), ("理", "り")) + "やり",
        ruby(("理", "り"), ("論", "ろん"), ("物", "ぶつ"), ("理", "り"), ("学", "がく")),
    ]
    assert "[" not in common and "]" not in common, "no notation reaches the card"
    assert '<span class="kanji-kana">むりやり</span>' in common, "still stated whole"
    assert '<span class="kanji-gloss">forcibly</span>' in common
    assert '<span class="kanji-usage-reading">り</span>' in common
    assert '<span class="kanji-usage-percent">(84%)</span>' in common
    assert '<span class="kanji-usage-percent">(&lt;1%)</span>' in common, (
        "a reading the provider quantified keeps its figure with no example"
    )


def test_a_bound_word_with_no_supplied_ruby_falls_back_to_its_spelling() -> None:
    """Reading the whole-word kana over the whole spelling would be janki
    deciding which kana sit over which characters — the one thing the reading
    page exists to state. The spelling is text from outside and reaches the card
    as text."""
    example = BoundExample(
        written="<b>理由</b>",
        pronounced="りゆう",
        gloss="reason",
        furigana="",
        source_url="https://jpdb.io/vocabulary/1550140/a#b",
    )
    facts = evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(usage("り", "(84%)", 84, (example,)),),
        )
    )

    common = card_fields(note("理", reading_evidence=facts))["CommonReadings"]

    assert bound_words(common) == ["&lt;b&gt;理由&lt;/b&gt;"]
    assert "<ruby" not in common, "no ruby guessed from the whole-word reading"
    assert '<span class="kanji-kana">りゆう</span>' in common
