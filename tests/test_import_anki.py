"""Reading a deck out of Anki and turning it into records.

A shared deck puts everything a card needs into free HTML: the word on the
front, and on the back a reading, a spoken variant, and a gloss, separated by
whatever markup its author typed. This is the parser for that, and its rules
are all one rule — read what is written, hold back what is ambiguous, and never
invent a reading, because the reading is half of the record id and the whole of
the Anki GUID derived from it.

The fixtures are the shapes the Yotsubato reading pack actually uses, taken off
the deck rather than imagined.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from japanese_anki.collection import CollectionError, DeckNote, read_deck_notes
from japanese_anki.importers.anki_deck import records_from_notes, staging_name


def note(front: str, back: str, note_id: int = 1, **fields: str) -> DeckNote:
    return DeckNote(
        id=note_id,
        guid=f"guid{note_id}",
        notetype="Basic (optional reversed card)+",
        fields={"Front": front, "Back": back, **fields},
        tags="",
    )


def only(front: str, back: str):
    """The single record a one-note deck produces."""
    result = records_from_notes([note(front, back)], "Pack")
    assert not result.held, result.held
    assert len(result.records) == 1
    return result.records[0]


# --- the shape the deck is mostly written in ----------------------------------


def test_a_word_a_reading_and_a_gloss() -> None:
    record = only(
        "<div>引っ越す</div>",
        '<span style="font-weight: normal;"><div>ひっこす</div>'
        "<div>To move (to a new residence)</div></span>",
    )

    assert record.id == "word:引っ越す:ひっこす"
    assert (record.expression, record.reading) == ("引っ越す", "ひっこす")
    assert record.meanings == ["To move (to a new residence)"]


def test_a_kana_word_is_its_own_reading() -> None:
    """`もう / もう` — the deck repeats it rather than leaving the back short,
    but a kana word needs no reading line at all."""
    record = only("<div>もう</div>", "<div>Soon, shortly</div>")

    assert (record.expression, record.reading) == ("もう", "もう")


def test_every_gloss_line_is_kept() -> None:
    record = only(
        "<div>寝る</div>",
        "<div>ねる</div><div>To sleep</div><div>Also: to lie down</div>",
    )

    assert record.meanings == ["To sleep", "Also: to lie down"]


def test_a_br_ends_a_line_like_a_div_does() -> None:
    record = only("話す", "はなす<br>To speak")

    assert (record.reading, record.meanings) == ("はなす", ["To speak"])


def test_entities_and_nbsp_are_read_as_text() -> None:
    record = only("&#x8A71;&#x3059;", "はなす&nbsp;<br>To speak &amp; talk")

    assert record.expression == "話す"
    assert record.meanings == ["To speak & talk"]


# --- the colloquial forms this deck exists for --------------------------------


def test_a_variant_beside_the_reading_becomes_a_usage_note() -> None:
    """`すごい　「すげえ」` is the reading and then the form the manga prints. A
    reading pack for a manga is *about* that second form, so it is kept — and
    kept as a usage note, which `enrich --ai` fills only when it is empty, so
    what the source said survives the model that would otherwise write there.

    Worded to stand alone: the card does not name the book it came from, so an
    earlier "…in this text" referred to nothing a learner could see — which is
    what `janki review` caught on three cards of a twenty-card pilot."""
    record = only("<div>凄い</div>", "<div>すごい　「すげえ」</div><div>Amazing, wow</div>")

    assert (record.expression, record.reading) == ("凄い", "すごい")
    assert record.usage_notes == "Colloquial form: 「すげえ」."
    assert record.meanings == ["Amazing, wow"]


def test_a_variant_on_its_own_line_is_not_a_second_reading() -> None:
    """The same deck writes it both ways. Treating a bracketed kana line as a
    competing reading held back every word that had one."""
    record = only(
        "<div>お姉ちゃん</div>",
        "<div>おねえちゃん</div><div>「おねーちゃん」</div><div>Older sister</div>",
    )

    assert record.reading == "おねえちゃん"
    assert record.usage_notes == "Colloquial form: 「おねーちゃん」."
    assert record.meanings == ["Older sister"]


def test_brackets_with_nothing_outside_them_are_the_word_itself() -> None:
    """`「げつもく」` alone is how the deck lists a phrase it only ever prints in
    its spoken form. There is no other candidate, so the brackets hold the word
    rather than a variant of one."""
    record = only("<div>「げつもく」</div>", "<div>「げつもく」</div><div>月曜日 and 木曜日</div>")

    assert (record.expression, record.reading) == ("げつもく", "げつもく")
    assert record.usage_notes == ""


def test_punctuation_is_not_part_of_a_pronunciation() -> None:
    """`あれ？` is read あれ. The expression keeps what the manga prints; the
    reading is half the record id, and a question mark there would ride into
    every GUID derived from it."""
    record = only("<div>あれ？</div>", "<div>Huh?</div>")

    assert record.expression == "あれ？"
    assert record.reading == "あれ"


def test_punctuation_on_the_reading_line_is_stripped_too() -> None:
    """The other route to a reading: taken off the back rather than from a kana
    expression. `まって！` is read まって, and the exclamation would otherwise
    ride into the record id and every GUID derived from it."""
    record = only("<div>待って！</div>", "<div>まって！</div><div>“Wait!”</div>")

    assert record.id == "word:待って!:まって"
    assert record.reading == "まって"


# --- what it refuses to guess -------------------------------------------------


@pytest.mark.parametrize(
    ("front", "back", "because"),
    [
        ("<div>難しい</div>", "<div>Difficult</div>", "no kana reading"),
        ("<div>そっとしておこう</div>",
         "<div>そのままにしておこう</div><div>ほうっておこう</div><div>Leave it be</div>",
         "unbracketed kana lines"),
        ("", "<div>ひと</div><div>Person</div>", "the front is empty"),
        ("<div>人</div>", "<div>ひと</div>", "no meaning"),
    ],
    ids=["kanji-with-no-reading", "two-japanese-lines", "no-front", "no-meaning"],
)
def test_an_unreadable_note_is_held_with_its_reason(
    front: str, back: str, because: str
) -> None:
    result = records_from_notes([note(front, back)], "Pack")

    assert not result.records
    assert len(result.held) == 1
    assert because in result.held[0].reason
    # The note's own text goes with it, so a reviewer can act without opening Anki.
    assert result.held[0].back == back


def test_the_same_word_twice_is_held_rather_than_merged() -> None:
    """Two notes minting one id would silently become one card, and which of
    the two glosses survived would depend on order."""
    result = records_from_notes(
        [note("<div>人</div>", "<div>ひと</div><div>Person</div>", note_id=1),
         note("<div>人</div>", "<div>ひと</div><div>People</div>", note_id=2)],
        "Pack",
    )

    assert len(result.records) == 1
    assert len(result.held) == 1
    assert "the same word as note 1" in result.held[0].reason


def test_the_held_note_lists_every_one_of_them() -> None:
    """It becomes `review_notes` in the staging file — the thing a reviewer
    reads later. A count that lives only in scrollback is a silent discard with
    an extra step."""
    result = records_from_notes(
        [note("<div>難しい</div>", "<div>Difficult</div>", note_id=7)], "Pack"
    )

    assert "1 note(s) could not be read" in result.held_note
    assert "[7] 難しい" in result.held_note


# --- provenance ---------------------------------------------------------------


def test_the_original_note_is_kept_verbatim() -> None:
    """Every field as it was, so nothing this parser did not understand is lost
    and the original card can be found again in Anki."""
    record = only("<div>猫</div>", "<div>ねこ</div><div>Cat</div>")
    raw = record.source.raw_fields

    assert record.source.type == "anki"
    assert record.source.imported_from == "Pack"
    assert raw["anki_note_id"] == "1"
    assert raw["anki_guid"] == "guid1"
    assert raw["anki_Front"] == "<div>猫</div>"
    assert "ねこ" in raw["anki_Back"]
    assert record.tags == ["anki"]


def test_the_fields_to_read_can_be_named() -> None:
    """A notetype whose fields are not called Front and Back."""
    deck_note = DeckNote(
        id=1, guid="g", notetype="Custom",
        fields={
            "Expression": "<div>猫</div>",
            "Notes": "",
            "Meaning": "<div>ねこ</div><div>Cat</div>",
        },
        tags="",
    )

    result = records_from_notes([deck_note], "Pack", front_field="Expression", back_field="Meaning")

    assert result.records[0].id == "word:猫:ねこ"


def test_the_staging_file_is_named_for_the_deck() -> None:
    assert staging_name("Yotsubato Volume 1 Reading Pack Vocab") == (
        "anki-yotsubato-volume-1-reading-pack-vocab.yaml"
    )
    assert staging_name("語彙::基礎") == "anki-語彙-基礎.yaml"


# --- reading the collection ---------------------------------------------------


def _collection(path: Path, deck: str, notes: list[tuple[str, str]]) -> Path:
    """A collection holding one deck and its notes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table notetypes (id integer primary key, name text not null,
                                mtime_secs integer not null default 0,
                                usn integer not null default 0, config blob);
        create table fields (ntid integer not null, ord integer not null,
                             name text not null, config blob);
        create table notes (id integer primary key, guid text not null,
                            mid integer not null, mod integer not null default 0,
                            usn integer not null default 0, tags text default '',
                            flds text default '', sfld text default '',
                            csum integer default 0, flags integer default 0,
                            data text default '');
        create table cards (id integer primary key, nid integer not null,
                            did integer not null, ord integer not null default 0,
                            mod integer not null default 0, usn integer not null default 0,
                            type integer default 0, queue integer default 0,
                            due integer default 0, ivl integer default 0,
                            factor integer default 0, reps integer default 0,
                            lapses integer default 0, left integer default 0,
                            odue integer default 0, odid integer default 0,
                            flags integer default 0, data text default '');
        create table decks (id integer primary key, name text not null,
                            mtime_secs integer not null default 0,
                            usn integer not null default 0, common blob, kind blob);
        """
    )
    con.execute("insert into notetypes (id, name, config) values (1, 'Basic', ?)", (b"\x08\x01",))
    for ord_, name in enumerate(("Front", "Back")):
        con.execute("insert into fields (ntid, ord, name) values (1, ?, ?)", (ord_, name))
    con.execute("insert into decks (id, name) values (9, ?)", (deck,))
    for index, (front, back) in enumerate(notes, start=1):
        con.execute(
            "insert into notes (id, guid, mid, flds, tags) values (?, ?, 1, ?, '')",
            (index, f"g{index}", f"{front}\x1f{back}"),
        )
        con.execute("insert into cards (id, nid, did) values (?, ?, 9)", (index, index))
    con.commit()
    con.close()
    return path


def test_a_deck_is_read_through_a_copy(tmp_path: Path) -> None:
    """Same path `read_notetypes` takes: Anki holds an exclusive lock while it
    runs, and this is a thing somebody would run with Anki open."""
    path = _collection(tmp_path / "collection.anki2", "Pack", [("猫", "ねこ<br>Cat")])

    notes = read_deck_notes(path, "Pack")

    assert [n.fields["Front"] for n in notes] == ["猫"]
    assert notes[0].fields["Back"] == "ねこ<br>Cat"


def test_two_cards_of_one_note_return_one_note(tmp_path: Path) -> None:
    """A reversed card doubles the cards and not the notes. The importer wants
    the note; returning it twice would file every word as its own duplicate."""
    path = _collection(tmp_path / "collection.anki2", "Pack", [("猫", "ねこ<br>Cat")])
    con = sqlite3.connect(path)
    con.execute("insert into cards (id, nid, did, ord) values (99, 1, 9, 1)")
    con.commit()
    con.close()

    assert len(read_deck_notes(path, "Pack")) == 1


def test_a_nested_deck_is_found_by_the_name_a_person_types(tmp_path: Path) -> None:
    """Anki stores `Parent::Child` with a \\x1f between the components."""
    path = _collection(tmp_path / "collection.anki2", "Manga\x1fYotsuba", [("猫", "ねこ<br>Cat")])

    assert len(read_deck_notes(path, "Manga::Yotsuba")) == 1


def test_a_deck_that_is_not_there_lists_the_ones_that_are(tmp_path: Path) -> None:
    path = _collection(tmp_path / "collection.anki2", "Pack", [("猫", "ねこ<br>Cat")])

    with pytest.raises(CollectionError, match="No deck named 'Nope'") as raised:
        read_deck_notes(path, "Nope")

    assert "Pack" in str(raised.value), "the message says what it does have"
