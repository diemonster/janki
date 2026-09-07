"""What Anki actually draws for a *character* deck.

`tests/test_rendered_cards.py` does this for word cards and says why: reading
the template source on disk cannot tell you whether `{{furigana:...}}` put the
reading over the right characters, whether a `{{#Field}}` section fired, or
whether a field name still resolves. A character deck is a second notetype with
six of its own templates and fourteen of its own fields, and none of that was
drawn by anything.

So this builds a real character package with the exporter, imports it into a
scratch collection, and asks Anki to render every card — the same code path the
desktop reviewer uses.

**Every fact here is synthetic.** The five characters are the owner's targets,
and the words are ordinary ones, but the percentages, vocabulary ids, retrieval
times, hashes and stroke paths are invented for the fixture. Nothing is
fetched, nothing reaches the network, and no figure below should be read as
something jpdb reported. What the fixture reproduces faithfully is the *shape*
of the stores a build reads, written through the same dataclasses and the same
writer the real pipeline uses.

**It is not a device test.** It says nothing about AnkiMobile's rendering, CSS
or layout. It proves that the HTML janki ships for a character becomes the HTML
it intends: the front asks one thing, the answer reveals the rest together, and
what the provider bound to a reading is what the card shows.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("genanki")
anki_collection = pytest.importorskip(
    "anki.collection",
    reason="the `anki` library renders these; it is in the dev extra",
)
import genanki  # noqa: E402
from anki.import_export_pb2 import ImportAnkiPackageRequest  # noqa: E402

from japanese_anki import kanji_notes  # noqa: E402
from japanese_anki.config import ProjectConfig  # noqa: E402
from japanese_anki.exporters.kanji_cards import (  # noqa: E402
    KANJI_FIELDS,
    build_kanji_deck,
)
from japanese_anki.identifiers import character_record_id  # noqa: E402
from japanese_anki.jpdb_kanji import (  # noqa: E402
    COMMON_CELL_CLASS,
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)
from japanese_anki.kanji_notes import CharacterNote, KanjidicReading  # noqa: E402
from japanese_anki.models import VocabularyRecord  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The owner's explicit targets for this milestone, and the whole deck.
TARGETS = ("物", "特", "鳥", "料", "理")

#: Invented KanjiVG-shaped paths per character, three each. Three so a loop
#: that draws one cell is visibly wrong, and distinct per character so a card
#: showing another character's strokes fails rather than looking plausible.
STROKE_COUNT = 3


def strokes(character: str) -> tuple[str, ...]:
    origin = ord(character) % 100
    return tuple(
        f"M{origin},{top}L{origin + 5},{top + 20}" for top in (10, 40, 70)
    )


@dataclass(frozen=True, slots=True)
class Fixture:
    """One character's synthetic dictionary and provider facts."""

    meanings: tuple[str, ...]
    #: KANJIDIC's inventory: ``(kind, reading)`` in the store's display form.
    inventory: tuple[tuple[str, str], ...]
    #: The reading the provider printed a figure beside, and that figure.
    reported: str
    percent_text: str
    percent: int
    #: A reading the provider listed with no figure at all, or ``""``.
    unquantified: str
    #: The one word the provider's page for ``reported`` filed under it.
    written: str
    pronounced: str
    gloss: str
    #: Anki notation built from the provider's own ruby, per character.
    furigana: str
    vocabulary_id: str
    #: The owner's disambiguating production prompt. Never names the character
    #: it asks for — a cue that shows the answer is not a cue.
    cue: str


FIXTURES: dict[str, Fixture] = {
    "物": Fixture(
        meanings=("thing", "object"),
        inventory=(("on", "ブツ"), ("on", "モツ"), ("kun", "もの")),
        reported="もの",
        percent_text="(63%)",
        percent=63,
        unquantified="ブツ",
        written="食べ物",
        pronounced="たべもの",
        gloss="food",
        furigana="食[た]べ 物[もの]",
        vocabulary_id="1358280",
        cue="thing, object — the second half of たべもの (food)",
    ),
    "特": Fixture(
        meanings=("special", "particular"),
        inventory=(("on", "トク"),),
        reported="トク",
        percent_text="(98%)",
        percent=98,
        # One cell on the page: the provider printed no unquantified reading,
        # so this character's answer must carry no extra-readings disclosure.
        unquantified="",
        written="特別",
        pronounced="とくべつ",
        gloss="special",
        furigana="特[とく] 別[べつ]",
        vocabulary_id="1455470",
        cue="special, particular — as in とくべつ",
    ),
    "鳥": Fixture(
        meanings=("bird",),
        inventory=(("on", "チョウ"), ("kun", "とり")),
        reported="とり",
        percent_text="(79%)",
        percent=79,
        unquantified="チョウ",
        written="小鳥",
        pronounced="ことり",
        gloss="small bird",
        furigana="小[こ] 鳥[とり]",
        vocabulary_id="1345540",
        cue="bird — as in ことり (a small one)",
    ),
    "料": Fixture(
        meanings=("fee", "materials"),
        inventory=(("on", "リョウ"),),
        reported="リョウ",
        percent_text="(99%)",
        percent=99,
        unquantified="",
        written="料金",
        pronounced="りょうきん",
        gloss="fee, charge",
        furigana="料[りょう] 金[きん]",
        vocabulary_id="1550170",
        cue="fee, materials — as in りょうきん (a charge)",
    ),
    "理": Fixture(
        meanings=("logic", "reason"),
        inventory=(("on", "リ"), ("kun", "ことわり")),
        reported="リ",
        percent_text="(84%)",
        percent=84,
        unquantified="ことわり",
        written="料理",
        pronounced="りょうり",
        gloss="cooking",
        furigana="料[りょう] 理[り]",
        vocabulary_id="1550140",
        cue="logic, reason — as in りょうり (cooking)",
    ),
}

#: A word record, in its own store, to prove a character build never touches
#: it. 話す shares nothing with the five targets, so its expression or meaning
#: on a kanji card could only have come from the vocabulary collection.
WORD_RECORD = VocabularyRecord(
    id="word:話す:はなす",
    expression="話す",
    reading="はなす",
    furigana="話[はな]す",
    meanings=["to speak"],
)
WORD_STORE = json.dumps([WORD_RECORD.to_dict()], ensure_ascii=False) + "\n"

#: When the fixture pretends the snapshot was taken. Fixed, because a card
#: states when its figures were retrieved and a moving clock would move that.
FETCHED_AT = "2026-09-07T00:00:00Z"


def readings(character: str) -> CharacterReadings:
    """One character's provider snapshot, in the source's own group order.

    Two cells where the page had two: the quantified one the card puts on the
    face of its answer, and the unquantified one it discloses separately. They
    stay separate because the source printed them separately.
    """
    facts = FIXTURES[character]
    quoted = f"https://jpdb.io/kanji/{character}"
    groups = [
        ReadingGroup(
            source_class=COMMON_CELL_CLASS,
            readings=(
                ReadingUsage(
                    label=facts.reported,
                    href=f"/kanji-reading/{character}/{facts.reported}",
                    percent_text=facts.percent_text,
                    percent=facts.percent,
                    percent_less_than=False,
                    examples=(
                        BoundExample(
                            written=facts.written,
                            pronounced=facts.pronounced,
                            gloss=facts.gloss,
                            furigana=facts.furigana,
                            source_url=(
                                f"https://jpdb.io/vocabulary/{facts.vocabulary_id}"
                                f"/{facts.written}/{facts.pronounced}"
                            ),
                        ),
                    ),
                    detail_source_url=(
                        f"https://jpdb.io/kanji-reading/{character}/{facts.reported}"
                    ),
                    detail_fetched_at_utc=FETCHED_AT,
                    detail_sha256="b" * 64,
                ),
            ),
        )
    ]
    if facts.unquantified:
        groups.append(
            ReadingGroup(
                source_class="kanji-reading-list",
                readings=(
                    ReadingUsage(
                        label=facts.unquantified,
                        href=f"/kanji-reading/{character}/{facts.unquantified}",
                        percent_text=None,
                        percent=None,
                        percent_less_than=None,
                    ),
                ),
            )
        )
    return CharacterReadings(
        character=character,
        source_url=quoted,
        fetched_at_utc=FETCHED_AT,
        sha256="a" * 64,
        groups=tuple(groups),
    )


def note(character: str, **overrides: Any) -> CharacterNote:
    """One curated character note, built the way the preparation flow does.

    The reading prompt is `first_bound_example` — the first word the source
    printed — rather than a choice made here, because choosing a better one is
    a judgement about Japanese.
    """
    facts = FIXTURES[character]
    evidence = kanji_notes.evidence_from_readings(readings(character))
    values: dict[str, Any] = {
        "character": character,
        "id": character_record_id(character),
        "meanings": facts.meanings,
        "stroke_count": STROKE_COUNT,
        "strokes": strokes(character),
        "kanjidic_readings": tuple(
            KanjidicReading(kind=kind, reading=reading)
            for kind, reading in facts.inventory
        ),
        "reading_evidence": evidence,
        "reading_example": kanji_notes.first_bound_example(evidence),
        "production_cue": facts.cue,
        "tags": ("genki2",),
        "created_at": FETCHED_AT,
        "sources": ("kanjiapi.dev (KANJIDIC2)", "KanjiVG"),
    }
    values.update(overrides)
    return CharacterNote(**values)


def notes() -> dict[str, CharacterNote]:
    return {character: note(character) for character in TARGETS}


def _project(
    root: Path, store: Mapping[str, CharacterNote], cards: dict[str, bool] | None
) -> tuple[ProjectConfig, Path]:
    """A scratch repository holding both stores and one character deck."""
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'kanji_notes_file = "kanji_notes.json"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    # The exact bytes a plan would show before writing them: `render_notes` is
    # what proposes a store, and building from anything else would test a file
    # janki never produces.
    (root / "kanji_notes.json").write_text(
        kanji_notes.render_notes(store), encoding="utf-8"
    )
    (root / "vocabulary.json").write_text(WORD_STORE, encoding="utf-8")
    (root / "decks").mkdir()
    section: dict[str, Any] = {
        "kind": "kanji",
        "name": "Genki II Kanji",
        # Pinned in the file: the notetype id is what an existing collection
        # matches these notes against.
        "deck_id": 1600000001,
        "model_id": 1600000002,
        "model_name": "Japanese Kanji",
        "source": "../kanji_notes.json",
        "output": "kanji.apkg",
        "include_ids": [character_record_id(c) for c in sorted(store)],
    }
    if cards is not None:
        section["cards"] = cards
    deck_path = root / "decks" / "kanji.yaml"
    deck_path.write_text(
        yaml.safe_dump({"deck": section}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(root), deck_path


def _wait_for_the_clock_to_tick() -> None:
    """Put a whole second between two builds, because Anki counts in seconds.

    genanki stamps a note with the second it was written, and the importer
    updates a note whose GUID it already has only when the incoming one is
    newer. Two builds inside one second are not newer than each other, so a
    rebuild test that does not wait measures the clock rather than the deck.
    A real rebuild is minutes or days later; this is the same condition.
    """
    boundary = int(time.time()) + 1
    while time.time() < boundary:
        time.sleep(0.05)


def render(
    store: Mapping[str, CharacterNote] | None = None,
    *,
    cards: dict[str, bool] | None = None,
    rebuild_as: Mapping[str, CharacterNote] | None = None,
) -> dict[str, Any]:
    """Build the character deck, import it, and render every card Anki makes.

    With ``rebuild_as`` the store is rewritten and built again into a second
    package, imported into the *same* collection — which is where a GUID that
    moved shows up as a duplicated note rather than an updated one.

    The collection is scratch and thrown away; `tests/conftest.py` also guards
    the developer's real one globally.
    """
    first = dict(notes() if store is None else store)
    root = Path(tempfile.mkdtemp())
    # Around the whole body, as in `test_rendered_cards.py`: the build is what
    # raises here, and a cleanup that starts after it leaves a copy of the
    # templates and a package behind for exactly the red-test loop that runs
    # this most.
    try:
        config, deck_path = _project(root, first, cards)
        word_store_before = (root / "vocabulary.json").read_bytes()

        collection = anki_collection.Collection(str(root / "scratch.anki2"))
        try:
            result = build_kanji_deck(deck_path, config, root / "out-1.apkg")
            collection.import_anki_package(
                ImportAnkiPackageRequest(package_path=str(root / "out-1.apkg"))
            )
            if rebuild_as is not None:
                (root / "kanji_notes.json").write_text(
                    kanji_notes.render_notes(rebuild_as), encoding="utf-8"
                )
                _wait_for_the_clock_to_tick()
                result = build_kanji_deck(deck_path, config, root / "out-2.apkg")
                collection.import_anki_package(
                    ImportAnkiPackageRequest(package_path=str(root / "out-2.apkg"))
                )

            drawn: list[dict[str, Any]] = []
            for card_id in collection.find_cards(""):
                card = collection.get_card(card_id)
                output = card.render_output()
                fields = dict(zip(KANJI_FIELDS, card.note().fields, strict=True))
                drawn.append(
                    {
                        "question": output.question_text,
                        "answer": output.answer_text,
                        "character": fields["Character"],
                        "record_id": fields["RecordID"],
                        "template": card.template()["name"],
                        "guid": card.note().guid,
                    }
                )
            return {
                "cards": drawn,
                "result": result,
                "note_count": len(collection.find_notes("")),
                "guids": {card["guid"] for card in drawn},
                "word_store": (word_store_before, (root / "vocabulary.json").read_bytes()),
            }
        finally:
            collection.close()
    finally:
        # Actually thrown away: these live outside pytest's `tmp_path`, whose
        # retention policy would otherwise keep several copies of a collection.
        shutil.rmtree(root, ignore_errors=True)


# --- reading a drawn card ------------------------------------------------------

#: The two carets a character answer may carry, each labelled for the list it
#: holds. Named here because three tests reason about the same two strings.
EXTRA_READINGS = "Other JPDB readings"
INVENTORY = "KANJIDIC readings"

DETAILS = re.compile(r"<details\b.*?</details>", re.S)
#: One annotated run of Anki notation: `料[りょう]`.
FURIGANA_GROUP = re.compile(r"([^\[\]\s]+)\[([^\[\]]+)\]")
TAGS = re.compile(r"<[^>]+>")
#: Anki furigana notation that reached a card without being rendered: kana in
#: square brackets is `話[はな]す`, not something a learner should read.
BRACKETED_KANA = re.compile(r"\[[ぁ-ゖァ-ヺー]+\]")


def exposed(drawn: str) -> str:
    """The card with every disclosure removed — what Show Answer reveals."""
    return DETAILS.sub("", drawn)


def disclosures(drawn: str) -> list[str]:
    return DETAILS.findall(drawn)


def ruby(furigana: str) -> list[str]:
    """What each annotated run of the provider's notation has to become.

    Written out here rather than imported from the renderer, so a change to
    how janki draws ruby fails this file instead of agreeing with it.
    """
    return [
        f"<ruby><rb>{base}</rb><rt>{reading}</rt></ruby>"
        for base, reading in FURIGANA_GROUP.findall(furigana)
    ]


def visible(drawn: str) -> str:
    """The text a learner reads, with markup and entities resolved."""
    return " ".join(html.unescape(TAGS.sub(" ", drawn)).split())


def one(cards: list[dict[str, Any]], character: str, template: str) -> dict[str, Any]:
    matched = [
        card
        for card in cards
        if card["character"] == character and card["template"] == template
    ]
    assert len(matched) == 1, f"{character} has {len(matched)} {template} cards"
    return matched[0]


@pytest.fixture(scope="module")
def default_deck() -> dict[str, Any]:
    """The deck a character target builds when it says nothing about cards."""
    return render()


@pytest.fixture(scope="module")
def reading_deck() -> dict[str, Any]:
    return render(cards={"recognition": True, "reading": True})


@pytest.fixture(scope="module")
def production_deck() -> dict[str, Any]:
    return render(cards={"recognition": True, "reading": True, "production": True})


# --- the default deck ----------------------------------------------------------


def test_five_targets_draw_five_notes_and_five_cards(
    default_deck: dict[str, Any],
) -> None:
    """Recognition is the default direction, and the only one enabled by
    silence: the project-wide word default that turns production on must not
    reach a character deck, where the prompt is a cue only the owner writes."""
    drawn = default_deck["cards"]

    assert default_deck["result"].card_types == ("recognition",)
    assert default_deck["note_count"] == 5
    assert len(drawn) == 5
    assert {card["character"] for card in drawn} == set(TARGETS)
    assert {card["template"] for card in drawn} == {"Kanji Recognition"}
    assert {card["record_id"] for card in drawn} == {
        f"kanji:{character}" for character in TARGETS
    }


def test_the_front_shows_the_character_and_gives_nothing_away(
    default_deck: dict[str, Any],
) -> None:
    """Recall the meaning from the character. A front carrying the meaning, the
    reading, the reported figure or the strokes is a card that asks nothing —
    and the answer's fields are one `{{FrontSide}}` mistake away from it."""
    for card in default_deck["cards"]:
        facts = FIXTURES[card["character"]]
        question = card["question"]

        assert card["character"] in question
        assert visible(question) == f"Kanji {card['character']}"
        for leaked in (
            *facts.meanings,
            facts.percent_text,
            facts.pronounced,
            facts.gloss,
            facts.written,
        ):
            assert leaked not in question, f"{card['character']} front leaks {leaked}"
        assert "stroke-cell" not in question
        assert "<details" not in question


def test_show_answer_reveals_meaning_strokes_and_the_common_reading_together(
    default_deck: dict[str, Any],
) -> None:
    """One reveal, not a hunt. Meanings, the stroke strip, the reading the
    provider quantified and the word it bound to that reading are all on the
    face of the answer — outside every disclosure on the card."""
    for card in default_deck["cards"]:
        character = card["character"]
        facts = FIXTURES[character]
        face = exposed(card["answer"])

        assert ", ".join(facts.meanings) in face
        assert f"{STROKE_COUNT} strokes" in visible(face)
        assert face.count('class="stroke-cell"') == STROKE_COUNT, "one cell per stroke"
        assert f'd="{strokes(character)[0]}"' in face, "this character's own paths"
        assert f'<span class="kanji-usage-reading">{facts.reported}</span>' in face
        assert facts.percent_text in face
        assert "JPDB reported usage" in face, "whose figure it is"
        # The provider's own binding, drawn as the reading's example rather
        # than merely present somewhere on the card — and with the ruby the
        # provider supplied, so no bracket notation reaches the learner.
        assert 'class="kanji-bound-example"' in face
        for group in ruby(facts.furigana):
            assert group in face, f"{character}: {group} is not drawn"
        assert facts.pronounced in face, "the whole-word reading, beside it"
        assert facts.gloss in face


def test_extra_readings_and_the_inventory_sit_behind_their_own_disclosures(
    default_deck: dict[str, Any],
) -> None:
    """The provider's later groups and KANJIDIC's on/kun list are each one
    caret in, each labelled for what it is. A jpdb reading group is not an
    on/kun entry, so neither list is drawn as the other."""
    for card in default_deck["cards"]:
        character = card["character"]
        facts = FIXTURES[character]
        answer, face = card["answer"], exposed(card["answer"])
        blocks = disclosures(answer)

        assert INVENTORY in answer
        inventory = [block for block in blocks if INVENTORY in block]
        assert len(inventory) == 1
        for _kind, reading in facts.inventory:
            assert reading in inventory[0]
            assert reading not in face or reading == facts.reported
        assert facts.percent_text not in inventory[0], (
            "no provider figure is attached to a dictionary reading"
        )

        if facts.unquantified:
            extra = [block for block in blocks if EXTRA_READINGS in block]
            assert len(extra) == 1
            assert facts.unquantified in extra[0]
            assert facts.unquantified not in face
            # The same string may appear in both lists — ブツ is a reading jpdb
            # lists and one KANJIDIC lists. What must not happen is one being
            # drawn as the other, so the on/kun badge belongs to the inventory
            # block alone.
            assert "音" not in extra[0] and "訓" not in extra[0]
        else:
            # The other direction, so the test above cannot pass by the block
            # being broken for every character: a source that printed one
            # quantified reading and nothing else gets no extra caret at all.
            assert EXTRA_READINGS not in answer


def test_the_answer_itself_is_not_collapsed(default_deck: dict[str, Any]) -> None:
    """Anki's Show Answer is the reveal. A `<details>` wrapping the whole
    answer would hide behind a second click what the learner already asked
    for — so every disclosure on the card is one of the two named ones."""
    for card in default_deck["cards"]:
        facts = FIXTURES[card["character"]]
        summaries = re.findall(r"<summary>(.*?)</summary>", card["answer"], re.S)

        assert set(summaries) <= {EXTRA_READINGS, INVENTORY}
        assert len(summaries) == (2 if facts.unquantified else 1)
        assert visible(exposed(card["answer"])).startswith("Kanji "), (
            "the character and its meanings are still drawn"
        )


def test_the_answer_credits_the_page_it_copied(default_deck: dict[str, Any]) -> None:
    """A card states its figures' provenance itself. The link is the provider
    page the note's evidence was taken from, with when it was taken."""
    for card in default_deck["cards"]:
        character = card["character"]
        answer = card["answer"]

        assert "kanjiapi.dev (KANJIDIC2)" in answer and "KanjiVG" in answer
        assert f'<a href="https://jpdb.io/kanji/{character}">' in answer
        assert FETCHED_AT in answer


# --- what no character card may contain ---------------------------------------


def test_no_field_reference_survives_rendering(
    default_deck: dict[str, Any],
    reading_deck: dict[str, Any],
    production_deck: dict[str, Any],
) -> None:
    """A renamed or mistyped field renders as literal `{{Whatever}}` — visible
    on the card, invisible to a test that reads the template source, and
    invisible to the build, which never resolves a field name."""
    for deck in (default_deck, reading_deck, production_deck):
        for card in deck["cards"]:
            for side in ("question", "answer"):
                assert "{{" not in card[side], (card["character"], card["template"], side)


def test_no_card_shows_raw_furigana_notation(
    default_deck: dict[str, Any],
    reading_deck: dict[str, Any],
    production_deck: dict[str, Any],
) -> None:
    """`料[りょう] 理[り]` is notation for Anki, not text for a learner. It
    reaches a card through `{{furigana:...}}` or it does not reach it at all."""
    for deck in (default_deck, reading_deck, production_deck):
        for card in deck["cards"]:
            for side in ("question", "answer"):
                leak = BRACKETED_KANA.search(visible(card[side]))
                assert leak is None, (
                    f"{card['character']} {card['template']} {side}: "
                    f"{leak.group(0) if leak else ''} drawn as notation"
                )


def test_a_character_build_neither_reads_nor_writes_the_word_store(
    default_deck: dict[str, Any],
) -> None:
    """Kanji is a distinct content type. Its notes come from the curated
    character store and its identities are `kanji:<character>`; the vocabulary
    collection sitting beside it is untouched, and none of it is on a card."""
    before, after = default_deck["word_store"]

    assert before == after, "the word store was rewritten by a character build"
    for card in default_deck["cards"]:
        assert card["record_id"].startswith("kanji:")
        for side in ("question", "answer"):
            assert WORD_RECORD.expression not in card[side]
            assert "to speak" not in card[side]


# --- the optional directions ---------------------------------------------------


def test_reading_cards_ask_the_one_fixed_example(reading_deck: dict[str, Any]) -> None:
    """Enabled reading practice adds exactly one card per character, and asks
    the example the note fixed when it was written — a prompt that moved with
    a later refresh would change the question a card with review history is
    asking."""
    drawn = reading_deck["cards"]

    assert reading_deck["result"].card_types == ("recognition", "reading")
    assert reading_deck["note_count"] == 5
    assert len(drawn) == 10

    for character in TARGETS:
        facts = FIXTURES[character]
        question = one(drawn, character, "Kanji Reading")["question"]

        assert visible(question) == f"Kanji reading {facts.written} How is this read?"
        assert facts.pronounced not in question, "the reading is the answer"
        assert facts.gloss not in question


def test_the_reading_answer_draws_the_providers_ruby(
    reading_deck: dict[str, Any],
) -> None:
    """The furigana is the provider's own, character by character, and Anki
    puts each group over the characters it was supplied for. This is the whole
    reason the file exists: the field holds bracket notation, and only a real
    render says what the learner sees."""
    answer = one(reading_deck["cards"], "理", "Kanji Reading")["answer"]

    assert "<ruby><rb>料</rb><rt>りょう</rt></ruby>" in answer
    assert "<ruby><rb>理</rb><rt>り</rt></ruby>" in answer
    assert "<rb>料理</rb>" not in answer, "り is drawn over 理 alone"
    assert "cooking" in answer
    assert "logic, reason" in answer, "the character it is teaching"


def test_production_cards_ask_from_the_owners_cue(
    production_deck: dict[str, Any],
) -> None:
    """A production front is the cue and nothing else. janki writes no cue —
    the front would be blank — and a cue that names its own character asks a
    question it has already answered."""
    drawn = production_deck["cards"]

    assert production_deck["result"].card_types == (
        "recognition",
        "reading",
        "production",
    )
    assert production_deck["note_count"] == 5
    assert len(drawn) == 15

    for character in TARGETS:
        facts = FIXTURES[character]
        card = one(drawn, character, "Kanji Production")

        assert facts.cue in visible(card["question"])
        assert character not in card["question"], "the cue gives the answer away"
        assert character in card["answer"]
        assert ", ".join(facts.meanings) in card["answer"]


def test_every_enabled_direction_draws_a_card_with_something_to_ask(
    production_deck: dict[str, Any],
) -> None:
    """Anki draws no card for an empty front, so a direction whose prompt field
    is unset goes missing rather than blank. Fifteen cards, and every one of
    them asks something beyond the card-kind label it was printed under."""
    drawn = production_deck["cards"]
    chrome = {
        "Kanji Recognition": "Kanji",
        "Kanji Reading": "Kanji reading How is this read?",
        "Kanji Production": "Kanji production Which character?",
    }

    assert len(drawn) == 15
    for character in TARGETS:
        for template, label in chrome.items():
            question = visible(one(drawn, character, template)["question"])
            asked = " ".join(
                word for word in question.split() if word not in label.split()
            )
            assert asked.strip(), f"{character} {template} front is only its label"


# --- rebuilds ------------------------------------------------------------------


def test_a_rebuild_updates_the_same_notes_rather_than_duplicating_them() -> None:
    """The GUID is derived from the note's identity — `kanji:理` — and not from
    what the note says, so refreshing a meaning updates the card a learner has
    review history on instead of stranding it beside a new one.

    Imported into one collection twice, which is where Anki, rather than the
    package, decides whether these are the same notes.
    """
    refreshed = notes()
    refreshed["理"] = note("理", meanings=("reason",))

    rebuilt = render(rebuild_as=refreshed)

    assert rebuilt["note_count"] == 5
    assert len(rebuilt["cards"]) == 5
    assert rebuilt["guids"] == {
        genanki.guid_for(character_record_id(character)) for character in TARGETS
    }
    answer = one(rebuilt["cards"], "理", "Kanji Recognition")["answer"]
    assert "reason" in answer
    assert "logic, reason" not in answer, "the rebuild replaced the note's meanings"
