"""Kanji reference data: what a card says about the characters in a word.

Every test drives the transport seam, so none reaches the network. The shapes
here are the real ones — 前 really does return 740 words that open on 一歩前進,
and KANJIDIC really does list まえ and -まえ as separate readings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki.kanji import (
    KANJIAPI,
    KANJIVG,
    KanjiError,
    KanjiInfo,
    fetch_kanji,
    kanji_in,
    load_store,
    render_kanji_html,
    save_store,
)


def fake_transport(*, info: dict, words: list | None = None, svg: str | None = None):
    """Answer the three URLs `fetch_kanji` asks for."""

    def send(url: str) -> bytes:
        if url.startswith(f"{KANJIVG}/"):
            if svg is None:
                raise KanjiError("no stroke data")
            return svg.encode("utf-8")
        if "/words/" in url:
            return json.dumps(words or []).encode("utf-8")
        if url.startswith(f"{KANJIAPI}/kanji/"):
            return json.dumps(info).encode("utf-8")
        raise AssertionError(f"unexpected url {url}")

    return send


SVG = (
    '<svg viewBox="0 0 109 109">'
    '<path id="kvg:0524d-s1" kvg:type="a" d="M1,1L2,2"/>'
    '<path id="kvg:0524d-s2" kvg:type="b" d="M3,3L4,4"/>'
    "</svg>"
)


def word(written: str, pronounced: str, gloss: str, priorities: list[str]) -> dict:
    return {
        "meanings": [{"glosses": [gloss]}],
        "variants": [
            {"written": written, "pronounced": pronounced, "priorities": priorities}
        ],
    }


# --- picking the characters -------------------------------------------------


def test_kanji_are_returned_in_the_order_they_are_written() -> None:
    """A card shows them left to right as the word is written; a set would put
    them in whichever order the hash landed."""
    assert kanji_in("使用") == ["使", "用"]
    assert kanji_in("前線と名前") == ["前", "線", "名"], "and each only once"
    assert kanji_in("する") == [], "kana carries no character block"


# --- ranking the examples ---------------------------------------------------


def test_a_common_word_beats_an_obscure_one() -> None:
    """The raw list for 前 is 740 entries opening on 一歩前進, 前官礼遇, 前駆体 —
    accurate and useless. JMdict's nfXX band is a frequency decile, and the
    three a commercial paper card chose all carry one."""
    send = fake_transport(
        info={"stroke_count": 9, "on_readings": ["ゼン"], "kun_readings": []},
        words=[
            word("前官礼遇", "ぜんかんれいぐう", "privileges of a former post", []),
            word("前線", "ぜんせん", "front line", ["news1", "nf08"]),
            word("午前", "ごぜん", "morning", ["ichi1", "news1", "nf02"]),
        ],
    )

    info = fetch_kanji("前", transport=send)

    written = [example.written for example in info.readings[0].examples]
    assert written == ["午前", "前線"], "commonest first, and the untagged one dropped"


def test_an_untagged_word_is_never_chosen_over_a_tagged_one() -> None:
    send = fake_transport(
        info={"on_readings": ["ゼン"], "kun_readings": []},
        words=[
            word("前駆体", "ぜんくたい", "precursor", []),
            word("前年", "ぜんねん", "the preceding year", ["news1", "nf12"]),
        ],
    )

    info = fetch_kanji("前", transport=send)

    assert [e.written for e in info.readings[0].examples] == ["前年"]


def test_common_examples_put_a_beginner_reading_before_a_rare_one() -> None:
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["く.らう", "た.べる"]},
        words=[
            word("食らう", "くらう", "to receive", ["nf40"]),
            word("食べる", "たべる", "to eat", ["nf05"]),
        ],
    )

    info = fetch_kanji("食", transport=send)

    assert [reading.reading for reading in info.readings] == ["た(べる)", "く(らう)"]


def test_only_words_that_use_that_reading_are_offered() -> None:
    """A full-reading substring can still come from somewhere else. The いく
    in 低空飛行 spans てい + くう while 行 is コウ; it cannot prove 行's い.く."""
    send = fake_transport(
        info={"on_readings": ["コウ"], "kun_readings": ["い.く"]},
        words=[
            word("低空飛行", "ていくうひこう", "low-altitude flight", ["nf20"]),
            word("行く", "いく", "to go", ["ichi1", "nf02"]),
        ],
    )

    info = fetch_kanji("行", transport=send)

    by_kind = {r.kind: [e.written for e in r.examples] for r in info.readings}
    assert by_kind == {"on": ["低空飛行"], "kun": ["行く"]}


def test_a_katakana_on_reading_matches_a_hiragana_word() -> None:
    """KANJIDIC writes on'yomi in katakana and words in hiragana, so a literal
    containment test finds nothing at all."""
    send = fake_transport(
        info={"on_readings": ["ゼン"], "kun_readings": []},
        words=[word("前線", "ぜんせん", "front line", ["nf08"])],
    )

    assert fetch_kanji("前", transport=send).readings[0].examples[0].written == "前線"


def test_okurigana_separates_two_readings_that_share_a_stem() -> None:
    """使 has both つか.う and つか.い. Matching on the stem alone made 使う and
    使い方 examples of both — and merging them labelled 使う as つか.い, which is
    the wrong reading for that word. Reported from a real card."""
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["つか.う", "つか.い"]},
        words=[
            word("使う", "つかう", "to use", ["ichi1"]),
            word("使い方", "つかいかた", "way of using", ["nf20"]),
        ],
    )

    info = fetch_kanji("使", transport=send)

    got = {r.reading: [e.written for e in r.examples] for r in info.readings}
    assert got == {"つか(う)": ["使う"], "つか(い)": ["使い方"]}


def test_a_stem_match_does_not_reach_into_an_unrelated_word() -> None:
    """書's か.く matched 教科書 — きょうか*し*ょ contains か — so an on'yomi
    compound was offered as an example of a kun reading."""
    send = fake_transport(
        info={"on_readings": ["ショ"], "kun_readings": ["か.く"]},
        words=[
            word("教科書", "きょうかしょ", "textbook", ["ichi1"]),
            word("書く", "かく", "to write", ["ichi1"]),
        ],
    )

    info = fetch_kanji("書", transport=send)

    kun = next(r for r in info.readings if r.kind == "kun")
    assert [e.written for e in kun.examples] == ["書く"]


def test_the_okurigana_marker_is_not_shown_as_a_dot() -> None:
    """KANJIDIC's dot is machine notation for where the kanji stops. The
    information is worth keeping — it is why 使う has one kana after the
    character — but a bare dot on a card reads as a typo, which is how it was
    reported."""
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["つか.う"]}, words=[]
    )

    assert fetch_kanji("使", transport=send).readings[0].reading == "つか(う)"


def test_a_reading_listed_twice_is_shown_once() -> None:
    """KANJIDIC lists まえ and -まえ — the same reading, marked for a suffix
    position. Both match the same words, so a row each prints the same example
    twice."""
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["まえ", "-まえ"]},
        words=[word("名前", "なまえ", "name", ["ichi1"])],
    )

    info = fetch_kanji("前", transport=send)

    assert [r.reading for r in info.readings] == ["まえ"]
    assert [e.written for e in info.readings[0].examples] == ["名前"]


def test_boundary_notation_does_not_duplicate_a_spoken_reading() -> None:
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["-い.き", "-いき"]},
        words=[word("行き", "いき", "going", ["ichi1"])],
    )

    info = fetch_kanji("行", transport=send)

    assert [r.reading for r in info.readings] == ["〜い(き)"]


def test_a_suffix_only_reading_keeps_its_position_marker() -> None:
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["-づか.い"]}, words=[]
    )

    assert fetch_kanji("使", transport=send).readings[0].reading == "〜づか(い)"


# --- the strokes ------------------------------------------------------------


def test_stroke_paths_are_captured_in_writing_order() -> None:
    send = fake_transport(info={"stroke_count": 2}, svg=SVG)

    info = fetch_kanji("前", transport=send)

    assert info.strokes == ("M1,1L2,2", "M3,3L4,4")


def test_a_character_kanjivg_does_not_cover_still_works() -> None:
    """Readings without strokes are still most of the card, so a missing
    diagram must not cost the whole lookup."""
    send = fake_transport(info={"stroke_count": 9, "on_readings": ["ゼン"]}, svg=None)

    info = fetch_kanji("前", transport=send)

    assert info.strokes == ()
    assert info.stroke_count == 9, "KANJIDIC still knows how many there are"


def test_something_that_is_not_a_kanji_is_refused() -> None:
    with pytest.raises(KanjiError, match="Not a kanji"):
        fetch_kanji("あ")


# --- rendering --------------------------------------------------------------


def _info(**kwargs) -> KanjiInfo:
    return KanjiInfo(character="前", **kwargs)


def test_each_stroke_cell_adds_exactly_one_stroke() -> None:
    rendered = render_kanji_html([_info(strokes=("a", "b", "c"))])

    assert rendered.count("<path") == 3, "path data is defined once, not quadratically"
    for path in ("a", "b", "c"):
        assert rendered.count(f'd="{path}"') == 1
    assert rendered.count('class="new"') == 3, "each visible cell adds one stroke"


def test_a_character_with_no_strokes_draws_no_grid() -> None:
    rendered = render_kanji_html([_info(meanings=("before",))])

    assert "stroke-order" not in rendered
    assert "before" in rendered, "but the rest of the block is still there"


def test_the_block_names_its_character_and_level() -> None:
    rendered = render_kanji_html([_info(stroke_count=9, jlpt=5, grade=2, meanings=("before",))])

    assert "<summary>前</summary>" in rendered
    assert "N5" in rendered and "9画" in rendered and "grade 2" in rendered


def test_html_in_the_data_is_escaped() -> None:
    rendered = render_kanji_html([_info(meanings=("<script>x</script>",))])

    assert "<script>" not in rendered


def test_nothing_at_all_renders_nothing() -> None:
    assert render_kanji_html([]) == ""


# --- the store --------------------------------------------------------------


def test_the_store_round_trips(tmp_path: Path) -> None:
    from japanese_anki.kanji import Example, KanjiStore, Reading

    store = KanjiStore(entries={"前": KanjiInfo(
        character="前", stroke_count=9, jlpt=5, meanings=("before",),
        readings=(Reading(kind="on", reading="ゼン", examples=(
            Example(written="前線", pronounced="ぜんせん", gloss="front line"),
        )),),
        strokes=("M1,1",),
    )})
    path = tmp_path / "kanji.json"

    save_store(path, store)
    again = load_store(path)

    assert again.entries["前"] == store.entries["前"]


def test_a_missing_store_is_empty_not_an_error(tmp_path: Path) -> None:
    """A build must not require the lookup to have been run."""
    assert load_store(tmp_path / "nothing.json").entries == {}


def test_the_store_reports_what_it_has_not_seen(tmp_path: Path) -> None:
    from japanese_anki.kanji import KanjiStore

    store = KanjiStore(entries={"前": KanjiInfo(character="前")})

    assert store.missing(["前", "線", "線"]) == ["線"], "deduplicated"


def test_a_store_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "kanji.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(KanjiError, match="keyed by character"):
        load_store(path)


def test_a_character_with_no_common_word_offers_none(tmp_path: Path) -> None:
    """Ranking alone would bury untagged entries only while something tagged
    exists to bury them under. For a rare character it would surface
    前官礼遇-grade words as though they were the ones to learn — better to show
    the reading with no example than a wrong impression of usefulness."""
    send = fake_transport(
        info={"on_readings": ["ゼン"], "kun_readings": []},
        words=[
            word("前官礼遇", "ぜんかんれいぐう", "privileges of a former post", []),
            word("前駆体", "ぜんくたい", "precursor", []),
        ],
    )

    info = fetch_kanji("前", transport=send)

    assert info.readings[0].examples == ()


def test_every_reading_gets_a_row_before_any_gets_a_second() -> None:
    """Filling reading by reading spent the whole row budget on the first two:
    使's card showed つか(い) twice and left out つか(う) — the reading of 使う,
    the word the card is about."""
    from japanese_anki.kanji import Example, Reading

    def reading(text: str, *words: str) -> Reading:
        return Reading(
            kind="kun",
            reading=text,
            examples=tuple(Example(written=w, pronounced="x", gloss="y") for w in words),
        )

    rendered = render_kanji_html([KanjiInfo(character="使", readings=(
        reading("シ", "大使", "使用"),
        reading("つか(い)", "使い方", "使い"),
        reading("つか(う)", "使う"),
        reading("づか(い)", "無駄遣い", "言葉遣い"),
    ))])

    assert "使う" in rendered, "the reading of the word this card is about"
    for text in ("大使", "使い方", "無駄遣い"):
        assert text in rendered, f"one example each, so {text} is there too"
    # The container div is class="kanji-examples", which contains the row
    # class as a substring — count the rows themselves.
    assert rendered.count('<div class="kanji-example">') == 4, "and the cap holds"


def test_an_example_less_reading_never_takes_a_row_from_one_with_examples() -> None:
    """The renderer cannot assume the caller sorted anything. `data/kanji.json`
    is committed and hand-editable, and older copies are in KANJIDIC's kana
    order — which for 来 puts four example-less readings ahead of く(る), so the
    four-row budget went to three examples and one blank, and 来る, the reading
    of the word the card is about, never rendered."""
    from japanese_anki.kanji import Example, Reading

    def reading(text: str, *words: str) -> Reading:
        return Reading(
            kind="kun",
            reading=text,
            examples=tuple(Example(written=w, pronounced="x", gloss="y") for w in words),
        )

    rendered = render_kanji_html([KanjiInfo(character="来", readings=(
        reading("き(たす)"),          # example-less, and deliberately first
        reading("き(たる)"),
        reading("きた(す)"),
        reading("きた(る)"),
        reading("ライ", "来年"),
        reading("く(る)", "来る"),
    ))])

    assert "来る" in rendered, "the reading of the word this card is about"
    assert "来年" in rendered
    assert rendered.count('<div class="kanji-example">') == 4, "and the cap holds"


def test_a_reading_without_a_common_example_is_still_shown() -> None:
    from japanese_anki.kanji import Reading

    rendered = render_kanji_html([
        KanjiInfo(character="飲", readings=(Reading(kind="on", reading="オン"),))
    ])

    assert '<span class="kanji-reading">オン</span>' in rendered
    assert rendered.count('<div class="kanji-example">') == 1


def test_curated_examples_fill_the_render_budget_past_the_fetch_cap() -> None:
    from japanese_anki.kanji import Example, Reading

    examples = tuple(
        Example(written=written, pronounced=pronounced, gloss=gloss)
        for written, pronounced, gloss in (
            ("午前", "ごぜん", "morning"),
            ("前線", "ぜんせん", "front line"),
            ("前年", "ぜんねん", "preceding year"),
        )
    )
    rendered = render_kanji_html([
        KanjiInfo(
            character="前",
            readings=(Reading(kind="on", reading="ゼン", examples=examples),),
        )
    ])

    assert all(word in rendered for word in ("午前", "前線", "前年"))
    assert rendered.count('<div class="kanji-example">') == 3


# --- the character has to be provably the one being read --------------------


def test_a_word_the_character_is_not_even_in_is_refused() -> None:
    """JMdict lists spellings together, so 書's word list contains 絵を描く. A
    substring test over the kana accepted it; the position rule refuses it for
    the same reason it refuses a buried character — nothing proves the pairing."""
    send = fake_transport(
        info={"on_readings": [], "kun_readings": ["か.く"]},
        words=[
            word("絵を描く", "えをかく", "to draw a picture", ["ichi1"]),
            word("書く", "かく", "to write", ["ichi1"]),
        ],
    )

    info = fetch_kanji("書", transport=send)

    assert [e.written for e in info.readings[0].examples] == ["書く"]


def test_a_character_buried_in_a_compound_cannot_be_pinned() -> None:
    """書 sits in the middle of 図書館 (としょかん), and しょ really is in there —
    so containment accepts it. But nothing proves *that* しょ is 書's rather than
    part of と-しょ-かん's reading of another character, and the card would
    assert a pairing nobody verified. Refused rather than guessed at.

    The reading here has to be one the pronunciation genuinely contains, or the
    test passes on the full-reading match and never reaches the position rule."""
    send = fake_transport(
        info={"on_readings": ["ショ"], "kun_readings": []},
        words=[word("図書館", "としょかん", "library", ["ichi1"])],
    )

    assert fetch_kanji("書", transport=send).readings[0].examples == ()


def test_the_reading_must_sit_where_the_character_sits() -> None:
    """部分 (ぶぶん) ends in 分, so 分's reading must end the pronunciation. ブ
    does not — the word uses ブン — and the old containment test took it."""
    send = fake_transport(
        info={"on_readings": ["ブ"], "kun_readings": []},
        words=[word("部分", "ぶぶん", "part", ["ichi1"])],
    )

    assert fetch_kanji("分", transport=send).readings[0].examples == ()


def test_the_longest_reading_that_fits_claims_the_word() -> None:
    """One reading is often a prefix of another: 分's ブ and ブン both open
    分野 (ぶんや). Offering it under ブ teaches a reading the word does not
    use."""
    send = fake_transport(
        info={"on_readings": ["ブ", "ブン"], "kun_readings": []},
        words=[word("分野", "ぶんや", "field", ["ichi1"])],
    )

    info = fetch_kanji("分", transport=send)

    got = {r.reading: [e.written for e in r.examples] for r in info.readings}
    assert got == {"ブ": [], "ブン": ["分野"]}


def test_a_word_the_character_opens_matches_on_its_prefix() -> None:
    send = fake_transport(
        info={"on_readings": ["ゼン"], "kun_readings": []},
        words=[word("前線", "ぜんせん", "front line", ["nf08"])],
    )

    assert fetch_kanji("前", transport=send).readings[0].examples[0].written == "前線"
