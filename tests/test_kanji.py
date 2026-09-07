"""Kanji reference data: what a card says about the characters in a word.

Every test drives the transport seam, so none reaches the network. The shapes
here are the real ones — KANJIDIC really does list まえ and -まえ as separate
readings, and 使 really does have both つか.う and つか.い.

Example words are not here, and that is the point: the ranking that used to
pick them was janki deciding how Japanese is read. A word reaches a card only
where a provider bound it to a reading itself, which is
`tests/test_jpdb_kanji.py`'s subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)
from japanese_anki.kanji import (
    KANJIAPI,
    KANJIVG,
    KanjiError,
    KanjiInfo,
    Reading,
    assigns_a_known_reading,
    fetch_kanji,
    kanji_in,
    load_store,
    render_furigana,
    render_kanji_html,
    render_stroke_strip,
    save_store,
)


def fake_transport(*, info: dict, svg: str | None = None):
    """Answer the two URLs `fetch_kanji` asks for."""
    requested: list[str] = []

    def send(url: str) -> bytes:
        requested.append(url)
        if url.startswith(f"{KANJIVG}/"):
            if svg is None:
                raise KanjiError("no stroke data")
            return svg.encode("utf-8")
        if url.startswith(f"{KANJIAPI}/kanji/"):
            return json.dumps(info).encode("utf-8")
        raise AssertionError(f"unexpected url {url}")

    send.requested = requested  # type: ignore[attr-defined]
    return send


SVG = (
    '<svg viewBox="0 0 109 109">'
    '<path id="kvg:0524d-s1" kvg:type="a" d="M1,1L2,2"/>'
    '<path id="kvg:0524d-s2" kvg:type="b" d="M3,3L4,4"/>'
    "</svg>"
)


# --- picking the characters -------------------------------------------------


def test_kanji_are_returned_in_the_order_they_are_written() -> None:
    """A card shows them left to right as the word is written; a set would put
    them in whichever order the hash landed."""
    assert kanji_in("使用") == ["使", "用"]
    assert kanji_in("前線と名前") == ["前", "線", "名"], "and each only once"
    assert kanji_in("する") == [], "kana carries no character block"


@pytest.mark.parametrize(
    ("listed", "surface"),
    [("ガク", "がっ"), ("ニチ", "にっ"), ("くち", "ぐち"), ("かみ", "がみ")],
)
def test_known_reading_accepts_compound_sound_changes(
    listed: str, surface: str
) -> None:
    info = KanjiInfo(
        character="字",
        readings=(Reading(kind="on", reading=listed),),
    )

    assert assigns_a_known_reading(info, surface)


@pytest.mark.parametrize(
    ("listed", "surface"),
    [("ニチ", "した"), ("あ", "した"), ("ヒ", "あす")],
)
def test_a_reading_the_dictionary_never_lists_is_refused(
    listed: str, surface: str
) -> None:
    """The negative direction, and the reason the function survived M8.3: on
    the --jpdb word path it stops a jukujikun's per-character split from
    teaching readings that do not exist. jpdb hands 明日 back as
    明[あ]日[した], and した is no reading of 日 under any rendaku or sokuon
    tolerance — so `_furigana_for` falls back to whole-word 明日[あした],
    which is always true. A version of this function that answers yes to
    everything re-teaches the false decomposition."""
    info = KanjiInfo(
        character="日",
        readings=(Reading(kind="on", reading=listed),)
        if listed.isupper() or listed in ("ニチ", "ヒ")
        else (Reading(kind="kun", reading=listed),),
    )

    assert not assigns_a_known_reading(info, surface)


# --- the reading inventory --------------------------------------------------


def test_the_inventory_is_kanjidic_order_on_readings_then_kun() -> None:
    """No ranking. The old sort asked JMdict's priority tags which readings a
    learner should see first, which is janki deciding how Japanese is read;
    KANJIDIC's own order is what the dictionary published."""
    send = fake_transport(
        info={"on_readings": ["ショク", "ジキ"], "kun_readings": ["た.べる", "く.う"]}
    )

    info = fetch_kanji("食", transport=send)

    assert [(r.kind, r.reading) for r in info.readings] == [
        ("on", "ショク"),
        ("on", "ジキ"),
        ("kun", "た(べる)"),
        ("kun", "く(う)"),
    ]


def test_a_reading_kanjidic_writes_twice_is_listed_twice() -> None:
    """まえ and -まえ are two entries in the source. Whether they are the same
    reading is a question about Japanese, and the rule that used to answer it
    (equal once the boundary markers come off) was this module's own invention.
    The inventory is the dictionary's list, so both stay."""
    send = fake_transport(info={"on_readings": [], "kun_readings": ["まえ", "-まえ"]})

    info = fetch_kanji("前", transport=send)

    assert [r.reading for r in info.readings] == ["まえ", "〜まえ"]


def test_a_kanjidic_reading_carries_no_examples() -> None:
    """A word is bound to a reading by a provider that says so, never by this
    module matching kana. There is nowhere on a KANJIDIC reading to put one."""
    send = fake_transport(info={"on_readings": ["ゼン"], "kun_readings": []})

    (reading,) = fetch_kanji("前", transport=send).readings

    assert not hasattr(reading, "examples")


def test_no_word_list_is_requested() -> None:
    """kanjiapi's /words/ endpoint fed the ranking, and the ranking is gone.
    Fetching it anyway would spend a request on data nothing reads."""
    send = fake_transport(info={"on_readings": ["ゼン"]}, svg=SVG)

    fetch_kanji("前", transport=send)

    assert send.requested == [
        f"{KANJIAPI}/kanji/%E5%89%8D",
        f"{KANJIVG}/0524d.svg",
    ]


def test_the_okurigana_marker_is_not_shown_as_a_dot() -> None:
    """KANJIDIC's dot is machine notation for where the kanji stops. The
    information is worth keeping — it is why 使う has one kana after the
    character — but a bare dot on a card reads as a typo, which is how it was
    reported."""
    send = fake_transport(info={"on_readings": [], "kun_readings": ["つか.う"]})

    assert fetch_kanji("使", transport=send).readings[0].reading == "つか(う)"


def test_a_suffix_only_reading_keeps_its_position_marker() -> None:
    send = fake_transport(info={"on_readings": [], "kun_readings": ["-づか.い"]})

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


def test_the_stroke_strip_is_one_renderer_two_callers_can_share() -> None:
    """The character back and the word-card block draw the same strip. Two
    copies of this markup would drift, and the ids are document-wide, so the
    caller supplies the prefix that keeps two strips on one card apart."""
    strip = render_stroke_strip(_info(strokes=("M1,1", "M2,2")), prefix="back-0")

    assert strip.count('id="back-0-stroke-') == 2
    assert 'href="#back-0-stage-0"' in strip, "each stage references the one before it"
    assert render_stroke_strip(_info()) == "", "and nothing to draw draws nothing"


def test_a_stroke_path_and_prefix_are_escaped() -> None:
    strip = render_stroke_strip(_info(strokes=('M1,1"/><script>x</script>',)), prefix='a"b')

    assert "<script>" not in strip
    assert 'id="a&quot;b-stroke-0"' in strip


# --- what a provider reported ------------------------------------------------


def _usage(label: str, percent_text: str | None, *examples: BoundExample) -> ReadingUsage:
    percent = None if percent_text is None else int(percent_text.strip("(%)"))
    return ReadingUsage(
        label=label,
        href=f"/kanji-reading/理/{label}",
        percent_text=percent_text,
        percent=percent,
        percent_less_than=None if percent_text is None else False,
        examples=examples,
        detail_source_url=None if percent_text is None else f"https://jpdb.io/x/{label}",
    )


def _example(written: str, pronounced: str, gloss: str) -> BoundExample:
    return BoundExample(
        written=written,
        pronounced=pronounced,
        gloss=gloss,
        furigana=f"{written}[{pronounced}]",
        source_url=f"https://jpdb.io/vocabulary/1/{written}/{pronounced}#a",
    )


def _evidence(*groups: ReadingGroup, character: str = "理") -> dict[str, CharacterReadings]:
    return {
        character: CharacterReadings(
            character=character,
            source_url="https://jpdb.io/kanji/理",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256="0" * 64,
            groups=groups,
        )
    }


RI_EVIDENCE = _evidence(
    ReadingGroup(
        source_class="kanji-reading-list-common",
        readings=(
            _usage("り", "(84%)", _example("理由", "りゆう", "reason"),
                   _example("無理", "むり", "unreasonable")),
            _usage("わ", "(14%)", _example("理由", "わけ", "reason")),
        ),
    ),
    ReadingGroup(
        source_class="kanji-reading-list",
        readings=(_usage("ことわり", None), _usage("め", None)),
    ),
)

RI_INFO = KanjiInfo(
    character="理",
    stroke_count=11,
    meanings=("reason", "logic"),
    readings=(Reading(kind="on", reading="リ"), Reading(kind="kun", reading="ことわり")),
)


def test_every_quantified_reading_is_shown_with_its_printed_figure() -> None:
    """In jpdb's order, with jpdb's own strings. Nothing is renormalised, and
    nothing is dropped for being small."""
    rendered = render_kanji_html([RI_INFO], reading_evidence=RI_EVIDENCE)

    assert "JPDB reported usage" in rendered
    assert rendered.index("り") < rendered.index("わ")
    assert "(84%)" in rendered and "(14%)" in rendered
    assert rendered.count('<div class="kanji-usage">') == 2


def test_a_reading_shows_the_words_its_own_page_bound_to_it() -> None:
    rendered = render_kanji_html([RI_INFO], reading_evidence=RI_EVIDENCE)

    first = rendered.split('<div class="kanji-usage">')[1]
    assert "理由" in first and "りゆう" in first and "reason" in first
    assert "無理" in first, "both of the two the page supplied"


def test_a_quantified_reading_with_no_bound_word_still_shows_its_figure() -> None:
    """Hiding the quantity until an example exists would report a different
    fact from the one jpdb printed."""
    evidence = _evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(_usage("り", "(84%)"), _usage("わ", "(14%)")),
        )
    )

    rendered = render_kanji_html([RI_INFO], reading_evidence=evidence)

    assert "(84%)" in rendered and "(14%)" in rendered
    assert rendered.count('<div class="kanji-usage">') == 2
    assert "kanji-bound-example" not in rendered


def test_a_jpdb_reading_never_gets_an_on_or_kun_badge() -> None:
    """A jpdb reading group is not KANJIDIC's on/kun inventory, and printing
    one beside the other would say it is."""
    rendered = render_kanji_html([RI_INFO], reading_evidence=RI_EVIDENCE)

    evidence_block = rendered.split('<div class="kanji-evidence">')[1].split("</details>")[0]
    assert "音" not in evidence_block and "訓" not in evidence_block


def test_the_extra_readings_and_the_inventory_are_separate_disclosures() -> None:
    """Two different things, from two different sources, behind two labels."""
    rendered = render_kanji_html([RI_INFO], reading_evidence=RI_EVIDENCE)

    assert "<summary>Other JPDB readings</summary>" in rendered
    assert "<summary>KANJIDIC readings</summary>" in rendered
    other = rendered.split("Other JPDB readings</summary>")[1].split("</details>")[0]
    assert "ことわり" in other and "め" in other
    inventory = rendered.split("KANJIDIC readings</summary>")[1].split("</details>")[0]
    assert "リ" in inventory and "音" in inventory
    assert "(84%)" not in other and "(84%)" not in inventory


def test_without_jpdb_facts_the_inventory_is_all_there_is() -> None:
    """And it stays reference: no reading is promoted, ranked, or presented as
    the common one."""
    rendered = render_kanji_html([RI_INFO])

    assert "JPDB reported usage" not in rendered
    assert "Other JPDB readings" not in rendered
    assert "<summary>KANJIDIC readings</summary>" in rendered
    assert "リ" in rendered and "ことわり" in rendered


def test_reported_facts_are_escaped_like_everything_else() -> None:
    evidence = _evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(
                _usage("り", "(84%)", _example("<b>理由</b>", "り", "<script>x</script>")),
            ),
        )
    )

    rendered = render_kanji_html([RI_INFO], reading_evidence=evidence)

    assert "<script>" not in rendered and "<b>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_evidence_for_another_character_is_not_borrowed() -> None:
    rendered = render_kanji_html([RI_INFO], reading_evidence=_evidence(character="王"))

    assert "JPDB reported usage" not in rendered


# --- the provider's own ruby ---------------------------------------------------


def test_explicit_notation_becomes_ruby_and_leaves_no_brackets() -> None:
    """Anki does not apply its ``furigana:`` filter to notation stored inside
    another field's HTML, so the bracket form has to be rendered here or it
    reaches the card as literal text."""
    assert render_furigana("理[り] 由[ゆう]") == (
        "<ruby><rb>理</rb><rt>り</rt></ruby><ruby><rb>由</rb><rt>ゆう</rt></ruby>"
    )


def test_unannotated_kana_and_a_repeated_annotation_both_survive() -> None:
    """無理やり's やり carries no reading on jpdb's page, and 理論物理学 carries
    理[り] twice. The notation is the source's: neither is tidied up."""
    assert render_furigana("無[む] 理[り]やり") == (
        "<ruby><rb>無</rb><rt>む</rt></ruby><ruby><rb>理</rb><rt>り</rt></ruby>やり"
    )

    rendered = render_furigana("理[り] 論[ろん] 物[ぶつ] 理[り] 学[がく]")

    assert rendered.count("<rb>理</rb><rt>り</rt>") == 2
    assert "[" not in rendered and "]" not in rendered


def test_only_the_boundary_space_is_consumed() -> None:
    """The one space Anki writes in front of an annotated run says where the run
    begins, and goes with it. A second space is text."""
    assert render_furigana("お 茶[ちゃ]") == "お<ruby><rb>茶</rb><rt>ちゃ</rt></ruby>"
    assert render_furigana("お  茶[ちゃ]") == "お <ruby><rb>茶</rb><rt>ちゃ</rt></ruby>"


def test_markup_in_the_notation_is_escaped_rather_than_rendered() -> None:
    """Only the brackets are structure. Everything they hold, and everything
    between them, is text."""
    rendered = render_furigana("<script>x</script>理[<img src=y onerror=z>]")

    assert rendered == (
        "&lt;script&gt;x&lt;/script&gt;"
        "<ruby><rb>理</rb><rt>&lt;img src=y onerror=z&gt;</rt></ruby>"
    )


def test_a_bound_word_is_shown_with_the_ruby_its_page_supplied() -> None:
    evidence = _evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(
                _usage(
                    "り",
                    "(84%)",
                    BoundExample(
                        written="無理やり",
                        pronounced="むりやり",
                        gloss="forcibly",
                        furigana="無[む] 理[り]やり",
                        source_url="https://jpdb.io/vocabulary/1531030/x/y#a",
                    ),
                ),
            ),
        )
    )

    rendered = render_kanji_html([RI_INFO], reading_evidence=evidence)

    word = rendered.split('<span class="kanji-word">')[1].split("</span>")[0]
    assert word == (
        "<ruby><rb>無</rb><rt>む</rt></ruby><ruby><rb>理</rb><rt>り</rt></ruby>やり"
    )
    assert "[" not in word and "]" not in word
    assert '<span class="kanji-kana">むりやり</span>' in rendered, "still stated whole"


def test_a_word_with_no_supplied_ruby_falls_back_to_its_spelling() -> None:
    """Reading the whole-word kana over the whole spelling would be janki
    deciding which kana sit over which characters — the one thing the reading
    page exists to state and this renderer must never infer."""
    evidence = _evidence(
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(
                _usage(
                    "り",
                    "(84%)",
                    BoundExample(
                        written="<b>理由</b>",
                        pronounced="りゆう",
                        gloss="reason",
                        furigana="",
                        source_url="https://jpdb.io/vocabulary/1550140/x/y#a",
                    ),
                ),
            ),
        )
    )

    rendered = render_kanji_html([RI_INFO], reading_evidence=evidence)

    word = rendered.split('<span class="kanji-word">')[1].split("</span>")[0]
    assert word == "&lt;b&gt;理由&lt;/b&gt;"
    assert "<ruby>" not in rendered, "no ruby guessed from the whole-word reading"


# --- the store --------------------------------------------------------------


def test_the_store_round_trips(tmp_path: Path) -> None:
    from japanese_anki.kanji import KanjiStore

    store = KanjiStore(entries={"前": KanjiInfo(
        character="前", stroke_count=9, jlpt=5, meanings=("before",),
        readings=(Reading(kind="on", reading="ゼン"), Reading(kind="kun", reading="まえ")),
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
