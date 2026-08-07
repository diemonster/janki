import pytest

from japanese_anki.identifiers import (
    contains_kanji,
    normalize_identity_part,
    stable_record_id,
)


def test_stable_record_id_normalizes_width_and_space() -> None:
    assert stable_record_id(" 話す ", "はなす") == "word:話す:はなす"
    assert stable_record_id("Ａ", "エー") == "word:A:エー"


@pytest.mark.parametrize(
    "value",
    [
        "話す",  # U+8A71, CJK Unified
        "𠮟る",  # U+20B9F, Extension B: the JIS2004 form of しかる, exported by Shirabe
        "𩸽",  # U+29E3D, Extension B: hokke, an ordinary menu item
        "㐬",  # U+340C, Extension A
        "﨑",  # U+FA11, Compatibility Ideographs — common in surnames
        "々",  # U+3005, the iteration mark
        "〇",  # U+3007, ideographic number zero: a numeral kanji (れい/まる)
        "〻",  # U+303B, the vertical iteration mark, Script=Han like 々
        "\U0002ebf0",  # Extension I's first ideograph (plane 2, above Ext F)
        "\U0002ee5d",  # Extension I's last-but-two ideograph
        "\U00030000",  # Extension G's first ideograph (plane 3)
        "\U00031350",  # Extension H's first ideograph (plane 3)
        # The last code point of every range: an interior member proves nothing
        # about a boundary, and the way this has broken twice is a range that
        # stops one block short.
        "䶿",  # Extension A's last
        "鿿",  # CJK Unified Ideographs' last
        "﫿",  # Compatibility Ideographs' last
        "\U0002a6df",  # Extension B's last
        "\U0002ee5f",  # the plane-2 range's top
        "\U0002fa1d",  # Compatibility Ideographs Supplement's last
        "\U000323b0",  # Extension J's first ideograph — unpinned until now
        "\U0003347b",  # Extension J's last ideograph
    ],
)
def test_kanji_is_recognized_in_every_plane_it_lives_in(value: str) -> None:
    # Supplementary-plane kanji are the ones a BMP-only range misses, and
    # missing one does not lose a warning: it mints word:𠮟る:𠮟る, an id that
    # can never be corrected without orphaning Anki review history.
    assert contains_kanji(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "⼀",  # U+2F00 KANGXI RADICAL ONE -> 一 U+4E00
        "⼀る",  # and anywhere inside a longer value
        "⺟",  # U+2E9F, a CJK Radicals Supplement member that folds -> 母
        "〸",  # U+3038 HANGZHOU NUMERAL TEN -> 十, right beside U+303B
        "㊀",  # U+3280 CIRCLED IDEOGRAPH ONE -> 一
        "㍻",  # U+337B SQUARE ERA NAME HEISEI -> 平成
        "㋿",  # U+32FF SQUARE ERA NAME REIWA -> 令和
    ],
)
def test_anything_that_normalizes_into_a_kanji_is_kanji(value: str) -> None:
    # 450 assigned code points are not ideographs but NFKC-fold into one, and
    # `stable_record_id` normalizes. A gate that tested the raw string called
    # ⼀ non-kanji while the minter stamped word:一:一 — the exact permanent,
    # uncorrectable id this function exists to prevent. Testing the normalized
    # form is what makes the two incapable of disagreeing.
    assert contains_kanji(value) is True
    assert normalize_identity_part(value) != value


def test_the_gate_and_the_id_minter_agree_by_construction() -> None:
    reading = "⼀"  # U+2F00, which nobody would call a reading if they saw it

    assert stable_record_id("一", reading) == "word:一:一"
    assert contains_kanji(reading) is True


@pytest.mark.parametrize(
    "value",
    ["", "はなす", "アリガトウ", "ありがとう、", "hanasu", "ー〜", "１２３"],
)
def test_kana_punctuation_and_latin_are_not_kanji(value: str) -> None:
    assert contains_kanji(value) is False


def test_kanji_is_found_anywhere_in_the_value() -> None:
    assert contains_kanji("お話しする") is True
    assert contains_kanji("しかる（𠮟る）") is True
