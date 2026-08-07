import pytest

from japanese_anki.identifiers import contains_kanji, stable_record_id


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
    ],
)
def test_kanji_is_recognized_in_every_plane_it_lives_in(value: str) -> None:
    # Supplementary-plane kanji are the ones a BMP-only range misses, and
    # missing one does not lose a warning: it mints word:𠮟る:𠮟る, an id that
    # can never be corrected without orphaning Anki review history.
    assert contains_kanji(value) is True


@pytest.mark.parametrize(
    "value",
    ["", "はなす", "アリガトウ", "ありがとう、", "hanasu", "ー〜", "１２３"],
)
def test_kana_punctuation_and_latin_are_not_kanji(value: str) -> None:
    assert contains_kanji(value) is False


def test_kanji_is_found_anywhere_in_the_value() -> None:
    assert contains_kanji("お話しする") is True
    assert contains_kanji("しかる（𠮟る）") is True
