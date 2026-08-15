"""Mechanical checks on an example sentence.

No network (IMPLEMENTATION_PLAN rule 6): every check here is offline. The
example checks read a sentence against itself and against KANJIDIC since M7.6V
retired the dictionary oracle; the parse helpers that remain are notation
readers, driven from canned token lists rather than fetched.
"""

from __future__ import annotations

from typing import Any

import pytest

from japanese_anki import jpdb
from japanese_anki.models import ExampleSentence
from japanese_anki.qc import (
    furigana_pairs,
    furigana_reading,
    regenerate_example_romaji,
)


def parse_of(*tokens: Any) -> jpdb.ParseResult:
    """A ParseResult carrying the given token furigana values."""
    return jpdb.ParseResult(
        tokens=[
            {"vocabulary_index": index, "furigana": value}
            for index, value in enumerate(tokens)
        ],
        vocabulary=[],
    )


def example(**overrides: Any) -> ExampleSentence:
    values: dict[str, Any] = {
        "japanese": "毎日日本語を話します。",
        "furigana": "毎日[まいにち] 日本語[にほんご]を 話[はな]します。",
        "romaji": "",
        "english": "I speak Japanese every day.",
    }
    values.update(overrides)
    return ExampleSentence(**values)


# --- does the sentence use the word? -----------------------------------------


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("話[はな]す", (("話", "はな"),)),
        ("お 茶[ちゃ]", (("茶", "ちゃ"),)),
        ("日[にっ] 本[ぽん] 語[ご]", (("日", "にっ"), ("本", "ぽん"), ("語", "ご"))),
        ("ねこ", ()),
        ("", ()),
    ],
)
def test_furigana_pairs_reads_anki_notation(
    notation: str, expected: tuple[tuple[str, str], ...]
) -> None:
    assert furigana_pairs(notation) == expected


# --- romaji ------------------------------------------------------------------


def test_romaji_is_rebuilt_from_the_furigana() -> None:
    rebuilt = regenerate_example_romaji(
        example(furigana="毎日[まいにち] 日本語[にほんご]を 話[はな]します。")
    )

    assert rebuilt.romaji == "mainichinihongoohanashimasu."


def test_model_supplied_romaji_is_discarded_not_checked() -> None:
    # A wrong romaji is invisible to a learner who is reading it *because* they
    # cannot yet read the kana.
    rebuilt = regenerate_example_romaji(
        example(furigana="話[はな]す", romaji="totally wrong")
    )

    assert rebuilt.romaji == "hanasu"


def test_an_all_kana_sentence_needs_no_furigana() -> None:
    rebuilt = regenerate_example_romaji(
        example(japanese="ねこはかわいい。", furigana="", romaji="stale")
    )

    assert rebuilt.romaji == "nekohakawaii."


def test_kanji_with_no_furigana_yields_no_romaji_rather_than_a_guess() -> None:
    # Transliterating kanji is exactly the invention this function removes.
    rebuilt = regenerate_example_romaji(
        example(japanese="毎日話します。", furigana="", romaji="stale")
    )

    assert rebuilt.romaji == ""


def test_regenerating_changes_nothing_else() -> None:
    original = example(romaji="stale")

    rebuilt = regenerate_example_romaji(original)

    assert rebuilt.japanese == original.japanese
    assert rebuilt.furigana == original.furigana
    assert rebuilt.english == original.english
    # The input is left alone; the caller decides whether to keep the result.
    assert original.romaji == "stale"


@pytest.mark.parametrize(
    ("notation", "reading"),
    [
        ("話[はな]す", "はなす"),
        ("お 茶[ちゃ]", "おちゃ"),
        ("話[はな]すを 食[た]べる", "はなすをたべる"),
        ("ねこ", "ねこ"),
    ],
)
def test_furigana_reading_drops_the_notation_spaces(notation: str, reading: str) -> None:
    # The spaces Anki needs before a ruby group are notation, not sound.
    assert furigana_reading(notation) == reading


# --- the furigana has to describe *this* sentence -----------------------------


def test_per_kanji_furigana_keeps_its_sokuon() -> None:
    # The field's spaces are required notation, not word boundaries: jpdb
    # segments 日本語 per kanji, so treating them as boundaries splits one word
    # into three and deletes the っ, which has nothing to geminate at the end of
    # a run.
    rebuilt = regenerate_example_romaji(
        ExampleSentence(
            japanese="日本語を話す。",
            furigana="日[にっ] 本[ぽん] 語[ご]を 話[はな]す。",
        )
    )

    assert rebuilt.romaji == "nippongoohanasu."


def test_latin_text_in_a_sentence_keeps_its_spaces() -> None:
    # Only the space Anki's notation requires before a ruby group is notation;
    # a space between two ASCII words is content, and kana_to_romaji passes
    # Latin through verbatim.
    rebuilt = regenerate_example_romaji(
        ExampleSentence(
            japanese="「Hello World」と言った。",
            furigana="「Hello World」と 言[い]った。",
        )
    )

    assert "Hello World" in rebuilt.romaji


def test_a_full_width_space_outside_a_ruby_group_is_content() -> None:
    # Only the ASCII space Anki's notation requires before a group is dropped.
    # A full-width space someone typed between two runs is content, and
    # kana_to_romaji renders it as a separator.
    rebuilt = regenerate_example_romaji(
        ExampleSentence(japanese="話す　よ", furigana="話[はな]す　よ")
    )

    assert furigana_reading("話[はな]す　よ") == "はなす　よ"
    assert rebuilt.romaji == "hanasu yo"


def test_the_docstrings_romaji_examples_are_what_the_code_returns() -> None:
    # These values are the design record for this function, and both were
    # wrong once: nippongoo was carried over from a sentence where を supplied
    # the extra o.
    assert regenerate_example_romaji(
        ExampleSentence(japanese="日本語", furigana="日[にっ] 本[ぽん] 語[ご]")
    ).romaji == "nippongo"
    # A typed space immediately before a ruby group is indistinguishable from
    # notation and goes with it.
    assert furigana_reading("本を 食[た]べる") == "本をたべる"


# --- the separator space ------------------------------------------------------


