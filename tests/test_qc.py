"""Mechanical checks on an example sentence.

No network (IMPLEMENTATION_PLAN rule 6): the jpdb parse is built from the
committed capture and from canned token lists, never fetched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import jpdb
from japanese_anki.models import ExampleSentence
from japanese_anki.qc import (
    FuriganaVerdict,
    example_contains_target,
    furigana_pairs,
    furigana_reading,
    parse_pairs,
    regenerate_example_romaji,
    target_forms,
    verify_example_furigana,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jpdb-parse-sample.json"


def parse_of(*tokens: Any) -> jpdb.ParseResult:
    """A ParseResult carrying the given token furigana values."""
    return jpdb.ParseResult(
        tokens=[
            {"vocabulary_index": index, "furigana": value}
            for index, value in enumerate(tokens)
        ],
        vocabulary=[],
    )


def captured_parse() -> jpdb.ParseResult:
    """The real /parse capture, zipped the way the client zips it."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))["response"]
    rows = payload["tokens"][0]
    return jpdb.ParseResult(
        tokens=[dict(zip(jpdb.DEFAULT_TOKEN_FIELDS, row, strict=True)) for row in rows],
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


def test_the_dictionary_form_counts() -> None:
    assert example_contains_target(example(japanese="毎日話す。"), "話す", "godan")


@pytest.mark.parametrize(
    "sentence", ["話した", "話して", "話さない", "話さなかった", "話せる"]
)
def test_a_conjugated_form_counts(sentence: str) -> None:
    # A model asked for an example of 話す will usually inflect it, and a naive
    # substring test would reject every good sentence it wrote.
    assert example_contains_target(example(japanese=f"昨日{sentence}。"), "話す", "godan")


def test_a_sentence_about_a_different_word_is_rejected() -> None:
    # A fine sentence, but not an example of this word — and a card whose
    # sentence lacks its own headword teaches the wrong association.
    assert not example_contains_target(example(japanese="毎日言います。"), "話す", "godan")


def test_a_word_with_no_verb_class_has_to_appear_as_written() -> None:
    # The right answer rather than a guess: janki does not know how it inflects.
    assert example_contains_target(example(japanese="日本語を勉強する。"), "日本語")
    assert not example_contains_target(example(japanese="英語を勉強する。"), "日本語")


def test_the_furigana_is_not_searched() -> None:
    # It carries bracketed readings that would match text no reader sees.
    only_in_furigana = example(japanese="毎日勉強します。", furigana="話[はな]")

    assert not example_contains_target(only_in_furigana, "話す", "godan")


def test_an_empty_sentence_or_target_contains_nothing() -> None:
    assert not example_contains_target(example(japanese="  "), "話す", "godan")
    assert not example_contains_target(example(), "  ", "godan")


def test_target_forms_are_longest_first() -> None:
    # So a caller reporting which form matched names 話さなかった rather than
    # the 話す inside it.
    forms = target_forms("話す", "godan")

    assert forms[0] == max(forms, key=len)
    assert "話す" in forms and "話さなかった" in forms


def test_an_i_adjective_conjugates_through_its_part_of_speech() -> None:
    assert example_contains_target(example(japanese="とても高くない。"), "高い", "i-adjective")


# --- is the furigana the dictionary's? ---------------------------------------


def test_furigana_matching_the_parse_verifies() -> None:
    parse = parse_of([["話", "はな"], "す"])
    verdict = verify_example_furigana(example(furigana="話[はな]す"), parse)

    assert verdict.verified
    assert bool(verdict) is True
    assert verdict.differences == ()


def test_a_wrong_reading_is_flagged_with_both_sides() -> None:
    parse = parse_of([["話", "はな"], "す"])

    verdict = verify_example_furigana(example(furigana="話[か]す"), parse)

    assert not verdict
    assert verdict.differences == ("jpdb reads 話 as はな, not か",)
    assert verdict.expected == "話[はな]す"
    assert verdict.found == "話[か]す"


def test_a_different_segmentation_is_flagged() -> None:
    # Exactly what a model invents plausibly and wrongly, and what would go on
    # to drive sentence audio.
    parse = parse_of([["日", "にっ"], ["本", "ぽん"], ["語", "ご"]])

    verdict = verify_example_furigana(example(furigana="日本語[にほんご]"), parse)

    assert not verdict
    assert "jpdb splits 日 where this splits 日本語" in verdict.differences[0]


def test_missing_furigana_is_flagged_not_passed() -> None:
    parse = parse_of([["話", "はな"], "す"])

    verdict = verify_example_furigana(example(furigana=""), parse)

    assert not verdict
    assert "nothing here" in verdict.differences[0]


def test_furigana_the_parse_does_not_have_is_flagged() -> None:
    verdict = verify_example_furigana(example(furigana="猫[ねこ]"), parse_of("ねこ"))

    assert not verdict
    assert "is not in jpdb's reading" in verdict.differences[0]


def test_a_sentence_with_no_kanji_verifies_with_no_furigana() -> None:
    verdict = verify_example_furigana(
        example(japanese="ねこはかわいい。", furigana=""), parse_of("ねこ", "は", "かわいい")
    )

    assert verdict.verified


def test_punctuation_does_not_cause_a_false_mismatch() -> None:
    # jpdb does not tokenize a full stop, so comparing the rendered strings
    # would report a mismatch for one. The verdict is on the readings.
    parse = parse_of([["話", "はな"], "す"])

    assert verify_example_furigana(example(furigana="話[はな]す。"), parse).verified


def test_a_missing_space_fails_because_it_moves_the_reading() -> None:
    # お茶[ちゃ] puts ちゃ over both characters instead of over 茶: Anki renders
    # the wrong ruby, and the reading extracted for audio comes out as ちゃ
    # with the お simply gone.
    parse = parse_of(["お", ["茶", "ちゃ"]])

    verdict = verify_example_furigana(example(furigana="お茶[ちゃ]"), parse)

    assert not verdict
    assert verdict.expected == "お 茶[ちゃ]"
    assert verdict.found == "お茶[ちゃ]"
    assert furigana_reading("お茶[ちゃ]") == "ちゃ"
    assert furigana_reading("お 茶[ちゃ]") == "おちゃ"


def test_the_real_capture_verifies_against_its_own_furigana() -> None:
    # Against the committed live capture rather than a hand-written fixture:
    # jpdb segments 日本語 per kanji and reads it にっぽんご, which a
    # hand-written expectation would have quietly "corrected".
    parse = captured_parse()
    rendered = jpdb.furigana_to_anki(
        [
            segment
            for token in parse.tokens
            for segment in (token["furigana"] or [])
        ]
    )

    verdict = verify_example_furigana(example(furigana=rendered), parse)

    assert verdict.verified
    assert "日[にっ] 本[ぽん] 語[ご]" in rendered


def test_parse_pairs_ignores_tokens_with_no_furigana() -> None:
    # jpdb sends null for an all-kana token; it contributes no ruby group.
    assert parse_pairs(parse_of(None, [["話", "はな"], "す"], None)) == (("話", "はな"),)


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


def test_the_verdict_is_falsy_when_it_failed() -> None:
    assert not FuriganaVerdict(False, "a", "b")
    assert FuriganaVerdict(True, "a", "a")
