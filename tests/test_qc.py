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
    settle_example_romaji,
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
    rebuilt, _rejected = settle_example_romaji(
        example(furigana="毎日[まいにち] 日本語[にほんご]を 話[はな]します。")
    )

    assert rebuilt.romaji == "mainichinihongoohanashimasu."


def test_a_supplied_romaji_that_says_something_else_is_replaced_and_named() -> None:
    """A wrong romaji is invisible to a learner reading it *because* they
    cannot yet read the kana — so it is replaced rather than kept, and the
    disagreement is reported rather than repaired quietly."""
    rebuilt, rejected = settle_example_romaji(
        example(furigana="話[はな]す", romaji="totally wrong")
    )

    assert rebuilt.romaji == "hanasu"
    assert "does not transliterate" in rejected
    assert "totally wrong" in rejected


def test_a_supplied_romaji_that_agrees_keeps_its_word_spacing() -> None:
    """The whole point of checking instead of rebuilding.

    `kyou wa osake nomanai no?` and `kyouhaosakenomanaino?` say the same thing;
    only the first is readable. Word spacing is segmentation, segmentation is
    parsing, and parsing is the model's job — janki's job is making sure the
    letters still say what the kana says.
    """
    rebuilt, rejected = settle_example_romaji(
        example(
            japanese="今日はおさけ飲まないの？",
            furigana="今日[きょう]はおさけ 飲[の]まないの？",
            romaji="kyou wa osake nomanai no?",
        )
    )

    assert rebuilt.romaji == "kyou wa osake nomanai no?"
    assert rejected == ""


def test_a_proper_nouns_capital_and_a_suffixs_hyphen_are_not_disagreements() -> None:
    """The prompt asks for both, so the verifier has to accept both.

    Kana records neither: やまだくん is the same five morae whether it is
    written `yamadakun` or `Yamada-kun`, and the second is what a reader
    wants. Rejecting it made the prompt ask for a spelling the checker refused
    — found by running the pass, not by reading it.
    """
    kept, rejected = settle_example_romaji(
        example(
            japanese="山田くん、歌上手なの？",
            furigana="山田[やまだ]くん、 歌[うた] 上手[じょうず]なの？",
            romaji="Yamada-kun, uta jouzu na no?",
        )
    )

    assert kept.romaji == "Yamada-kun, uta jouzu na no?"
    assert rejected == ""


def test_a_different_word_is_still_a_disagreement() -> None:
    """The separators and the case are the only slack. 上手 is not 下手."""
    replaced, rejected = settle_example_romaji(
        example(
            japanese="山田くん、歌上手なの？",
            furigana="山田[やまだ]くん、 歌[うた] 上手[じょうず]なの？",
            romaji="Yamada-kun, uta heta na no?",
        )
    )

    assert replaced.romaji == "yamadakun, utajouzunano?"
    assert "does not transliterate" in rejected


def test_a_particle_may_be_spelled_either_way_but_nothing_else_may() -> None:
    """は is `ha` or `wa` and へ is `he` or `e`, because which one it is needs
    the segmentation janki does not have. Every other letter has to agree."""
    kept, _ = settle_example_romaji(
        example(japanese="がっこうへいく。", furigana="", romaji="gakkou e iku.")
    )
    assert kept.romaji == "gakkou e iku.", "へ as the particle e"

    replaced, rejected = settle_example_romaji(
        example(japanese="がっこうへいく。", furigana="", romaji="gakkou e kuru.")
    )
    assert replaced.romaji == "gakkouheiku.", "a different verb is not a spelling"
    assert rejected


def test_an_all_kana_sentence_needs_no_furigana() -> None:
    rebuilt, _rejected = settle_example_romaji(
        example(japanese="ねこはかわいい。", furigana="", romaji="stale")
    )

    assert rebuilt.romaji == "nekohakawaii."


def test_kanji_with_no_furigana_yields_no_romaji_rather_than_a_guess() -> None:
    # Transliterating kanji is exactly the invention this function removes.
    rebuilt, _rejected = settle_example_romaji(
        example(japanese="毎日話します。", furigana="", romaji="stale")
    )

    assert rebuilt.romaji == ""


def test_regenerating_changes_nothing_else() -> None:
    original = example(romaji="stale")

    rebuilt, _rejected = settle_example_romaji(original)

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
    rebuilt, _rejected = settle_example_romaji(
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
    rebuilt, _rejected = settle_example_romaji(
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
    rebuilt, _rejected = settle_example_romaji(
        ExampleSentence(japanese="話す　よ", furigana="話[はな]す　よ")
    )

    assert furigana_reading("話[はな]す　よ") == "はなす　よ"
    assert rebuilt.romaji == "hanasu yo"


def test_the_docstrings_romaji_examples_are_what_the_code_returns() -> None:
    # These values are the design record for this function, and both were
    # wrong once: nippongoo was carried over from a sentence where を supplied
    # the extra o.
    settled, _rejected = settle_example_romaji(
        ExampleSentence(japanese="日本語", furigana="日[にっ] 本[ぽん] 語[ご]")
    )
    assert settled.romaji == "nippongo"
    # A typed space immediately before a ruby group is indistinguishable from
    # notation and goes with it.
    assert furigana_reading("本を 食[た]べる") == "本をたべる"


# --- the separator space ------------------------------------------------------




def test_a_romaji_that_cannot_be_checked_is_dropped_out_loud() -> None:
    """The third branch, and the only one that used to say nothing.

    A sentence with kanji and no furigana has no reading janki knows, so a
    supplied romaji cannot be verified and is not kept — trusting it would be
    trusting it for exactly the reason it cannot be trusted. But it *is* a
    loss, and the silent version meant a record could come back from the
    romaji pass with one sentence improved and another emptied, with no
    warning between them.
    """
    dropped, reason = settle_example_romaji(
        example(
            japanese="日本語を勉強します。",
            furigana="",
            romaji="nihongo o benkyou shimasu.",
        )
    )

    assert dropped.romaji == "", "not kept, because nothing can check it"
    assert "could not be checked" in reason
    assert "no furigana" in reason


def test_the_spoken_particle_spelling_is_accepted_only_in_the_shape_asked_for() -> None:
    """`は` may be `wa`, and `prompts/romaji.md` asks for particles as their own
    token — so the verifier accepts that spelling only in that shape.

    A check on the requested *format*, not on Japanese. Keyed on the kana
    alone, every は in the language could be spelled `wa`, so 花がきれい
    verified as `wana ga kirei` and reached the one reader who cannot check it
    against the kana.

    What it does not and cannot catch is pinned below: janki does not know
    which は is a particle, so a `wa` standing alone where `ha` was meant
    passes. Closing that needs a parse, and parsing is not janki's half of
    this project.
    """
    accepted, rejected = settle_example_romaji(
        example(japanese="きょうはおさけ。", furigana="", romaji="kyou wa osake.")
    )
    assert accepted.romaji == "kyou wa osake." and rejected == ""

    replaced, named = settle_example_romaji(
        example(japanese="はながきれい。", furigana="", romaji="wana ga kirei.")
    )
    assert replaced.romaji == "hanagakirei.", "wana is not はな"
    assert "does not transliterate" in named


def test_a_standalone_wa_that_should_be_ha_is_the_hole_that_stays_open() -> None:
    """Pinned so it is a known limit rather than a surprise.

    はは is a word, not two particles, and `wa wa` is the shape a particle
    takes — so it verifies. Every way of closing this needs to know which は
    is a particle, which is a parse. An earlier attempt refused a
    sentence-initial `wa` instead, reasoning that a particle attaches to what
    precedes it: true of Japanese, and grammar written into a Python file,
    which is the thing `DESIGN.md` says janki does not do.

    If this ever needs closing, close it in `prompts/romaji.md` — the side of
    the line that is allowed to know.
    """
    kept, rejected = settle_example_romaji(
        example(japanese="はは。", furigana="", romaji="wa wa.")
    )

    assert kept.romaji == "wa wa.", "accepted, and known to be"
    assert rejected == ""
