"""Mechanical checks on an example sentence.

No network (IMPLEMENTATION_PLAN rule 6): the jpdb parse is built from the
committed capture and from canned token lists, never fetched.
"""

from __future__ import annotations

import json
import unicodedata
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
    repair_spilled_punctuation,
    stray_furigana_spaces,
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
    verdict = verify_example_furigana(example(japanese="話す", furigana="話[はな]す"), parse)

    assert verdict.verified
    assert bool(verdict) is True
    assert verdict.differences == ()


def test_a_wrong_reading_is_flagged_with_both_sides() -> None:
    parse = parse_of([["話", "はな"], "す"])

    verdict = verify_example_furigana(example(japanese="話す", furigana="話[か]す"), parse)

    assert not verdict
    assert verdict.differences[0] == "jpdb reads this as はなす; the furigana reads かす"
    assert verdict.expected == "話[はな]す"
    assert verdict.found == "話[か]す"


def test_a_finer_split_saying_the_same_thing_verifies() -> None:
    """jpdb returns furigana per *character*; an example is written per *word*,
    which is how a card is read. Comparing the group sequences pairwise failed
    on every multi-kanji compound — measured against a real import, 12 of 17
    correct examples were flagged and their audio suppressed, while the 5 that
    passed did so only because their words happened to be single kanji."""
    parse = parse_of([["週", "しゅう"], ["末", "まつ"]])

    verdict = verify_example_furigana(
        example(japanese="週末", furigana="週末[しゅうまつ]"), parse
    )

    assert verdict, verdict.differences


def test_a_reading_that_actually_differs_is_still_flagged() -> None:
    """The same split, a different reading: jpdb reads 日本語 as にっぽんご. Both
    are real, but a disagreement about the *sound* is what this exists for."""
    parse = parse_of([["日", "にっ"], ["本", "ぽん"], ["語", "ご"]])

    verdict = verify_example_furigana(
        example(japanese="日本語", furigana="日本語[にほんご]"), parse
    )

    assert not verdict
    assert verdict.differences[0] == (
        "jpdb reads this as にっぽんご; the furigana reads にほんご"
    )


def test_a_missing_space_still_fails_though_the_split_is_free() -> None:
    """The one grouping difference that is not free. `お茶[ちゃ]` puts ちゃ over
    both characters, so Anki renders the wrong ruby and `furigana_reading`
    yields ちゃ with the お gone — a difference in the *text* under the ruby,
    which the joined comparison still sees."""
    parse = parse_of(["お", ["茶", "ちゃ"]])

    verdict = verify_example_furigana(
        example(japanese="お茶", furigana="お茶[ちゃ]"), parse
    )

    assert not verdict
    assert "the furigana reads ちゃ" in verdict.differences[0], "the お is gone"


def test_missing_furigana_is_flagged_not_passed() -> None:
    parse = parse_of([["話", "はな"], "す"])

    verdict = verify_example_furigana(example(japanese="話す", furigana=""), parse)

    assert not verdict
    assert "the furigana reads 話す" in verdict.differences[0], (
        "the kanji passes through unread, which is the missing reading"
    )


def test_furigana_matching_a_kana_parse_verifies() -> None:
    """Kept as the inverse of its old self. The parse here is synthetic — a bare
    kana token for the sentence 猫 — and under a reading comparison it *agrees*:
    jpdb reads ねこ and the furigana says ねこ. There is nothing to flag."""
    verdict = verify_example_furigana(example(japanese="猫", furigana="猫[ねこ]"), parse_of("ねこ"))

    assert verdict, verdict.differences


def test_a_sentence_with_no_kanji_verifies_with_no_furigana() -> None:
    verdict = verify_example_furigana(
        example(japanese="ねこはかわいい。", furigana=""), parse_of("ねこ", "は", "かわいい")
    )

    assert verdict.verified


def test_punctuation_does_not_cause_a_false_mismatch() -> None:
    # jpdb does not tokenize a full stop, so comparing the rendered strings
    # would report a mismatch for one. The verdict is on the readings.
    parse = parse_of([["話", "はな"], "す"])

    verdict = verify_example_furigana(
        example(japanese="話す。", furigana="話[はな]す。"), parse
    )
    assert verdict.verified


def test_a_missing_space_fails_because_it_moves_the_reading() -> None:
    # お茶[ちゃ] puts ちゃ over both characters instead of over 茶: Anki renders
    # the wrong ruby, and the reading extracted for audio comes out as ちゃ
    # with the お simply gone.
    parse = parse_of(["お", ["茶", "ちゃ"]])

    verdict = verify_example_furigana(example(japanese="お茶", furigana="お茶[ちゃ]"), parse)

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

    from japanese_anki.qc import furigana_base

    verdict = verify_example_furigana(
        example(japanese=furigana_base(rendered), furigana=rendered), parse
    )

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


# --- the furigana has to describe *this* sentence -----------------------------


def test_a_furigana_field_that_rewrites_the_sentence_is_flagged() -> None:
    # The bug this closes: only the bracketed groups were compared, so a model
    # could change a particle inside the furigana field — the field that drives
    # sentence audio — and pass the check built to catch model invention.
    parse = parse_of([["本", "ほん"]], "を", [["読", "よ"], "む"])
    rewritten = ExampleSentence(japanese="本を読む。", furigana="本[ほん]が 読[よ]む。")

    verdict = verify_example_furigana(rewritten, parse)

    assert not verdict
    assert "the furigana spells 本が読む。, but the sentence is 本を読む。" in (
        verdict.differences[0]
    )


def test_okurigana_changed_inside_the_furigana_is_flagged() -> None:
    parse = parse_of([["話", "はな"], "します"])
    altered = ExampleSentence(japanese="話します。", furigana="話[はな]しました。")

    assert not verify_example_furigana(altered, parse)


def test_a_full_width_space_does_not_pass_as_a_separator() -> None:
    # Anki's furigana filter separates on the ASCII space alone, so お　茶[ちゃ]
    # renders ちゃ over both characters — the same wrong-ruby failure a missing
    # space causes, through a different character.
    parse = parse_of(["お", ["茶", "ちゃ"]])
    wide = ExampleSentence(japanese="お茶", furigana="お　茶[ちゃ]")

    assert not verify_example_furigana(wide, parse)


def test_kana_tokens_appear_in_the_rendering_shown_to_a_human() -> None:
    # jpdb sends null furigana for an all-kana token, so a rendering built from
    # furigana alone reads as though the dictionary dropped half the sentence.
    parse = jpdb.ParseResult(
        tokens=[
            {"vocabulary_index": 0, "furigana": [["話", "はな"], "す"]},
            {"vocabulary_index": 1, "furigana": None},
        ],
        vocabulary=[{"spelling": "話す"}, {"spelling": "を"}],
    )

    verdict = verify_example_furigana(
        ExampleSentence(japanese="話す", furigana="話[はな]す"), parse
    )

    assert "を" in verdict.expected


def test_a_segment_that_reads_as_itself_is_not_a_ruby_group() -> None:
    # The same collapse furigana_to_anki applies when janki writes these
    # fields; without it, furigana janki rendered from a parse could fail
    # verification against that very parse.
    parse = parse_of([["は", "は"], ["話", "はな"], ["す", ""]])

    assert parse_pairs(parse) == (("話", "はな"),)


def test_a_decomposed_dakuten_still_matches_its_composed_form() -> None:
    # conjugate returns NFKC forms while an example's text is only stripped, and
    # a failed contains-target check rejects the example outright — so the miss
    # would be a silent drop of a good sentence.
    decomposed = unicodedata.normalize("NFD", "食べた")
    assert decomposed != "食べた"  # the dakuten really is a separate character
    assert example_contains_target(
        ExampleSentence(japanese=f"昨日{decomposed}。"), "食べる", "ichidan"
    )


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


def test_a_decomposed_sentence_does_not_reject_correct_furigana() -> None:
    # Both sides render identically, so this failure would also have been
    # undiagnosable from the message.
    parse = parse_of([["食", "た"], "べた"])
    decomposed = ExampleSentence(
        japanese=unicodedata.normalize("NFD", "食べた。"), furigana="食[た]べた。"
    )

    assert verify_example_furigana(decomposed, parse).verified


def test_a_decomposed_headword_with_no_verb_class_still_matches() -> None:
    # conjugate normalizes internally, so the derived forms were already fine —
    # but a word with no verb class contributes only the headword, which is
    # most vocabulary.
    assert example_contains_target(
        ExampleSentence(japanese="かばんを買う。"), unicodedata.normalize("NFD", "かばん")
    )


def test_a_token_that_resolves_to_nothing_is_marked_not_dropped() -> None:
    parse = jpdb.ParseResult(
        tokens=[
            {"vocabulary_index": 0, "furigana": [["話", "はな"], "す"]},
            {"vocabulary_index": None, "furigana": None},
        ],
        vocabulary=[{"spelling": "話す"}],
    )

    verdict = verify_example_furigana(
        ExampleSentence(japanese="話す", furigana="話[か]す"), parse
    )

    assert "〈?〉" in verdict.expected


def test_a_decomposed_reading_does_not_reject_correct_furigana() -> None:
    # The message would have read "jpdb reads 語 as ご, not ご" — two strings
    # that render identically, so the rejection could not be diagnosed.
    parse = parse_of([["語", "ご"], ["学", "がく"]])
    decomposed = ExampleSentence(
        japanese="語学",
        furigana=f"語[{unicodedata.normalize('NFD', 'ご')}] 学[がく]",
    )

    assert verify_example_furigana(decomposed, parse).verified


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


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("週末[しゅうまつ]、何[なに]するの？", "週末[しゅうまつ]、 何[なに]するの？"),
        (
            "風邪[かぜ]なの？ 薬[くすり]、飲[の]んだ？",
            "風邪[かぜ]なの？ 薬[くすり]、 飲[の]んだ？",
        ),
        ("今[いま]、何[なん]て 言[い]ったの？", "今[いま]、 何[なん]て 言[い]ったの？"),
    ],
    ids=["a-comma", "after-a-question-mark", "twice-in-one-sentence"],
)
def test_punctuation_a_group_swallowed_gets_the_separator_back(
    written: str, expected: str
) -> None:
    """Punctuation cannot belong to the annotated word and the word plainly
    starts after it, so the space goes between. Nothing about the reading or the
    segmentation is inferred — which is what makes this safe to run unattended
    on model output."""
    assert repair_spilled_punctuation(written) == expected


def test_a_spill_that_would_need_a_guess_is_left_alone() -> None:
    """`、妻と日本語[にほんご]` has swallowed a noun and a particle too. Deciding
    that にほんご annotates 日本語 rather than 妻と日本語 is choosing where the
    word begins, and this project does not guess a segmentation.

    Repairing it would also destroy the evidence: the run would then start with
    a Han character, which `spilled_furigana_groups` never flags, so a reported
    defect would become an unreportable one in a field that now looks tidy."""
    from japanese_anki.qc import spilled_furigana_groups

    # No space before 日本語 — that is the whole point. With one, `_GROUP` gives
    # the group the run 日本語 and there is no spill to leave alone, so the test
    # passed without ever reaching the guard it names.
    written = "毎日[まいにち]、妻と日本語[にほんご]を 話[はな]す"

    assert repair_spilled_punctuation(written) == written
    assert spilled_furigana_groups(written), "and it is still reported"


@pytest.mark.parametrize(
    "written",
    ["お茶[おちゃ]を 飲[の]む", "日[にっ]本[ぽん]", "話[はな]す 人[ひと]", "ご飯[ごはん]"],
    ids=["whole-word-ruby", "abutting-groups", "ordinary", "an-honorific"],
)
def test_correct_furigana_is_returned_unchanged(written: str) -> None:
    """Whole-word ruby and legitimately abutting groups must survive: `お 茶[おちゃ]`
    reads おおちゃ, so a repair that touched them would manufacture the defect."""
    assert repair_spilled_punctuation(written) == written


def test_the_repaired_field_no_longer_reports_a_spill() -> None:
    """The two functions have to agree, or `enrich` repairs something `validate`
    still condemns."""
    from japanese_anki.qc import spilled_furigana_groups

    repaired = repair_spilled_punctuation("週末[しゅうまつ]、何[なに]するの？")

    assert spilled_furigana_groups(repaired) == ()


def test_the_comma_survives_into_the_reading_once_repaired() -> None:
    """Which is the point: `furigana_reading` drops the space before a group, so
    an unrepaired field loses the comma from the romaji and the sentence audio
    as well as drawing the ruby wrongly."""
    repaired = repair_spilled_punctuation("週末[しゅうまつ]、何[なに]するの？")

    assert "、" in furigana_reading(repaired)


# --- spaces that are content --------------------------------------------------


def test_a_space_no_group_follows_is_reported() -> None:
    """In a furigana field a space means "the next group starts here".
    `furigana_reading` removes it only when a group follows, so this one lives
    on into the reading, the romaji and the audio, and draws on the card as a
    gap the plain sentence does not have."""
    assert stray_furigana_spaces("日本語[にほんご]の ニュースが 少[すこ]し 分[わ]かります。") == (
        "ニュースが",
    )


def test_notation_spaces_are_not_reported() -> None:
    assert stray_furigana_spaces("毎晩[まいばん]、 音楽[おんがく]を 聞[き]いて") == ()


def test_a_field_with_no_spaces_at_all_is_quiet() -> None:
    assert stray_furigana_spaces("日本語[にほんご]") == ()


def test_punctuation_followed_by_a_swallowed_particle_is_left_alone() -> None:
    """`、と日本語[にほんご]` has swallowed a particle as well as the comma.
    Inserting the separator after the comma alone gives `、 と日本語[にほんご]`,
    which is still a spill — it would rewrite the record without fixing it, and
    make the field look attended to. Left for a human, and still reported."""
    from japanese_anki.qc import spilled_furigana_groups

    written = "毎日[まいにち]、と日本語[にほんご]を 話[はな]す"

    assert repair_spilled_punctuation(written) == written
    assert spilled_furigana_groups(written), "and it stays flagged"


def test_a_space_the_sentence_itself_contains_is_not_reported() -> None:
    """`furigana_reading` keeps non-notation spaces on purpose, and this one is
    part of the text. Warning about it sends a reader to delete it, and the
    romaji becomes HelloWorld."""
    assert stray_furigana_spaces("「Hello World」と 言[い]った。") == ()


def test_two_spaces_in_a_row_name_the_word_after_them() -> None:
    """A doubled space is a plausible typo and exactly what this catches, but
    read one character at a time the first one's "following word" was the empty
    string — reported as '(end of field)' for a space nowhere near the end."""
    assert stray_furigana_spaces("日本語[にほんご]の  ニュースが 少[すこ]し") == ("ニュースが",)


def test_two_spaces_before_a_group_are_still_notation_gone_wrong() -> None:
    """One space before a group is the notation; two is not."""
    assert stray_furigana_spaces("毎晩[まいばん]、  音楽[おんがく]を") == ("音楽[おんがく]を",)


def test_a_stray_space_right_after_a_group_is_reported() -> None:
    """`語[ご] を` is the ordinary way a model mis-spaces the notation, and so
    the check's most common trigger. Classifying `]` as ASCII *content* silenced
    it for exactly the field the check was written for, and no test noticed:
    the existing cases put their stray space after a kana, or before a real
    group."""
    assert stray_furigana_spaces("私[わたし] は 学生[がくせい]です") == ("は",)


@pytest.mark.parametrize(
    ("written", "expected"),
    [("日本語[にほんご]の ", ("(end of field)",)), (" を 話[はな]す", ("を",))],
    ids=["trailing", "leading"],
)
def test_a_space_at_either_edge_of_the_field_is_reported(
    written: str, expected: tuple[str, ...]
) -> None:
    """The case that is *most* provably notation gone wrong: there is no next
    group for it to start. Treating an absent neighbour as content read that
    backwards and reported nothing."""
    assert stray_furigana_spaces(written) == expected


def test_doubled_spaces_inside_latin_content_stay_unreported() -> None:
    """`run > 1` used to short-circuit the content test, so the doubled-space
    typo inside quoted Latin was reported with a message that is false for it —
    and acting on it makes the romaji HelloWorld, the harm the single-space
    branch exists to avoid."""
    assert stray_furigana_spaces("「Hello  World」と 言[い]った。") == ()


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("いい 天気[てんき]ですね!  散歩[さんぽ]しましょう。", "散歩[さんぽ]しましょう。"),
        ("9  時[じ]に 起[お]きます。", "時[じ]に"),
    ],
    ids=["after-punctuation", "after-a-digit"],
)
def test_a_doubled_space_beside_ascii_that_is_not_a_word_is_still_reported(
    written: str, expected: str
) -> None:
    """At most one space can ever be notation, so a run of two is wrong wherever
    it is not inside Latin text. Requiring only one ASCII neighbour to suppress
    it silenced ASCII punctuation and digits, which a model writes as readily as
    it writes letters — and those runs are real defects."""
    assert stray_furigana_spaces(written) == (expected,)


@pytest.mark.parametrize(
    "written",
    ["iPhone を 使[つか]う", "と Twitter", '"ありがとう" と 言[い]った'],
    ids=["latin-then-japanese", "japanese-then-latin", "after-a-quote"],
)
def test_a_single_space_at_a_latin_boundary_stays_quiet(written: str) -> None:
    """A single space with ASCII on one side is the Latin↔Japanese boundary,
    where the sentence may well carry the space too — so the warning would tell
    a reader to delete something the card really does contain. Only a *run* uses
    the both-sides rule, because at most one space can ever be notation."""
    assert stray_furigana_spaces(written) == ()
