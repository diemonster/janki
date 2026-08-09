"""Golden tables for :mod:`japanese_anki.conjugation`.

Every form asserted here is written out in full rather than derived, because a
test that rebuilds the answer the way the module builds it proves only that the
module is self-consistent. A wrong conjugation is the worst bug this project
can ship — it goes onto a card and gets memorised — so the assertions are the
hand-checked forms, and the module has to match them.

Three coverage tests at the bottom (``test_every_godan_ending_...`` and
friends) fail when a table or exception list grows an entry no case above
pins, so a new hand-written exception cannot land unproven.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki.conjugation import (
    ADJECTIVE_FORMS,
    CONJUGATION_FORMS,
    conjugate,
    conjugate_i_adjective,
    polite_stem,
)
from japanese_anki.jpdb import GODAN, ICHIDAN, KURU, SURU

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_RECORDS = PROJECT_ROOT / "data" / "normalized" / "vocabulary.json"


def table(*forms: str) -> dict[str, str]:
    """A full seven-form table, named positionally in :data:`CONJUGATION_FORMS` order."""
    return dict(zip(CONJUGATION_FORMS, forms, strict=True))


# --------------------------------------------------------------------------
# The key set
# --------------------------------------------------------------------------


def test_the_key_set_and_its_order_are_what_the_stored_records_already_carry() -> None:
    # The exporter labels each row from the dict's own keys in iteration order,
    # so a generated table has to key and order itself like a hand-typed one or
    # the cards change shape. This is that contract, read off the shipped data.
    #
    # An *ordered subset*, not equality: this module omits a form a verb does
    # not have rather than inventing one — ある has no standard potential or
    # passive, because ありえる is a separate lexeme — so demanding all seven
    # keys asserted the opposite of what the module documents. It passed only
    # while the file happened to hold three regular verbs, and failed the first
    # time real vocabulary arrived.
    records = json.loads(CURATED_RECORDS.read_text(encoding="utf-8"))
    stored = [tuple(record["conjugations"]) for record in records if record.get("conjugations")]
    assert stored, "the stored records are the fixture for this test; they cannot be empty"
    assert any(keys == CONJUGATION_FORMS for keys in stored), "at least one full table"
    for keys in stored:
        assert set(keys) <= set(CONJUGATION_FORMS), f"unknown form in {keys}"
        order = [form for form in CONJUGATION_FORMS if form in set(keys)]
        assert list(keys) == order, f"{keys} is not in the canonical order"


def test_the_curated_records_are_reproduced_form_for_form() -> None:
    # The three hand-written verbs in data/normalized/vocabulary.json were typed
    # by a human before this module existed. Regenerating them is the strongest
    # golden test available: it is an answer key nobody wrote for the code.
    records = json.loads(CURATED_RECORDS.read_text(encoding="utf-8"))
    checked = 0
    for record in records:
        if not record.get("conjugations") or not record.get("verb_group"):
            continue
        assert (
            conjugate(record["expression"], record["reading"], record["verb_group"])
            == record["conjugations"]
        ), record["expression"]
        checked += 1
    assert checked >= 3


def test_an_adjective_has_the_first_five_forms_and_no_potential_or_passive() -> None:
    assert CONJUGATION_FORMS[:5] == ADJECTIVE_FORMS
    assert "potential" not in ADJECTIVE_FORMS
    assert "passive" not in ADJECTIVE_FORMS


# --------------------------------------------------------------------------
# Godan: one verb per ending
# --------------------------------------------------------------------------

GODAN_GOLDEN: list[tuple[str, str, dict[str, str]]] = [
    # う — the あ-row stem is わ, not あ. 買あない is the classic bug.
    ("買う", "かう",
     table("買う", "買わない", "買った", "買わなかった", "買って", "買える", "買われる")),
    ("待つ", "まつ",
     table("待つ", "待たない", "待った", "待たなかった", "待って", "待てる", "待たれる")),
    ("帰る", "かえる",
     table("帰る", "帰らない", "帰った", "帰らなかった", "帰って", "帰れる", "帰られる")),
    ("飲む", "のむ",
     table("飲む", "飲まない", "飲んだ", "飲まなかった", "飲んで", "飲める", "飲まれる")),
    ("遊ぶ", "あそぶ",
     table("遊ぶ", "遊ばない", "遊んだ", "遊ばなかった", "遊んで", "遊べる", "遊ばれる")),
    # ぬ — the only one, and its past negative doubles the な: 死ななかった.
    ("死ぬ", "しぬ",
     table("死ぬ", "死なない", "死んだ", "死ななかった", "死んで", "死ねる", "死なれる")),
    ("書く", "かく",
     table("書く", "書かない", "書いた", "書かなかった", "書いて", "書ける", "書かれる")),
    ("泳ぐ", "およぐ",
     table("泳ぐ", "泳がない", "泳いだ", "泳がなかった", "泳いで", "泳げる", "泳がれる")),
    ("話す", "はなす",
     table("話す", "話さない", "話した", "話さなかった", "話して", "話せる", "話される")),
]


@pytest.mark.parametrize(("expression", "reading", "expected"), GODAN_GOLDEN)
def test_each_godan_ending_conjugates_its_whole_row(
    expression: str, reading: str, expected: dict[str, str]
) -> None:
    assert conjugate(expression, reading, GODAN) == expected


def test_the_te_form_splits_five_ways_across_the_nine_godan_endings() -> None:
    # The single reason the godan endings are a table and not a vowel shift.
    te_forms = {
        expression[-1]: expected["te_form"][-2:] for expression, _, expected in GODAN_GOLDEN
    }
    assert te_forms == {
        "う": "って",
        "つ": "って",
        "る": "って",
        "む": "んで",
        "ぶ": "んで",
        "ぬ": "んで",
        "く": "いて",
        "ぐ": "いで",
        "す": "して",
    }


def test_every_godan_ending_in_the_table_has_a_golden_verb() -> None:
    from japanese_anki.conjugation import _GODAN_ENDINGS

    assert {expression[-1] for expression, _, _ in GODAN_GOLDEN} == set(_GODAN_ENDINGS)


# --------------------------------------------------------------------------
# Ichidan
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reading", "expected"),
    [
        (
            "食べる",
            "たべる",
            table(
                "食べる", "食べない", "食べた", "食べなかった", "食べて", "食べられる", "食べられる"
            ),
        ),
        # 上一段 as well as 下一段, and a two-character word where the stem is
        # a single kanji: 見る → 見ない, not 見らない.
        ("見る", "みる",
         table("見る", "見ない", "見た", "見なかった", "見て", "見られる", "見られる")),
    ],
)
def test_ichidan_drops_the_ru_and_takes_rareru_for_both_potential_and_passive(
    expression: str, reading: str, expected: dict[str, str]
) -> None:
    assert conjugate(expression, reading, ICHIDAN) == expected
    assert expected["potential"] == expected["passive"]


def test_the_verb_group_is_the_only_thing_that_separates_kaeru_from_taberu() -> None:
    # 帰る and 食べる end in the same kana and inflect differently. Nothing in
    # the spelling says which, which is why an unknown group must refuse.
    assert conjugate("帰る", "かえる", GODAN)["negative"] == "帰らない"
    assert conjugate("食べる", "たべる", ICHIDAN)["negative"] == "食べない"


# --------------------------------------------------------------------------
# する and くる
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reading", "expected"),
    [
        # する on its own: the potential is できる, a different word entirely.
        ("する", "する", table("する", "しない", "した", "しなかった", "して", "できる", "される")),
        (
            "勉強する",
            "べんきょうする",
            table(
                "勉強する",
                "勉強しない",
                "勉強した",
                "勉強しなかった",
                "勉強して",
                "勉強できる",
                "勉強される",
            ),
        ),
        (
            "コピーする",
            "こぴーする",
            table(
                "コピーする",
                "コピーしない",
                "コピーした",
                "コピーしなかった",
                "コピーして",
                "コピーできる",
                "コピーされる",
            ),
        ),
    ],
)
def test_suru_compounds_keep_the_stem_and_conjugate_the_suru(
    expression: str, reading: str, expected: dict[str, str]
) -> None:
    assert conjugate(expression, reading, SURU) == expected


@pytest.mark.parametrize("expression", ["愛する", "察する", "関する"])
def test_a_single_character_suru_stem_is_refused_because_it_may_follow_the_su_pattern(
    expression: str,
) -> None:
    # 愛する → 愛せる, never 愛できる, and nothing mechanical separates 愛 from
    # 勉強. The whole single-character class is declined rather than guessed.
    assert conjugate(expression, "", SURU) == {}


@pytest.mark.parametrize(
    ("expression", "reading", "expected"),
    [
        # The kanji spelling hides three reading changes (来る/来ない/来た is
        # くる/こない/きた) — exactly what a card should show.
        ("来る", "くる",
         table("来る", "来ない", "来た", "来なかった", "来て", "来られる", "来られる")),
        ("くる", "くる",
         table("くる", "こない", "きた", "こなかった", "きて", "こられる", "こられる")),
        (
            "持ってくる",
            "もってくる",
            table(
                "持ってくる",
                "持ってこない",
                "持ってきた",
                "持ってこなかった",
                "持ってきて",
                "持ってこられる",
                "持ってこられる",
            ),
        ),
        (
            "連れて来る",
            "つれてくる",
            table(
                "連れて来る",
                "連れて来ない",
                "連れて来た",
                "連れて来なかった",
                "連れて来て",
                "連れて来られる",
                "連れて来られる",
            ),
        ),
    ],
)
def test_kuru_conjugates_per_spelling_and_carries_its_prefix(
    expression: str, reading: str, expected: dict[str, str]
) -> None:
    assert conjugate(expression, reading, KURU) == expected


# --------------------------------------------------------------------------
# The hand-written exceptions — one test each, or they are unpinned
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reading", "te_form", "past"),
    [
        ("行く", "いく", "行って", "行った"),  # never 行いて
        ("いく", "いく", "いって", "いった"),
        ("逝く", "いく", "逝って", "逝った"),
        ("出て行く", "でていく", "出て行って", "出て行った"),  # matched as a suffix
        ("連れていく", "つれていく", "連れていって", "連れていった"),
        # The う-verbs that keep the classical euphonic form.
        ("問う", "とう", "問うて", "問うた"),
        ("請う", "こう", "請うて", "請うた"),
        ("乞う", "こう", "乞うて", "乞うた"),
    ],
)
def test_the_godan_te_form_exceptions(
    expression: str, reading: str, te_form: str, past: str
) -> None:
    forms = conjugate(expression, reading, GODAN)
    assert forms["te_form"] == te_form
    assert forms["past"] == past


def test_only_the_te_and_ta_forms_of_iku_are_irregular() -> None:
    assert conjugate("行く", "いく", GODAN) == table(
        "行く", "行かない", "行った", "行かなかった", "行って", "行ける", "行かれる"
    )
    assert conjugate("問う", "とう", GODAN) == table(
        "問う", "問わない", "問うた", "問わなかった", "問うて", "問える", "問われる"
    )


def test_aru_takes_a_different_word_for_its_negative() -> None:
    forms = conjugate("ある", "ある", GODAN)
    assert forms == {
        "plain": "ある",
        "negative": "ない",  # not あらない
        "past": "あった",
        "past_negative": "なかった",
        "te_form": "あって",
    }


def test_aru_omits_the_forms_it_does_not_have_rather_than_inventing_them() -> None:
    forms = conjugate("ある", "ある", GODAN)
    assert "potential" not in forms  # ありえる is a separate lexeme
    assert "passive" not in forms  # あられる is not used
    assert tuple(forms) == CONJUGATION_FORMS[: len(forms)]


@pytest.mark.parametrize(
    ("expression", "reading"),
    [
        ("ゆく", "ゆく"),  # ゆいて and 行って are both attested; pick neither
        ("行く", "ゆく"),  # the same problem spelled in kanji
        ("有る", "ある"),  # the negative is written 無い, not 有らない
        ("在る", "ある"),
        ("である", "である"),  # ではない, not でらない
        # jpdb maps JMDict v5uru to godan and that code names 得る: the
        # classical うる, whose negative is 得ない and never 得らない.
        ("得る", "うる"),
    ],
)
def test_the_expressions_with_no_safe_mechanical_answer_return_nothing(
    expression: str, reading: str
) -> None:
    assert conjugate(expression, reading, GODAN) == {}


def test_refusing_uru_does_not_refuse_the_ichidan_reading_of_the_same_spelling() -> None:
    assert conjugate("得る", "える", ICHIDAN)["negative"] == "得ない"


def test_a_compound_built_on_a_hand_written_verb_is_not_that_verb_plus_a_prefix() -> None:
    # 置いてある's negative is 置いてない. The regular godan table would produce
    # 置いてあらない, which is not Japanese, so the compound is refused.
    assert conjugate("置いてある", "おいてある", GODAN) == {}
    assert conjugate("書いてある", "かいてある", GODAN) == {}


def test_a_verb_ending_in_uru_that_really_is_godan_still_conjugates() -> None:
    # 売る shares 得る's reading and must not be caught by the refusal.
    assert conjugate("売る", "うる", GODAN) == table(
        "売る", "売らない", "売った", "売らなかった", "売って", "売れる", "売られる"
    )


# --------------------------------------------------------------------------
# The three ways this module says "I don't know"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verb_group", ["irregular", "", "   ", "verb", "godanish", "adjective"])
def test_a_verb_group_the_module_does_not_know_returns_nothing(verb_group: str) -> None:
    assert conjugate("食べる", "たべる", verb_group) == {}


@pytest.mark.parametrize(
    ("expression", "verb_group"),
    [
        ("高い", GODAN),  # い is not a godan ending
        ("きれい", GODAN),
        ("学校", GODAN),  # okurigana left off: the expression ends in kanji
        ("る", GODAN),  # an ending with no stem
        ("", GODAN),
        ("話す", ICHIDAN),  # ichidan verbs end in る
        ("食べる", SURU),  # not a する compound
        ("食べる", KURU),  # not a くる compound
        ("勉強", SURU),
    ],
)
def test_an_expression_that_cannot_belong_to_its_group_returns_nothing(
    expression: str, verb_group: str
) -> None:
    assert conjugate(expression, "", verb_group) == {}


@pytest.mark.parametrize(
    ("expression", "reading"),
    [
        ("話す", "はなした"),  # a conjugated reading against a plain expression
        ("食べる", "たべ"),  # a stem, not a dictionary form
        ("行く", "いった"),
    ],
)
def test_an_expression_and_reading_that_inflect_differently_return_nothing(
    expression: str, reading: str
) -> None:
    assert conjugate(expression, reading, GODAN) == {}
    assert conjugate(expression, reading, ICHIDAN) == {}


def test_an_absent_reading_is_not_a_disagreement() -> None:
    assert conjugate("話す", "", GODAN)["negative"] == "話さない"


# --------------------------------------------------------------------------
# い-adjectives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("高い", ("高い", "高くない", "高かった", "高くなかった", "高くて")),
        ("新しい", ("新しい", "新しくない", "新しかった", "新しくなかった", "新しくて")),
        # 良い spelled in kanji is regular and needs no exception entry.
        ("良い", ("良い", "良くない", "良かった", "良くなかった", "良くて")),
    ],
)
def test_an_i_adjective_swaps_its_final_i_for_ku(
    expression: str, expected: tuple[str, ...]
) -> None:
    assert conjugate_i_adjective(expression) == dict(zip(ADJECTIVE_FORMS, expected, strict=True))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("いい", ("いい", "よくない", "よかった", "よくなかった", "よくて")),
        ("かっこいい",
         ("かっこいい", "かっこよくない", "かっこよかった", "かっこよくなかった", "かっこよくて")),
        ("ちょうどいい",
         ("ちょうどいい", "ちょうどよくない", "ちょうどよかった", "ちょうどよくなかった",
          "ちょうどよくて")),
    ],
)
def test_ii_inflects_as_yoi_and_carries_its_prefix(
    expression: str, expected: tuple[str, ...]
) -> None:
    assert conjugate_i_adjective(expression) == dict(zip(ADJECTIVE_FORMS, expected, strict=True))


@pytest.mark.parametrize(
    ("expression", "reading"),
    [
        ("きれい", "きれい"),
        ("綺麗", "きれい"),
        ("奇麗", "きれい"),
        ("きらい", "きらい"),
        ("嫌い", "きらい"),
        ("大嫌い", "だいきらい"),  # the suffix match is what catches this one
        ("小綺麗", "こぎれい"),
    ],
)
def test_the_na_adjectives_that_end_in_i_are_refused(expression: str, reading: str) -> None:
    assert conjugate_i_adjective(expression, reading) == {}


@pytest.mark.parametrize("expression", ["高", "い", "", "静か", "食べる"])
def test_a_word_that_is_not_shaped_like_an_i_adjective_returns_nothing(expression: str) -> None:
    assert conjugate_i_adjective(expression) == {}


def test_an_i_adjective_reaches_the_same_table_through_conjugate() -> None:
    assert conjugate("高い", "たかい", "i-adjective") == conjugate_i_adjective("高い", "たかい")
    assert conjugate("高い", "たかい", "i-adjective")["negative"] == "高くない"


# --------------------------------------------------------------------------
# Housekeeping the callers depend on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verb_group", ["godan", "Godan", "  GODAN ", "u-verb", "五段", "group1"])
def test_the_group_name_is_read_leniently(verb_group: str) -> None:
    assert conjugate("話す", "はなす", verb_group)["negative"] == "話さない"


def test_every_alias_resolves_to_a_group_with_a_table() -> None:
    from japanese_anki.conjugation import _BUILDERS, _VERB_GROUP_ALIASES, I_ADJECTIVE

    for alias, group in _VERB_GROUP_ALIASES.items():
        assert group in _BUILDERS or group == I_ADJECTIVE, alias


def test_a_returned_table_is_fresh_so_a_caller_may_store_and_mutate_it() -> None:
    first = conjugate("ある", "ある", GODAN)
    first["negative"] = "あらない"
    assert conjugate("ある", "ある", GODAN)["negative"] == "ない"


@pytest.mark.parametrize(
    ("expression", "reading", "verb_group"),
    [
        ("買う", "かう", GODAN),
        ("食べる", "たべる", ICHIDAN),
        ("勉強する", "べんきょうする", SURU),
        ("来る", "くる", KURU),
        ("ある", "ある", GODAN),
    ],
)
def test_a_table_only_ever_holds_known_keys_in_the_known_order(
    expression: str, reading: str, verb_group: str
) -> None:
    keys = tuple(conjugate(expression, reading, verb_group))
    assert set(keys) <= set(CONJUGATION_FORMS)
    assert keys == tuple(form for form in CONJUGATION_FORMS if form in keys)


# --------------------------------------------------------------------------
# Coverage: an exception with no test above is an unpinned exception
# --------------------------------------------------------------------------


def test_the_honorific_verbs_take_an_i_row_polite_stem() -> None:
    """いらっしゃる is godan by class and い-row by inflection. The regular rule
    invents いらっしゃります and misses the form Genki teaches first — and a
    Shirabe export carries the kanji headword, so both spellings are listed."""
    assert polite_stem("いらっしゃる", "godan") == "いらっしゃい"
    assert polite_stem("くださる", "godan") == "ください"
    assert polite_stem("下さる", "godan") == "下さい"
    assert polite_stem("おっしゃる", "godan") == "おっしゃい"
    assert polite_stem("仰る", "godan") == "仰い"
    assert polite_stem("仰有る", "godan") == "仰有い"
    assert polite_stem("なさる", "godan") == "なさい"
    assert polite_stem("為さる", "godan") == "為さい"
    assert polite_stem("ござる", "godan") == "ござい"
    assert polite_stem("御座る", "godan") == "御座い"
    # 〜てくださる inflects on the honorific ending, so the suffix match covers it.
    assert polite_stem("読んでくださる", "godan") == "読んでください"
    # ください is already the stem, not a る verb to take one from.
    assert polite_stem("ください", "godan") == ""


def test_the_polite_stem_refuses_what_conjugate_refuses() -> None:
    """得る is うる, whose polite form is 得ます — never 得ります. The regular
    godan rule builds the second, so the same guard has to sit on both paths."""
    assert polite_stem("得る", "godan") == ""


def test_every_hand_written_exception_is_pinned_by_a_case_above() -> None:
    from japanese_anki.conjugation import (
        _GODAN_TE_OVERRIDES,
        _HONORIFIC_MASU_STEMS,
        _IRREGULAR,
        _IRREGULAR_ADJECTIVE_SUFFIXES,
        _KURU_FORMS,
        _NA_ADJECTIVE_SUFFIXES_ENDING_IN_I,
        _UNSAFE_GODAN_SUFFIXES,
        _UNSAFE_SUFFIXES,
    )

    tested_te_overrides = {"行く", "いく", "逝く", "問う", "請う", "乞う"}
    tested_unsafe = {"ゆく", "有る", "在る", "である"}
    tested_unsafe_godan = {"得る"}
    tested_irregular = {"ある"}
    tested_kuru = {"来る", "くる"}
    tested_adjective = {"いい"}
    tested_na_adjectives = {"きれい", "綺麗", "奇麗", "きらい", "嫌い"}
    tested_honorifics = {
        "いらっしゃる",
        "おっしゃる",
        "仰る",
        "仰有る",
        "くださる",
        "下さる",
        "なさる",
        "為さる",
        "ござる",
        "御座る",
    }

    assert set(_GODAN_TE_OVERRIDES) == tested_te_overrides
    assert set(_UNSAFE_SUFFIXES) == tested_unsafe
    assert set(_UNSAFE_GODAN_SUFFIXES) == tested_unsafe_godan
    assert set(_IRREGULAR) == tested_irregular
    assert set(_KURU_FORMS) == tested_kuru
    assert set(_IRREGULAR_ADJECTIVE_SUFFIXES) == tested_adjective
    assert set(_NA_ADJECTIVE_SUFFIXES_ENDING_IN_I) == tested_na_adjectives
    assert set(_HONORIFIC_MASU_STEMS) == tested_honorifics
