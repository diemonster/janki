"""Conjugation tables: the rule-computed half of DESIGN_V2's division of labor.

Conjugation is a rule, not a fact and not a judgment, so it belongs in plain
code: no tokens, no API key, no network, no dependency. Every form here is a
table lookup plus string concatenation.

The contract that shapes the whole module is **unknown or irregular returns
an empty dict**. A conjugation janki declines to produce leaves the record's
``conjugations`` field empty, which ``janki status`` counts and a human fills
in; a conjugation janki gets wrong ships onto a flashcard and is memorised.
So every table below is either exact for the whole class it covers or absent,
and the four ways this module says "I don't know" are all the same way:

* the verb group is not one it knows (``irregular``, ``""``, a typo),
* the expression does not end the way that group must end (a godan verb whose
  dictionary form ends in ``い``, an expression that ends in kanji because the
  okurigana was left off),
* the expression and the reading do not inflect alike (``話す`` / ``はなした``
  are two different words, whichever one the record meant),
* the expression is on one of the small hand-written exception lists — either
  a form the rule would get wrong (``有る`` → not ``有らない``, ``得る`` read
  ``うる`` → not ``得らない``) or one where usage is genuinely split (``ゆく``
  → ``ゆいて`` and ``行って`` are both attested; pick neither) — or it is a
  compound built on a verb that has one (``置いてある`` is ``置いてない``, not
  ``置いてあらない``).

Where an exception *does* have one exact answer it is written out as data
rather than refused: ``ある``'s negative is ``ない``, ``行く``'s te-form is
``行って``, ``問う``'s is ``問うて``. Forms that do not exist at all (``ある``
has no standard potential or passive) are omitted from the dict rather than
invented — callers iterate whatever keys they get, so a partial table renders
fine and claims nothing false.

The key set is fixed by what already ships: the curated records in
``data/normalized/vocabulary.json`` carry ``plain``, ``negative``, ``past``,
``past_negative``, ``te_form``, ``potential``, ``passive``, in that order. The
note template renders the whole table as one ``{{Conjugations}}`` blob and the
exporter labels each row from the dict's own keys in iteration order, so these
tables are built in the order the cards read and generated values sit beside
hand-typed ones without a diff.

Forms are built from the **expression**, not the reading: cards show
``行った``, not ``いった``. The reading is used to reject a record whose two
halves disagree, and to catch exceptions that are visible only in kana.
"""

from __future__ import annotations

from japanese_anki.identifiers import normalize_identity_part
from japanese_anki.jpdb import GODAN, ICHIDAN, KURU, SURU

# Not a verb group, but the value ``pos_to_part_of_speech`` gives an い-adjective
# and the one a caller holding a single "how does this inflect" field will pass.
I_ADJECTIVE = "i-adjective"

# The rendered key set and its order (see the module docstring). Exported so
# callers and tests name the forms once.
CONJUGATION_FORMS: tuple[str, ...] = (
    "plain",
    "negative",
    "past",
    "past_negative",
    "te_form",
    "potential",
    "passive",
)

# い-adjectives take the first five and have no potential or passive.
ADJECTIVE_FORMS: tuple[str, ...] = CONJUGATION_FORMS[:5]


# Godan endings. Per ending: the あ-row stem (negative and passive), the え-row
# stem (potential), and the て/た forms — which are the reason this is a table
# at all, since they split five ways across nine endings while everything else
# is a straight vowel shift.
#
# The う row's あ-row stem is わ, not あ: 買う → 買わない. It is the one cell
# here that a "shift the vowel" implementation gets wrong, and it is the most
# common ending in the language.
_GODAN_ENDINGS: dict[str, tuple[str, str, str, str]] = {
    "う": ("わ", "え", "って", "った"),
    "つ": ("た", "て", "って", "った"),
    "る": ("ら", "れ", "って", "った"),
    "む": ("ま", "め", "んで", "んだ"),
    "ぶ": ("ば", "べ", "んで", "んだ"),
    "ぬ": ("な", "ね", "んで", "んだ"),
    "く": ("か", "け", "いて", "いた"),
    "ぐ": ("が", "げ", "いで", "いだ"),
    "す": ("さ", "せ", "して", "した"),
}

# Godan verbs whose て/た forms alone are irregular; everything else about them
# follows the table above. Matched as a **suffix** so compounds come along:
# 出て行く → 出て行って, 連れていく → 連れていって.
_GODAN_TE_OVERRIDES: dict[str, tuple[str, str]] = {
    "行く": ("って", "った"),  # the famous one: 行って, never 行いて
    "いく": ("って", "った"),
    "逝く": ("って", "った"),
    # The う-verbs that keep the classical euphonic form: 問うて, not 問って.
    "問う": ("うて", "うた"),
    "請う": ("うて", "うた"),
    "乞う": ("うて", "うた"),
}

# Godan suffixes whose dictionary form is really a different verb wearing a
# godan ending. Checked only on the godan path, so the same spelling still
# conjugates when the caller says it is something else.
#
# 得る is the one that matters: jpdb maps JMDict's `v5uru` to ``godan``, and
# that code names exactly this word — the classical うる, whose negative is
# 得ない (えない) and never 得らない. Read える it is an ordinary ichidan verb
# and this list does not touch it.
_UNSAFE_GODAN_SUFFIXES: tuple[str, ...] = ("得る",)

# Whole verbs with a hand-written table, matched on the exact expression. A
# suffix match would swallow である and 〜てある, which are not this verb;
# compounds ending in one of these are refused instead (see :func:`conjugate`).
_IRREGULAR: dict[str, dict[str, str]] = {
    "ある": {
        "plain": "ある",
        "negative": "ない",  # not あらない — the negative is a different word
        "past": "あった",
        "past_negative": "なかった",
        "te_form": "あって",
        # No potential and no passive: ありえる is a separate lexeme and
        # あられる is not used. Omitted rather than invented.
    },
}

# Expressions where no mechanical answer is safe. Matched as a suffix against
# both the expression and the reading, so the kanji and kana spellings of the
# same problem are one entry.
_UNSAFE_SUFFIXES: tuple[str, ...] = (
    "ゆく",  # 行く read ゆく: ゆいて and 行って are both attested
    "有る",  # ある in kanji: the negative is written 無い, not 有らない
    "在る",
    "である",  # the copula wearing a る ending: ではない, not でらない
)

# する verbs that follow the ～す pattern instead: 愛する → 愛せる, never
# 愛できる. They are not mechanically separable from the regular compounds
# (愛 and 勉強 are both nouns), but they are essentially all single-character
# stems, so janki declines the whole single-character class rather than
# guessing per word. Two-character-and-longer compounds — 勉強する, 掃除する,
# コピーする — are uniformly regular.
_SURU_AMBIGUOUS_STEM_LENGTH = 1

# くる, per spelling. The kanji form hides the reading change (来た is きた),
# which is exactly what a card should show.
_KURU_FORMS: dict[str, tuple[str, ...]] = {
    "来る": ("来る", "来ない", "来た", "来なかった", "来て", "来られる", "来られる"),
    "くる": ("くる", "こない", "きた", "こなかった", "きて", "こられる", "こられる"),
}

# い-adjectives whose stem changes, matched as a suffix: かっこいい → かっこよくない.
# 良い spelled in kanji is regular (良くない) and needs no entry.
_IRREGULAR_ADJECTIVE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "いい": ("よくない", "よかった", "よくなかった", "よくて"),
}

# な-adjectives that end in い and would otherwise be conjugated as い-adjectives
# (嫌い → 嫌いくない). The caller decides part of speech; this list only stops
# the two or three that get miscategorised often enough to be worth naming.
# Matched as a suffix, so the intensified and prefixed spellings — 大嫌い,
# 小綺麗 — come along instead of falling through to the regular table.
_NA_ADJECTIVE_SUFFIXES_ENDING_IN_I: tuple[str, ...] = (
    "きれい",
    "綺麗",
    "奇麗",
    "きらい",
    "嫌い",
)

# Accepted spellings of each verb group, after casefolding and dropping spaces,
# hyphens and underscores. jpdb's POS mapping emits the four canonical names;
# the rest are what a human types into a deck file or a Shirabe column.
#
# The four canonical names are imported from :mod:`japanese_anki.jpdb`, which
# owns the JMDict→verb-group mapping that produces them; restating the literals
# here would be a second copy free to drift from the one that writes the field.
# "irregular" is deliberately absent: it names two different verbs.
_VERB_GROUP_ALIASES: dict[str, str] = {
    GODAN: GODAN,
    "godanverb": GODAN,
    "uverb": GODAN,
    "五段": GODAN,
    "五段動詞": GODAN,
    "class1": GODAN,
    "group1": GODAN,
    ICHIDAN: ICHIDAN,
    "ichidanverb": ICHIDAN,
    "ruverb": ICHIDAN,
    "一段": ICHIDAN,
    "上一段": ICHIDAN,
    "下一段": ICHIDAN,
    "class2": ICHIDAN,
    "group2": ICHIDAN,
    SURU: SURU,
    "suruverb": SURU,
    "する": SURU,
    "サ変": SURU,
    "サ行変格": SURU,
    KURU: KURU,
    "kuruverb": KURU,
    "くる": KURU,
    "来る": KURU,
    "カ変": KURU,
    "カ行変格": KURU,
    # An adjective is not a verb group, but a caller holding one field for
    # "how does this inflect" will put it here, and dispatching costs nothing.
    # (``i-adjective`` itself normalizes to ``iadjective``: the hyphen is noise.)
    "iadjective": I_ADJECTIVE,
    "iadj": I_ADJECTIVE,
    "adji": I_ADJECTIVE,
    "い形容詞": I_ADJECTIVE,
}

_ALIAS_NOISE = str.maketrans("", "", " \t-_・")


def _normalize_group(verb_group: str) -> str:
    return normalize_identity_part(verb_group).translate(_ALIAS_NOISE).casefold()


def _endings_disagree(expression: str, reading: str) -> bool:
    """Whether a non-empty reading inflects differently from the expression.

    Every group handled here inflects on the final kana, and a record whose
    expression and reading do not share it (話す / はなした, 食べる / たべ) is
    data janki cannot conjugate from — the two halves describe different words.
    """
    return bool(reading) and reading[-1] != expression[-1]


def _is_unsafe(expression: str, reading: str) -> bool:
    return any(
        expression.endswith(suffix) or reading.endswith(suffix)
        for suffix in _UNSAFE_SUFFIXES
    )


def _godan(expression: str) -> dict[str, str]:
    stem, ending = expression[:-1], expression[-1]
    rows = _GODAN_ENDINGS.get(ending)
    if rows is None or not stem:
        return {}
    if any(expression.endswith(suffix) for suffix in _UNSAFE_GODAN_SUFFIXES):
        return {}
    a_row, e_row, te_form, past = rows
    for suffix, override in _GODAN_TE_OVERRIDES.items():
        if expression.endswith(suffix):
            te_form, past = override
            break
    return {
        "plain": expression,
        "negative": f"{stem}{a_row}ない",
        "past": f"{stem}{past}",
        "past_negative": f"{stem}{a_row}なかった",
        "te_form": f"{stem}{te_form}",
        "potential": f"{stem}{e_row}る",
        "passive": f"{stem}{a_row}れる",
    }


def _ichidan(expression: str) -> dict[str, str]:
    stem = expression[:-1]
    if not expression.endswith("る") or not stem:
        return {}
    return {
        "plain": expression,
        "negative": f"{stem}ない",
        "past": f"{stem}た",
        "past_negative": f"{stem}なかった",
        "te_form": f"{stem}て",
        "potential": f"{stem}られる",
        "passive": f"{stem}られる",
    }


def _suru(expression: str) -> dict[str, str]:
    if not expression.endswith("する"):
        return {}
    stem = expression[:-2]
    if len(stem) == _SURU_AMBIGUOUS_STEM_LENGTH:
        return {}
    return {
        "plain": f"{stem}する",
        "negative": f"{stem}しない",
        "past": f"{stem}した",
        "past_negative": f"{stem}しなかった",
        "te_form": f"{stem}して",
        "potential": f"{stem}できる",
        "passive": f"{stem}される",
    }


def _kuru(expression: str) -> dict[str, str]:
    for tail, forms in _KURU_FORMS.items():
        if expression.endswith(tail):
            prefix = expression[: len(expression) - len(tail)]
            return {
                name: f"{prefix}{form}"
                for name, form in zip(CONJUGATION_FORMS, forms, strict=True)
            }
    return {}


_BUILDERS = {
    GODAN: _godan,
    ICHIDAN: _ichidan,
    SURU: _suru,
    KURU: _kuru,
}


def conjugate(expression: str, reading: str, verb_group: str) -> dict[str, str]:
    """The conjugation table for one word, or ``{}`` when janki is not sure.

    ``verb_group`` selects the rules — ``godan``, ``ichidan``, ``suru``,
    ``kuru`` (and ``i-adjective``, forwarded to
    :func:`conjugate_i_adjective`). Any other value, including ``irregular``
    and the empty string, returns ``{}``: the group is what tells 帰る (godan,
    帰らない) from 食べる (ichidan, 食べない), and nothing in the spelling does.

    Keys are :data:`CONJUGATION_FORMS`, in that order, minus any form the word
    does not have. Returned dicts are fresh, so callers may store and mutate
    them.
    """
    expression = normalize_identity_part(expression)
    reading = normalize_identity_part(reading)
    group = _VERB_GROUP_ALIASES.get(_normalize_group(verb_group))
    if group is None or not expression:
        return {}
    if group == I_ADJECTIVE:
        return conjugate_i_adjective(expression, reading)
    if _endings_disagree(expression, reading) or _is_unsafe(expression, reading):
        return {}
    if expression in _IRREGULAR:
        return dict(_IRREGULAR[expression])
    if any(expression.endswith(word) for word in _IRREGULAR):
        # A compound ending in a hand-written verb is not that verb plus a
        # prefix: 置いてある's negative is 置いてない, not 置いてあらない, and the
        # regular table would happily produce the second. One word has one
        # table; anything built on top of it is a human's call.
        return {}
    return _BUILDERS[group](expression)


def conjugate_i_adjective(expression: str, reading: str = "") -> dict[str, str]:
    """The い-adjective table for one word, or ``{}`` when janki is not sure.

    Keys are :data:`ADJECTIVE_FORMS`: an adjective has no potential or passive.
    Part of speech is the caller's call — this function only refuses the
    な-adjectives that end in い often enough to be worth naming (嫌い, 綺麗),
    since nothing in their spelling distinguishes them from 高い.
    """
    expression = normalize_identity_part(expression)
    reading = normalize_identity_part(reading)
    if not expression.endswith("い") or len(expression) < 2:
        return {}
    if any(
        expression.endswith(suffix) or reading.endswith(suffix)
        for suffix in _NA_ADJECTIVE_SUFFIXES_ENDING_IN_I
    ):
        return {}
    if _endings_disagree(expression, reading):
        return {}
    for suffix, forms in _IRREGULAR_ADJECTIVE_SUFFIXES.items():
        if expression.endswith(suffix):
            prefix = expression[: len(expression) - len(suffix)]
            inflected = (expression, *(f"{prefix}{form}" for form in forms))
            return dict(zip(ADJECTIVE_FORMS, inflected, strict=True))
    stem = expression[:-1]
    return {
        "plain": expression,
        "negative": f"{stem}くない",
        "past": f"{stem}かった",
        "past_negative": f"{stem}くなかった",
        "te_form": f"{stem}くて",
    }


# The い-row a godan verb's polite stem takes: 話す → 話し, 買う → 買い. Kept
# beside the あ/え-row table above rather than in a caller, because "which kana
# does this ending become" is this module's question wherever it is asked.
_GODAN_MASU_STEM: dict[str, str] = {
    "う": "い",
    "つ": "ち",
    "る": "り",
    "む": "み",
    "ぶ": "び",
    "ぬ": "に",
    "く": "き",
    "ぐ": "ぎ",
    "す": "し",
}

# The five honorific godan verbs whose polite stem is い-row, not り-row:
# いらっしゃいます, くださいます — never いらっしゃります. They are godan by
# class, so the table above would build the wrong form *and* miss the right
# one, and these are words a beginner textbook teaches early and politely.
# Matched as a suffix so 〜てくださる is covered too, since the honorific ending
# is what inflects there.
# Every spelling of each: a Shirabe export carries JMDict headwords, and four of
# these five verbs have kanji ones — おっしゃる has two. A record spelled 下さる
# would otherwise take the regular branch and build 下さり, which is exactly the
# form this table exists to prevent.
_HONORIFIC_MASU_STEMS: tuple[str, ...] = (
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
)


def polite_stem(expression: str, verb_group: str) -> str:
    """The stem ``ます`` attaches to, or ``""`` when janki cannot say.

    Not part of :data:`CONJUGATION_FORMS` and deliberately not stored on any
    record: this exists so a *check* can recognise 話します as 話す, which
    matters because the style guide asks for beginner examples and a beginner
    textbook teaches polite forms first. Adding it to the stored table would
    change every existing record's conjugations, which is a different decision
    from being able to match a sentence.

    ``""`` for anything janki has no class for, on the same principle as
    :func:`conjugate`: an unknown inflection is not guessed at.
    """
    expression = normalize_identity_part(expression)
    group = _VERB_GROUP_ALIASES.get(_normalize_group(verb_group))
    if not expression or group is None:
        return ""
    if group == GODAN:
        if any(expression.endswith(suffix) for suffix in _UNSAFE_GODAN_SUFFIXES):
            return ""
        for suffix in _HONORIFIC_MASU_STEMS:
            if expression.endswith(suffix):
                return f"{expression[: -len(suffix)]}{suffix[:-1]}い"
        stem, ending = expression[:-1], expression[-1]
        row = _GODAN_MASU_STEM.get(ending)
        return f"{stem}{row}" if stem and row else ""
    if group == ICHIDAN:
        return expression[:-1] if expression.endswith("る") and len(expression) > 1 else ""
    if group == SURU:
        # 勉強する → 勉強し; a bare する → し.
        return expression[:-2] + "し" if expression.endswith("する") else ""
    if group == KURU:
        for tail, stem in (("来る", "来"), ("くる", "き")):
            if expression.endswith(tail):
                return expression[: -len(tail)] + stem
    return ""
