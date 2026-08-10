"""What a document teaches — `janki patterns`.

Every test drives the client seam, so none reaches a model. The shapes are the
real ones: `teform_song.pdf` really is a rule chart with no lyrics, and
`104 Week 11 Slide.pdf` really teaches 〜の？, 〜んだ and つもり with no
vocabulary slide anywhere in it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

# Imported as a module so tests can patch the `claude_client` object `patterns`
# actually holds. `tests/test_claude_client.py` deletes that module from
# `sys.modules` to test import behaviour, so a fresh `from japanese_anki import
# claude_client` here would return a *different* object and patching it would
# leave the real `parse_call` in place — which is a live API call from the test
# suite, and how this was found.
from japanese_anki import patterns as patterns_module
from japanese_anki.patterns import (
    Pattern,
    PatternError,
    PatternSet,
    extract_patterns,
    format_patterns,
    load_store,
    reviewed_patterns,
    save_store,
)
from japanese_anki.patterns import (
    check_pattern_rules as _check_pattern_rules,
)

#: The verb classes a real collection would hold for the words these tests use.
#: `check_pattern_rules` refuses to guess a class — a conjugation chart teaches
#: the outliers — so a test that supplies none is testing the hold-back path.
CLASSES: dict[str, str] = {
    "かう": "godan", "買う": "godan", "いく": "godan", "行く": "godan",
    "まつ": "godan", "とる": "godan", "のむ": "godan", "飲む": "godan",
    "およぐ": "godan", "泳ぐ": "godan", "はなす": "godan", "話す": "godan",
    "書く": "godan", "帰る": "godan", "だます": "godan", "済ます": "godan",
    "𠮟る": "godan", "ゆく": "godan",
    "たべる": "ichidan", "食べる": "ichidan", "おきる": "ichidan",
    "みる": "ichidan", "きる": "ichidan",
    "くる": "kuru", "来る": "kuru",
    "する": "suru", "勉強する": "suru",
}


def check_pattern_rules(entry: PatternSet, groups: dict[str, str] | None = None):
    """The real check, with a collection that knows these tests' verbs.

    Pass `{}` to exercise the no-class-on-record path deliberately.
    """
    return _check_pattern_rules(entry, CLASSES if groups is None else groups)


class FakeInput:
    """Stands in for a prepared PDF."""

    def __init__(self, name: str = "week11.pdf") -> None:
        self.origin_path = Path(name)

    def content_block(self) -> dict[str, Any]:
        return {"type": "document", "source": {"type": "base64", "data": ""}}


class Parsed:
    def __init__(self, kind: str, title: str, patterns: list[Any]) -> None:
        self.kind, self.title, self.patterns = kind, title, patterns


class Item:
    def __init__(self, template: str, gloss: str = "", examples: list[str] | None = None,
                 where: str = "") -> None:
        self.template, self.gloss = template, gloss
        self.examples, self.where = examples or [], where


def fake_call(parsed: Any, stop_reason: str = "end_turn", refusal: Any = None):
    """Replaces `claude_client.parse_call`."""
    from japanese_anki.claude_client import CallResult

    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        return CallResult(parsed=parsed, stop_reason=stop_reason, refusal=refusal)

    return call


# --- reading a document -----------------------------------------------------


def test_a_lesson_deck_yields_the_grammar_it_teaches(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = Parsed("lesson", "Week 11", [
        Item("〜の？", "casual explanatory question", ["どうしたの?"], "slides 5-14"),
        Item("ない form + つもり", "an intention not to do something", [], "slides 15-31"),
    ])
    monkeypatch.setattr(patterns_module.claude_client, "parse_call", fake_call(parsed))

    result = extract_patterns(FakeInput(), model="test-model")

    assert result.kind == "lesson"
    assert [p.template for p in result.patterns] == ["〜の？", "ない form + つもり"]
    assert result.patterns[0].examples == ("どうしたの?",)
    assert result.reviewed is False, "inferred, so nothing uses it yet"


def test_a_pattern_document_is_labelled_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A te-form chart contains almost no vocabulary and is entirely about a
    form. Reading it for its word list throws away what it was written for."""
    parsed = Parsed("pattern", "Te-form Song", [Item("う・つ・る → って", "godan て-form")])
    monkeypatch.setattr(patterns_module.claude_client, "parse_call", fake_call(parsed))

    assert extract_patterns(FakeInput("teform.pdf"), model="m").kind == "pattern"


def test_a_kind_the_model_invents_becomes_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        fake_call(Parsed("worksheet", "", [Item("〜たい")])),
    )

    assert extract_patterns(FakeInput(), model="m").kind == "unknown"


def test_a_pattern_with_no_template_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gloss with nothing to recognise it by cannot steer a sentence, and
    would sit in the store looking like content."""
    parsed = Parsed("lesson", "", [Item("", "something"), Item("〜んだ", "explains")])
    monkeypatch.setattr(patterns_module.claude_client, "parse_call", fake_call(parsed))

    assert [p.template for p in extract_patterns(FakeInput(), model="m").patterns] == ["〜んだ"]


def test_a_truncated_answer_is_refused_not_salvaged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cut-off answer looks exactly like a complete one with fewer patterns,
    and nothing downstream could tell the difference."""
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", fake_call(None, "max_tokens")
    )

    with pytest.raises(PatternError, match="max_tokens"):
        extract_patterns(FakeInput(), model="m")


def test_a_refusal_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        fake_call(None, "refusal", "declined"),
    )

    with pytest.raises(PatternError, match="declined"):
        extract_patterns(FakeInput(), model="m")


# --- nothing unreviewed steers a sentence -----------------------------------


def test_only_reviewed_documents_reach_a_prompt() -> None:
    """An unreviewed set is a model's reading of a slide deck nobody checked.
    Letting it steer every card would spread one bad inference across the whole
    collection."""
    store = {
        "week11.pdf": PatternSet("week11.pdf", "lesson", patterns=(Pattern("〜んだ"),),
                                 reviewed=True),
        "week12.pdf": PatternSet("week12.pdf", "lesson", patterns=(Pattern("〜ながら"),),
                                 reviewed=False),
    }

    assert [p.template for p in reviewed_patterns(store)] == ["〜んだ"]


def test_a_conjugation_chart_does_not_steer_sentences() -> None:
    """The te-form chart's rows are production rules, not sentence patterns.
    "Prefer `く → いて` when a sentence can use one naturally" is not a coherent
    instruction, and eight such rows drowned the six real lesson patterns while
    biasing every generated example toward the て-form."""
    store = {
        "teform.pdf": PatternSet("teform.pdf", "pattern", reviewed=True,
                                 patterns=(Pattern("く → いて"), Pattern("む・ぶ・ぬ → んで"))),
        "week11.pdf": PatternSet("week11.pdf", "lesson", patterns=(Pattern("〜んだ"),),
                                 reviewed=True),
    }

    assert [p.template for p in reviewed_patterns(store)] == ["〜んだ"]


def test_a_word_list_does_not_steer_sentences_either() -> None:
    store = {
        "words.pdf": PatternSet("words.pdf", "vocabulary", patterns=(Pattern("〜たい"),),
                                reviewed=True),
    }

    assert reviewed_patterns(store) == []


def test_the_steering_kinds_can_be_asked_for_explicitly() -> None:
    """The te-form chart is meant to become its own cards, so something has to
    be able to ask for it — just not the sentence writer."""
    store = {
        "teform.pdf": PatternSet("teform.pdf", "pattern", patterns=(Pattern("く → いて"),),
                                 reviewed=True),
    }

    assert [p.template for p in reviewed_patterns(store, kinds=("pattern",))] == ["く → いて"]


def test_a_named_document_narrows_it_further() -> None:
    store = {
        "a.pdf": PatternSet("a.pdf", "lesson", patterns=(Pattern("〜んだ"),), reviewed=True),
        "b.pdf": PatternSet("b.pdf", "lesson", patterns=(Pattern("〜つもり"),), reviewed=True),
    }

    assert [p.template for p in reviewed_patterns(store, ["b.pdf"])] == ["〜つもり"]


def test_no_reviewed_patterns_is_an_empty_block() -> None:
    """`enrich --ai` must behave exactly as before when nothing is reviewed —
    an empty instruction block, not an empty list rendered as one."""
    assert format_patterns([]) == ""


def test_the_block_says_the_patterns_are_a_preference() -> None:
    """A sentence forced into a pattern that does not suit the word is worse
    than one in ordinary Japanese."""
    block = format_patterns([Pattern("〜んだ", "explains background")])

    assert "〜んだ" in block and "explains background" in block
    assert "never force" in block


# --- the store --------------------------------------------------------------


def test_the_store_round_trips(tmp_path: Path) -> None:
    store = {"week11.pdf": PatternSet(
        source="week11.pdf", kind="lesson", title="Week 11", reviewed=True,
        patterns=(Pattern("〜の？", "casual question", ("どうしたの?",), "slide 5"),),
    )}
    path = tmp_path / "patterns.json"

    save_store(path, store)
    again = load_store(path)

    assert again["week11.pdf"] == store["week11.pdf"]


def test_a_missing_store_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_store(tmp_path / "nothing.json") == {}


def test_a_store_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "patterns.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(PatternError, match="keyed by document name"):
        load_store(path)


def test_an_entry_that_is_not_an_object_is_refused_not_skipped(tmp_path: Path) -> None:
    """Skipping it erased it. `save_store` rewrites the whole file from what was
    loaded, so an entry the loader quietly dropped was gone from the committed
    store on the next command that writes — reported as success."""
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps({"teform.pdf": None, "week11.pdf": {"kind": "lesson"}}),
        encoding="utf-8",
    )

    with pytest.raises(PatternError, match="teform.pdf"):
        load_store(path)


def test_a_pattern_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """`"〜んだ".get(...)` raises AttributeError, which the CLI does not catch
    and cannot format — a traceback rather than a message naming the file."""
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps({"week11.pdf": {"kind": "lesson", "patterns": ["〜んだ"]}}),
        encoding="utf-8",
    )

    with pytest.raises(PatternError, match="each pattern must be an object"):
        load_store(path)


def test_examples_given_as_a_string_are_refused(tmp_path: Path) -> None:
    """A bare string iterates one character at a time, so "どうしたの?" became
    ten single-character "examples", each looking like a sentence someone could
    check against the document."""
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps({"week11.pdf": {"patterns": [{"template": "〜の？",
                                                 "examples": "どうしたの?"}]}}),
        encoding="utf-8",
    )

    with pytest.raises(PatternError, match="examples must be a list"):
        load_store(path)


def test_the_store_is_written_sorted(tmp_path: Path) -> None:
    """So re-reading a document produces no diff by itself."""
    path = tmp_path / "patterns.json"
    save_store(path, {
        "b.pdf": PatternSet("b.pdf", "lesson"),
        "a.pdf": PatternSet("a.pdf", "lesson"),
    })

    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["a.pdf", "b.pdf"]


# --- a chart is checked, not believed ----------------------------------------


def chart(*patterns_in: Pattern) -> PatternSet:
    return PatternSet("teform.pdf", "pattern", patterns=tuple(patterns_in))


def test_a_worked_example_is_checked_against_janki_s_own_rules() -> None:
    """The chart shows its work and janki computes て-forms, so there is
    something to check against — the same dictionary-checks-writer shape the
    furigana path uses against jpdb."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → って", examples=("かう ⇨ かって (to buy)",)),
    ))

    assert len(checks) == 1
    assert checks[0].verb == "かう" and checks[0].claimed == "かって"
    assert checks[0].agrees and checks[0].group == "godan"


def test_a_rule_the_model_garbled_is_reported() -> None:
    """The point of checking. Nothing conjugates かう to かいて, so a chart
    claiming it disagrees with janki under every group."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → いて", examples=("かう ⇨ かいて",)),
    ))

    assert not checks[0].agrees
    assert any("godan: かって" in text for text in checks[0].computed)


def test_the_irregulars_and_the_iku_exception_are_checked_too() -> None:
    """いく ends in く but takes って, and it is the one thing on the page a
    reader is most likely to have copied down wrong."""
    checks = check_pattern_rules(chart(
        Pattern("いく → いって", examples=("いく is an exception, te-form いって",)),
        Pattern("くる → きて / する → して", examples=("くる ⇨ きて", "する ⇨ して")),
        Pattern("る-verb: 〜る → 〜て", examples=("ex. たべる ⇨ たべて (to eat)",)),
    ))

    assert {(c.verb, c.claimed, c.group) for c in checks} == {
        ("いく", "いって", "godan"),
        ("くる", "きて", "kuru"),
        ("する", "して", "suru"),
        ("たべる", "たべて", "ichidan"),
    }


def test_a_rule_shape_is_not_something_to_check() -> None:
    """`う・つ・る → って` is the shape of a rule, and `く → いて` names an
    ending. Neither is a verb `conjugate` can be asked about, and inventing one
    to test them with would be checking janki against itself."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → って"),
        Pattern("く → いて"),
        Pattern("る-verb: 〜る → 〜て"),
    ))

    assert checks == ()


def test_a_lesson_deck_has_nothing_to_check() -> None:
    """A lesson deck's "examples" are sentences from a slide, not conjugation
    claims — so the kind is what decides, not whether a sentence happens to
    contain an arrow. The fixture carries a pair that *would* be checked on a
    chart, because with a pattern that has no arrow the test passed without the
    kind gate existing at all."""
    lesson = PatternSet(
        "week11.pdf", "lesson",
        patterns=(Pattern("〜んだ", "explains", examples=("たべる ⇨ たべて",)),),
    )

    assert check_pattern_rules(lesson) == ()
    assert check_pattern_rules(replace(lesson, kind="pattern")), "the same rows on a chart"


def test_the_same_pair_written_twice_is_checked_once() -> None:
    """The chart repeats itself between its template and its examples."""
    checks = check_pattern_rules(chart(
        Pattern("かう ⇨ かって", examples=("かう ⇨ かって", "かう ⇨ かって (to buy)")),
    ))

    assert len(checks) == 1


def test_a_pair_that_is_not_a_dictionary_form_is_not_checked() -> None:
    """`って ⇨ んで` contrasts two endings; `かって ⇨ かった` relates two
    inflected forms. Neither is a "this verb's て-form is X" claim, and asking
    `conjugate` about かって — treating an inflected form as a dictionary form —
    would invent a disagreement out of a line the chart got right."""
    checks = check_pattern_rules(chart(
        Pattern("って ⇨ んで"),
        Pattern("past tense", examples=("かって ⇨ かった",)),
    ))

    assert checks == ()


def test_a_chart_written_in_kanji_is_checked() -> None:
    """Which is how a chart is actually written. A hiragana-only match made the
    whole feature a no-op on its ordinary input, and reported that as success —
    including 行く, the row a reader is most likely to have copied down wrong."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → って", examples=("買う ⇨ 買って",)),
        Pattern("行く ⇨ 行って"),
    ))

    assert {(c.verb, c.claimed, c.agrees) for c in checks} == {
        ("買う", "買って", True),
        ("行く", "行って", True),
    }


def test_a_ta_form_chart_is_checked_against_the_past_not_the_te_form() -> None:
    """Genki's た-form table repeats the て-form chart's rule shapes verbatim.
    A checker that only ever consulted `te_form` contradicted every correct row
    of it and exited non-zero — false alarms on right input, which is the one
    thing a checker must not do."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → った", examples=("かう ⇨ かった (to buy)",)),
    ))

    assert checks[0].agrees and checks[0].form == "past"


def test_a_nai_form_row_is_checked_too() -> None:
    checks = check_pattern_rules(chart(Pattern("ない form", examples=("かう ⇨ かわない",))))

    assert checks[0].agrees and checks[0].form == "negative"


def test_a_row_listing_several_verbs_is_not_paired_positionally() -> None:
    """`かう・まつ・とる ⇨ かって・まって・とって` is a correct row. The pattern
    matched the last verb against the first result — とる ⇨ かって — and reported
    the row as wrong. Deciding which result belongs to which verb is a guess."""
    checks = check_pattern_rules(chart(
        Pattern("かう・まつ・とる ⇨ かって・まって・とって"),
        Pattern("みる・きる → みて・きて"),
    ))

    assert checks == ()


def test_a_word_janki_declines_to_conjugate_is_not_a_disagreement() -> None:
    """`conjugate` refuses ゆく because both ゆいて and 行って are attested. janki
    having no opinion is not the chart being wrong, and reporting it as one
    failed a document that is right."""
    checks = check_pattern_rules(chart(Pattern("く → いて", examples=("ゆく ⇨ ゆいて",))))

    assert len(checks) == 1
    assert not checks[0].examined and not checks[0].agrees
    assert "no conjugation for this word" in checks[0].held_back


def test_a_form_janki_has_no_table_for_is_not_a_disagreement() -> None:
    """`CONJUGATION_FORMS` stops at seven, so a ます / たい / ば chart — a
    `pattern` document by the extractor's own definition — matched nothing and
    was reported as *wrong*, offering a て-form as the correction — 食 and 買 are
    both inside the old hand-written kanji range, so this was reachable for
    every kana and BMP-kanji chart, not only since the class was widened."""
    checks = check_pattern_rules(chart(
        Pattern("ます form", examples=("食べる ⇨ 食べます",)),
        Pattern("たい form", examples=("買う ⇨ 買いたい",)),
    ))

    # Held back, not dropped: neither agreement nor disagreement, and named in
    # the report so the row does not vanish under an all-clear.
    assert [c.examined for c in checks] == [False, False]
    assert not any(c.agrees for c in checks)


def test_several_complete_pairs_on_one_line_are_all_checked() -> None:
    """`かう ⇨ かって、まつ ⇨ まって` is two unambiguous claims. Requiring a
    separator on only one side discarded both, with no message — and if the
    document had one other row, --check then said "all 1 agree"."""
    checks = check_pattern_rules(chart(
        Pattern("かう ⇨ かって、まつ ⇨ まって"),
        Pattern("irregulars", examples=("くる ⇨ きて / する ⇨ して",)),
    ))

    assert {(c.verb, c.claimed) for c in checks} == {
        ("かう", "かって"), ("まつ", "まって"), ("くる", "きて"), ("する", "して"),
    }
    assert all(c.agrees for c in checks)


def test_a_spaced_out_list_row_is_still_not_paired_positionally() -> None:
    """The same row as the compact spelling. Looking at the immediately adjacent
    character saw a space and let まつ ⇨ かって through as a disagreement."""
    assert check_pattern_rules(chart(Pattern("かう ・ まつ ⇨ かって ・ まって"))) == ()


def test_which_form_a_row_matched_is_recorded() -> None:
    """This declines to parse which form a chart teaches, so it has to report
    what it found: `のむ ⇨ のんだ` on a て-form chart agrees — as a *past* — and
    without naming the form, the likeliest garble reads as a pass."""
    checks = check_pattern_rules(chart(
        Pattern("む・ぶ・ぬ → んで", examples=("のむ ⇨ のんだ",)),
    ))

    assert checks[0].agrees and checks[0].form == "past"


def test_a_decomposed_kanji_chart_is_normalized_before_matching() -> None:
    """A decomposed ぐ is く plus U+3099, which is outside the word class, so
    `_PAIR` cannot cross it to reach the arrow and the row matches nothing.

    Only the normalization is pinned here. 泳 is U+6CF3, inside the old
    hand-written `一-龥` too, so taking the kanji class from `identifiers`
    changed nothing for this row — that switch mattered for 𠮟 and Extension-A,
    which the test below covers."""
    import unicodedata

    checks = check_pattern_rules(chart(
        Pattern("ぐ → いで", examples=(unicodedata.normalize("NFD", "泳ぐ ⇨ 泳いで"),)),
    ))

    assert len(checks) == 1 and checks[0].agrees


def test_a_supplementary_plane_kanji_row_is_checked() -> None:
    """`_WORD` had `一-龥` written out by hand, which misses 𠮟 — a word real
    exports carry, and one `identifiers` names for exactly this reason — so the
    row matched nothing and was silently uncounted."""
    checks = check_pattern_rules(chart(Pattern("𠮟る ⇨ 𠮟って")))

    assert len(checks) == 1 and checks[0].agrees


def test_three_complete_pairs_on_one_line_are_all_checked() -> None:
    """The interior pair of an N≥3 line had a separator on both sides, so the
    both-sides guard dropped it — uncounted, unwarned, and not in the skipped
    list either, so a garbled middle row vanished under "all 2 agree"."""
    checks = check_pattern_rules(chart(
        Pattern("くる ⇨ きて / する ⇨ して / いく ⇨ いって"),
    ))

    assert {(c.verb, c.claimed) for c in checks} == {
        ("くる", "きて"), ("する", "して"), ("いく", "いって"),
    }
    assert all(c.agrees for c in checks)


@pytest.mark.parametrize(
    "written",
    ["かう・まつ ⇨ かって", "かう・まつ・とる ⇨ って"],
    ids=["a-dropped-result", "a-truncated-result"],
)
def test_an_asymmetric_list_row_is_still_not_paired(written: str) -> None:
    """What a model produces when it drops a result or a line breaks. The
    both-sides rule let まつ ⇨ かって through as a disagreement — janki's wrong
    pairing reported as the chart's error, which is a false alarm on right
    input."""
    assert check_pattern_rules(chart(Pattern(written))) == ()


def test_a_polite_past_chart_is_held_back_not_failed() -> None:
    """`食べました` ends in た, so reading the ending alone called it a *past* and
    the commonest polite chart there is was reported wrong. janki computes no
    polite form at all."""
    checks = check_pattern_rules(chart(
        Pattern("ます form", examples=("食べる ⇨ 食べました",)),
    ))

    assert len(checks) == 1
    assert not checks[0].examined and not checks[0].agrees


def test_a_row_janki_has_no_opinion_about_is_recorded_not_dropped() -> None:
    """A found-but-unexamined row used to disappear, so a chart with one
    readable row beside it reported "all 1 agree" and mentioned nothing else.
    Held back is a third outcome, and it has to reach the report."""
    checks = check_pattern_rules(chart(
        Pattern("て form", examples=("買う ⇨ 買って", "買う ⇨ 買います")),
    ))

    assert [(c.verb, c.claimed, c.examined) for c in checks] == [
        ("買う", "買って", True),
        ("買う", "買います", False),
    ]
    assert checks[1].held_back


def test_prose_or_a_gloss_beside_a_pair_does_not_discard_the_line() -> None:
    """`う, つ, る verbs: かう ⇨ かいて` is a verbatim chart row, and refusing
    the whole line because a segment lacked an arrow threw the garbled claim
    away entirely — reported as "all 1 agree", the vanishing-under-an-all-clear
    this function exists to prevent."""
    checks = check_pattern_rules(chart(
        Pattern("う・つ・る → って", examples=("う, つ, る verbs: かう ⇨ かいて",)),
        Pattern("て", examples=("買う ⇨ 買って, to buy",)),
    ), {"かう": "godan", "買う": "godan"})

    assert [(c.verb, c.claimed, c.agrees) for c in checks] == [
        ("かう", "かいて", False),
        ("買う", "買って", True),
    ]


def test_a_polite_form_of_a_masu_stem_verb_is_not_confused_with_a_garble() -> None:
    """`だます ⇨ だしまして` is a plausible transcription slip of だまして, which
    janki can disprove — but it ends in まして, so matching the polite endings
    against the whole claim excused it. The endings are matched against the tail
    beyond the verb's own stem now."""
    checks = check_pattern_rules(chart(
        Pattern("て", examples=("だます ⇨ だしまして",)),
    ))

    assert len(checks) == 1
    assert checks[0].examined and not checks[0].agrees


def test_a_real_polite_form_of_the_same_verb_is_still_held_back() -> None:
    """The other side of it: 済ましました really is a polite past, and janki
    computes no polite form at all."""
    checks = check_pattern_rules(chart(
        Pattern("ます", examples=("済ます ⇨ 済ましました",)),
    ))

    assert len(checks) == 1 and not checks[0].examined


def test_a_potential_claim_is_checked_rather_than_excused() -> None:
    """`CONJUGATION_FORMS` has seven entries and the ending table covered five,
    so a claim janki *could* judge was reported as one it had no opinion about —
    a false statement, reading as reassurance on a row it could have failed."""
    checks = check_pattern_rules(
        chart(Pattern("potential", examples=("書く ⇨ 書けれる",))), {"書く": "godan"}
    )

    assert len(checks) == 1
    assert checks[0].examined, "janki computes a potential; it has an opinion"
    # And names the potential, not the passive. Both 書ける and 書かれる end in
    # れる, so picking by suffix told the reader to write a passive onto a
    # potential row.
    assert checks[0].computed == ("godan: 書ける",)


def test_the_verbs_real_class_is_used_when_anything_knows_it() -> None:
    """`enrich --jpdb` records a verb_group from jpdb's own codes, so the class
    a chart states in English prose is already on disk. With it, the check is
    exact: 食べる is ichidan, its potential is 食べられる, and ら抜き is wrong."""
    checks = check_pattern_rules(
        chart(Pattern("potential", examples=("食べる ⇨ 食べれる",))),
        {"食べる": "ichidan"},
    )

    assert len(checks) == 1
    assert checks[0].examined and not checks[0].agrees
    # And names the potential it was reaching for. 食べる, 食べられる and 食べない
    # all share the prefix 食べ, so the dictionary form wins on prefix alone —
    # and "not what janki computes (ichidan: 食べる)" answers nothing.
    assert checks[0].computed == ("ichidan: 食べられる",)


def test_a_verb_with_no_class_on_record_is_held_back_not_guessed_at() -> None:
    """Trying every class to see if one fits gets both directions wrong. 食べる
    run through the *godan* rules gives 食べれる, so the ら抜き row would pass;
    and where janki has no override for an exception a chart is teaching, the
    regular rules would contradict a correct row. A conjugation chart exists
    because of those outliers, so an unknown class is no opinion."""
    checks = check_pattern_rules(
        chart(Pattern("potential", examples=("食べる ⇨ 食べれる",))), {}
    )

    assert len(checks) == 1
    assert not checks[0].examined and not checks[0].agrees
    assert "no verb class on record" in checks[0].held_back


def test_a_class_is_found_by_reading_as_well_as_spelling() -> None:
    """A chart writes its examples in kana — かう ⇨ かって — while the record is
    買う with reading かう."""
    checks = check_pattern_rules(
        chart(Pattern("う・つ・る → って", examples=("かう ⇨ かって",))),
        {"買う": "godan", "かう": "godan"},
    )

    assert checks[0].agrees and checks[0].examined


def test_a_wrong_class_on_record_makes_the_row_fail_rather_than_pass() -> None:
    """The cost of trusting the collection, stated: if the record is wrong the
    check inherits it. That is the right trade — the collection's verb_group is
    already what `conjugate` runs for every card janki builds, so a chart and a
    card now disagree loudly instead of the chart quietly agreeing with itself."""
    checks = check_pattern_rules(
        chart(Pattern("て", examples=("たべる ⇨ たべて",))), {"たべる": "godan"}
    )

    assert not checks[0].agrees


@pytest.mark.parametrize(
    "example",
    ["勉強する ⇨ 勉強しました", "くる ⇨ きました", "する ⇨ しました"],
    ids=["a-compound", "kuru", "suru"],
)
def test_a_polite_row_on_an_irregular_verb_is_held_back(example: str) -> None:
    """する's ます-stem is し, not す, so a claim testing against the dictionary
    stem shared no prefix and the guard silently did not apply — reporting the
    irregular rows of the commonest polite chart there is as wrong. Every
    Genki-style chart lists all three classes."""
    checks = check_pattern_rules(
        chart(Pattern("ます form", examples=(example,))),
        {"勉強する": "suru", "する": "suru", "くる": "kuru"},
    )

    assert len(checks) == 1 and not checks[0].examined


@pytest.mark.parametrize(
    "example",
    ["飲む ⇨ 飲まされる", "食べる ⇨ 食べさせられる", "書く ⇨ 書かせられる"],
    ids=["godan-causative-passive", "ichidan-causative-passive", "godan-rareru"],
)
def test_a_causative_row_is_held_back_not_contradicted(example: str) -> None:
    """`CONJUGATION_FORMS` has neither causative nor causative-passive, but
    adding られる/れる to the ending table gave `_claimed_form` an opinion about
    them anyway — so a correct 使役受身 chart was reported wrong on every row.

    Every case here ends in an ending `_FORM_BY_ENDING` *does* match, so each
    fails against the old code. `食べる ⇨ 食べさせる` was dropped from this list:
    せる matches nothing in that table either way, so it was already held back
    and pinned nothing."""
    checks = check_pattern_rules(
        chart(Pattern("使役受身", examples=(example,))),
        {"飲む": "godan", "食べる": "ichidan", "書く": "godan"},
    )

    assert len(checks) == 1 and not checks[0].examined
    # The reason, not just the absence of a verdict: three different paths set
    # `held_back`, and asserting only `not examined` cannot tell them apart.
    assert "no form with this ending" in checks[0].held_back


def test_a_disagreement_names_the_form_the_row_was_reaching_for() -> None:
    """Not the dictionary form, which answers a question nobody asked, and not
    whichever form shares the claim's last two kana."""
    checks = check_pattern_rules(
        chart(Pattern("て", examples=("帰る ⇨ 帰えて",))), {"帰る": "godan"}
    )

    assert checks[0].computed == ("godan: 帰って",)


def test_the_verbs_a_chart_names_can_be_listed_for_looking_up() -> None:
    """So a caller can fetch the classes before checking, using the same scan
    the check itself runs — two scans would disagree about which words matter."""
    from japanese_anki.patterns import chart_verbs

    verbs = chart_verbs(chart(
        Pattern("う・つ・る → って", examples=("かう ⇨ かって",)),
        Pattern("くる ⇨ きて / する ⇨ して"),
        Pattern("く → いて"),
    ))

    assert verbs == ["かう", "くる", "する"], "rule shapes name no verb"


def test_jpdb_answers_the_class_the_chart_states_in_prose() -> None:
    """One `/parse` call: its vocabulary entries already carry part_of_speech,
    and `pos_to_verb_group` already reads those codes."""
    from japanese_anki.patterns import verb_groups_from_jpdb

    class FakeClient:
        def __init__(self) -> None:
            self.asked = ""

        def parse(self, text: str):
            self.asked = text

            class Result:
                vocabulary = [
                    {"spelling": "おきる", "part_of_speech": ["vi", "v1"]},
                    {"spelling": "およぐ", "part_of_speech": ["vi", "v5", "v5g"]},
                    {"spelling": "あれ", "part_of_speech": ["pn"]},
                ]

            return Result()

    client = FakeClient()

    found = verb_groups_from_jpdb(["おきる", "およぐ", "あれ"], client)

    assert found == {"おきる": "ichidan", "およぐ": "godan"}, "no class, no entry"
    assert client.asked.count("。") == 3, "one call for the lot"


@pytest.mark.parametrize(
    ("example", "expected"),
    [("帰る ⇨ 反って", "godan: 帰って"), ("かう ⇨ 買って", "godan: かって")],
    ids=["an-ocr-misread", "mixed-orthography"],
)
def test_a_claim_sharing_no_prefix_still_gets_its_correction(
    example: str, expected: str
) -> None:
    """These are the garbles the checker exists for. Ranking candidates by
    shared prefix scored every form at zero and discarded the winner, so the
    correction vanished and the message read "no group applies" — false, since
    the class was known and produced a full table."""
    checks = check_pattern_rules(
        chart(Pattern("て", examples=(example,))), {"帰る": "godan", "かう": "godan"}
    )

    assert checks[0].computed == (expected,)


def test_a_lesson_decks_verbs_are_not_scraped_for_looking_up() -> None:
    """`check_pattern_rules` returns nothing for a lesson deck, so sending its
    words to the dictionary buys classes nothing will ever consult."""
    from japanese_anki.patterns import chart_verbs

    lesson = PatternSet(
        "week11.pdf", "lesson", patterns=(Pattern("〜んだ", examples=("かう ⇨ かって",)),)
    )

    assert chart_verbs(lesson) == []
    assert chart_verbs(replace(lesson, kind="pattern")) == ["かう"]
