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
    check_pattern_rules,
    extract_patterns,
    format_patterns,
    load_store,
    reviewed_patterns,
    save_store,
)


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

    assert checks == ()
