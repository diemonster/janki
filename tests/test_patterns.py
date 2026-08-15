"""What a document teaches — `janki patterns`.

Every test drives the client seam, so none reaches a model. The shapes are the
real ones: `teform_song.pdf` really is a rule chart with no lyrics, and
`104 Week 11 Slide.pdf` really teaches 〜の？, 〜んだ and つもり with no
vocabulary slide anywhere in it.
"""

from __future__ import annotations

import json
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
    verb_pairs_in,
    worked_examples_in,
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


def test_a_multi_verb_row_is_not_paired_positionally() -> None:
    """`かう・まつ・とる ⇨ かって・まって・とって` cannot be paired without
    deciding which result belongs to which verb, so the scan refuses the
    segment outright. Since M8.3 the scanned pairs ship onto rule cards
    verbatim — nothing downstream adjudicates them — so this refusal is the
    only thing standing between a multi-verb row and a fabricated
    とる ⇨ かって on a card the human never reviewed in that form."""
    assert verb_pairs_in("かう・まつ・とる ⇨ かって・まって・とって") == []


def test_an_ending_contrast_row_is_not_a_worked_example() -> None:
    """`って ⇨ んで` contrasts endings; neither side is a verb. The
    dictionary-ending filter is what keeps it off a card as a fake pair."""
    entry = PatternSet(
        "chart.pdf", "pattern", "て-form",
        patterns=(Pattern("って ⇨ んで", "voiced ending"),), reviewed=True,
    )

    assert worked_examples_in(entry) == {}


def test_a_mid_row_parenthetical_does_not_swallow_the_worked_example() -> None:
    """The scan strips parentheticals *before* pairing, and the row that
    depends on it is the mid-row annotation: `かう (exception) → かって`.
    Unstripped, `_PAIR` matches nothing across the parenthesis and the whole
    row's worked example vanishes silently — a chart the human reviewed ships
    a rule card with no example and no error. (A trailing `(to buy)` never
    reaches the pair regex at all, so it cannot pin this.)"""
    entry = PatternSet(
        "chart.pdf", "pattern", "て-form",
        patterns=(Pattern("かう (exception) → かって"),),
        reviewed=True,
    )

    assert worked_examples_in(entry) == {
        "かう (exception) → かって": [("かう", "かう ⇨ かって")]
    }
