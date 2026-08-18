"""Durable storage and reviewed use of patterns found during extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki.patterns import (
    Pattern,
    PatternError,
    PatternSet,
    format_patterns,
    load_store,
    reviewed_patterns,
    save_store,
    verb_pairs_in,
    with_prompt_provenance,
    worked_examples_in,
)

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


def test_the_block_is_labelled_pattern_data_not_hidden_instruction() -> None:
    block = format_patterns([Pattern("〜んだ", "explains background")])

    assert block.startswith("Reviewed lesson patterns:")
    assert "〜んだ" in block and "explains background" in block
    assert "prefer" not in block.lower()


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


def test_prompt_provenance_is_optional_and_round_trips(tmp_path: Path) -> None:
    entry = with_prompt_provenance(
        PatternSet(
            source="week11.pdf",
            kind="lesson",
            patterns=(Pattern("〜んだ", "explanation"),),
        ),
        {"model": "claude-opus-5", "response_schema_version": 3},
    )
    path = tmp_path / "patterns.json"

    save_store(path, {entry.source: entry})

    assert load_store(path)[entry.source] == entry
    historical = PatternSet.from_dict(
        "old.pdf", {"kind": "lesson", "patterns": []}
    )
    assert historical.prompt_provenance == {}


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
