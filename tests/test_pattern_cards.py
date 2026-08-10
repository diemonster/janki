"""Cards for a rule — ``janki build`` on a `kind: pattern` deck.

The chart is inference, so the rule that reaches a card is the one janki checked
and agreed with. These tests are mostly about what is *refused*: an unreviewed
document, a document of the wrong kind, and a worked example janki disagrees
with.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.pattern_cards import (
    PatternDeckError,
    build_pattern_deck,
    cards_for,
)
from japanese_anki.io import DataError
from japanese_anki.patterns import Pattern, PatternSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSES = {"かう": "godan", "くる": "kuru", "する": "suru", "たべる": "ichidan"}


def chart(*patterns: Pattern, reviewed: bool = True, kind: str = "pattern") -> PatternSet:
    return PatternSet("teform.pdf", kind, "Te-form", tuple(patterns), reviewed)


# --- which rules become which cards -------------------------------------------


def test_a_rule_becomes_a_question_and_an_answer() -> None:
    cards = cards_for(chart(Pattern("う・つ・る → って", "godan て-form")), CLASSES)

    assert len(cards) == 1
    assert (cards[0].trigger, cards[0].result) == ("う・つ・る", "って")
    assert cards[0].gloss == "godan て-form"


def test_a_row_stating_two_rules_becomes_two_cards() -> None:
    """`くる → きて / する → して` is one row of the chart and two things to
    know, which is how they are drilled."""
    cards = cards_for(chart(Pattern("くる → きて / する → して", "the irregulars")), CLASSES)

    assert [(c.trigger, c.result) for c in cards] == [("くる", "きて"), ("する", "して")]


def test_a_rule_with_no_arrow_keeps_its_gloss_as_the_answer() -> None:
    """Stated in prose rather than as a transformation. Guessing where to cut it
    would invent a question the document does not ask."""
    cards = cards_for(chart(Pattern("Group II verbs are regular", "drop る, add て")))

    assert (cards[0].trigger, cards[0].result) == ("Group II verbs are regular", "")
    assert cards[0].gloss == "drop る, add て"


# --- only what janki agreed with ----------------------------------------------


def test_only_a_verified_example_reaches_a_card() -> None:
    """The chart is a model's reading of a page. An example janki disagrees with
    is one it has reason to think was mis-transcribed, and drilling it would
    teach the error."""
    cards = cards_for(
        chart(Pattern("う・つ・る → って", examples=("かう ⇨ かって", "かう ⇨ かいて"))),
        CLASSES,
    )

    assert cards[0].examples == ("かう ⇨ かって",)


def test_a_rule_with_nothing_checkable_still_ships() -> None:
    """`く → いて` names an ending, not a verb, so there is nothing to disagree
    with — and the rule is the thing being taught."""
    cards = cards_for(chart(Pattern("く → いて", "godan く")), CLASSES)

    assert len(cards) == 1 and cards[0].examples == ()


def test_each_rule_on_a_row_keeps_only_its_own_example() -> None:
    """Asking about くる while showing する ⇨ して is noise on the card."""
    cards = cards_for(
        chart(Pattern("くる → きて / する → して", examples=("くる ⇨ きて", "する ⇨ して"))),
        CLASSES,
    )

    assert cards[0].examples == ("くる ⇨ きて",)
    assert cards[1].examples == ("する ⇨ して",)


def test_a_rule_stated_by_ending_takes_the_rows_examples() -> None:
    """No example names `う・つ・る`, so the row's whole verified set is its."""
    cards = cards_for(
        chart(Pattern("う・つ・る → って", examples=("かう ⇨ かって",))), CLASSES
    )

    assert cards[0].examples == ("かう ⇨ かって",)


# --- what will not be built ----------------------------------------------------


def test_an_unreviewed_document_makes_no_cards() -> None:
    """Nothing inferred ships unread — the rule this project applies to every
    written-rather-than-looked-up thing."""
    with pytest.raises(PatternDeckError, match="has not been reviewed"):
        cards_for(chart(Pattern("う・つ・る → って"), reviewed=False))


def test_a_lesson_deck_states_no_rules_to_build() -> None:
    with pytest.raises(PatternDeckError, match="only pattern documents"):
        cards_for(chart(Pattern("〜んだ"), kind="lesson"))


# --- the package ---------------------------------------------------------------


def project(root: Path) -> None:
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()


def deck_file(root: Path, **overrides: object) -> Path:
    body = {
        "kind": "pattern",
        "name": "Te-form",
        "deck_id": 2059400113,
        "model_id": 1607392351,
        "document": "teform.pdf",
    }
    body.update(overrides)
    lines = "\n".join(f"  {key}: {value!r}" for key, value in body.items())
    path = root / "decks" / "teform.yaml"
    path.write_text(f"deck:\n{lines}\n", encoding="utf-8")
    return path


def test_the_package_carries_one_note_per_rule(tmp_path: Path) -> None:
    project(tmp_path)
    store = {"teform.pdf": chart(
        Pattern("う・つ・る → って", "godan"), Pattern("くる → きて / する → して", "irregular")
    )}

    target, count = build_pattern_deck(
        deck_file(tmp_path), ProjectConfig.load(tmp_path), store,
        tmp_path / "out.apkg", CLASSES,
    )

    assert count == 3, "two rules, one of which states two"
    assert target.exists()


def test_a_deck_naming_an_unread_document_says_which_are_known(tmp_path: Path) -> None:
    project(tmp_path)

    with pytest.raises(PatternDeckError, match="no document has been read"):
        build_pattern_deck(
            deck_file(tmp_path, document="nothing.pdf"),
            ProjectConfig.load(tmp_path), {}, tmp_path / "out.apkg",
        )


def test_a_pattern_deck_needs_a_document(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "decks" / "teform.yaml"
    path.write_text(
        "deck:\n  kind: pattern\n  name: T\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(PatternDeckError, match="needs 'document:'"):
        build_pattern_deck(path, ProjectConfig.load(tmp_path), {}, tmp_path / "o.apkg")


@pytest.mark.parametrize("key", ["deck_id", "model_id"], ids=["deck-id", "model-id"])
def test_an_identifier_that_is_not_an_integer_is_refused(tmp_path: Path, key: str) -> None:
    """YAML 1.1 again: `deck_id: yes` is `True`, and `int(True)` is 1 — a deck
    that quietly merges into whatever owns deck 1."""
    project(tmp_path)
    store = {"teform.pdf": chart(Pattern("う・つ・る → って"))}

    with pytest.raises(DataError, match=f"{key} must be an integer"):
        build_pattern_deck(
            deck_file(tmp_path, **{key: True}),
            ProjectConfig.load(tmp_path), store, tmp_path / "out.apkg",
        )


def test_the_guid_does_not_move_when_a_chart_is_corrected(tmp_path: Path) -> None:
    """A rule whose answer was mis-transcribed and later fixed is the same card
    with a corrected back. A GUID derived from the answer would orphan its
    review history on the day the deck got better."""
    first = cards_for(chart(Pattern("う・つ・る → いて", "godan")), CLASSES)
    fixed = cards_for(chart(Pattern("う・つ・る → って", "godan")), CLASSES)

    assert first[0].identity == fixed[0].identity
    assert first[0].result != fixed[0].result


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("う/つ/る → って", [("う/つ/る", "って")]),
        ("う・つ・る → って", [("う・つ・る", "って")]),
        ("くる → きて / する → して", [("くる", "きて"), ("する", "して")]),
        ("くる → きて、する → して", [("くる", "きて"), ("する", "して")]),
    ],
    ids=["slashed-triggers", "dotted-triggers", "two-rules-slash", "two-rules-comma"],
)
def test_a_separator_divides_rules_only_when_every_piece_has_an_arrow(
    template: str, expected: list[tuple[str, str]]
) -> None:
    """The same character does both jobs. `patterns.INSTRUCTIONS` asks the model
    to write a rule's triggers as `う/つ/る → って`, and a chart also puts two
    whole rules on one line. Splitting unconditionally turned the first into
    three cards — one of them drilling `る → って`, which is the *ichidan*
    ending and takes て. A card teaching an error is what this module exists
    not to ship."""
    cards = cards_for(chart(Pattern(template)), CLASSES)

    assert [(c.trigger, c.result) for c in cards] == expected
