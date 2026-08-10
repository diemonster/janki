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
    build_conjugation_deck,
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
        # A chart cell that compresses two rows, mixing both roles on one line.
        # Testing the whole separator set at once had no branch for it and sent
        # it down the *prose* path: one card whose question was the entire line
        # with an empty back, and the second rule never made a card at all.
        ("う・つ・る → って / く → いて", [("う・つ・る", "って"), ("く", "いて")]),
        # A row the model truncated, or a cell copied with its divider. `strip()`
        # removes whitespace only, so the answer read "いて /".
        ("く → いて /", [("く", "いて")]),
    ],
    ids=[
        "slashed-triggers", "dotted-triggers", "two-rules-slash", "two-rules-comma",
        "mixed-roles", "a-trailing-separator",
    ],
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


# --- the drill deck ------------------------------------------------------------


def verb(expression: str, reading: str, group: str, meanings: list[str] | None = None):
    from japanese_anki.models import VocabularyRecord

    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=meanings or ["to do something"],
        verb_group=group,
    )


def test_the_answer_is_computed_never_transcribed() -> None:
    """The same rules every vocabulary card is built with, so a drill card and
    its word card can never disagree — including on the exceptions, which is
    the whole reason a te-form deck exists."""
    from japanese_anki.exporters.pattern_cards import drill_cards

    cards = drill_cards(
        [verb("行く", "いく", "godan"), verb("来る", "くる", "kuru"),
         verb("食べる", "たべる", "ichidan")],
        "te_form",
    )

    assert [c.result for c, _ in cards] == ["行って", "来て", "食べて"]


def test_a_verb_janki_declines_produces_no_card() -> None:
    """ゆく's て-form is genuinely contested and `conjugate` refuses it; a class
    name janki does not know is refused too. This deck's whole value is that
    its answers are right, so a verb it cannot compute is left out rather than
    guessed at."""
    from japanese_anki.exporters.pattern_cards import drill_cards

    assert drill_cards([verb("ゆく", "ゆく", "godan")], "te_form") == []
    assert drill_cards([verb("食べる", "たべる", "一段活用")], "te_form") == []


def test_the_reading_rides_along_when_it_adds_something() -> None:
    """A kanji verb cannot be conjugated without its reading, and hiding it
    would make the card test the reading instead of the form. A kana verb
    already shows it."""
    from japanese_anki.exporters.pattern_cards import drill_cards

    kanji, _ = drill_cards([verb("買う", "かう", "godan")], "te_form")[0]
    kana, _ = drill_cards([verb("ある", "ある", "godan")], "te_form")[0]

    assert kanji.trigger == "買う（かう）"
    assert kana.trigger == "ある"


def test_another_form_can_be_drilled(tmp_path: Path) -> None:
    from japanese_anki.exporters.pattern_cards import drill_cards

    cards = drill_cards([verb("買う", "かう", "godan")], "past")

    assert cards[0][0].result == "買った"


def test_a_form_janki_does_not_compute_is_refused(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  form: polite\n  name: D\n"
        "  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(PatternDeckError, match="form must be one of"):
        build_conjugation_deck(
            path, ProjectConfig.load(tmp_path), [verb("買う", "かう", "godan")],
            tmp_path / "o.apkg",
        )


def test_the_drill_guid_survives_a_corrected_reading(tmp_path: Path) -> None:
    """Correcting a reading rewrites the front of the card. A GUID that moved
    with it would orphan the review history on the day the deck got better."""
    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    import genanki

    first = drill_guid(tmp_path, path, verb("買う", "かう", "godan"))
    fixed = drill_guid(tmp_path, path, verb("買う", "かう", "godan", ["to buy"]))

    assert first == fixed == genanki.guid_for("drill:te_form:word:買う:かう")


def drill_guid(root: Path, deck: Path, record) -> str:
    import genanki

    from japanese_anki.exporters.pattern_cards import drill_cards

    _card, record_id = drill_cards([record], "te_form")[0]
    return genanki.guid_for(f"drill:te_form:{record_id}")


def test_a_collection_with_no_conjugable_verb_says_so(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(PatternDeckError, match="no record janki can conjugate"):
        build_conjugation_deck(
            path, ProjectConfig.load(tmp_path), [verb("ゆく", "ゆく", "godan")],
            tmp_path / "o.apkg",
        )


# --- what the rest of janki sees ----------------------------------------------


def test_the_notetype_check_asks_what_a_pattern_build_writes(tmp_path: Path) -> None:
    """`deck_notetype` exists so the collection check asks the same question a
    build answers. Answering 27 for a deck that writes 6 made `janki status`
    warn "has 6 fields where this deck writes 27" on every run, advising a Merge
    Notetypes re-import that would fix nothing — and a false warning that never
    clears is how the detector guarding the real notetype-append invariant gets
    ignored."""
    from japanese_anki.exporters.anki import deck_notetype
    from japanese_anki.exporters.pattern_cards import FIELDS

    project(tmp_path)
    path = deck_file(tmp_path)

    model_id, name, fields = deck_notetype(path, ProjectConfig.load(tmp_path))

    assert (model_id, name, fields) == (1607392351, "Japanese Pattern", len(FIELDS))


def test_validate_refuses_a_pattern_deck_the_build_would(tmp_path: Path) -> None:
    """`janki validate` is the command whose job is catching a broken deck
    before a build, and it read a pattern deck as an ordinary one, found no
    records and called it clean."""
    from japanese_anki.exporters.pattern_cards import deck_problems

    project(tmp_path)
    store = {"teform.pdf": chart(Pattern("う・つ・る → って"))}

    assert deck_problems(deck_file(tmp_path), store) == []
    assert deck_problems(deck_file(tmp_path, document="typo.pdf"), store) == [
        "no document has been read under 'typo.pdf'. Known: teform.pdf"
    ]
    assert deck_problems(deck_file(tmp_path, deck_id=True), store)[0].startswith(
        "deck.deck_id must be an integer"
    )


def test_validate_refuses_an_unreviewed_document(tmp_path: Path) -> None:
    from japanese_anki.exporters.pattern_cards import deck_problems

    project(tmp_path)
    store = {"teform.pdf": chart(Pattern("う・つ・る → って"), reviewed=False)}

    assert deck_problems(deck_file(tmp_path), store) == [
        "teform.pdf has not been reviewed"
    ]


def test_validate_checks_a_drill_decks_form(tmp_path: Path) -> None:
    from japanese_anki.exporters.pattern_cards import deck_problems

    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  form: polite\n  name: D\n"
        "  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )

    assert deck_problems(path, {})[0].startswith("deck.form must be one of")


# --- the GUID -------------------------------------------------------------------


def test_the_guid_ignores_the_gloss() -> None:
    """The gloss is model-extracted prose — rewritten by any re-extraction and
    routinely shortened by hand during `janki patterns --review`. In the
    identity it made a rebuild duplicate the card and strand its history, which
    is the failure the deterministic-GUID rule exists to prevent."""
    long_gloss = cards_for(
        chart(Pattern("う・つ・る → って", "godan verbs ending in う, つ, or る take って")),
        CLASSES,
    )
    shortened = cards_for(chart(Pattern("う・つ・る → って", "godan う/つ/る → って")), CLASSES)

    assert long_gloss[0].identity == shortened[0].identity


def test_two_rules_sharing_a_trigger_are_refused(tmp_path: Path) -> None:
    """They would collide on a GUID and silently drop a card. Adding prose to
    tell them apart is exactly what made the GUID unstable."""
    project(tmp_path)
    store = {"teform.pdf": chart(
        Pattern("く → いて", "godan"), Pattern("く → いた", "past")
    )}

    with pytest.raises(PatternDeckError, match="more than one rule for く"):
        build_pattern_deck(
            deck_file(tmp_path), ProjectConfig.load(tmp_path), store,
            tmp_path / "o.apkg", CLASSES,
        )
