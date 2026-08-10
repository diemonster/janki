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
        # The trigger list and the rule divider using the *same* character —
        # which is the style INSTRUCTIONS actually asks for. No per-separator
        # test can split this; counting arrows can.
        ("う/つ/る → って / く → いて", [("う/つ/る", "って"), ("く", "いて")]),
        # Two different separators dividing rules on one line, because the model
        # is told to write it the way the page does.
        (
            "くる → きて / する → して、いく → いって",
            [("くる", "きて"), ("する", "して"), ("いく", "いって")],
        ),
        # A parenthetical carrying an arrow of its own, which `patterns.py`
        # documents as ordinary chart content.
        ("く → いて / ぐ → いで (voiced → で)", [("く", "いて"), ("ぐ", "いで")]),
        # A rule whose *result* is a list, followed by a second rule. An
        # arrow-less run was always attached forward, so `った` left the first
        # answer and became the second card's question: `った / く`. Which
        # separator introduced the run is what says where it belongs — `・` here
        # ties it back, `/` opens the next rule.
        (
            "う・つ・る → って・った / く → いて",
            [("う・つ・る", "って・った"), ("く", "いて")],
        ),
        (
            "かう・まつ ⇨ かって・まって / く → いて",
            [("かう・まつ", "かって・まって"), ("く", "いて")],
        ),
        # The same character on both sides of that run: the line gives no way to
        # tell a result list from the next rule's trigger list. One prose card
        # carrying the whole line, rather than a confident rule teaching the
        # wrong trigger — which is also the note's GUID.
        (
            "う/つ/る → って/った / く → いて",
            [("う/つ/る → って/った / く → いて", "")],
        ),
        # A quoted chart cell. The parenthetical is stripped to *count* arrows,
        # and letting the stripped text reach the output made an empty trigger —
        # a blank card, with the guid `pattern:<document>:`.
        ("（〜てもいい）", [("（〜てもいい）", "")]),
        # Same rule, for a trigger that is also the GUID: shortening it would
        # add a duplicate note on the next build of an already-shipped deck and
        # strand the original's review history.
        (
            "〜てもいいですか (asking permission)",
            [("〜てもいいですか (asking permission)", "")],
        ),
    ],
    ids=[
        "slashed-triggers", "dotted-triggers", "two-rules-slash", "two-rules-comma",
        "mixed-roles", "a-trailing-separator", "same-character-both-jobs",
        "two-different-dividers", "a-parenthetical-arrow",
        "a-result-list-before-a-second-rule", "a-kana-result-list-before-a-rule",
        "an-ambiguous-result-list", "a-parenthesised-cell",
        "a-parenthetical-on-a-prose-rule",
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


def verb(
    expression: str,
    reading: str,
    group: str,
    meanings: list[str] | None = None,
    record_id: str = "",
    part_of_speech: str = "",
):
    from japanese_anki.models import VocabularyRecord

    return VocabularyRecord(
        id=record_id or f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=meanings or ["to do something"],
        verb_group=group,
        part_of_speech=part_of_speech,
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


def guids_in(package: Path) -> list[str]:
    """The GUIDs a built package actually carries."""
    import json
    import sqlite3
    import tempfile
    from zipfile import ZipFile

    with ZipFile(package) as archive:
        name = (
            "collection.anki21"
            if "collection.anki21" in archive.namelist()
            else "collection.anki2"
        )
        db = Path(tempfile.mkdtemp()) / "c.db"
        db.write_bytes(archive.read(name))
    con = sqlite3.connect(db)
    guids = [row[0] for row in con.execute("select guid from notes order by id")]
    con.close()
    assert json  # keeps the import honest for readers of this helper
    return guids


def build_drill(tmp_path: Path, records: list, name: str) -> Path:
    if not (tmp_path / "janki.toml").exists():
        project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    target = tmp_path / name
    build_conjugation_deck(path, ProjectConfig.load(tmp_path), records, target)
    return target


def test_the_drill_guid_survives_a_corrected_reading(tmp_path: Path) -> None:
    """Correcting a reading rewrites the front of the card. A GUID that moved
    with it would orphan the review history on the day the deck got better.

    Read off the built notes, not recomputed from the production f-string: the
    helper this replaces re-typed that string, so mutating it left the test
    green. And the id is held fixed while the *reading* changes, which is the
    case the name describes — both calls used to pass the same reading."""
    # The id is the record's durable identity and does not move when a reading
    # is corrected — that is what `stable_record_id` is for — so the card built
    # from it must not move either.
    fixed = "word:買う:かう"
    before = verb("買う", "かう", "godan", record_id=fixed)
    # A *different reading*, so the front of the card really moves — こう rather
    # than かう. Holding the reading constant and changing only the meanings, as
    # this test first did, pinned gloss-independence and left the case in its
    # own name untested.
    after = verb("買う", "こう", "godan", record_id=fixed)

    first = guids_in(build_drill(tmp_path, [before], "a.apkg"))
    corrected = guids_in(build_drill(tmp_path, [after], "b.apkg"))

    assert first == corrected, "the reading moved; the identity did not"


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


def test_every_card_in_a_drill_deck_gets_its_own_guid(tmp_path: Path) -> None:
    """The other half, so the test above cannot pass by every card sharing one
    constant. Read off the built notes: comparing the `record_id` this helper
    returns compared two strings the *test* had just constructed, exercised no
    production identity logic at all, and left GUID uniqueness unpinned on a
    commit about note identity."""
    guids = guids_in(build_drill(
        tmp_path,
        [verb("買う", "かう", "godan"), verb("待つ", "まつ", "godan"),
         verb("読む", "よむ", "godan")],
        "many.apkg",
    ))

    assert len(guids) == 3
    assert len(set(guids)) == 3, "three cards, three identities"


def test_an_i_adjective_gets_a_drill_card() -> None:
    """jpdb has no verb class for one, so 高い carries its class in
    `part_of_speech` — which every other `conjugate` caller passes as a
    fallback. Without it 高い got no card while its word card rendered 高くて,
    the two disagreeing by omission."""
    from japanese_anki.exporters.pattern_cards import drill_cards

    cards = drill_cards(
        [verb("高い", "たかい", "", ["expensive"], part_of_speech="i-adjective")],
        "te_form",
    )

    assert [c.result for c, _ in cards] == ["高くて"]


@pytest.mark.parametrize(
    ("key", "written"),
    [("exclude_ids", '"word:買う:かう"'), ("include_ids", "3")],
    ids=["a-bare-string", "a-bare-number"],
)
def test_a_malformed_filter_is_refused_like_any_other_deck(
    tmp_path: Path, key: str, written: str
) -> None:
    """Written by hand, `exclude_ids:` as a bare string became a set of single
    characters and excluded nothing — so the record the user held back shipped
    onto a card, where a vocabulary deck with the identical typo is refused."""
    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        f"deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n"
        f"  {key}: {written}\n",
        encoding="utf-8",
    )

    with pytest.raises(DataError, match=f"deck.{key} must be a list"):
        build_conjugation_deck(
            path, ProjectConfig.load(tmp_path), [verb("買う", "かう", "godan")],
            tmp_path / "o.apkg",
        )


def test_an_empty_include_list_means_no_filter(tmp_path: Path) -> None:
    """The vocabulary path's semantics. Treating it as "include nothing" filtered
    every record out and blamed a missing verb_group."""
    project(tmp_path)
    path = tmp_path / "decks" / "drill.yaml"
    path.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n"
        "  include_ids: []\n",
        encoding="utf-8",
    )

    _target, count = build_conjugation_deck(
        path, ProjectConfig.load(tmp_path), [verb("買う", "かう", "godan")],
        tmp_path / "o.apkg",
    )

    assert count == 1


# --- through the CLI ------------------------------------------------------------


def cli_project(tmp_path: Path, records: list) -> Path:
    import json

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "decks" / "drill.yaml").write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    return tmp_path / "decks" / "drill.yaml"


def test_the_build_command_dispatches_on_the_new_kind(tmp_path: Path) -> None:
    """A new deck `kind:` is a new supported input format, and nothing reached
    the dispatch: every test called the builder directly, and `make gates`
    builds one vocabulary deck with `--output`."""
    from japanese_anki import cli

    deck = cli_project(tmp_path, [verb("買う", "かう", "godan")])

    code = cli.main([
        "--root", str(tmp_path), "build", str(deck),
        "--output", str(tmp_path / "out.apkg"),
    ])

    assert code == 0
    assert guids_in(tmp_path / "out.apkg"), "the package carries notes"


def test_a_drill_deck_refuses_only_new(tmp_path: Path, capsys) -> None:
    """It records no exports, so the flag cannot mean anything — and accepting
    it while doing a full rebuild reports a flag as honoured that never was."""
    from japanese_anki import cli

    deck = cli_project(tmp_path, [verb("買う", "かう", "godan")])

    assert cli.main(["--root", str(tmp_path), "build", str(deck), "--only-new"]) == 1
    assert "--only-new needs export history" in capsys.readouterr().err


def test_a_missing_collection_says_so(tmp_path: Path, capsys) -> None:
    """Not "a verb needs a verb_group janki knows", which sent the user to
    enrich a collection that is not there."""
    from japanese_anki import cli

    deck = cli_project(tmp_path, [])
    (tmp_path / "vocabulary.json").unlink()

    assert cli.main(["--root", str(tmp_path), "build", str(deck)]) == 1
    assert "no collection at" in capsys.readouterr().err


def test_a_drill_deck_honours_its_own_source(tmp_path: Path) -> None:
    """Every vocabulary deck resolves `source:` against its own directory, and
    this path read the project's collection and ignored the key — so a deck
    naming another collection silently drilled the wrong one."""
    import json

    from japanese_anki.config import ProjectConfig
    from japanese_anki.exporters.pattern_cards import collection_for

    project(tmp_path)
    (tmp_path / "other.json").write_text(
        json.dumps([verb("待つ", "まつ", "godan").to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    deck = tmp_path / "decks" / "drill.yaml"
    deck.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n"
        '  source: "../other.json"\n',
        encoding="utf-8",
    )

    assert collection_for(deck, ProjectConfig.load(tmp_path)).name == "other.json"


def test_a_drill_deck_passes_the_same_review_gate_a_word_deck_does(
    tmp_path: Path, capsys
) -> None:
    """A drill card carries the expression, the reading and a meaning straight
    off the record, so shipping one janki has not read is the thing the gate
    exists to stop. The `pattern` branch skips it because it ships no record
    content; this branch had no such excuse and returned before reaching it."""
    from japanese_anki import cli

    deck = cli_project(tmp_path, [verb("買う", "かう", "godan")])

    assert cli.main(["--root", str(tmp_path), "build", str(deck)]) == 1

    err = capsys.readouterr().err
    assert "not ready to ship" in err
    assert "janki review" in err


def test_the_gate_asks_only_about_records_the_deck_can_ship(tmp_path: Path) -> None:
    """A noun has no verb class, so no drill card could ever carry it, and a
    record `exclude_ids` holds back is one the deck has already declined.
    Gating the whole collection refused a valid build over both."""
    from japanese_anki.exporters.pattern_cards import shipping_records

    project(tmp_path)
    deck = tmp_path / "decks" / "drill.yaml"
    deck.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n"
        '  exclude_ids:\n  - "word:待つ:まつ"\n',
        encoding="utf-8",
    )
    noun = verb("猫", "ねこ", "")
    held_back = verb("待つ", "まつ", "godan")
    shipped = verb("買う", "かう", "godan")

    kept = shipping_records(deck, [noun, held_back, shipped])

    assert [r.id for r in kept] == ["word:買う:かう"]


def test_a_sweep_builds_a_drill_deck_that_cannot_narrow(tmp_path: Path, capsys) -> None:
    """`janki refresh` runs `build --all --only-new`, so refusing the flag
    outright stopped the documented pipeline at the first drill deck and the
    decks after it never built. On a sweep the flag is a mode, not an assertion
    about each deck."""
    import json

    from japanese_anki import cli

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [verb("買う", "かう", "godan", part_of_speech="verb").to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "decks" / "drill.yaml").write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "janki.toml").write_text(
        (tmp_path / "janki.toml").read_text(encoding="utf-8")
        + "\n[review]\nrequire = false\n",
        encoding="utf-8",
    )

    code = cli.main(["--root", str(tmp_path), "build", "--all", "--only-new"])

    assert code == 0
    assert "cannot narrow it" in capsys.readouterr().out


def test_refresh_builds_one_named_drill_deck(tmp_path: Path, capsys) -> None:
    """The other half of the same rule. `refresh --deck drill` injects
    `--only-new` too, so a drill deck hit the refusal on the *last* stage of a
    run that had already spent its jpdb, `--ai` and audio calls — and the remedy
    the message offers, dropping the flag, is one no `janki refresh` invocation
    can follow, because refresh always adds it."""
    import json

    from japanese_anki import cli

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [verb("買う", "かう", "godan", part_of_speech="verb").to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "decks" / "drill.yaml").write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "janki.toml").write_text(
        (tmp_path / "janki.toml").read_text(encoding="utf-8")
        + "\n[review]\nrequire = false\n",
        encoding="utf-8",
    )

    code = cli.main([
        "--root", str(tmp_path), "refresh", "--deck", "drill",
        "--no-jpdb", "--no-ai", "--no-recheck", "--no-audio", "--no-review",
    ])

    assert code == 0, "the stage refresh always injects the flag for"
    assert "cannot narrow it" in capsys.readouterr().out


def test_a_hand_typed_only_new_still_refuses_on_a_drill_deck(
    tmp_path: Path, capsys
) -> None:
    """The flag means two different things on the two paths, and the refusal is
    the point of the hand-typed one: it is an assertion about *this* deck, and
    quietly building everything instead reports a flag as honoured that never
    was."""
    from japanese_anki import cli

    deck = cli_project(tmp_path, [verb("買う", "かう", "godan")])

    assert cli.main(["--root", str(tmp_path), "build", str(deck), "--only-new"]) == 1
    assert "--only-new needs export history" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('  exclude_ids: "word:買う:かう"', "deck.exclude_ids must be a list"),
        ('  source: "../nope.json"', "no collection at"),
    ],
    ids=["a-bare-string-filter", "a-missing-collection"],
)
def test_validate_catches_what_the_build_would_refuse(
    tmp_path: Path, line: str, expected: str
) -> None:
    """`janki validate && janki build` passing the first and failing the second
    is the failure `deck_problems` exists to prevent."""
    from japanese_anki.exporters.pattern_cards import deck_problems

    project(tmp_path)
    deck = tmp_path / "decks" / "drill.yaml"
    deck.write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n"
        + line + "\n",
        encoding="utf-8",
    )

    problems = deck_problems(deck, {}, ProjectConfig.load(tmp_path))

    assert any(expected in problem for problem in problems), problems


def test_an_unreviewed_noun_does_not_block_the_drill_build(tmp_path: Path) -> None:
    """Through the CLI, with the gate on. A noun has no verb class so no drill
    card could carry it — refusing the build over one is a false refusal, and
    the collection a real learner has is mostly nouns."""
    import json

    from japanese_anki import cli
    from japanese_anki.review import CardReview, card_fingerprint, save_store

    project(tmp_path)
    drilled = verb("買う", "かう", "godan")
    noun = verb("猫", "ねこ", "")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([drilled.to_dict(), noun.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "decks" / "drill.yaml").write_text(
        "deck:\n  kind: conjugation\n  name: D\n  deck_id: 1\n  model_id: 2\n",
        encoding="utf-8",
    )
    # Only the verb has been read. The noun has not, and must not matter.
    save_store(
        tmp_path / "review.json",
        {card_fingerprint(drilled): CardReview(drilled.id, card_fingerprint(drilled))},
    )
    (tmp_path / "janki.toml").write_text(
        (tmp_path / "janki.toml").read_text(encoding="utf-8")
        + '\nreview_file = "review.json"\n',
        encoding="utf-8",
    )

    assert cli.main(["--root", str(tmp_path), "build", str(tmp_path / "decks" / "drill.yaml")]) == 0


def test_validate_reports_a_pattern_deck_without_swallowing_the_records(
    tmp_path: Path, capsys
) -> None:
    """Through the CLI. Every `validate` test called `deck_problems` directly, so
    the routing itself — which decides whether a file's records are validated at
    all — went unexercised, and a dispatch on truthiness passed the suite."""
    import json

    from japanese_anki import cli

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [verb("買う", "かう", "godan", part_of_speech="verb").to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "decks" / "words.yaml").write_text(
        'name: W\ndeck:\n  source: "../vocabulary.json"\n', encoding="utf-8"
    )
    (tmp_path / "decks" / "rules.yaml").write_text(
        "deck:\n  kind: pattern\n  name: R\n  deck_id: 1\n  model_id: 2\n"
        '  document: "unread.pdf"\n',
        encoding="utf-8",
    )

    assert cli.main(["--root", str(tmp_path), "validate"]) == 1

    out = capsys.readouterr().out
    assert "no document has been read under 'unread.pdf'" in out
    assert "Validated 1 records" in out, "the word deck was still checked"


def test_a_deck_kind_janki_does_not_know_is_left_to_the_ordinary_path(
    tmp_path: Path, capsys
) -> None:
    """A typo, or a deliberate `kind: vocabulary`. Dispatching on truthiness
    invented two errors that are false for a word deck — a missing `model_id`
    an ordinary deck never pins, and a missing `document:` — and skipped every
    record the file holds, while `janki build` built them all."""
    import json

    from japanese_anki import cli

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [verb("買う", "かう", "godan", part_of_speech="verb").to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    deck = tmp_path / "decks" / "words.yaml"
    deck.write_text(
        'name: W\ndeck:\n  kind: vocabulary\n  source: "../vocabulary.json"\n',
        encoding="utf-8",
    )

    assert cli.main(["--root", str(tmp_path), "validate", str(deck)]) == 0
    assert "Validated 1 records" in capsys.readouterr().out


def test_validate_does_not_read_the_pattern_store_for_a_staging_file(
    tmp_path: Path, capsys
) -> None:
    """`patterns.json` is machine-written and committed, so it can carry a merge
    marker — and reading it up front made that cancel a command with nothing to
    do with it."""
    import json

    from japanese_anki import cli

    project(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [verb("買う", "かう", "godan", part_of_speech="verb").to_dict()],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "patterns.json").write_text("<<<<<<< HEAD\n", encoding="utf-8")
    (tmp_path / "janki.toml").write_text(
        (tmp_path / "janki.toml").read_text(encoding="utf-8")
        + 'patterns_file = "patterns.json"\n',
        encoding="utf-8",
    )

    code = cli.main([
        "--root", str(tmp_path), "validate", str(tmp_path / "vocabulary.json")
    ])

    assert code == 0, "the unreadable pattern store was never touched"


@pytest.mark.parametrize(
    ("patterns_in", "expected"),
    [
        ([], "states no rules"),
        ([Pattern("く → いて"), Pattern("く → いた")], "more than one rule for く"),
    ],
    ids=["no-rules", "a-duplicate-trigger"],
)
def test_validate_refuses_a_well_formed_deck_the_build_cannot_use(
    tmp_path: Path, patterns_in: list, expected: str
) -> None:
    """Both refusals live past the document checks, so a deck naming a reviewed
    document passed validate and failed the build."""
    from japanese_anki.exporters.pattern_cards import deck_problems

    project(tmp_path)
    store = {"teform.pdf": chart(*patterns_in)}

    problems = deck_problems(deck_file(tmp_path), store, ProjectConfig.load(tmp_path))

    assert any(expected in problem for problem in problems), problems


def test_a_pattern_deck_with_a_bad_model_id_still_warns_in_status(tmp_path: Path) -> None:
    """Zeroing it matched no notetype, so `status` skipped the deck in silence
    where it used to print "could not read <deck>: model_id must be an integer"."""
    from japanese_anki.exporters.anki import deck_notetype

    project(tmp_path)
    path = tmp_path / "decks" / "bad.yaml"
    path.write_text(
        "deck:\n  kind: pattern\n  name: R\n  deck_id: 1\n  model_id: yes\n"
        '  document: "x.pdf"\n',
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="model_id must be an integer"):
        deck_notetype(path, ProjectConfig.load(tmp_path))


def test_a_pattern_build_stops_on_an_unreadable_collection(tmp_path: Path, capsys) -> None:
    """The worked examples on a rule card come from checking the chart against
    the collection's own `verb_group` values. When the collection could not be
    read that lookup returned nothing, every row was held back for want of a
    class, and the deck shipped with every `Examples` field empty — while the
    command printed an unchanged rule-card count and exited 0.

    Worse than an empty deck: the note GUID is `trigger\x1fgloss` and
    deliberately excludes the examples, so importing that package *updates* the
    rule cards already in Anki and blanks their examples. A transient bad
    `vocabulary.json` erases verified content on a run that reported success."""
    from japanese_anki import cli
    from japanese_anki import patterns as patterns_module

    project(tmp_path)
    patterns_module.save_store(
        ProjectConfig.load(tmp_path).patterns_file,
        {"teform.pdf": chart(Pattern("う・つ・る → って", "godan て-form", ("かう ⇨ かって",)))},
    )
    deck = deck_file(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        '[{"id": "word:x:x", "expression": "x", "reading": "x", "examples": 3}]',
        encoding="utf-8",
    )

    code = cli.main([
        "--root", str(tmp_path), "build", str(deck),
        "--output", str(tmp_path / "out.apkg"),
    ])

    assert code == 1, "the build stopped"
    assert not (tmp_path / "out.apkg").exists(), "and shipped nothing"
    assert "rule card(s)" not in capsys.readouterr().out
