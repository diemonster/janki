"""W1.1b: the shared services the CLI and the workbench both call.

The property that matters most here is **agreement**. A preview exists to tell
someone what a command would do; a preview computed by a second implementation
of the same rule is a lie with a progress bar, and the failure mode is silent —
it looks right until the day the two copies disagree, which is the day nobody
is checking. So the last test in each group builds the real thing and asserts
the preview named exactly what it shipped.

Everything here is offline.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_workbench_fixtures import materialize

from japanese_anki import cli, operations, promote
from japanese_anki.application import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    OUTCOME_UNKNOWN,
    ExtractionPlan,
    authorize_dispatch,
    capture_hook,
    classify_dispatch_failure,
    deck_membership,
    plan_extraction,
    plan_promotion,
    settle_dispatch,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import load_records
from japanese_anki.models import SourceReference, VocabularyRecord

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
ledger_file = "ledger.json"
media_dir = "media"
staging_dir = "staging"
patterns_file = "patterns.json"
scan_inbox = "inbox"
"""


def _record(**overrides: object) -> VocabularyRecord:
    values: dict[str, object] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "verb_group": "godan",
        "tags": ["lesson-8"],
        "source": SourceReference(type="extract", imported_from="lesson-8.pdf"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)  # type: ignore[arg-type]


def _project(tmp_path: Path, decks: dict[str, dict[str, object]]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    deck_dir = tmp_path / "decks"
    deck_dir.mkdir(exist_ok=True)
    for stem, deck in decks.items():
        (deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump({"deck": deck}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return ProjectConfig.load(tmp_path)


def _word_deck(**overrides: object) -> dict[str, object]:
    deck: dict[str, object] = {
        "name": "Japanese::Lesson 8",
        "deck_id": 2059400001,
        "output": "lesson-8.apkg",
        "source": "../vocabulary.json",
        "cards": {"recognition": True, "production": True},
    }
    deck.update(overrides)
    return deck


def _by_stem(config: ProjectConfig, record: VocabularyRecord) -> dict[str, object]:
    return {item.stem: item for item in deck_membership(config, record)}


# --- which deck would take this card ----------------------------------------


def test_a_tagged_card_lands_in_the_deck_that_asks_for_its_tag(tmp_path: Path) -> None:
    config = _project(
        tmp_path,
        {
            "lesson-8": _word_deck(include_tags=["lesson-8"]),
            "lesson-9": _word_deck(include_tags=["lesson-9"]),
        },
    )

    found = _by_stem(config, _record())

    assert found["lesson-8"].takes is True
    assert found["lesson-8"].refusal is None
    assert found["lesson-9"].takes is False


def test_every_deck_is_reported_not_only_the_matching_ones(tmp_path: Path) -> None:
    """"It lands nowhere" is an answer somebody needs, and an empty list is not
    a readable way to give it — it is indistinguishable from "I did not look"."""
    config = _project(tmp_path, {"lesson-9": _word_deck(include_tags=["lesson-9"])})

    found = deck_membership(config, _record())

    assert [item.stem for item in found] == ["lesson-9"]
    assert found[0].takes is False


def test_the_refusal_names_the_tags_on_both_sides(tmp_path: Path) -> None:
    """A verdict alone cannot be acted on. Someone deciding where a new lesson's
    words go needs to see the tag the deck wants and the tag the card carries,
    which is the edit that would fix it."""
    config = _project(tmp_path, {"lesson-9": _word_deck(include_tags=["lesson-9"])})

    [item] = deck_membership(config, _record())

    assert "lesson-9" in item.refusal
    assert "lesson-8" in item.refusal


def test_an_untagged_card_is_refused_in_words_not_by_an_empty_list(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path, {"lesson-9": _word_deck(include_tags=["lesson-9"])})

    [item] = deck_membership(config, _record(tags=[]))

    assert item.takes is False
    assert "tagged nothing" in item.refusal


def test_a_deck_with_no_filter_claims_every_card_and_says_so(tmp_path: Path) -> None:
    """Worth its own flag: promoting any word puts it in this deck whether or
    not anybody meant it to, and that is a thing to know *before* promoting."""
    config = _project(tmp_path, {"everything": _word_deck()})

    [item] = deck_membership(config, _record())

    assert item.takes is True
    assert item.unfiltered is True


@pytest.mark.parametrize(
    ("deck", "takes", "reason"),
    [
        ({"include_ids": ["word:話す:はなす"]}, True, None),
        ({"include_ids": ["word:食べる:たべる"]}, False, "not one"),
        ({"exclude_ids": ["word:話す:はなす"]}, False, "by name"),
        ({"exclude_tags": ["lesson-8"]}, False, "excludes cards tagged"),
        ({"include_tags": ["lesson-8"], "exclude_tags": ["lesson-8"]}, False,
         "excludes cards tagged"),
    ],
)
def test_each_filter_decides_and_explains(
    tmp_path: Path, deck: dict[str, object], takes: bool, reason: str | None
) -> None:
    config = _project(tmp_path, {"deck": _word_deck(**deck)})

    [item] = deck_membership(config, _record())

    assert item.takes is takes
    if reason is None:
        assert item.refusal is None
    else:
        assert reason in item.refusal


def test_a_deck_that_will_not_parse_is_unknown_rather_than_no(
    tmp_path: Path,
) -> None:
    """The same distinction `promote` draws when a deck will not read: nothing
    can be proved absent from a file nobody could open. Reporting `takes=False`
    would quietly claim it was checked."""
    config = _project(tmp_path, {"broken": _word_deck()})
    (tmp_path / "decks" / "broken.yaml").write_text(
        "deck:\n  cards: not-a-mapping\n", encoding="utf-8"
    )

    [item] = deck_membership(config, _record())

    assert item.unreadable
    assert item.takes is False


# --- the decks that do not select by tag ------------------------------------


def test_a_drill_deck_takes_what_it_can_conjugate(tmp_path: Path) -> None:
    """The blind spot this service exists to close.

    `teform-drill` declares no tags and no ids: it holds every record janki can
    build a te-form for, and puts a meaning on each card. A membership answer
    that understood only tag rules called that "no deck" — which is why 14
    corrected glosses sat in a drill package nobody thought to rebuild.
    """
    config = _project(
        tmp_path,
        {
            "lesson-8": _word_deck(include_tags=["lesson-8"]),
            "teform-drill": {
                "kind": "conjugation",
                "form": "te_form",
                "name": "Japanese::Te-form Practice",
                "deck_id": 2059400114,
                "model_id": 1607392351,
                "model_name": "Japanese Pattern",
                "output": "teform-drill.apkg",
                "source": "../vocabulary.json",
            },
        },
    )

    found = _by_stem(config, _record())

    # Both, and that is the point: one card, two packages to rebuild.
    assert found["lesson-8"].takes is True
    assert found["teform-drill"].takes is True
    assert found["teform-drill"].kind == "conjugation"


def test_a_drill_deck_refuses_a_word_it_cannot_conjugate(tmp_path: Path) -> None:
    """Asked of the real builder rather than of `part_of_speech`, because the
    deck file itself records that they disagree: records carrying
    `verb_group: suru` still produce no card unless the expression ends in する.
    """
    config = _project(
        tmp_path,
        {
            "teform-drill": {
                "kind": "conjugation",
                "form": "te_form",
                "name": "Japanese::Te-form Practice",
                "deck_id": 2059400114,
                "model_id": 1607392351,
                "model_name": "Japanese Pattern",
                "output": "teform-drill.apkg",
                "source": "../vocabulary.json",
            }
        },
    )
    noun = _record(
        id="word:本:ほん", expression="本", reading="ほん", meanings=["book"],
        part_of_speech="noun", verb_group="",
    )

    [item] = deck_membership(config, noun)

    assert item.takes is False
    assert "te-form" in item.refusal


def test_a_pattern_deck_holds_no_word_cards_at_all(tmp_path: Path) -> None:
    """Not applicable rather than refused: a grammar deck has no vocabulary to
    hold, so "this card failed its rule" would be the wrong sentence."""
    config = _project(
        tmp_path,
        {
            "teform": {
                "kind": "pattern",
                "name": "Japanese::Te-form Rules",
                "deck_id": 2059400113,
                "model_id": 1607392351,
                "model_name": "Japanese Pattern",
                "output": "teform-rules.apkg",
                "document": "teform_song.pdf",
            }
        },
    )

    [item] = deck_membership(config, _record())

    assert item.takes is False
    assert "grammar patterns" in item.refusal


# --- agreement with the build -----------------------------------------------


def test_the_preview_names_exactly_what_the_deck_resolves(tmp_path: Path) -> None:
    """The anti-drift test, and the reason `DeckSelection` is one value.

    The preview asks each deck's rule about a record that is not in the
    collection; the build filters the collection through the same rule. If
    those ever diverge the preview keeps rendering confidently and is simply
    wrong, so this drives both over the same records and demands the same
    answer — including for the filters no other test here combines.
    """
    decks = {
        "tagged": _word_deck(include_tags=["lesson-8"]),
        "other": _word_deck(include_tags=["lesson-9"]),
        "everything": _word_deck(),
        "minus-one": _word_deck(exclude_ids=["word:話す:はなす"]),
        "narrowed": _word_deck(
            include_tags=["lesson-8", "lesson-9"], exclude_tags=["draft"]
        ),
    }
    config = _project(tmp_path, decks)
    population = [
        _record(),
        _record(id="word:食べる:たべる", expression="食べる", reading="たべる",
                meanings=["to eat"], verb_group="ichidan", tags=["lesson-9"]),
        _record(id="word:飲む:のむ", expression="飲む", reading="のむ",
                meanings=["to drink"], tags=["lesson-9", "draft"]),
        _record(id="word:本:ほん", expression="本", reading="ほん",
                meanings=["book"], part_of_speech="noun", verb_group="", tags=[]),
    ]
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in population], ensure_ascii=False),
        encoding="utf-8",
    )

    for stem in decks:
        _config, resolved = resolve_deck_records(tmp_path / "decks" / f"{stem}.yaml")
        shipped = {record.id for record in resolved}
        predicted = {
            record.id
            for record in population
            if _by_stem(config, record)[stem].takes
        }
        assert predicted == shipped, stem


def test_the_preview_answers_for_a_record_the_collection_does_not_have(
    tmp_path: Path,
) -> None:
    """The whole reason this is not "filter the collection": the card being
    asked about is staged, so it is in no collection yet and `resolve_deck_records`
    cannot see it."""
    config = _project(tmp_path, {"lesson-8": _word_deck(include_tags=["lesson-8"])})
    _deck, resolved = resolve_deck_records(tmp_path / "decks" / "lesson-8.yaml")
    assert resolved == []

    staged = replace(_record(), id="word:新しい:あたらしい", expression="新しい",
                     reading="あたらしい", meanings=["new"])

    [item] = deck_membership(config, staged)

    assert item.takes is True


# --- what adding a source would do ------------------------------------------
#
# Driven off the W0 fixtures, materialized through the real extraction
# pipeline, so the preview is answering about staging files `extract` actually
# writes rather than hand-typed YAML that could drift from them.


def _staged(tmp_path: Path, scenario: str, *, filename: str | None = None) -> Path:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    if not (tmp_path / "vocabulary.json").exists():
        (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "inbox").mkdir(exist_ok=True)
    return Path(materialize(tmp_path, scenario, filename=filename)["staging_path"])


def test_a_fresh_source_is_all_new_cards(tmp_path: Path) -> None:
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert plan.landing
    assert plan.adding == plan.landing
    assert plan.merging == ()
    assert plan.reminted == {}


def test_an_unresolved_coverage_block_stops_the_plan_and_says_which(
    tmp_path: Path,
) -> None:
    """The gate `promote` applies before it reads a client or writes a byte,
    reported in the same words. A preview that listed the cards this file would
    add would be describing a promote that cannot happen — and `--accept-coverage`
    is a paid model call, so it is a decision to put in front of a person rather
    than a step to run on their behalf."""
    path = _staged(tmp_path, "table_exhaustive", filename="table.pdf")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert "coverage" in plan.blocked
    assert plan.landing == ()
    assert plan.held == ()


def test_a_word_the_collection_already_has_merges_and_keeps_its_meanings(
    tmp_path: Path,
) -> None:
    """The existing-wins surprise, previewed. `promote` keeps the meaning
    already on the card and files this lesson's wording as source evidence, so
    a person who expected their new gloss to appear needs telling first."""
    path = _staged(tmp_path, "shared_word_source_a", filename="a.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging

    staged_records, _meta = read_staging(path)
    owned = replace(staged_records[0], meanings=["the wording already on my card"])
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([owned.to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    plan = plan_promotion(config, path)

    merging = {card.staged.id: card for card in plan.merging}
    assert owned.id in merging, [card.staged.id for card in plan.landing]
    card = merging[owned.id]
    assert card.landing.meanings == ["the wording already on my card"]
    assert card.keeps_existing_meanings is True


def test_a_card_whose_meanings_survive_untouched_is_not_reported_as_overridden(
    tmp_path: Path,
) -> None:
    """The negative half. Flagging every merge as "your wording won" would make
    the warning meaningless on the sources where nothing was lost."""
    path = _staged(tmp_path, "shared_word_source_a", filename="a.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging

    staged_records, _meta = read_staging(path)
    same = staged_records[0]
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([same.to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    plan = plan_promotion(config, path)

    card = next(item for item in plan.merging if item.staged.id == same.id)
    assert card.keeps_existing_meanings is False


def test_a_held_row_is_reported_with_the_reason_it_was_held(tmp_path: Path) -> None:
    path = _staged(tmp_path, "reading_holds", filename="holds.pdf")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.held, "the reading_holds fixture stages a row promote holds"
    assert all(card.reason for card in plan.held)
    landing_ids = {card.staged.id for card in plan.landing}
    assert not landing_ids & {card.record.id for card in plan.held}


def test_the_preview_says_the_dictionary_has_not_been_asked(tmp_path: Path) -> None:
    """Load-bearing, not decorative. The real promote holds back a row whose
    reading no dictionary lists, and that check costs a paid lookup per row —
    so this plan never makes it. A row listed as landing may still be held when
    the source is actually added, and the flag is what stops the page claiming
    otherwise."""
    path = _staged(tmp_path, "table_exhaustive", filename="table.pdf")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.readings_unchecked is True


def test_the_preview_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page refresh must not be able to spend money. Asserted against the
    client itself rather than by inspecting arguments, because the argument
    that matters (`client=None`) is exactly the one a refactor drops."""
    from japanese_anki import jpdb

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the promotion preview contacted jpdb")

    monkeypatch.setattr(jpdb, "JpdbClient", refuse)
    monkeypatch.setattr(jpdb, "api_key_from_env", refuse)
    path = _staged(tmp_path, "reading_holds", filename="holds.pdf")
    config = ProjectConfig.load(tmp_path)

    plan_promotion(config, path)


def test_the_preview_writes_nothing(tmp_path: Path) -> None:
    """Including the staging file it reads: a preview that pruned or annotated
    would turn opening a page into a half-finished promote."""
    path = _staged(tmp_path, "reading_holds", filename="holds.pdf")
    config = ProjectConfig.load(tmp_path)
    before = {
        item: item.read_bytes()
        for item in sorted(tmp_path.rglob("*"))
        if item.is_file()
    }

    plan_promotion(config, path)

    after = {
        item: item.read_bytes()
        for item in sorted(tmp_path.rglob("*"))
        if item.is_file()
    }
    assert after == before


def test_an_unreadable_staging_file_is_reported_not_raised(tmp_path: Path) -> None:
    """"You cannot add this yet, here is why" is the answer a page needs. A
    traceback from a view that changes nothing is not."""
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "staging").mkdir(exist_ok=True)
    broken = tmp_path / "staging" / "broken.yaml"
    broken.write_text("records: [oh dear\n", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, broken)

    assert plan.is_blocked
    assert plan.landing == ()


def test_the_plan_names_exactly_what_promote_then_does(tmp_path: Path) -> None:
    """The anti-drift test for the promotion preview.

    Every decision in the plan comes from a function `promote` itself calls, so
    the two should never disagree — but "should never" is what a second
    implementation always says. This runs the plan, then runs the real
    `janki promote` over the same file, and demands the collection contain
    exactly the records the plan named, under exactly the ids it predicted,
    with exactly the meanings it previewed.

    `--skip-reading-check` keeps it offline, which also makes the comparison
    fair: that is the one check the preview declines to make.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging

    staged_records, _meta = read_staging(path)
    assert len(staged_records) > 1, (
        "a one-row fixture makes the under-report direction vacuous: dropping "
        "one of N rows needs N > 1 to be visible"
    )
    owned = replace(staged_records[0], meanings=["the wording already on my card"])
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([owned.to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    plan = plan_promotion(config, path)
    assert not plan.is_blocked, plan.blocked
    # The whole record, not just id and meanings: a landing card with a wrong
    # reading or a dropped example is exactly the drift this test is named for,
    # and comparing two fields would sail past it.
    predicted = {card.landing.id: card.landing.to_dict() for card in plan.landing}
    assert predicted

    code = cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ])
    assert code == 0

    stored = {
        record.id: record.to_dict()
        for record in load_records(tmp_path / "vocabulary.json")
    }
    for record_id, expected in predicted.items():
        assert record_id in stored, f"{record_id} was predicted but did not land"
        assert stored[record_id] == expected, record_id
    # And nothing landed that the plan did not name — a preview that
    # under-reports is as wrong as one that over-reports, and only this
    # direction catches it.
    assert set(stored) == set(predicted) | {owned.id}

    # A row the plan held is still in the staging file, not in the collection.
    for held in plan.held:
        assert held.record.id not in stored


def test_a_corrected_reading_is_reported_as_a_change_of_identity(
    tmp_path: Path,
) -> None:
    """The case `promote.remint` exists for, and the one worth warning about.

    A reviewer fixes a reading, but the id was minted from the wrong one and
    ids are uncorrectable by design — so promote files the card under a
    different id than it was staged with. That is a *different Anki note*: any
    card already shipped under the old id keeps its review history and this one
    starts from zero. Nobody guesses that from a card that looks the same.
    """
    path = _staged(tmp_path, "shared_word_source_a", filename="a.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    # The id keeps the reading it was minted from; the record now carries the
    # corrected one. Only the id still remembers, which is the docstring's point.
    corrected = replace(records[0], reading="ちがうよみ")
    write_staging(path, [corrected, *records[1:]], meta, force=True)

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert plan.reminted, "a corrected reading should change the identity"
    assert records[0].id in plan.reminted
    new_id = plan.reminted[records[0].id]
    assert new_id != records[0].id
    assert "ちがうよみ" in new_id
    card = next(item for item in plan.landing if item.reminted_from == records[0].id)
    assert card.landing.id == new_id

    # And promote agrees: the record lands under the predicted id, not the
    # staged one.
    assert cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ]) == 0
    stored = {record.id for record in load_records(tmp_path / "vocabulary.json")}
    assert new_id in stored
    assert records[0].id not in stored


def test_rows_already_in_this_runs_archive_are_not_offered_again(
    tmp_path: Path,
) -> None:
    """A partial promote writes what landed to `staging/done/` and prunes only
    those rows. Re-running is meant to be safe — and this axis shipped
    untested, so the plan could have ignored the archive entirely and no test
    would have noticed.

    The rows already there are reported as `already_archived`, not as cards to
    add: offering them again is how a retry doubles a review.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    assert len(records) > 1
    done = tmp_path / "staging" / "done"
    done.mkdir(parents=True, exist_ok=True)
    # The same run's archive, holding the first row exactly as it was staged.
    write_staging(done / path.name, [records[0]], promote.archive_meta(meta, 1))

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert plan.already_archived == (records[0].id,)
    landing_ids = {card.staged.id for card in plan.landing}
    assert records[0].id not in landing_ids
    assert {record.id for record in records[1:]} == landing_ids


def test_a_file_inside_the_promoted_archive_is_refused(tmp_path: Path) -> None:
    """Promoting the archive would duplicate a committed file that is the only
    copy of a finished review. promote refuses it; so does the preview, rather
    than listing every row as ready to add."""
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    done = tmp_path / "staging" / "done"
    done.mkdir(parents=True, exist_ok=True)
    archived = done / path.name
    archived.write_bytes(path.read_bytes())

    plan = plan_promotion(config, archived)

    assert plan.is_blocked
    assert "already in the collection" in plan.blocked
    assert plan.landing == ()


def test_a_suffix_write_staging_could_not_rewrite_is_refused(tmp_path: Path) -> None:
    """The archive is written under the same name, so a suffix `write_staging`
    refuses is a review that can never be finished however often it is retried.

    The content has to be valid JSON, not YAML renamed: `read_staging` parses
    by suffix, so a renamed YAML file is refused by the *parser* and never
    reaches the gate this test is named for.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    renamed = path.with_suffix(".json")
    renamed.write_text(
        json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")),
                   ensure_ascii=False),
        encoding="utf-8",
    )

    plan = plan_promotion(config, renamed)

    assert plan.is_blocked
    assert "rename it" in plan.blocked.lower()


def test_a_staging_file_ruamel_could_not_rewrite_is_refused(tmp_path: Path) -> None:
    """`check_rewritable`, which the suffix gate above does not reach.

    `read_staging` goes through PyYAML, which accepts a duplicate key silently;
    the rewrite goes through ruamel, which does not. Finding that out *after*
    the records and the archive were written leaves the promoted rows still in
    the staging file, and the re-run appends them to the archive a second time
    — so promote checks it up front, and so does the plan.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\nsource_file: duplicated\n",
        encoding="utf-8",
    )

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert plan.landing == ()


def test_a_collection_that_will_not_load_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    """`vocabulary.json` is a file people hand-edit. A preview that raised on a
    broken one would put a traceback on a page whose whole job is to say what
    is wrong."""
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    (tmp_path / "vocabulary.json").write_text("{not json", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert plan.landing == ()


def test_a_deck_that_will_not_parse_holds_a_re_mint_and_says_why(
    tmp_path: Path,
) -> None:
    """`remint_blocked`: with a deck unreadable, the ids the collection holds
    cannot be completed, so a row whose id would change is held rather than
    promoted under an id that would then be permanent. The plan carries the
    same sentence the command prints."""
    path = _staged(tmp_path, "shared_word_source_a", filename="a.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    write_staging(
        path, [replace(records[0], reading="ちがうよみ"), *records[1:]], meta,
        force=True,
    )
    # A deck whose `deck:` section is not a mapping: `deck_declared_ids`
    # refuses it, so the ids it holds are unknown.
    (tmp_path / "decks" / "broken.yaml").write_text(
        "deck: [1, 2, 3]\n", encoding="utf-8"
    )

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert any("cannot be checked" in warning for warning in plan.warnings)
    # Held, not re-minted: the id it arrived with must not become permanent.
    assert plan.reminted == {}
    assert records[0].id in {card.record.id for card in plan.held}


def test_two_rows_that_would_mint_one_id_block_the_plan(tmp_path: Path) -> None:
    """The collision only exists after the re-mint, which is why the plan has
    to run the accounting check a second time.

    A reviewer corrects two rows to the same expression and reading. Their
    staged ids still differ — they were minted from the wrong readings — so the
    first check passes. Both then mint the same id, and `promote` refuses the
    whole file. Checking once left the plan cheerfully listing two cards that
    could never be added.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    assert len(records) > 1
    assert records[0].id != records[1].id
    collide = [
        replace(records[0], expression="話す", reading="はなす"),
        replace(records[1], expression="話す", reading="はなす"),
        *records[2:],
    ]
    write_staging(path, collide, meta, force=True)

    plan = plan_promotion(config, path)

    assert plan.is_blocked, [card.landing.id for card in plan.landing]
    assert "collision" in plan.blocked
    assert plan.landing == ()

    # And promote agrees: it refuses the file rather than adding either row.
    assert cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ]) != 0


def test_a_stale_binding_on_a_held_row_blocks_the_whole_plan(tmp_path: Path) -> None:
    """A reading hold narrows what may *land*, not what review the old-value
    binding covers.

    `merge_staged_records` refuses the entire merge when any replacement target
    was edited since the review — that atomicity is the point, so a stale
    review cannot land half its rows. The command therefore validates every
    live row, held ones included. Validating only the promotable subset hid the
    held row's stale binding, and the plan previewed a card from a promote that
    refuses outright.

    Both rows carry a binding, because the provenance maps must name identical
    ids. The landing row's still matches the collection; the held row's does
    not. So the *only* thing that can notice is validating the held row.
    """
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "staging").mkdir(exist_ok=True)
    from japanese_anki.staging import write_staging

    lands = _record(id="word:話す:はなす")
    held = _record(
        id="word:食べる:たべる", expression="食べる", reading="たべる",
        meanings=["to eat"], verb_group="ichidan",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [
                lands.to_dict(),
                # Edited since the review, so the binding recorded below no
                # longer matches what the collection holds.
                replace(held, meanings=["to eat (hand-checked)"]).to_dict(),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ids = [lands.id, held.id]
    changes = {
        lands.id: {"meanings": (["to speak"], ["to speak, to talk"])},
        held.id: {"meanings": (["to eat"], ["to consume"])},
    }
    meta = {
        "source_file": "vocabulary.json",
        "model": "claude-opus-5",
        "provider": "anthropic",
        "ai_enrichment": {
            "version": 1,
            "model": "claude-opus-5",
            "provider": "anthropic",
            "request_fingerprints": {record_id: "a" * 64 for record_id in ids},
            "input_fingerprints": {record_id: "b" * 64 for record_id in ids},
            "fields": {record_id: ["meanings"] for record_id in ids},
        },
        "field_replacements": promote.field_replacement_block(
            [lands, held], changes
        ),
    }
    path = tmp_path / "staging" / "ai.yaml"
    write_staging(
        path,
        [
            replace(lands, meanings=["to speak, to talk"]),
            # Held: a blank reading cannot mint an id, whatever a dictionary says.
            replace(held, reading="", meanings=["to consume"]),
        ],
        meta,
    )
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.is_blocked, [card.landing.id for card in plan.landing]
    assert "field-replacements-stale" in plan.blocked
    assert held.id in plan.blocked
    assert plan.landing == ()

    # And promote agrees: it refuses rather than landing the other row.
    assert cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ]) != 0
    stored = {
        record.id: tuple(record.meanings)
        for record in load_records(tmp_path / "vocabulary.json")
    }
    assert stored[lands.id] == ("to speak",), "nothing landed"


def test_a_divergent_same_run_archive_is_refused(tmp_path: Path) -> None:
    """`validate_record_archive`. A same-run archive whose own record of what
    it archived does not match what it holds is not a partial promotion this
    run can complete — promote refuses rather than guessing, and a plan that
    previewed cards from it would be describing a promote that never happens."""
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    done = tmp_path / "staging" / "done"
    done.mkdir(parents=True, exist_ok=True)
    archived_meta = promote.archive_meta(meta, 1)
    # Says it archived one row; the archive's own note no longer records it.
    archived_meta["review_notes"] = "no count here"
    write_staging(done / path.name, [records[0]], archived_meta)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert plan.landing == ()


def test_a_ledger_that_will_not_load_blocks_a_promote_that_would_write(
    tmp_path: Path,
) -> None:
    """promote reads the ledger before the records land, so a corrupt one stops
    the whole thing — and the preview has to say so rather than listing cards
    that cannot be added."""
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    (tmp_path / "ledger.json").write_text("{not json", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert plan.landing == ()


def test_a_ledger_that_will_not_load_does_not_block_a_promote_that_writes_nothing(
    tmp_path: Path,
) -> None:
    """The other half, and the one that makes the gate honest.

    promote reads the ledger only once it knows rows will land. When every row
    is held it returns having written hold reasons and never opens the ledger —
    so a plan that read it up front reported *every* zero-landing source as
    blocked on one corrupt file, which is telling somebody their work is
    unusable when it is not.
    """
    path = _staged(tmp_path, "reading_holds", filename="holds.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    # Every row held: a blank reading cannot mint an id.
    write_staging(
        path, [replace(record, reading="") for record in records], meta, force=True
    )
    (tmp_path / "ledger.json").write_text("{not json", encoding="utf-8")

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert plan.landing == ()
    assert len(plan.held) == len(records)

    # And promote agrees: it succeeds, because it never opens the ledger.
    assert cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ]) == 0


def test_an_enrichment_block_naming_rows_this_file_lacks_is_refused(
    tmp_path: Path,
) -> None:
    """`staged_ai_enrichment`. Nothing downstream catches this: the merge's own
    binding check never reads `ai_enrichment`, so without this gate the plan
    previewed cards from a file promote refuses on provenance."""
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    ghost = "word:存在しない:そんざいしない"
    meta = dict(meta)
    meta["ai_enrichment"] = {
        "version": 1,
        "model": "claude-opus-5",
        "provider": "anthropic",
        "request_fingerprints": {ghost: "a" * 64},
        "input_fingerprints": {ghost: "b" * 64},
        "fields": {ghost: ["meanings"]},
    }
    ghost_record = replace(records[0], id=ghost)
    meta["field_replacements"] = promote.field_replacement_block(
        [ghost_record],
        {ghost: {"meanings": (list(ghost_record.meanings), ["new"])}},
    )
    write_staging(path, records, meta, force=True)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert plan.landing == ()


def test_a_grammar_only_source_nobody_reviewed_cannot_be_added(
    tmp_path: Path,
) -> None:
    """A zero-record rich extraction has its own completion contract, and it
    refuses a source whose grammar has not been reviewed.

    Without this the plan called such a file clean — no cards, nothing blocked
    — and an Add button keyed on that invoked a promote that refuses.
    """
    path = _staged(tmp_path, "pattern_only_chart", filename="chart.pdf")
    config = ProjectConfig.load(tmp_path)

    plan = plan_promotion(config, path)

    assert plan.is_blocked
    assert "patterns-unreviewed" in plan.blocked

    # And promote agrees.
    assert cli.main([
        "--root", str(tmp_path), "promote", str(path), "--skip-reading-check",
    ]) != 0


def test_an_empty_file_beside_its_own_archive_says_the_rows_already_landed(
    tmp_path: Path,
) -> None:
    """The shape a crash between the prune and the unlink leaves behind.

    Every row is in this run's archive and the live file is empty. promote
    completes the retry and removes the empty review. The plan has to say the
    rows already landed — reporting an empty file with nothing archived reads
    as "nothing to do here", which is true of the file and false of the source.
    """
    path = _staged(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    config = ProjectConfig.load(tmp_path)
    from japanese_anki.staging import read_staging, write_staging

    records, meta = read_staging(path)
    done = tmp_path / "staging" / "done"
    done.mkdir(parents=True, exist_ok=True)
    write_staging(done / path.name, records, promote.archive_meta(meta, len(records)))
    write_staging(path, [], meta, force=True)

    plan = plan_promotion(config, path)

    assert not plan.is_blocked, plan.blocked
    assert plan.landing == ()
    assert plan.already_archived == tuple(record.id for record in records)


# --- what a run would send, before anyone agrees to it ----------------------
#
# W3 needs this as a value: adding a file to the corpus and sending it to a
# model are two separate actions, and the consent button has to name exactly
# what leaves the computer. A dialog cannot render that from a function that
# asks and dispatches in one breath.


def _input(tmp_path: Path, name: str, *, copied: bool = False) -> PreparedInput:
    path = tmp_path / "inbox" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake\n")
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=path,
        copied=copied,
    )


def _plan(tmp_path: Path, *inputs: PreparedInput, mode: str | None = None,
          force: bool = False) -> ExtractionPlan:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    if not (tmp_path / "vocabulary.json").exists():
        (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "staging").mkdir(exist_ok=True)
    return plan_extraction(
        ProjectConfig.load(tmp_path),
        list(inputs),
        mode=mode,
        model="claude-opus-5",
        style_guide="style",
        system="system",
        force=force,
    )


def test_the_plan_names_the_permanent_filename_and_the_model(
    tmp_path: Path,
) -> None:
    """What the consent sentence is built from: this file, this model."""
    plan = _plan(tmp_path, _input(tmp_path, "lesson-8.pdf"))

    assert plan.names == ("lesson-8.pdf",)
    assert plan.model == "claude-opus-5"
    assert plan.targets[0].staging_path.name.startswith("lesson-8.pdf")


def test_a_batch_that_would_collide_is_refused_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    """Refused now rather than after paying for both. Two inputs resolving to
    one staging file means one answer overwrites the other, and finding that
    out after the second call has already been billed.

    Two *different* files sharing a basename, which is the case worth refusing
    and the one a real batch hits — `lesson.pdf` from two folders. The same
    file listed twice collides as well, but through a path anyone would guess.
    """
    first = _input(tmp_path, "lesson.pdf")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    other = elsewhere / "lesson.pdf"
    other.write_bytes(b"%PDF-1.4 a different document\n")
    second = PreparedInput(
        kind="document", media_type="application/pdf", data_b64="b3RoZXI=",
        origin_path=other,
    )

    with pytest.raises(JankiError, match="lesson.pdf"):
        _plan(tmp_path, first, second)


def test_prose_mode_is_told_what_the_collection_already_has(
    tmp_path: Path,
) -> None:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([_record().to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    plan = _plan(tmp_path, _input(tmp_path, "lesson.pdf"), mode="prose")

    assert plan.skip_list == ("話す",)


def test_a_table_is_never_told_what_to_skip(tmp_path: Path) -> None:
    """A table is transcribed row by row. Telling the model to skip rows would
    put holes in a faithful transcription — so the skip list is prose-only, and
    this is the half that is easy to lose in a refactor."""
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([_record().to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    for mode in ("table", None):
        plan = _plan(tmp_path, _input(tmp_path, "sheet.pdf"), mode=mode)
        assert plan.skip_list == (), mode


def test_only_files_this_run_copied_are_reported_as_kept(tmp_path: Path) -> None:
    """The half a refusal message forgets. The inbox copy happens before
    consent, so "nothing was sent" alone reads as "nothing happened" while a
    private document sits staged for the next `git add`.

    Asked of `copied`, not of the path: every branch of the copy returns a path
    under the inbox, including the ones that copy nothing, so a containment
    test announces files the run never touched.
    """
    plan = _plan(
        tmp_path,
        _input(tmp_path, "brought-in.pdf", copied=True),
        _input(tmp_path, "already-there.pdf", copied=False),
    )

    assert [path.name for path in plan.kept_in_inbox] == ["brought-in.pdf"]


def test_an_unreadable_pattern_store_refuses_before_the_call(
    tmp_path: Path,
) -> None:
    """A preflight that exists so a paid answer is never thrown away for a
    reason that was knowable beforehand."""
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "patterns.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(JankiError):
        _plan(tmp_path, _input(tmp_path, "lesson.pdf"))


def test_every_target_carries_the_request_identity_it_will_be_journalled_under(
    tmp_path: Path,
) -> None:
    """Journalled before dispatch, so a crash between sending and parsing can
    still say which call was made. The plan resolves it, and the command
    journals what the plan resolved rather than deriving its own."""
    plan = _plan(tmp_path, _input(tmp_path, "lesson-8.pdf"))

    fingerprint = plan.targets[0].provenance["request_fingerprint"]
    assert isinstance(fingerprint, str) and fingerprint

    # Same inputs, same identity: a fingerprint that moved between the plan and
    # the dispatch would journal one call and make another.
    again = _plan(tmp_path, _input(tmp_path, "lesson-8.pdf"))
    assert again.targets[0].provenance["request_fingerprint"] == fingerprint


def test_planning_contacts_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deciding what a run *would* send must never be able to send it."""
    from japanese_anki import claude_client

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("planning an extraction contacted the provider")

    monkeypatch.setattr(claude_client, "parse_call", refuse)

    _plan(tmp_path, _input(tmp_path, "lesson.pdf"))


# --- the paid call's journal ------------------------------------------------
#
# The journal exists so that a crash between sending and parsing can still say
# whether money was spent. W3 dispatches from a browser and must not
# reimplement any of this, so the lifecycle is pinned here directly rather than
# only through the command that currently drives it.


def _journal(tmp_path: Path) -> tuple[ProjectConfig, Any, Any]:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "staging").mkdir(exist_ok=True)
    config = ProjectConfig.load(tmp_path)
    plan = plan_extraction(
        config,
        [_input(tmp_path, "lesson.pdf")],
        mode=None,
        model="claude-opus-5",
        style_guide="style",
        system="system",
    )
    return config, operations.OperationJournal.load(config.operations_file), plan


def test_authority_is_recorded_before_the_request_exists(tmp_path: Path) -> None:
    """Not after it succeeds. The interval this protects is the one where the
    money is spent and nothing on disk remembers — a crash between the send and
    the parse otherwise leaves no way to tell "never sent" from "sent and
    lost"."""
    config, journal, plan = _journal(tmp_path)

    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)

    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "dispatching"
    assert entry.source_file == "lesson.pdf"
    # The rest of the record, because W3's recovery surface reads it and an
    # empty field in a durable record of money is not a small thing.
    assert entry.kind == "extract"
    assert entry.model == "claude-opus-5"
    assert entry.source_sha256 == plan.targets[0].source_sha256
    # The identity the plan resolved, so the journal names the call that will
    # be made rather than one this frame derived for itself.
    assert entry.request_fp == plan.targets[0].provenance["request_fingerprint"]


def test_a_reply_that_carries_an_answer_is_reported_as_paid_for(
    tmp_path: Path,
) -> None:
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)
    capture_hook(config, journal, operation_id)({"content": [{"type": "text",
                                                             "text": "cards"}]})

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("schema mismatch")
    )

    assert failure.outcome == ANSWER_SAVED
    assert failure.was_paid_for
    assert failure.artifact
    # Left where it is. The paid bytes are on disk and a person has to decide
    # what to do with them; calling that "unknown" would hide an answer already
    # bought.
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "result_captured"


def test_a_reply_holding_only_reasoning_is_not_called_an_answer(
    tmp_path: Path,
) -> None:
    """A call that reaches `max_tokens` while still thinking returns a real,
    billed reply containing nothing to recover. Saying "the answer was saved"
    sends somebody hunting for cards in a file that holds none."""
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)
    capture_hook(config, journal, operation_id)(
        {"content": [{"type": "thinking", "thinking": "considering the page"}]}
    )

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("truncated")
    )

    assert failure.outcome == ANSWER_EMPTY
    assert failure.was_paid_for
    # The pointer to the billed file, which is the only thing a person can act
    # on here — the message that carries it says there is nothing to recover,
    # so without the path it says nothing useful at all.
    assert failure.artifact


def test_a_call_that_vanished_says_a_retry_risks_a_second_charge(
    tmp_path: Path,
) -> None:
    """Sent, nothing captured. This is the only outcome where re-running costs
    money that may already have been spent, so it is the one the journal has to
    get right — and it is answered from the journal's own state rather than
    guessed from the exception."""
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("connection reset")
    )

    assert failure.outcome == OUTCOME_UNKNOWN
    assert failure.money_may_have_been_spent is True
    assert not failure.was_paid_for
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "outcome_unknown"
    assert "connection reset" in entry.detail


def test_an_answer_the_capture_hook_never_saw_is_still_recorded(
    tmp_path: Path,
) -> None:
    """The hook fires inside the client. A provider wrapper that never calls it
    would leave the entry at `dispatching` for ever, describing a call still in
    flight that in fact returned."""
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)

    settle_dispatch(config, journal, operation_id, {"candidates": []})

    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "result_captured"
    assert entry.artifact


def test_settling_does_not_overwrite_the_exact_bytes_the_hook_captured(
    tmp_path: Path,
) -> None:
    """The hook stores the provider's own reply; settling stores janki's
    normalized value. Overwriting the first with the second would replace the
    evidence with a rendering of it."""
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)
    capture_hook(config, journal, operation_id)({"content": [{"type": "text",
                                                             "text": "exact"}]})
    before = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]

    settle_dispatch(config, journal, operation_id, {"candidates": []})

    after = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert after.artifact == before.artifact
    assert (config.operations_file.parent / after.artifact).read_bytes() == (
        config.operations_file.parent / before.artifact
    ).read_bytes()


def test_a_paid_reply_is_found_even_when_another_journal_captured_it(
    tmp_path: Path,
) -> None:
    """The state of record is the file, not whichever object this frame holds.

    `advance` updates the journal it is called on, so a run that captures
    through the same object sees its own write either way. Nothing makes that
    the only shape — a request handler holding one journal while the client
    advances another is exactly how the browser will drive this — and asking a
    copy that never saw the answer reports a reply already paid for as an
    unknown outcome, which is the one classification that tells somebody a
    retry is safe.
    """
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)
    # A second, independently loaded journal captures the answer — the first
    # one's in-memory copy still says `dispatching`.
    elsewhere = operations.OperationJournal.load(config.operations_file)
    capture_hook(config, elsewhere, operation_id)(
        {"content": [{"type": "text", "text": "cards"}]}
    )
    assert journal.operations[operation_id].state == "dispatching"

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("schema mismatch")
    )

    assert failure.outcome == ANSWER_SAVED
    assert failure.was_paid_for


def test_the_exact_bytes_reach_disk_before_the_journal_claims_they_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-ahead order the journal's whole design rests on.

    A crash between the two must leave an unreferenced file rather than an
    entry pointing at an answer that was never written — the second is a lie,
    and it is a lie about something that has been paid for. Every other test
    here observes the settled end state, which is identical either way, so
    this is the one that can tell the orders apart.
    """
    config, journal, plan = _journal(tmp_path)
    operation_id = authorize_dispatch(journal, plan.targets[0], model=plan.model)

    def explode(*args: object, **kwargs: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(
        "japanese_anki.application.extraction.operations.capture_artifact", explode
    )

    with pytest.raises(OSError):
        capture_hook(config, journal, operation_id)({"content": []})

    # Still dispatching: nothing claims an answer arrived, because none was
    # stored. Advancing first would have left an entry naming a file that does
    # not exist.
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "dispatching"
    assert entry.artifact == ""
