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

import pytest
import yaml

from japanese_anki.application import deck_membership
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import resolve_deck_records
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
