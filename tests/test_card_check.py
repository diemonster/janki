"""W4.1's read-only structural-card check projection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from japanese_anki.application import CardCheckAction, CheckActionKind, check_cards
from japanese_anki.application import card_check as card_check_module
from japanese_anki.config import ProjectConfig
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    EXAMPLE_AUTHORITY_STAGING,
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
)
from japanese_anki.staging import (
    HOLD_MISSING_READING,
    HOLD_UNKNOWN_READING,
    annotate,
)


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\ndeck_dir = "decks"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "decks").mkdir()
    return ProjectConfig.load(tmp_path)


def _deck(
    config: ProjectConfig,
    stem: str,
    *,
    intake_tag: str,
    include_tags: list[str],
) -> None:
    document = {
        "deck": {
            "name": stem.title(),
            "source": "../vocabulary.json",
            "intake_tag": intake_tag,
            "include_tags": include_tags,
        }
    }
    (config.deck_dir / f"{stem}.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _examples() -> list[ExampleSentence]:
    return [
        ExampleSentence(
            japanese="話します。",
            furigana="話[はな]します。",
            english="I speak.",
            romaji="Hanashimasu.",
            register="polite",
        ),
        ExampleSentence(
            japanese="話す。",
            furigana="話[はな]す。",
            english="I speak.",
            romaji="Hanasu.",
            register="casual",
        ),
    ]


def _record(
    *,
    record_id: str,
    reading: str,
    meanings: list[str],
    tags: list[str] | None = None,
    raw_fields: dict[str, str] | None = None,
) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id,
        expression="話す",
        reading=reading,
        meanings=meanings,
        examples=_examples(),
        tags=tags or [],
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            row=1,
            raw_fields=raw_fields or {},
        ),
    )


def _details(action: CardCheckAction) -> tuple[tuple[str, str], ...]:
    return tuple((item.code, item.reason) for item in action.details)


def _kind(value: CheckActionKind) -> str:
    return value


def test_check_cards_combines_snapshot_facts_into_ordered_learner_actions(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    _deck(config, "lesson", intake_tag="lesson", include_tags=["lesson"])
    incomplete = annotate(
        _record(record_id="word:話す:", reading="", meanings=[]),
        hold_reason=HOLD_MISSING_READING,
    )
    held = annotate(
        _record(
            record_id="word:話す:はなす",
            reading="はなす",
            meanings=["to speak"],
            tags=["lesson"],
            raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
        ),
        hold_reason=HOLD_UNKNOWN_READING,
    )
    before = [record.to_dict() for record in (incomplete, held)]

    report = check_cards(
        config,
        [incomplete, held],
        source=config.staging_dir / "lesson.pdf.yaml",
    )

    assert [card.record.id for card in report.cards] == [
        incomplete.id,
        held.id,
    ]
    assert [action.label for action in report.cards[0].actions] == [
        "Add a reading",
        "Add a meaning",
        "Review both Japanese examples",
    ]
    assert [code for code, _reason in _details(report.cards[0].actions[0])] == [
        "reading-hold",
        "missing-reading",
    ]
    assert [action.label for action in report.cards[1].actions] == [
        "This reading is not listed by jpdb; keep it for another decision."
    ]
    assert _details(report.cards[1].actions[0]) == (
        ("reading-hold", HOLD_UNKNOWN_READING),
    )
    assert [record.to_dict() for record in (incomplete, held)] == before


def test_check_cards_exposes_validation_authority_and_overlap_details(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    _deck(
        config,
        "first",
        intake_tag="first",
        include_tags=["first", "shared"],
    )
    _deck(
        config,
        "second",
        intake_tag="second",
        include_tags=["second", "shared"],
    )
    record = _record(
        record_id="word:話す:はなす",
        reading="はなす",
        meanings=["to speak"],
        tags=["shared"],
        raw_fields={EXAMPLE_AUTHORITY_KEY: "not-an-authority"},
    )
    record.furigana = "話[はなす"

    [checked] = check_cards(config, [record], source="review.yaml").cards

    assert [action.label for action in checked.actions] == [
        "Correct this field",
        "Repair the example approval",
        "Choose a study deck",
    ]
    assert [detail.code for action in checked.actions for detail in action.details] == [
        "unbalanced-furigana",
        "example-authority-invalid",
        "deck-overlap",
    ]
    assert checked.actions[0].details[0].level == "error"
    assert "First" in checked.actions[-1].details[0].reason
    assert "Second" in checked.actions[-1].details[0].reason


def test_check_cards_keeps_invalid_identity_findings_on_their_snapshot_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    _deck(config, "lesson", intake_tag="lesson", include_tags=["lesson"])
    first = _record(
        record_id="word:話す:はなす",
        reading="はなす",
        meanings=["to speak"],
        tags=["lesson"],
        raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
    )
    duplicate = _record(
        record_id=first.id,
        reading="はなす",
        meanings=[],
        raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
    )
    missing_id = _record(
        record_id="",
        reading="はなす",
        meanings=["to speak"],
        raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
    )
    readingless_id = _record(
        record_id="word:話す:",
        reading="はなす",
        meanings=["to speak"],
        raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
    )
    missing_expression = replace(
        _record(
            record_id="word::はなす",
            reading="はなす",
            meanings=["to speak"],
            raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
        ),
        expression="",
    )
    held_identity = annotate(
        _record(
            record_id="word:話す:ほか",
            reading="ほか",
            meanings=["to speak"],
            tags=["lesson"],
            raw_fields={EXAMPLE_AUTHORITY_KEY: EXAMPLE_AUTHORITY_STAGING},
        ),
        hold_reason=HOLD_UNKNOWN_READING,
    )
    evaluated: list[tuple[str, ...]] = []
    real_evaluate = card_check_module.evaluate_prospective_deck_ownership

    def evaluate(config: ProjectConfig, records: object):
        supplied = tuple(records)  # type: ignore[arg-type]
        evaluated.append(tuple(record.id for record in supplied))
        return real_evaluate(config, supplied)

    monkeypatch.setattr(
        card_check_module,
        "evaluate_prospective_deck_ownership",
        evaluate,
    )

    report = check_cards(
        config,
        [
            first,
            duplicate,
            missing_id,
            readingless_id,
            missing_expression,
            held_identity,
        ],
        source="review.yaml",
    )

    assert report.cards[0].actions == ()
    assert [action.label for action in report.cards[1].actions] == [
        "Add a meaning",
        "Re-identify or remove this duplicate",
    ]
    assert [action.label for action in report.cards[2].actions] == [
        "Re-identify this card"
    ]
    assert [action.label for action in report.cards[3].actions] == [
        "Re-identify this card"
    ]
    assert [action.label for action in report.cards[4].actions] == [
        "Re-identify this card"
    ]
    assert [action.label for action in report.cards[5].actions] == [
        "This reading is not listed by jpdb; keep it for another decision."
    ]
    assert _kind(report.cards[3].actions[0].kind) == "identity"
    assert evaluated == [()]

    held = check_cards(
        config,
        [annotate(missing_id, hold_reason=HOLD_MISSING_READING)],
        source="review.yaml",
        reidentifiable=False,
    )
    assert [action.label for action in held.cards[0].actions] == [
        "Keep this card for another identity decision"
    ]
    assert held.cards[0].actions[0].kind == "hold"

    review_blocked = check_cards(
        config,
        [
            _record(
                record_id="word:話す:はなす",
                reading="はなす",
                meanings=["to speak"],
                tags=["lesson"],
            )
        ],
        source="review.yaml",
        approvable=False,
    )
    assert [action.label for action in review_blocked.cards[0].actions] == [
        "Keep these examples for another review decision"
    ]
    assert review_blocked.cards[0].actions[0].kind == "hold"
