"""W4.1's shared, read-only deck-assignment plan."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from japanese_anki.application.assignment import (
    AssignmentError,
    assignable_word_decks,
    plan_deck_assignment,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import save_records_json
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


def _project(tmp_path: Path, decks: dict[str, dict[str, object]]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    deck_dir = tmp_path / "decks"
    deck_dir.mkdir()
    for stem, deck in decks.items():
        (deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump({"deck": deck}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return ProjectConfig.load(tmp_path)


def _word_deck(name: str, tag: str, **overrides: object) -> dict[str, object]:
    deck: dict[str, object] = {
        "name": name,
        "source": "../vocabulary.json",
        "include_tags": [tag],
        "intake_tag": tag,
    }
    deck.update(overrides)
    return deck


def _record(*, tags: list[str], imported_from: str = "page-a.pdf") -> VocabularyRecord:
    return VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        tags=tags,
        source=SourceReference(type="extract", imported_from=imported_from),
    )


def test_only_assignable_word_decks_are_destinations_and_stem_is_authority(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {
            "lesson": _word_deck("Lesson deck", "lesson-intake"),
            "archive": {
                "name": "Archive",
                "source": "../vocabulary.json",
                "include_tags": ["archive"],
            },
            "patterns": {
                "name": "Patterns",
                "kind": "pattern",
                "include_tags": ["pattern-intake"],
                "intake_tag": "pattern-intake",
            },
            "drills": {"name": "Drills", "kind": "conjugation"},
        },
    )

    decks = assignable_word_decks(config)

    assert [(deck.stem, deck.name, deck.intake_tag) for deck in decks] == [
        ("lesson", "Lesson deck", "lesson-intake")
    ]
    plan = plan_deck_assignment(config, _record(tags=[]), "lesson")
    assert plan.destination.stem == "lesson"
    assert plan.destination.intake_tag == "lesson-intake"
    with pytest.raises(AssignmentError, match="deck stem"):
        plan_deck_assignment(config, _record(tags=[]), "lesson-intake")


def test_plan_preserves_duplicate_proposals_and_replaces_only_ownership_tags(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    current = _record(tags=["week-a", "personal"], imported_from="page-a.pdf")
    sibling = _record(tags=["from-second-source"], imported_from="page-b.pdf")

    plan = plan_deck_assignment(config, current, "week-b", sibling_proposals=[sibling])

    assert [item.imported_from for item in plan.proposals] == [
        "page-a.pdf",
        "page-b.pdf",
    ]
    assert plan.tag_diff.before == ("week-a", "personal")
    assert plan.tag_diff.after == ("personal", "week-b")
    assert plan.tag_diff.removed == ("week-a",)
    assert plan.tag_diff.added == ("week-b",)
    assert [item.stem for item in plan.memberships if item.takes] == ["week-b"]
    assert plan.assigned_record.tags == list(plan.tag_diff.after)
    assert plan.prospective_record.tags == list(plan.tag_diff.after)


def test_existing_single_owner_cannot_be_silently_reassigned(tmp_path: Path) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    existing = _record(tags=["week-a", "curated"])
    save_records_json(config.normalized_file, [existing])

    with pytest.raises(AssignmentError, match="already belongs to Week A"):
        plan_deck_assignment(
            config,
            _record(tags=["week-b"]),
            "week-b",
        )

    unchanged_owner = plan_deck_assignment(
        config, _record(tags=["new-source"]), "week-a"
    )
    assert unchanged_owner.existing_owner == "week-a"
    assert unchanged_owner.tag_diff.before == ("new-source",)
    assert unchanged_owner.assigned_record.tags == ["new-source", "week-a"]
    assert unchanged_owner.prospective_record.tags == [
        "curated",
        "new-source",
        "week-a",
    ]

    save_records_json(
        config.normalized_file, [_record(tags=["week-a", "week-b"])]
    )
    with pytest.raises(AssignmentError, match="more than one word deck"):
        plan_deck_assignment(
            config,
            _record(tags=[]),
            "week-a",
        )

    save_records_json(config.normalized_file, [_record(tags=["curated"])])
    first_owner = plan_deck_assignment(
        config, _record(tags=[]), "week-b"
    )
    assert first_owner.existing_owner is None


def test_plan_projects_inline_overrides_through_the_real_deck_resolver(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {"week-a": _word_deck("Week A", "week-a")},
    )
    deck_path = config.deck_dir / "week-a.yaml"
    document = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
    document["notes"] = [_record(tags=["inline-only"]).to_dict()]
    deck_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(AssignmentError, match="does not select the card"):
        plan_deck_assignment(config, _record(tags=[]), "week-a")


def test_a_word_deck_reading_another_source_still_counts_as_an_owner(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "legacy": {
                "name": "Legacy",
                "source": "../legacy.json",
                "include_tags": ["legacy"],
            },
        },
    )
    save_records_json(tmp_path / "legacy.json", [_record(tags=["legacy"])])

    with pytest.raises(AssignmentError, match="would also select Legacy"):
        plan_deck_assignment(config, _record(tags=[]), "week-a")


@pytest.mark.parametrize(
    ("other_overrides", "proposal_tags", "message"),
    [
        ({"include_tags": ["week-b", "shared"]}, ["shared"], "also select Week B"),
        ({}, ["blocked"], "does not select the card"),
    ],
)
def test_plan_proves_exactly_one_destination_with_real_deck_rules(
    tmp_path: Path,
    other_overrides: dict[str, object],
    proposal_tags: list[str],
    message: str,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a", exclude_tags=["blocked"]),
            "week-b": _word_deck("Week B", "week-b", **other_overrides),
        },
    )

    with pytest.raises(AssignmentError, match=message):
        plan_deck_assignment(config, _record(tags=proposal_tags), "week-a")
