"""Assigning a staged card to a standalone deck makes an independent copy.

The shared original is never re-tagged, re-identified or overwritten: the copy
is a new canonical record under the destination deck's scope, and the staged
row the owner selected is the one that becomes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from japanese_anki import staging
from japanese_anki.application import assistant_assignment
from japanese_anki.application.assignment import (
    AssignmentError,
    assignable_word_decks,
    plan_deck_assignment,
)
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records, save_records_json
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

SCOPE = "ab12"
COPY_ID = f"standalone:{SCOPE}:話す:はなす"


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
    config = ProjectConfig.load(tmp_path)
    config.staging_dir.mkdir(exist_ok=True)
    return config


def _word_deck(name: str, tag: str, **overrides: object) -> dict[str, object]:
    deck: dict[str, object] = {
        "name": name,
        "source": "../vocabulary.json",
        "include_tags": [tag],
        "intake_tag": tag,
    }
    deck.update(overrides)
    return deck


def _record(
    record_id: str = "word:話す:はなす",
    *,
    tags: list[str] | None = None,
    row: int | None = 3,
) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id,
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        tags=list(tags or []),
        source=SourceReference(type="extract", imported_from="page-a.pdf", row=row),
    )


def _standalone_project(tmp_path: Path) -> ProjectConfig:
    return _project(
        tmp_path,
        {
            "lesson": _word_deck("Lesson deck", "lesson"),
            "verbs": _word_deck("Class verbs", "verbs", scope_id=SCOPE),
        },
    )


def test_a_standalone_destination_copies_the_selected_row_under_its_own_scope(
    tmp_path: Path,
) -> None:
    config = _standalone_project(tmp_path)
    shared = _record(tags=["lesson"])
    save_records_json(config.normalized_file, [shared])
    before = config.normalized_file.read_bytes()
    staged = staging.annotate(_record(tags=["personal"]), already_known=True)

    plan = plan_deck_assignment(config, staged, "verbs")

    # The selected staged row still resolves by the id the owner chose it under.
    assert plan.record_id == "word:話す:はなす"
    assert plan.destination.scope_id == SCOPE
    assert plan.assigned_record.id == COPY_ID
    assert plan.prospective_record.id == COPY_ID
    assert plan.assigned_record.expression == "話す"
    assert plan.assigned_record.reading == "はなす"
    assert plan.assigned_record.meanings == ["to speak"]
    # Provenance from the copy back to the row it was made from.
    raw_fields = plan.assigned_record.source.raw_fields
    assert raw_fields["standalone_copy_from"] == "word:話す:はなす"
    assert plan.assigned_record.source.imported_from == "page-a.pdf"
    assert plan.assigned_record.source.row == 3
    # The copy is new here, so "janki already has this word" is stale on it.
    assert "already_known" not in raw_fields
    # Non-ownership tags ride along; the destination's intake tag is added.
    assert plan.tag_diff.after == ("personal", "verbs")
    assert plan.existing_owner is None
    assert [item.stem for item in plan.memberships if item.takes] == ["verbs"]
    # The shared original is untouched, and planning writes nothing.
    assert config.normalized_file.read_bytes() == before
    assert load_records(config.normalized_file)[0].tags == ["lesson"]


def test_repeating_the_assignment_reuses_the_copy_instead_of_minting_another(
    tmp_path: Path,
) -> None:
    config = _standalone_project(tmp_path)
    save_records_json(
        config.normalized_file,
        [_record(tags=["lesson"]), _record(COPY_ID, tags=["personal", "verbs"])],
    )
    staged = _record(tags=["personal"])

    plan = plan_deck_assignment(config, staged, "verbs")

    assert plan.assigned_record.id == COPY_ID
    assert plan.existing_owner == "verbs"
    assert [item.stem for item in plan.memberships if item.takes] == ["verbs"]


def test_a_row_already_in_the_destination_scope_keeps_its_stored_identity(
    tmp_path: Path,
) -> None:
    config = _standalone_project(tmp_path)
    staged = _record(COPY_ID, tags=[])

    plan = plan_deck_assignment(config, staged, "verbs")

    assert plan.assigned_record.id == COPY_ID
    # Nothing was copied from anything: this row already is the scoped card.
    assert "standalone_copy_from" not in plan.assigned_record.source.raw_fields


def test_a_shared_destination_does_not_strip_a_scoped_identity(tmp_path: Path) -> None:
    config = _standalone_project(tmp_path)
    staged = _record(COPY_ID, tags=[])

    with pytest.raises(AssignmentError) as caught:
        plan_deck_assignment(config, staged, "lesson")

    assert "standalone" in str(caught.value)


def test_shared_reassignment_is_still_refused(tmp_path: Path) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    save_records_json(config.normalized_file, [_record(tags=["week-a"])])

    with pytest.raises(AssignmentError, match="already belongs to Week A"):
        plan_deck_assignment(config, _record(tags=["week-a"]), "week-b")


def test_a_standalone_deck_is_an_ordinary_assignable_destination(tmp_path: Path) -> None:
    config = _standalone_project(tmp_path)

    decks = {deck.stem: deck for deck in assignable_word_decks(config)}

    assert set(decks) == {"lesson", "verbs"}
    assert decks["verbs"].scope_id == SCOPE
    assert decks["lesson"].scope_id == ""


# --- the Assistant's exact staged-card assignment -------------------------


def _resources(config: ProjectConfig, deck_title: str) -> tuple[str, str]:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)["data"][
        "resources"
    ]
    proposal_id = next(
        str(item["resource_id"])
        for item in catalog
        if item["kind"] == "proposal" and item.get("proposal_kind") == "source_extraction"
    )
    destination_id = next(
        str(item["resource_id"])
        for item in catalog
        if item["kind"] == "deck" and item["title"] == deck_title
    )
    return proposal_id, destination_id


def _assistant_project(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    config = _standalone_project(tmp_path)
    save_records_json(config.normalized_file, [_record(tags=["lesson"])])
    proposal_path = config.staging_dir / "page-a.pdf.yaml"
    staging.write_staging(
        proposal_path,
        [
            staging.annotate(_record(tags=["personal"]), already_known=True),
            _record("word:食べる:たべる", tags=["leave-alone"], row=4),
        ],
        {"source_file": "page-a.pdf", "review_notes": "preserve this metadata"},
    )
    return config, proposal_path


def test_assistant_assignment_exposes_the_copy_and_returns_its_new_id(
    tmp_path: Path,
) -> None:
    config, proposal_path = _assistant_project(tmp_path)
    proposal_id, destination_id = _resources(config, "Class verbs")
    canonical_before = config.normalized_file.read_bytes()

    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=["word:話す:はなす"],
        instruction="Put this verb in the class verbs deck.",
    )

    projection = plan.projection
    assert projection["destination"]["scope_id"] == SCOPE
    assert projection["destination"]["deck_sha256"]
    assert projection["selection"]["record_ids"] == ["word:話す:はなす"]
    assert projection["selection"]["target_record_ids"] == [COPY_ID]
    assignment_value = projection["selection"]["assignments"][0]
    assert assignment_value["record_id"] == "word:話す:はなす"
    assert assignment_value["target_record_id"] == COPY_ID

    execution = assistant_assignment.execute_assignment(config, plan)

    assert execution.assigned_record_ids == (COPY_ID,)
    rows, metadata = staging.read_staging(proposal_path)
    assert [row.id for row in rows] == [COPY_ID, "word:食べる:たべる"]
    assert rows[0].tags == ["personal", "verbs"]
    assert rows[0].source.raw_fields["standalone_copy_from"] == "word:話す:はなす"
    assert metadata["review_notes"] == "preserve this metadata"
    # Assignment edits the staging file and nothing else.
    assert config.normalized_file.read_bytes() == canonical_before


def test_assistant_assignment_refuses_when_the_deck_scope_changes_after_display(
    tmp_path: Path,
) -> None:
    config, _proposal_path = _assistant_project(tmp_path)
    proposal_id, destination_id = _resources(config, "Class verbs")
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=["word:話す:はなす"],
        instruction="Put this verb in the class verbs deck.",
    )
    (config.deck_dir / "verbs.yaml").write_text(
        yaml.safe_dump(
            {"deck": _word_deck("Class verbs", "verbs", scope_id="cd34")},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(assistant_assignment.AssistantAssignmentError, match="changed"):
        assistant_assignment.execute_assignment(config, plan)
