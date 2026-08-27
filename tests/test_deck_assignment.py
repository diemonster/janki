"""W4.1's shared, read-only deck-assignment plan."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import japanese_anki.application.assignment as assignment_module
import japanese_anki.application.promotion as promotion_application
from japanese_anki.application.assignment import (
    AssignmentError,
    assignable_word_decks,
    evaluate_deck_ownership,
    evaluate_prospective_deck_ownership,
    plan_deck_assignment,
)
from japanese_anki.application.promotion import (
    POST_READING_GATES,
    decide_promotion,
    execute_promotion,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import save_records_json
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.promote import PromoteError
from japanese_anki.staging import write_staging

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


def test_ownership_evaluation_distinguishes_exact_zero_multiple_and_unreadable(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    exact = _record(tags=["week-a"])
    unassigned = VocabularyRecord.from_dict(
        {
            **_record(tags=[]).to_dict(),
            "id": "word:聞く:きく",
            "expression": "聞く",
            "reading": "きく",
        }
    )
    multiple = VocabularyRecord.from_dict(
        {
            **_record(tags=["week-a", "week-b"]).to_dict(),
            "id": "word:読む:よむ",
            "expression": "読む",
            "reading": "よむ",
        }
    )
    collection = [exact, unassigned, multiple]

    evaluations = evaluate_deck_ownership(config, collection, collection)

    assert [item.state for item in evaluations] == [
        "exactly_one",
        "unassigned",
        "multiple",
    ]
    assert evaluations[0].owner_stems == ("week-a",)
    assert evaluations[2].owner_stems == ("week-a", "week-b")

    save_records_json(config.normalized_file, [exact])
    merged = evaluate_prospective_deck_ownership(
        config, [_record(tags=["week-b"])]
    )
    assert merged[0].state == "multiple"

    (config.deck_dir / "week-b.yaml").write_text(
        "deck: [not, a, mapping]\n", encoding="utf-8"
    )
    [unreadable] = evaluate_deck_ownership(config, [exact], [exact])
    assert unreadable.state == "unreadable"
    assert "week-b.yaml" in " ".join(unreadable.unreadable_decks)


def test_ownership_projects_each_deck_once_for_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dashboard-sized batches must not reload the corpus per staged card."""
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    records = [
        _record(tags=["week-a"]),
        VocabularyRecord.from_dict(
            {
                **_record(tags=["week-b"]).to_dict(),
                "id": "word:聞く:きく",
                "expression": "聞く",
                "reading": "きく",
            }
        ),
        VocabularyRecord.from_dict(
            {
                **_record(tags=["week-a"]).to_dict(),
                "id": "word:読む:よむ",
                "expression": "読む",
                "reading": "よむ",
            }
        ),
    ]
    real_project = assignment_module.project_deck_records
    projected: list[str] = []

    def observe_project(
        deck_path: Path,
        source_path: Path,
        source_records: list[VocabularyRecord],
    ) -> tuple[dict[str, object], list[VocabularyRecord]]:
        projected.append(deck_path.name)
        return real_project(deck_path, source_path, source_records)

    monkeypatch.setattr(
        assignment_module,
        "project_deck_records",
        observe_project,
    )

    evaluations = evaluate_deck_ownership(config, records, records)

    assert [item.state for item in evaluations] == [
        "exactly_one",
        "exactly_one",
        "exactly_one",
    ]
    assert projected == ["week-a.yaml", "week-b.yaml"]


def test_batched_ownership_keeps_static_source_membership_without_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    target = _record(tags=["week-a"])
    save_records_json(tmp_path / "legacy.json", [_record(tags=["legacy"])])
    real_project = assignment_module.project_deck_records
    projected: list[str] = []

    def observe_project(
        deck_path: Path,
        source_path: Path,
        source_records: list[VocabularyRecord],
    ) -> tuple[dict[str, object], list[VocabularyRecord]]:
        projected.append(deck_path.name)
        return real_project(deck_path, source_path, source_records)

    monkeypatch.setattr(
        assignment_module,
        "project_deck_records",
        observe_project,
    )

    [evaluation] = evaluate_deck_ownership(config, [target], [target])

    assert evaluation.state == "multiple"
    assert evaluation.owner_stems == ("legacy", "week-a")
    assert projected == ["week-a.yaml"]


@pytest.mark.parametrize("message", ["prospective deck projection failed", ""])
def test_one_batch_projection_failure_marks_every_target_unreadable_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    first = _record(tags=["week-a"])
    second = VocabularyRecord.from_dict(
        {
            **_record(tags=["week-b"]).to_dict(),
            "id": "word:聞く:きく",
            "expression": "聞く",
            "reading": "きく",
        }
    )
    attempts = 0

    def refuse_projection(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise AssignmentError(message)

    monkeypatch.setattr(
        assignment_module,
        "project_deck_records",
        refuse_projection,
    )

    evaluations = evaluate_deck_ownership(
        config,
        [first, second],
        [first, second],
    )

    assert attempts == 1
    assert [item.state for item in evaluations] == ["unreadable", "unreadable"]
    assert all(item.memberships == () for item in evaluations)
    assert all(
        item.unreadable_decks == (message,)
        for item in evaluations
    )


@pytest.mark.parametrize(
    ("existing_tags", "incoming_tags", "state", "gate"),
    [
        (["week-a"], ["week-a"], "lands", ""),
        ([], [], "blocked", "deck"),
        (["week-a"], ["week-b"], "blocked", "deck"),
    ],
)
def test_promotion_requires_one_owner_after_the_actual_merge(
    tmp_path: Path,
    existing_tags: list[str],
    incoming_tags: list[str],
    state: str,
    gate: str,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    save_records_json(config.normalized_file, [_record(tags=existing_tags)])
    path = config.staging_dir / "page.yaml"
    write_staging(
        path,
        [_record(tags=incoming_tags)],
        {"source_file": "page-a.pdf", "review_notes": "reviewed"},
    )

    decision = decide_promotion(config, path, skip_reading_check=True)

    assert decision.state == state
    assert decision.gate == gate
    if gate:
        assert gate in POST_READING_GATES
        if existing_tags:
            assert "Week A" in str(decision.error)
            assert "Week B" in str(decision.error)
        else:
            assert "deck-unassigned" in str(decision.error)


def test_execution_refuses_deck_rules_changed_after_the_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(
        tmp_path,
        {
            "week-a": _word_deck("Week A", "week-a"),
            "week-b": _word_deck("Week B", "week-b"),
        },
    )
    path = config.staging_dir / "page.yaml"
    write_staging(
        path,
        [_record(tags=["week-a"])],
        {"source_file": "page-a.pdf", "review_notes": "reviewed"},
    )
    decision = decide_promotion(config, path, skip_reading_check=True)
    before_collection = config.normalized_file.read_bytes()
    before_staging = path.read_bytes()

    (config.deck_dir / "week-b.yaml").write_text(
        yaml.safe_dump(
            {
                "deck": _word_deck(
                    "Week B",
                    "week-b",
                    include_tags=["week-a", "week-b"],
                )
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssignmentError, match="deck-overlap"):
        execute_promotion(config, decision)

    assert config.normalized_file.read_bytes() == before_collection
    assert path.read_bytes() == before_staging

    (config.deck_dir / "week-a.yaml").write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": "Week A",
                    "source": "../vocabulary.json",
                    "include_tags": ["other"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PromoteError, match="deck-ownership-stale"):
        execute_promotion(config, decision)

    assert config.normalized_file.read_bytes() == before_collection
    assert path.read_bytes() == before_staging

    for stem, deck in {
        "week-a": _word_deck("Week A", "week-a"),
        "week-b": _word_deck("Week B", "week-b"),
    }.items():
        (config.deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump({"deck": deck}, sort_keys=False),
            encoding="utf-8",
        )

    real_lock = promotion_application.exclusive_path_lock
    real_require = promotion_application.require_exact_deck_ownership
    real_save = promotion_application.save_records_json
    deck_lock_held = False

    @contextmanager
    def tracking_lock(target: Path):
        nonlocal deck_lock_held
        with real_lock(target):
            is_deck_lock = target == config.deck_dir
            if is_deck_lock:
                deck_lock_held = True
            try:
                yield
            finally:
                if is_deck_lock:
                    deck_lock_held = False

    def tracking_require(*args: object, **kwargs: object):
        assert deck_lock_held
        return real_require(*args, **kwargs)

    def tracking_save(*args: object, **kwargs: object) -> None:
        assert deck_lock_held
        real_save(*args, **kwargs)

    monkeypatch.setattr(promotion_application, "exclusive_path_lock", tracking_lock)
    monkeypatch.setattr(
        promotion_application,
        "require_exact_deck_ownership",
        tracking_require,
    )
    monkeypatch.setattr(promotion_application, "save_records_json", tracking_save)

    result = execute_promotion(config, decision)

    assert result.state == "landed"
