"""Exact Assistant plans for assigning staged cards to study decks."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import staging
from japanese_anki.application import assistant_assignment
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    ProposalContext,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import (
    exclusive_path_lock,
    save_records_json,
    save_records_json_locked,
)
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.workbench import review


def _record(expression: str, reading: str, *, tags: list[str]) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["fixture meaning"],
        tags=tags,
        usage_notes="A fixture note that assignment must preserve.",
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            row=7,
        ),
    )


def _write_deck(config: ProjectConfig, *, intake_tag: str = "lesson-intake") -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / "lesson.yaml"
    path.write_text(
        "deck:\n"
        "  name: Lesson deck\n"
        "  source: ../normalized.json\n"
        f"  include_tags: [{intake_tag}]\n"
        f"  intake_tag: {intake_tag}\n",
        encoding="utf-8",
    )
    return path


def _project(
    tmp_path: Path,
) -> tuple[ProjectConfig, Path, Path, tuple[VocabularyRecord, ...]]:
    (tmp_path / "janki.toml").write_text(
        "[project]\nname = \"Assignment fixture\"\n"
        "[paths]\n"
        "scan_inbox = \"inbox\"\n"
        "staging_dir = \"staging\"\n"
        "normalized_file = \"normalized.json\"\n"
        "deck_dir = \"decks\"\n"
        "ledger_file = \"ledger.json\"\n"
        "patterns_file = \"patterns.json\"\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.staging_dir.mkdir()
    save_records_json(config.normalized_file, [])
    deck_path = _write_deck(config)
    records = (
        _record("食べる", "たべる", tags=["personal"]),
        _record("話す", "はなす", tags=["leave-alone"]),
    )
    proposal_path = config.staging_dir / "lesson.pdf.yaml"
    staging.write_staging(
        proposal_path,
        records,
        {"source_file": "lesson.pdf", "review_notes": "preserve this metadata"},
    )
    return config, proposal_path, deck_path, records


def _resources(config: ProjectConfig) -> tuple[str, str]:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)["data"][
        "resources"
    ]
    proposal_id = next(
        str(item["resource_id"])
        for item in catalog
        if item["kind"] == "proposal"
        and item.get("proposal_kind") == "source_extraction"
    )
    destination_id = next(
        str(item["resource_id"])
        for item in catalog
        if item["kind"] == "deck" and item["title"] == "Lesson deck"
    )
    return proposal_id, destination_id


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_plan_is_read_only_and_binds_exact_tag_diff_ownership_and_snapshots(
    tmp_path: Path,
) -> None:
    config, proposal_path, deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    before = _files(tmp_path)

    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[0].id,),
        instruction="Put this card in Lesson deck.",
    )

    assert _files(tmp_path) == before
    assert plan.proposal_path == proposal_path.resolve()
    assert plan.destination_path == deck_path.resolve()
    assert plan.selected_record_ids == (records[0].id,)
    assert plan.records == records
    assert plan.assigned_records == (
        replace(records[0], tags=["personal", "lesson-intake"]),
        records[1],
    )
    assert plan.staging_snapshot == hashlib.sha256(
        proposal_path.read_bytes()
    ).hexdigest()
    assert plan.projection["proposal"] == {
        "resource_id": proposal_id,
        "kind": "source_extraction",
        "source_name": "lesson.pdf",
        "configured_file": "staging/lesson.pdf.yaml",
        "sha256": plan.staging_snapshot,
    }
    assert plan.projection["destination"] == {
        "resource_id": destination_id,
        "name": "Lesson deck",
        "stem": "lesson",
        "intake_tag": "lesson-intake",
        # A shared deck: no record scope, so every assignment below is the same
        # card gaining a tag rather than an independent copy.
        "scope_id": "",
        "configured_file": "decks/lesson.yaml",
        "deck_sha256": hashlib.sha256(deck_path.read_bytes()).hexdigest(),
    }
    assert plan.projection["selection"]["target_record_ids"] == [records[0].id]
    [change] = plan.projection["selection"]["assignments"]
    assert change["record_id"] == records[0].id
    assert change["target_record_id"] == records[0].id
    assert change["expression"] == "食べる"
    assert change["reading"] == "たべる"
    assert change["tags"] == {
        "before": ["personal"],
        "after": ["personal", "lesson-intake"],
        "removed": [],
        "added": ["lesson-intake"],
    }
    assert len(change["assigned_record_sha256"]) == 64
    assert len(change["prospective_record_sha256"]) == 64
    assert [
        item["deck"]
        for item in change["resulting_memberships"]
        if item["takes"]
    ] == ["Lesson deck"]
    assert plan.projection["writes"] == {
        "staging_proposal": "staging/lesson.pdf.yaml",
        "canonical_cards": False,
        "deck_definition": False,
    }
    assert plan.projection["service_fingerprint"] == plan.service_fingerprint


def test_execute_replans_then_changes_only_selected_staging_rows(
    tmp_path: Path,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[1].id,),
        instruction="Put the second card in Lesson deck.",
    )

    result = assistant_assignment.execute_assignment(config, plan)

    updated, metadata = staging.read_staging(proposal_path)
    assert result.assigned_record_ids == (records[1].id,)
    assert result.plan.fingerprint == plan.fingerprint
    assert updated == [
        records[0],
        replace(records[1], tags=["leave-alone", "lesson-intake"]),
    ]
    assert metadata["review_notes"] == "preserve this metadata"
    assert config.normalized_file.read_text(encoding="utf-8") == "[]\n"


def test_one_plan_assigns_an_explicit_multi_card_batch_atomically(
    tmp_path: Path,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[1].id, records[0].id),
        instruction="Assign both selected cards together.",
    )

    result = assistant_assignment.execute_assignment(config, plan)

    updated, _metadata = staging.read_staging(proposal_path)
    assert result.assigned_record_ids == (records[1].id, records[0].id)
    assert updated == [
        replace(records[0], tags=["personal", "lesson-intake"]),
        replace(records[1], tags=["leave-alone", "lesson-intake"]),
    ]


@pytest.mark.parametrize(
    "drift",
    ["staging", "destination", "canonical", "sibling"],
)
def test_execute_refuses_any_fresh_assignment_drift_without_overwriting(
    tmp_path: Path,
    drift: str,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[0].id,),
        instruction="Put this card in Lesson deck.",
    )
    if drift == "staging":
        current, metadata = staging.read_staging(proposal_path)
        current[1] = replace(current[1], tags=["concurrent"])
        staging.write_staging(proposal_path, current, metadata, force=True)
    elif drift == "destination":
        _write_deck(config, intake_tag="replacement-intake")
    elif drift == "canonical":
        save_records_json(
            config.normalized_file,
            [replace(records[0], tags=["canonical-only"])],
        )
    else:
        sibling = replace(
            records[0],
            source=replace(records[0].source, imported_from="other.pdf", row=2),
        )
        staging.write_staging(
            config.staging_dir / "other.pdf.yaml",
            [sibling],
            {"source_file": "other.pdf"},
        )
    changed = proposal_path.read_bytes()

    with pytest.raises(assistant_assignment.AssistantAssignmentError, match="changed"):
        assistant_assignment.execute_assignment(config, plan)

    assert proposal_path.read_bytes() == changed


@pytest.mark.parametrize("dependency", ["destination", "canonical", "sibling"])
def test_execute_waits_for_every_assignment_dependency_then_replans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    config, proposal_path, deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    sibling_path = config.staging_dir / "other.pdf.yaml"
    sibling_record = replace(
        records[0],
        source=replace(records[0].source, imported_from="other.pdf", row=2),
    )
    staging.write_staging(
        sibling_path,
        [sibling_record],
        {"source_file": "other.pdf"},
    )
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[0].id,),
        instruction="Put this card in Lesson deck.",
    )
    selected_before = proposal_path.read_bytes()
    dependency_path = {
        "destination": deck_path,
        "canonical": config.normalized_file,
        "sibling": sibling_path,
    }[dependency]
    lock_attempted = threading.Event()
    real_lock = exclusive_path_lock

    @contextmanager
    def observed_lock(path: Path) -> Iterator[None]:
        if Path(os.path.realpath(path)) == Path(os.path.realpath(dependency_path)):
            lock_attempted.set()
        with real_lock(path):
            yield

    monkeypatch.setattr(
        assistant_assignment,
        "exclusive_path_lock",
        observed_lock,
        raising=False,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with real_lock(dependency_path):
            pending = pool.submit(
                assistant_assignment.execute_assignment,
                config,
                plan,
            )
            assert lock_attempted.wait(timeout=2), (
                f"assignment execution never locked its {dependency} dependency"
            )
            if dependency == "destination":
                _write_deck(config, intake_tag="replacement-intake")
            elif dependency == "canonical":
                save_records_json_locked(
                    config.normalized_file,
                    [replace(records[0], tags=["canonical-only"])],
                )
            else:
                current_bytes = sibling_path.read_bytes()
                replacement_text = staging.render_staging_update(
                    current_bytes,
                    [
                        replace(
                            sibling_record,
                            source=replace(sibling_record.source, row=99),
                        )
                    ],
                    source=str(sibling_path),
                )
                review.bound_replace_under_lock(
                    sibling_path,
                    replacement_text,
                    current_bytes,
                    label="sibling staging file",
                )

        with pytest.raises(
            assistant_assignment.AssistantAssignmentError,
            match="changed|cannot be assigned|Could not plan",
        ):
            pending.result(timeout=5)

    assert proposal_path.read_bytes() == selected_before


@pytest.mark.parametrize("dependency", ["destination", "canonical", "sibling"])
def test_execute_keeps_assignment_dependencies_locked_through_staging_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    config, proposal_path, deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    sibling_path = config.staging_dir / "other.pdf.yaml"
    sibling_record = replace(
        records[0],
        source=replace(records[0].source, imported_from="other.pdf", row=2),
    )
    staging.write_staging(
        sibling_path,
        [sibling_record],
        {"source_file": "other.pdf"},
    )
    plan = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=(records[0].id,),
        instruction="Put this card in Lesson deck.",
    )
    dependency_path = {
        "destination": deck_path,
        "canonical": config.normalized_file,
        "sibling": sibling_path,
    }[dependency]
    real_lock = exclusive_path_lock
    real_replace = review.bound_replace_under_lock
    writer_started = threading.Event()
    writer_acquired = threading.Event()
    writers: list[threading.Thread] = []
    writer_errors: list[BaseException] = []

    def competing_write() -> None:
        try:
            writer_started.set()
            with real_lock(dependency_path):
                writer_acquired.set()
                if dependency == "destination":
                    _write_deck(config, intake_tag="replacement-intake")
                elif dependency == "canonical":
                    save_records_json_locked(
                        config.normalized_file,
                        [replace(records[0], tags=["canonical-only"])],
                    )
                else:
                    current_bytes = sibling_path.read_bytes()
                    replacement_text = staging.render_staging_update(
                        current_bytes,
                        [
                            replace(
                                sibling_record,
                                source=replace(sibling_record.source, row=99),
                            )
                        ],
                        source=str(sibling_path),
                    )
                    real_replace(
                        sibling_path,
                        replacement_text,
                        current_bytes,
                        label="sibling staging file",
                    )
        except BaseException as exc:  # pragma: no cover - asserted in main thread
            writer_errors.append(exc)

    def replace_at_commit(
        target: Path,
        text: str,
        snapshot: bytes,
        *,
        label: str,
    ) -> None:
        writer = threading.Thread(target=competing_write, daemon=True)
        writers.append(writer)
        writer.start()
        assert writer_started.wait(timeout=2)
        assert not writer_acquired.wait(timeout=0.2), (
            f"assignment released its {dependency} dependency before staging commit"
        )
        real_replace(target, text, snapshot, label=label)

    monkeypatch.setattr(review, "bound_replace_under_lock", replace_at_commit)

    result = assistant_assignment.execute_assignment(config, plan)
    assert result.assigned_record_ids == (records[0].id,)
    assert len(writers) == 1
    writers[0].join(timeout=5)
    assert not writers[0].is_alive()
    assert writer_acquired.is_set()
    assert writer_errors == []
    updated, _metadata = staging.read_staging(proposal_path)
    assert updated[0].tags == ["personal", "lesson-intake"]


def test_plan_requires_source_proposal_exact_record_and_assignable_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)

    with pytest.raises(assistant_assignment.AssistantAssignmentError, match="absent"):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=("word:missing",),
            instruction="Assign it.",
        )

    duplicate = [records[0], replace(records[0], tags=["second occurrence"])]
    staging.write_staging(
        proposal_path,
        duplicate,
        {"source_file": "lesson.pdf"},
        force=True,
    )
    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="appears more than once",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(records[0].id,),
            instruction="Assign it.",
        )

    class EnrichmentBroker:
        def __init__(self, _config: ProjectConfig) -> None:
            pass

        def proposal_context(self, resource_id: str) -> ProposalContext:
            return ProposalContext(
                resource_id=resource_id,
                proposal_kind="ai_enrichment",
                path=proposal_path.absolute(),
                proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
            )

    monkeypatch.setattr(
        assistant_assignment,
        "AssistantContextBroker",
        EnrichmentBroker,
    )
    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="source-extraction proposals only",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(records[0].id,),
            instruction="Assign it.",
        )


def test_plan_rejects_forged_targets_empty_selection_and_existing_reassignment(
    tmp_path: Path,
) -> None:
    config, _proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)

    with pytest.raises(assistant_assignment.AssistantAssignmentError, match="at least one"):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(),
            instruction="Assign cards.",
        )
    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="(?i)destination",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id="resource_forged",
            record_ids=(records[0].id,),
            instruction="Assign cards.",
        )

    old = config.deck_dir / "old.yaml"
    old.write_text(
        "deck:\n"
        "  name: Old deck\n"
        "  source: ../normalized.json\n"
        "  include_tags: [old-intake]\n"
        "  intake_tag: old-intake\n",
        encoding="utf-8",
    )
    save_records_json(
        config.normalized_file,
        [replace(records[0], tags=["old-intake"])],
    )
    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="explicit reassignment review",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(records[0].id,),
            instruction="Move this card.",
        )


def test_plan_never_follows_a_proposal_swapped_to_a_symlink_after_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    saved = proposal_path.with_name("saved-proposal.txt")
    real_resolve_destination = assistant_assignment._resolve_destination

    def resolve_then_swap(fresh: ProjectConfig, resource_id: str) -> Path:
        destination = real_resolve_destination(fresh, resource_id)
        proposal_path.rename(saved)
        proposal_path.symlink_to(saved)
        return destination

    monkeypatch.setattr(
        assistant_assignment,
        "_resolve_destination",
        resolve_then_swap,
    )

    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="Could not read the selected staging proposal",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(records[0].id,),
            instruction="Assign this card.",
        )

    assert proposal_path.is_symlink()
    assert saved.read_bytes()


def test_plan_rechecks_the_brokers_exact_proposal_byte_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, proposal_path, _deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    monkeypatch.setattr(
        assistant_assignment,
        "_resolve_proposal",
        lambda _config, resource_id: ProposalContext(
            resource_id=resource_id,
            proposal_kind="source_extraction",
            path=proposal_path.absolute(),
            proposal_sha256="0" * 64,
        ),
    )

    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="changed while it was being resolved",
    ):
        assistant_assignment.plan_assignment(
            config,
            proposal_resource_id=proposal_id,
            destination_resource_id=destination_id,
            record_ids=(records[0].id,),
            instruction="Assign this card.",
        )


# --- the path-keyed entry the CLI and a study job call ------------------------
#
# Contracts §9.3: factor `plan_assignment_for_paths` out of the existing
# preparation, have the resource-id-keyed entry delegate to it, and give the
# CLI the same call. `execute_assignment` stays the sole writer.


def test_a_path_keyed_plan_matches_the_resource_keyed_one_and_writes_once(
    tmp_path: Path,
) -> None:
    """Two ways to name the same two ends; one planner, one writer."""

    config, proposal_path, deck_path, records = _project(tmp_path)
    proposal_id, destination_id = _resources(config)
    selected = (records[0].id,)

    by_id = assistant_assignment.plan_assignment(
        config,
        proposal_resource_id=proposal_id,
        destination_resource_id=destination_id,
        record_ids=selected,
        instruction="Assign this card.",
    )
    by_path = assistant_assignment.plan_assignment_for_paths(
        config,
        proposal_path=proposal_path,
        destination_path=deck_path,
        record_ids=selected,
        instruction="Assign this card.",
    )

    # Same resolution, same service plan, same bytes to write.
    assert by_path.proposal_path == by_id.proposal_path
    assert by_path.destination_path == by_id.destination_path
    assert by_path.service_fingerprint == by_id.service_fingerprint
    assert by_path.assigned_records == by_id.assigned_records
    # The plan says by which key it must be re-prepared, and a path-keyed plan
    # carries no resource ids to re-prepare from.
    assert by_path.proposal_resource_id == ""
    assert by_path.destination_resource_id == ""
    assert by_id.proposal_resource_id == proposal_id

    execution = assistant_assignment.execute_assignment(config, by_path)
    assert execution.assigned_record_ids == selected
    written, meta = staging.read_staging(proposal_path)
    assert meta["review_notes"] == "preserve this metadata"
    assert "lesson-intake" in written[0].tags


def test_the_path_keyed_entry_keeps_every_refusal_the_broker_route_has(
    tmp_path: Path,
) -> None:
    """A path outside the staging census, and a proposal of the wrong kind."""

    config, proposal_path, deck_path, records = _project(tmp_path)
    outside = tmp_path / "elsewhere.yaml"
    staging.write_staging(outside, records, {"source_file": "lesson.pdf"})

    with pytest.raises(assistant_assignment.AssistantAssignmentError) as error:
        assistant_assignment.plan_assignment_for_paths(
            config,
            proposal_path=outside,
            destination_path=deck_path,
            record_ids=(records[0].id,),
            instruction="Assign this card.",
        )
    assert "staging" in str(error.value)

    # The real proposal is untouched by that refusal.
    assert staging.read_staging(proposal_path)[0] == list(records)


def test_the_path_keyed_entry_refuses_a_proposal_of_the_wrong_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one proposal-kind rule applies wherever a proposal is resolved."""

    config, proposal_path, deck_path, records = _project(tmp_path)

    class EnrichmentBroker:
        def __init__(self, _config: ProjectConfig) -> None:
            pass

        def resource_id_for_proposal(self, _path: Path) -> str:
            return "resource_proposal"

        def proposal_context(self, resource_id: str) -> ProposalContext:
            return ProposalContext(
                resource_id=resource_id,
                proposal_kind="ai_enrichment",
                path=proposal_path.absolute(),
                proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
            )

    monkeypatch.setattr(
        assistant_assignment, "AssistantContextBroker", EnrichmentBroker
    )
    with pytest.raises(
        assistant_assignment.AssistantAssignmentError,
        match="source-extraction proposals only",
    ):
        assistant_assignment.plan_assignment_for_paths(
            config,
            proposal_path=proposal_path,
            destination_path=deck_path,
            record_ids=(records[0].id,),
            instruction="Assign this card.",
        )
    assert staging.read_staging(proposal_path)[0] == list(records)


def test_a_plan_names_both_ends_by_resource_id_or_neither(tmp_path: Path) -> None:
    """Half a key cannot be re-prepared, so it is not a plan at all.

    The invariant is what makes `execute_assignment`'s re-preparation exact:
    a plan resolved from two local paths and then re-planned from a resource
    id would be a different resolution wearing the same fingerprint, which is
    the one thing the comparison there cannot catch.
    """

    config, proposal_path, deck_path, records = _project(tmp_path)
    plan = assistant_assignment.plan_assignment_for_paths(
        config,
        proposal_path=proposal_path,
        destination_path=deck_path,
        record_ids=(records[0].id,),
        instruction="Assign this card.",
    )

    with pytest.raises(ValueError) as error:
        replace(plan, proposal_resource_id="resource_something")
    assert "both ends by resource id or neither" in str(error.value)
    assert staging.read_staging(proposal_path)[0] == list(records)
