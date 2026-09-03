"""Plan-bound owner actions over opaque Assistant staging resources."""

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

from japanese_anki import extract, ledger, patterns, staging
from japanese_anki.application import coverage as coverage_application
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.application.assistant_staging_actions import (
    AssistantStagingActionError,
    execute_coverage_approval,
    execute_reidentification,
    execute_staged_deletion,
    plan_coverage_approval,
    plan_reidentification,
    plan_staged_deletion,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.extract import ExtractionResult, SourceUnit
from japanese_anki.io import (
    exclusive_path_lock,
    save_records_json_locked,
)
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.workbench import review

RUN_ID = "11111111-1111-4111-8111-111111111111"


def _record(expression: str, reading: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["fixture"],
        examples=[
            ExampleSentence(
                japanese=f"{expression}例です。",
                furigana=f"{expression}[{reading}]例[れい]です。",
                romaji="fixture rei desu",
                english="This is a fixture example.",
                spoken_japanese=f"{expression}例です。",
                register="polite",
            )
        ],
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            raw_fields={"page": "1"},
        ),
    )


def _provenance(source_sha256: str) -> dict[str, object]:
    return {
        "source_sha256": source_sha256,
        "mode": "table",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "response_schema_version": 3,
        "system_prompt_fingerprint": "2" * 64,
        "style_guide_fingerprint": "3" * 64,
        "user_prompt_fingerprint": "4" * 64,
        "response_schema_fingerprint": "5" * 64,
        "request_fingerprint": "6" * 64,
    }


def _project(tmp_path: Path) -> tuple[ProjectConfig, Path, tuple[VocabularyRecord, ...]]:
    (tmp_path / "janki.toml").write_text(
        '[project]\nname = "Staging actions"\n'
        '[paths]\nnormalized_file = "data/normalized/vocabulary.json"\n'
        'staging_dir = "data/staging"\nscan_inbox = "data/inbox"\n'
        'patterns_file = "data/patterns.json"\nledger_file = "data/ledger.json"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.normalized_file.parent.mkdir(parents=True)
    config.normalized_file.write_text("[]\n", encoding="utf-8")
    config.scan_inbox.mkdir(parents=True)
    source = config.scan_inbox / "lesson.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    config.patterns_file.parent.mkdir(parents=True, exist_ok=True)
    patterns.save_store(config.patterns_file, {})
    records = (_record("食べる", "たべる"), _record("撮る", "とる"))
    units = tuple(
        SourceUnit(
            page=1,
            section="vocabulary",
            ordinal=index,
            context=f"row {index}",
            context_fingerprint=hashlib.sha256(f"row {index}".encode()).hexdigest(),
            disposition="candidate",
            reason="",
        )
        for index in (1, 2)
    )
    result = ExtractionResult(
        candidates=(),
        source_units=units,
        model_reported_unit_count=len(units),
    )
    pattern_set = replace(
        patterns.with_prompt_provenance(
            patterns.PatternSet(
                source="lesson.pdf",
                kind="lesson",
                title="Lesson",
                patterns=(),
            ),
            _provenance(source_sha),
        ),
        review_run_id=RUN_ID,
    )
    metadata = {
        "source_file": "lesson.pdf",
        "extracted_at": "2026-09-02",
        "model": "claude-opus-5",
        "review_run_id": RUN_ID,
        "prompt_provenance": _provenance(source_sha),
        "pattern_set": pattern_set.to_dict(),
        "coverage": extract.coverage_block(
            result,
            source_sha256=source_sha,
            mode="table",
        ),
    }
    path = config.staging_dir / "lesson.pdf.yaml"
    staging.write_staging(path, records, metadata)
    return config, path, records


def _proposal_resource(config: ProjectConfig) -> str:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)
    proposals = [
        item
        for item in catalog["data"]["resources"]
        if item["kind"] == "proposal"
    ]
    assert len(proposals) == 1
    return str(proposals[0]["resource_id"])


def test_staged_deletion_renders_exact_rows_and_uses_existing_cas(
    tmp_path: Path,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_staged_deletion(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[1].id,),
        instruction="Remove the 撮る proposal.",
    )

    assert plan.projection["selection"]["records"][0]["expression"] == "撮る"
    assert plan.projection["effects"] == {
        "rows_before": 2,
        "rows_removed": 1,
        "rows_after": 1,
        "canonical_cards_changed": False,
        "paid_provider_call": False,
    }
    result = execute_staged_deletion(config, plan)

    remaining, _meta = staging.read_staging(path)
    assert result.removed_record_ids == (records[1].id,)
    assert [record.id for record in remaining] == [records[0].id]


def test_staged_deletion_refuses_stale_or_ambiguous_selection(tmp_path: Path) -> None:
    config, path, records = _project(tmp_path)
    resource_id = _proposal_resource(config)
    plan = plan_staged_deletion(
        config,
        proposal_resource_id=resource_id,
        record_ids=(records[0].id,),
        instruction="Remove one proposal.",
    )
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(AssistantStagingActionError, match="changed|stale|proposal"):
        execute_staged_deletion(config, plan)

    with pytest.raises(AssistantStagingActionError, match="supplied twice"):
        plan_staged_deletion(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_ids=(records[1].id, records[1].id),
            instruction="Remove it twice.",
        )


def test_reidentification_renders_identity_history_and_preserves_examples(
    tmp_path: Path,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_reidentification(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_id=records[1].id,
        new_expression="写真を撮る",
        new_reading="しゃしんをとる",
        instruction="This proposal is the full expression 写真を撮る.",
    )

    assert plan.projection["identity"]["old"]["record_id"] == records[1].id
    assert plan.projection["identity"]["new"] == {
        "record_id": "word:写真を撮る:しゃしんをとる",
        "expression": "写真を撮る",
        "reading": "しゃしんをとる",
    }
    assert plan.projection["effects"]["japanese_examples_changed"] is False
    result = execute_reidentification(config, plan)

    changed, _meta = staging.read_staging(path)
    assert result.old_record_id == records[1].id
    assert result.new_record_id == "word:写真を撮る:しゃしんをとる"
    assert changed[1].id == "word:写真を撮る:しゃしんをとる"
    assert changed[1].expression == "写真を撮る"
    assert changed[1].reading == "しゃしんをとる"
    assert changed[1].examples == records[1].examples


def test_reidentification_refuses_a_source_collision(tmp_path: Path) -> None:
    config, _path, records = _project(tmp_path)

    with pytest.raises(AssistantStagingActionError, match="same identity|claim one word"):
        plan_reidentification(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_id=records[1].id,
            new_expression=records[0].expression,
            new_reading=records[0].reading,
            instruction="Merge these staging identities.",
        )


def test_reidentification_refuses_corrupt_export_history(tmp_path: Path) -> None:
    config, _path, records = _project(tmp_path)
    config.ledger_file.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(
        AssistantStagingActionError,
        match="ledger|export history",
    ):
        plan_reidentification(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_id=records[1].id,
            new_expression="写真を撮る",
            new_reading="しゃしんをとる",
            instruction="This proposal is the full expression 写真を撮る.",
        )


def test_reidentification_refuses_symlinked_export_history(tmp_path: Path) -> None:
    config, _path, records = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-ledger.json"
    outside.write_text("{}\n", encoding="utf-8")
    config.ledger_file.symlink_to(outside)

    with pytest.raises(
        AssistantStagingActionError,
        match="ledger|export history|safely",
    ):
        plan_reidentification(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_id=records[1].id,
            new_expression="写真を撮る",
            new_reading="しゃしんをとる",
            instruction="This proposal is the full expression 写真を撮る.",
        )


def test_reidentification_refuses_a_dangling_collection_symlink(
    tmp_path: Path,
) -> None:
    config, _path, records = _project(tmp_path)
    config.normalized_file.unlink()
    config.normalized_file.symlink_to(tmp_path / "missing-collection.json")

    with pytest.raises(
        AssistantStagingActionError,
        match="collection|symlink|safely|Could not plan",
    ):
        plan_reidentification(
            config,
            proposal_resource_id=_proposal_resource(config),
            record_id=records[1].id,
            new_expression="写真を撮る",
            new_reading="しゃしんをとる",
            instruction="This proposal is the full expression 写真を撮る.",
        )


@pytest.mark.parametrize("dependency", ["canonical", "ledger"])
def test_reidentification_waits_for_identity_dependencies_then_replans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_reidentification(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_id=records[1].id,
        new_expression="写真を撮る",
        new_reading="しゃしんをとる",
        instruction="This proposal is the full expression 写真を撮る.",
    )
    staging_before = path.read_bytes()
    dependency_path = (
        config.normalized_file if dependency == "canonical" else config.ledger_file
    )
    lock_attempted = threading.Event()
    real_lock = exclusive_path_lock

    @contextmanager
    def observed_lock(target: Path) -> Iterator[None]:
        if Path(os.path.realpath(target)) == Path(os.path.realpath(dependency_path)):
            lock_attempted.set()
        with real_lock(target):
            yield

    import japanese_anki.application.assistant_staging_actions as action_module

    monkeypatch.setattr(
        action_module,
        "exclusive_path_lock",
        observed_lock,
        raising=False,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with real_lock(dependency_path):
            pending = pool.submit(execute_reidentification, config, plan)
            assert lock_attempted.wait(timeout=2), (
                f"reidentification never locked its {dependency} dependency"
            )
            if dependency == "canonical":
                save_records_json_locked(
                    config.normalized_file,
                    [_record("写真を撮る", "しゃしんをとる")],
                )
            else:
                book = ledger.Ledger(path=config.ledger_file)
                book.record_export(records[1].id, "lesson", at="2026-09-02")
                book._save_locked()

        with pytest.raises(
            AssistantStagingActionError,
            match="changed|Could not plan|same identity|already exists",
        ):
            pending.result(timeout=5)

    assert path.read_bytes() == staging_before


@pytest.mark.parametrize("dependency", ["canonical", "ledger"])
def test_reidentification_keeps_dependencies_locked_through_staging_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_reidentification(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_id=records[1].id,
        new_expression="写真を撮る",
        new_reading="しゃしんをとる",
        instruction="This proposal is the full expression 写真を撮る.",
    )
    dependency_path = (
        config.normalized_file if dependency == "canonical" else config.ledger_file
    )
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
                if dependency == "canonical":
                    save_records_json_locked(
                        config.normalized_file,
                        [_record("写真を撮る", "しゃしんをとる")],
                    )
                else:
                    book = ledger.Ledger(path=config.ledger_file)
                    book.record_export(records[1].id, "lesson", at="2026-09-02")
                    book._save_locked()
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
            f"reidentification released its {dependency} dependency before commit"
        )
        real_replace(target, text, snapshot, label=label)

    monkeypatch.setattr(review, "bound_replace_under_lock", replace_at_commit)

    result = execute_reidentification(config, plan)
    assert result.new_record_id == "word:写真を撮る:しゃしんをとる"
    assert len(writers) == 1
    writers[0].join(timeout=5)
    assert not writers[0].is_alive()
    assert writer_acquired.is_set()
    assert writer_errors == []
    changed, _metadata = staging.read_staging(path)
    assert changed[1].id == result.new_record_id


@pytest.mark.parametrize("intended_bytes_are_live", [False, True])
def test_staged_deletion_reports_indeterminate_write_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intended_bytes_are_live: bool,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_staged_deletion(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_ids=(records[1].id,),
        instruction="Remove the second proposal.",
    )
    real_replace = review.bound_replace

    def replace_then_fail(
        target: Path,
        text: str,
        snapshot: bytes,
        *,
        label: str,
    ) -> None:
        if intended_bytes_are_live:
            real_replace(target, text, snapshot, label=label)
        raise review.IndeterminateWriteError(
            "write acknowledgement failed",
            intended_bytes_are_live=intended_bytes_are_live,
        )

    monkeypatch.setattr(review, "bound_replace", replace_then_fail)
    expected = "did reach" if intended_bytes_are_live else "could not determine"
    with pytest.raises(AssistantStagingActionError, match=expected):
        execute_staged_deletion(config, plan)

    remaining, _meta = staging.read_staging(path)
    assert [record.id for record in remaining] == (
        [records[0].id]
        if intended_bytes_are_live
        else [records[0].id, records[1].id]
    )


def test_reidentification_reports_landed_indeterminate_write_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, path, records = _project(tmp_path)
    plan = plan_reidentification(
        config,
        proposal_resource_id=_proposal_resource(config),
        record_id=records[1].id,
        new_expression="写真を撮る",
        new_reading="しゃしんをとる",
        instruction="This proposal is the full expression 写真を撮る.",
    )
    real_replace = review.bound_replace_under_lock

    def replace_then_fail(
        target: Path,
        text: str,
        snapshot: bytes,
        *,
        label: str,
    ) -> None:
        real_replace(target, text, snapshot, label=label)
        raise review.IndeterminateWriteError(
            "write acknowledgement failed",
            intended_bytes_are_live=True,
        )

    monkeypatch.setattr(review, "bound_replace_under_lock", replace_then_fail)
    with pytest.raises(AssistantStagingActionError, match="did reach"):
        execute_reidentification(config, plan)

    changed, _meta = staging.read_staging(path)
    assert changed[1].id == "word:写真を撮る:しゃしんをとる"


def test_owner_coverage_approval_renders_account_and_records_exact_reason(
    tmp_path: Path,
) -> None:
    config, path, _records = _project(tmp_path)
    plan = plan_coverage_approval(
        config,
        proposal_resource_id=_proposal_resource(config),
        reason="I compared both source rows with these proposals.",
        instruction="Approve this coverage account.",
    )

    assert "row 1" in plan.projection["coverage"]["account"]
    assert plan.projection["owner_decision"] == {
        "authority": "repository-owner",
        "reason": "I compared both source rows with these proposals.",
    }
    result = execute_coverage_approval(config, plan)

    _records, meta = staging.read_staging(path)
    assert result.staging_path == path.resolve()
    assert meta["coverage"]["approval"]["authority"] == "repository-owner"
    assert (
        meta["coverage"]["approval"]["reason"]
        == "I compared both source rows with these proposals."
    )


def test_owner_coverage_approval_never_infers_reason_or_reuses_stale_account(
    tmp_path: Path,
) -> None:
    config, path, _records = _project(tmp_path)
    resource_id = _proposal_resource(config)
    with pytest.raises(AssistantStagingActionError, match="owner's nonblank reason"):
        plan_coverage_approval(
            config,
            proposal_resource_id=resource_id,
            reason=" ",
            instruction="Approve coverage.",
        )
    plan = plan_coverage_approval(
        config,
        proposal_resource_id=resource_id,
        reason="I checked both rows.",
        instruction="Approve coverage.",
    )
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(AssistantStagingActionError, match="changed|stale|proposal"):
        execute_coverage_approval(config, plan)


def test_owner_coverage_error_after_landing_reports_unknown_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, path, _records = _project(tmp_path)
    plan = plan_coverage_approval(
        config,
        proposal_resource_id=_proposal_resource(config),
        reason="I checked both rows.",
        instruction="Approve coverage.",
    )
    real_approve = coverage_application.approve_coverage_as_owner

    def approve_then_fail(
        current: ProjectConfig,
        decision: coverage_application.CoverageDecision,
        *,
        reason: str,
    ) -> None:
        real_approve(current, decision, reason=reason)
        raise coverage_application.CoverageApplicationError(
            "approval acknowledgement failed"
        )

    monkeypatch.setattr(
        coverage_application,
        "approve_coverage_as_owner",
        approve_then_fail,
    )
    with pytest.raises(
        AssistantStagingActionError,
        match="could not prove whether|inspect the current coverage",
    ):
        execute_coverage_approval(config, plan)

    _records, meta = staging.read_staging(path)
    assert meta["coverage"]["approval"]["authority"] == "repository-owner"
