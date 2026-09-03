"""The generic ``revise`` pass captures once and stops at unapproved staging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import seed_prompts
from japanese_anki import operations
from japanese_anki.application import card_change_staging, card_revision
from japanese_anki.config import ProjectConfig
from japanese_anki.staging import read_staging

SELECTED = "word:話す:はなす"
OUTSIDE = "word:遊ぶ:あそぶ"
OPERATION_ID = "62c87504-0351-4a9d-ad6a-c80d5e874cc2"


def _record(record_id: str, expression: str, reading: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "expression": expression,
        "reading": reading,
        "meanings": ["to speak" if record_id == SELECTED else "to play"],
        "part_of_speech": "godan verb",
        "verb_group": "godan",
        "tags": ["lesson-8"],
        "usage_notes": "Existing owner-reviewed note.",
        "source": {
            "type": "manual",
            "imported_from": "private-lesson.csv",
        },
    }


def _project(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        'revise_provider = "anthropic-api"\n'
        'revise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    normalized = tmp_path / "data" / "normalized"
    normalized.mkdir(parents=True)
    (normalized / "vocabulary.json").write_text(
        json.dumps(
            [
                _record(SELECTED, "話す", "はなす"),
                _record(OUTSIDE, "遊ぶ", "あそぶ"),
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    deck_dir = tmp_path / "data" / "decks"
    deck_dir.mkdir(parents=True)
    deck = deck_dir / "lesson-8.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson 8\n"
        "  deck_id: 1234\n"
        "  model_id: 5678\n"
        '  source: "../normalized/vocabulary.json"\n'
        "  include_ids:\n"
        f"    - {SELECTED}\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path), deck


def _answer(
    plan: card_revision.CardRevisionPlan,
    *,
    field: str = "usage_notes",
    value_json: str = '"Use this when someone speaks or talks."',
) -> Any:
    return plan.provider_plan.schema.model_validate(
        {
            "summary": "Clarify the selected card's usage.",
            "card_changes": [
                {
                    "record_id": SELECTED,
                    "reason": "The owner asked for a clearer usage explanation.",
                    "updates": [{"field": field, "value_json": value_json}],
                }
            ],
        }
    )


def _api_response(answer: Any) -> dict[str, Any]:
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": answer.model_dump_json()}],
    }


def _api_call(
    config: ProjectConfig,
    plan: card_revision.CardRevisionPlan,
    events: list[str],
    *,
    answer: Any | None = None,
):
    def call(*_args: Any, capture=None, **_kwargs: Any) -> None:
        held = operations.OperationJournal.load(config.operations_file).operations[
            plan.operation_id
        ]
        events.append(held.state)
        assert held.kind == "revise"
        assert held.source_file == plan.deck_relative_path
        assert held.source_sha256 == plan.deck_sha256
        assert held.request_fp == plan.request_fingerprint
        request = json.loads(plan.request_manifest_path.read_text(encoding="utf-8"))
        assert request["state"] == "request"
        assert not plan.staging_path.exists()
        assert capture is not None
        capture(_api_response(answer or _answer(plan)))
        events.append(
            operations.OperationJournal.load(config.operations_file)
            .operations[plan.operation_id]
            .state
        )

    return call


def test_plan_binds_exact_configured_deck_canonical_bytes_and_membership(
    tmp_path: Path,
) -> None:
    config, deck = _project(tmp_path)

    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card's usage note.",
        operation_id=OPERATION_ID,
        focus_resource_id="deck:lesson-8",
    )

    assert plan.operation_id == OPERATION_ID
    assert plan.selected_record_ids == (SELECTED,)
    assert tuple(record.id for record in plan.current_records) == (SELECTED,)
    assert plan.deck_sha256
    assert plan.canonical_sha256
    assert plan.can_dispatch
    assert SELECTED in plan.user_turn
    assert OUTSIDE not in plan.user_turn
    assert "private-lesson.csv" in plan.user_turn
    assert not config.operations_file.exists()
    assert not config.staging_dir.exists()

    with pytest.raises(card_revision.CardRevisionError, match="not members"):
        card_revision.plan_card_revision(
            config,
            deck,
            [OUTSIDE],
            "Change this card.",
            operation_id="2e278fb0-45e0-47ce-83cf-faadbd7f8a4b",
        )


def test_conjugation_deck_without_source_uses_the_default_canonical_collection(
    tmp_path: Path,
) -> None:
    config, _deck = _project(tmp_path)
    deck = tmp_path / "data" / "decks" / "te-form.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: te_form\n"
        "  name: Te-form Practice\n"
        "  deck_id: 1234\n"
        "  model_id: 5678\n"
        "  include_ids:\n"
        f"    - {SELECTED}\n",
        encoding="utf-8",
    )

    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card's usage note.",
        operation_id=OPERATION_ID,
    )

    assert plan.selected_record_ids == (SELECTED,)
    assert tuple(record.id for record in plan.current_records) == (SELECTED,)
    assert plan.canonical_path == config.normalized_file.resolve()


def test_card_revision_schema_and_stager_refuse_protected_or_invalid_values(
    tmp_path: Path,
) -> None:
    config, deck = _project(tmp_path)
    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Change this card.",
        operation_id=OPERATION_ID,
    )

    with pytest.raises(ValidationError):
        _answer(plan, field="reading", value_json='"しゃべる"')

    invalid = _answer(plan, field="meanings", value_json='"not a meaning list"')
    with pytest.raises(card_revision.CardRevisionRunError, match="canonical JSON field"):
        card_revision.run_card_revision(
            config,
            plan,
            client=object(),
            api_call=_api_call(config, plan, [], answer=invalid),
        )

    held = operations.OperationJournal.load(config.operations_file).operations[
        OPERATION_ID
    ]
    assert held.state == "result_captured"
    assert plan.request_manifest_path.exists()
    assert not plan.staging_path.exists()


def test_run_captures_before_staging_then_commits_separate_exact_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card's usage note.",
        operation_id=OPERATION_ID,
        focus_resource_id="deck:lesson-8",
    )
    original_canonical = config.normalized_file.read_bytes()
    events: list[str] = []
    real_stage = card_change_staging.stage_card_change_staging_under_lock

    def observing_stage(*args: Any, **kwargs: Any):
        held = operations.OperationJournal.load(config.operations_file).operations[
            OPERATION_ID
        ]
        events.append("stage:" + held.state)
        manifest = json.loads(plan.request_manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "request"
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(
        card_revision.card_change_staging,
        "stage_card_change_staging_under_lock",
        observing_stage,
    )

    result = card_revision.run_card_revision(
        config,
        plan,
        client=object(),
        api_call=_api_call(config, plan, events),
    )

    assert result.staging_path == plan.staging_path
    assert events == ["dispatching", "result_captured", "stage:result_captured"]
    assert config.normalized_file.read_bytes() == original_canonical
    held = operations.OperationJournal.load(config.operations_file).operations[
        OPERATION_ID
    ]
    assert held.state == "committed"
    records, meta = read_staging(result.staging_path)
    assert records[0].usage_notes == "Use this when someone speaks or talks."
    assert meta["card_revision"]["operation_id"] == OPERATION_ID
    assert "acceptance" not in meta
    manifest = json.loads(result.request_manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "result"
    assert manifest["result"]["answer"]["summary"].startswith("Clarify")
    assert manifest["result"]["staging_sha256"]


def test_run_replans_and_refuses_changed_canonical_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card.",
        operation_id=OPERATION_ID,
    )
    values = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    values[0]["usage_notes"] = "Concurrent owner edit."
    config.normalized_file.write_text(
        json.dumps(values, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prepared = False

    def prepare(*_args: Any, **_kwargs: Any) -> None:
        nonlocal prepared
        prepared = True

    provider = card_revision.revision_provider.provider_for(plan.provider)
    monkeypatch.setattr(type(provider), "prepare", prepare)

    with pytest.raises(card_revision.CardRevisionError, match="request-stale"):
        card_revision.run_card_revision(config, plan)

    assert not prepared
    assert not config.operations_file.exists()
    assert not plan.request_manifest_path.exists()


def test_recovery_consumes_captured_reply_once_and_finishes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = card_revision.plan_card_revision(
        config,
        deck,
        [SELECTED],
        "Clarify this card.",
        operation_id=OPERATION_ID,
    )
    calls: list[str] = []
    real_persist = card_revision._persist_result_under_locks

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise card_revision.CardRevisionError("simulated staging interruption")

    monkeypatch.setattr(card_revision, "_persist_result_under_locks", interrupted)
    with pytest.raises(card_revision.CardRevisionRunError, match="interruption"):
        card_revision.run_card_revision(
            config,
            plan,
            client=object(),
            api_call=_api_call(config, plan, calls),
        )
    assert calls == ["dispatching", "result_captured"]
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[OPERATION_ID]
        .state
        == "result_captured"
    )

    monkeypatch.setattr(card_revision, "_persist_result_under_locks", real_persist)
    recovered = card_revision.recover_card_revision(config, OPERATION_ID)
    assert recovered.staging_path.exists()
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[OPERATION_ID]
        .state
        == "committed"
    )
    same = card_revision.recover_card_revision(config, OPERATION_ID)
    assert same == recovered
    assert calls == ["dispatching", "result_captured"]
