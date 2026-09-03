"""Journaled bare-card AI enrichment stops at an unapproved staging review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from conftest import seed_prompts
from japanese_anki import operations
from japanese_anki.application import ai_enrichment, promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.staging import read_staging

FIRST = "word:話す:はなす"
SECOND = "word:遊ぶ:あそぶ"
FIRST_OPERATION = "579fd68e-f84b-44f8-a949-800fc0d93eb7"
SECOND_OPERATION = "3cb5026d-824e-4209-89b4-fd9c66f74966"


def _record(record_id: str, expression: str, reading: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "expression": expression,
        "reading": reading,
        "meanings": [],
        "examples": [],
        "usage_notes": "",
        "tags": ["lesson-8"],
        "source": {
            "type": "manual",
            "imported_from": "private-lesson.csv",
        },
    }


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        '[ai]\nenrich_provider = "anthropic"\nenrich_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    normalized = tmp_path / "data" / "normalized"
    normalized.mkdir(parents=True)
    (normalized / "vocabulary.json").write_text(
        json.dumps(
            [
                _record(FIRST, "話す", "はなす"),
                _record(SECOND, "遊ぶ", "あそぶ"),
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _answer(call: ai_enrichment.AiEnrichmentCallPlan, *, gloss: str) -> Any:
    return call.provider_plan.schema.model_validate(
        {
            "meanings": [gloss],
            "examples": [
                {
                    "japanese": f"{call.record.expression}ことができます。",
                    "furigana": f"{call.record.expression}[{call.record.reading}]ことができます。",
                    "romaji": "example desu",
                    "english": "I can do it.",
                    "speech_level": "polite",
                },
                {
                    "japanese": f"{call.record.expression}ことができる。",
                    "furigana": f"{call.record.expression}[{call.record.reading}]ことができる。",
                    "romaji": "example da",
                    "english": "I can do it.",
                    "speech_level": "casual",
                },
            ],
            "usage_notes": "A useful usage note.",
        }
    )


def _api_response(answer: Any) -> dict[str, Any]:
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": answer.model_dump_json()}],
    }


def test_plan_binds_exact_cards_prompts_provider_requests_and_staging_targets(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [SECOND, FIRST],
        operation_ids=[SECOND_OPERATION, FIRST_OPERATION],
        focus_resource_id="deck:lesson-8",
    )

    assert tuple(call.record_id for call in plan.calls) == (SECOND, FIRST)
    assert tuple(call.operation_id for call in plan.calls) == (
        SECOND_OPERATION,
        FIRST_OPERATION,
    )
    assert plan.provider == "anthropic"
    assert plan.billing_display == "Anthropic API billing"
    assert plan.model == "claude-opus-5"
    assert plan.focus_resource_id == "deck:lesson-8"
    assert plan.canonical_sha256
    assert plan.patterns_sha256
    assert plan.plan_fingerprint
    assert all(call.request_fingerprint for call in plan.calls)
    assert all(call.input_fingerprint for call in plan.calls)
    assert all(call.record.expression in call.user_turn for call in plan.calls)
    assert all(
        call.staging_path.name == f"ai-enrichment-{call.operation_id}.yaml" for call in plan.calls
    )
    assert not config.operations_file.exists()
    assert not config.staging_dir.exists()


def test_run_captures_each_paid_reply_before_parsing_and_stages_without_canonical_write(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST, SECOND],
        operation_ids=[FIRST_OPERATION, SECOND_OPERATION],
        focus_resource_id="resource_deck_lesson_8",
    )
    original_canonical = config.normalized_file.read_bytes()
    events: list[tuple[str, str]] = []

    answers = {
        FIRST_OPERATION: _answer(plan.calls[0], gloss="to speak"),
        SECOND_OPERATION: _answer(plan.calls[1], gloss="to play"),
    }

    def api_call(*_args: Any, capture=None, **_kwargs: Any) -> None:
        assert capture is not None
        journal = operations.OperationJournal.load(config.operations_file)
        active = [
            operation
            for operation in journal.operations.values()
            if operation.state == "dispatching"
        ]
        assert len(active) == 1
        held = active[0]
        events.append((held.operation_id, held.state))
        assert held.kind == "enrich"
        expected_call = next(call for call in plan.calls if call.operation_id == held.operation_id)
        request_manifest = json.loads(
            expected_call.request_manifest_path.read_text(encoding="utf-8")
        )
        assert request_manifest["state"] == "request"
        assert request_manifest["operation_id"] == held.operation_id
        assert not (config.staging_dir / f"ai-enrichment-{held.operation_id}.yaml").exists()
        capture(_api_response(answers[held.operation_id]))
        events.append(
            (
                held.operation_id,
                operations.OperationJournal.load(config.operations_file)
                .operations[held.operation_id]
                .state,
            )
        )

    result = ai_enrichment.run_ai_enrichment(
        config,
        plan,
        client=object(),
        api_call=api_call,
    )

    assert events == [
        (FIRST_OPERATION, "dispatching"),
        (FIRST_OPERATION, "result_captured"),
        (SECOND_OPERATION, "dispatching"),
        (SECOND_OPERATION, "result_captured"),
    ]
    assert config.normalized_file.read_bytes() == original_canonical
    assert tuple(item.operation_id for item in result.results) == (
        FIRST_OPERATION,
        SECOND_OPERATION,
    )
    assert all(item.state == "staged" for item in result.results)
    journal = operations.OperationJournal.load(config.operations_file)
    assert [journal.operations[item].state for item in (FIRST_OPERATION, SECOND_OPERATION)] == [
        "committed",
        "committed",
    ]
    for call, item in zip(plan.calls, result.results, strict=True):
        records, meta = read_staging(item.staging_path)
        assert [record.id for record in records] == [call.record_id]
        assert meta["ai_enrichment"]["request_fingerprints"] == {
            call.record_id: call.request_fingerprint
        }
        assert meta["ai_enrichment"]["input_fingerprints"] == {
            call.record_id: call.input_fingerprint
        }
        assert meta["ai_enrichment"]["focus_resource_id"] == ("resource_deck_lesson_8")
        assert set(meta["ai_enrichment"]["fields"][call.record_id]) == {
            "examples",
            "meanings",
            "usage_notes",
        }
        assert "reviewed_pattern_set" not in meta
        manifest = json.loads(item.request_manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "result"
        assert manifest["result"]["staging_sha256"]


@pytest.mark.parametrize("invalid_focus", ["", " ", 7, False])
def test_ai_enrichment_focus_is_optional_but_malformed_persisted_focus_refuses(
    tmp_path: Path,
    invalid_focus: object,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )
    answer = _answer(plan.calls[0], gloss="to speak")

    result = ai_enrichment.run_ai_enrichment(
        config,
        plan,
        client=object(),
        api_call=lambda *_args, capture, **_kwargs: capture(_api_response(answer)),
    )

    [record], meta = read_staging(result.results[0].staging_path)
    assert "focus_resource_id" not in meta["ai_enrichment"]
    assert promotion.staged_ai_enrichment(meta, [record]) is not None

    malformed = dict(meta)
    malformed_provenance = dict(meta["ai_enrichment"])
    malformed_provenance["focus_resource_id"] = invalid_focus
    malformed["ai_enrichment"] = malformed_provenance
    with pytest.raises(promotion.PromoteError, match="focus_resource_id.*nonblank"):
        promotion.staged_ai_enrichment(malformed, [record])


def test_run_refuses_changed_card_before_authority_or_provider_dispatch(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )
    values = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    values[0]["usage_notes"] = "Concurrent owner edit."
    config.normalized_file.write_text(
        json.dumps(values, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dispatched = False

    def api_call(*_args: Any, **_kwargs: Any) -> None:
        nonlocal dispatched
        dispatched = True

    with pytest.raises(ai_enrichment.AiEnrichmentError, match="request-stale"):
        ai_enrichment.run_ai_enrichment(
            config,
            plan,
            client=object(),
            api_call=api_call,
        )

    assert not dispatched
    assert not config.operations_file.exists()
    assert not plan.calls[0].request_manifest_path.exists()


def test_run_refuses_a_changed_enrichment_prompt_before_authority_or_dispatch(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )
    prompt_path = tmp_path / "prompts" / "enrich-bare-word.md"
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + "\nOne changed instruction.\n",
        encoding="utf-8",
    )
    dispatched = False

    def api_call(*_args: Any, **_kwargs: Any) -> None:
        nonlocal dispatched
        dispatched = True

    with pytest.raises(ai_enrichment.AiEnrichmentError, match="request-stale"):
        ai_enrichment.run_ai_enrichment(
            config,
            plan,
            client=object(),
            api_call=api_call,
        )

    assert not dispatched
    assert not config.operations_file.exists()
    assert not plan.calls[0].request_manifest_path.exists()


def test_run_refuses_tampered_bound_prompt_text_before_authority_or_dispatch(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )
    tampered = replace(
        plan,
        task_template=plan.task_template + "\nUnconfirmed instruction.",
    )

    with pytest.raises(ai_enrichment.AiEnrichmentError, match="fingerprint"):
        ai_enrichment.run_ai_enrichment(
            config,
            tampered,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("a tampered plan must not dispatch"),
        )

    assert not config.operations_file.exists()


def test_invalid_paid_answer_remains_recoverable_and_never_stages(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )

    def invalid(*_args: Any, capture=None, **_kwargs: Any) -> None:
        assert capture is not None
        capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"meanings":[]}'}],
            }
        )

    with pytest.raises(ai_enrichment.AiEnrichmentRunError) as caught:
        ai_enrichment.run_ai_enrichment(
            config,
            plan,
            client=object(),
            api_call=invalid,
        )

    assert caught.value.operation_id == FIRST_OPERATION
    assert caught.value.provider_dispatched
    held = operations.OperationJournal.load(config.operations_file).operations[FIRST_OPERATION]
    assert held.state == "result_captured"
    assert not plan.calls[0].staging_path.exists()
    assert plan.calls[0].request_manifest_path.exists()


def test_card_change_after_reply_capture_preserves_reply_and_refuses_staging(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST],
        operation_ids=[FIRST_OPERATION],
    )
    answer = _answer(plan.calls[0], gloss="to speak")

    def api_call(*_args: Any, capture=None, **_kwargs: Any) -> None:
        assert capture is not None
        capture(_api_response(answer))
        values = json.loads(config.normalized_file.read_text(encoding="utf-8"))
        values[0]["usage_notes"] = "Concurrent owner edit after capture."
        config.normalized_file.write_text(
            json.dumps(values, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ai_enrichment.AiEnrichmentRunError, match="selected card changed"):
        ai_enrichment.run_ai_enrichment(
            config,
            plan,
            client=object(),
            api_call=api_call,
        )

    held = operations.OperationJournal.load(config.operations_file).operations[FIRST_OPERATION]
    assert held.state == "result_captured"
    assert not plan.calls[0].staging_path.exists()
    assert plan.calls[0].request_manifest_path.exists()


def test_later_failure_reports_the_durable_completed_prefix(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = ai_enrichment.plan_ai_enrichment(
        config,
        [FIRST, SECOND],
        operation_ids=[FIRST_OPERATION, SECOND_OPERATION],
    )
    first_answer = _answer(plan.calls[0], gloss="to speak")

    def api_call(*_args: Any, capture=None, **_kwargs: Any) -> None:
        assert capture is not None
        active = [
            operation
            for operation in operations.OperationJournal.load(
                config.operations_file
            ).operations.values()
            if operation.state == "dispatching"
        ]
        assert len(active) == 1
        if active[0].operation_id == FIRST_OPERATION:
            capture(_api_response(first_answer))
        else:
            capture(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"meanings":[]}'}],
                }
            )

    with pytest.raises(ai_enrichment.AiEnrichmentRunError) as caught:
        ai_enrichment.run_ai_enrichment(
            config,
            plan,
            client=object(),
            api_call=api_call,
        )

    assert caught.value.operation_id == SECOND_OPERATION
    assert caught.value.provider_dispatched is True
    assert len(caught.value.completed_results) == 1
    [completed] = caught.value.completed_results
    assert completed.operation_id == FIRST_OPERATION
    assert completed.state == "staged"
    assert completed.staging_path is not None
    assert completed.staging_path.name in str(caught.value)
    assert "1 earlier enrichment call reached its durable destination" in str(caught.value)
    journal = operations.OperationJournal.load(config.operations_file)
    assert journal.operations[FIRST_OPERATION].state == "committed"
    assert journal.operations[SECOND_OPERATION].state == "result_captured"


def test_codex_enrichment_refuses_until_its_transport_has_exact_reply_capture(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    (tmp_path / "janki.toml").write_text(
        '[ai]\nenrich_provider = "codex"\nenrich_model = "gpt-5.6-sol"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    with pytest.raises(ai_enrichment.AiEnrichmentError, match="exact reply capture"):
        ai_enrichment.plan_ai_enrichment(config, [FIRST])

    assert not config.operations_file.exists()


def test_plan_refuses_an_effectively_empty_enrichment_prompt_without_authority(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    (tmp_path / "prompts" / "enrich-bare-word.md").write_text(
        "\u200b\ufeff\n",
        encoding="utf-8",
    )

    with pytest.raises(ai_enrichment.AiEnrichmentError, match="prompt.*empty"):
        ai_enrichment.plan_ai_enrichment(config, [FIRST])

    assert not config.operations_file.exists()
