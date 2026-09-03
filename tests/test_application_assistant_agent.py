"""Repository-aware ordinary Assistant turns and their durable recovery."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from conftest import seed_prompts
from japanese_anki import ai_schema, operations
from japanese_anki.application import assistant_agent
from japanese_anki.claude_client import CallResult
from japanese_anki.config import ProjectConfig
from japanese_anki.models import VocabularyRecord


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[assistant]\n"
        'enabled = true\n'
        'provider = "anthropic-api"\n'
        'model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    return ProjectConfig.load(tmp_path)


def _record() -> VocabularyRecord:
    return VocabularyRecord(
        id="word:食べる:たべる",
        expression="食べる",
        reading="たべる",
        meanings=["to eat"],
    )


def _context(
    *,
    marker: str = "current",
    focused: bool = True,
    editable: bool = True,
) -> assistant_agent.AgentContext:
    resource_ids = (
        "resource_deck_01",
        "resource_card_01",
        "resource_status_01",
    )
    wire = json.dumps(
        {
            "schema_version": 1,
            "kind": "assistant_turn_context",
            "marker": marker,
            "resources": [
                {"resource_id": item, "kind": item.split("_")[1]}
                for item in resource_ids
            ],
            "cards": [_record().to_dict()] if editable else [],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return assistant_agent.AgentContext(
        wire=wire,
        fingerprint=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
        resource_ids=resource_ids,
        editable_records=(_record(),) if editable else (),
        focus_resource_id="resource_deck_01" if focused else None,
    )


def _api_response(
    answer: str,
    *,
    intents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "answer": answer,
                        "action_intents": intents or [],
                    },
                    ensure_ascii=False,
                ),
            }
        ],
    }


def _unanswered_api_response() -> dict[str, Any]:
    """A reply the provider completes but that carries no answer.

    An incomplete stop reason is how both providers really report this; a
    provider error means a reply janki could not read, which is a different
    outcome with a different owner.
    """

    return {"stop_reason": "max_tokens", "content": []}


def _unreadable_api_response() -> dict[str, Any]:
    """A reply that finished normally but does not match the schema."""

    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Not structured output at all."}],
    }


def _api_call(response: dict[str, Any]):
    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(response)
        return CallResult(None, "end_turn", None)

    return call


def _capture_without_commit(
    config: ProjectConfig,
    plan: assistant_agent.AgentPlan,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete_manifest_landed: bool = False,
) -> str:
    response = _api_response(
        "The captured answer.",
        intents=[
            {
                "kind": "revise_cards",
                "resource_ids": ["resource_deck_01"],
                "record_ids": [_record().id],
                "instruction": "Add one polite example through revise.",
            }
        ],
    )
    if complete_manifest_landed:
        real_commit = operations.OperationJournal.commit_result

        def persist_then_fail(
            self: operations.OperationJournal,
            operation_id: str,
            persist: Any,
        ) -> Any:
            persist()
            raise operations.OperationError("simulated journal commit failure")

        monkeypatch.setattr(
            operations.OperationJournal,
            "commit_result",
            persist_then_fail,
        )
    else:
        real_write = assistant_agent.atomic_write_text_bound

        def fail_complete(path: Path, text: str, **kwargs: Any) -> Any:
            if json.loads(text).get("state") == "complete":
                raise OSError("simulated complete-manifest failure")
            return real_write(path, text, **kwargs)

        monkeypatch.setattr(assistant_agent, "atomic_write_text_bound", fail_complete)

    with pytest.raises(assistant_agent.AgentRunError) as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: _context(),
            client=object(),
            api_call=_api_call(response),
        )

    if complete_manifest_landed:
        monkeypatch.setattr(
            operations.OperationJournal,
            "commit_result",
            real_commit,
        )
    else:
        monkeypatch.setattr(
            assistant_agent,
            "atomic_write_text_bound",
            real_write,
        )
    operation_id = raised.value.operation_id
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "result_captured"
    return operation_id


def test_plan_binds_exact_context_history_message_and_allowlists(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context()
    history = (("user", "Which deck?"), ("assistant", "The focused one."))

    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Add a polite example.",
        history=history,
    )

    turn = json.loads(plan.user_turn)
    assert turn == {
        "repository_context": json.loads(context.wire),
        "context_fingerprint": context.fingerprint,
        "resource_ids": list(context.resource_ids),
        "active_focus_resource_id": context.focus_resource_id,
        "history": [
            {"role": "user", "content": "Which deck?"},
            {"role": "assistant", "content": "The focused one."},
        ],
        "user_message": "Add a polite example.",
    }
    assert plan.context.editable_records[0] is not context.editable_records[0]
    assert plan.context.editable_records[0].to_dict() == _record().to_dict()
    request_body = json.loads(plan.provider_plan.request_bytes)
    assert request_body["messages"] == [{"role": "user", "content": plan.user_turn}]
    assert not config.operations_file.exists()
    assert not config.assistant_dir.exists()


def test_plan_refuses_context_wire_that_does_not_match_fingerprint(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    context = replace(_context(), fingerprint="0" * 64)

    with pytest.raises(assistant_agent.AgentApplicationError, match="fingerprint"):
        assistant_agent.plan_agent(
            config,
            context=context,
            message="What is in this project?",
        )

    assert not config.operations_file.exists()


def test_agent_prompt_keeps_ordinary_turn_out_of_card_writing_lane(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    plan = assistant_agent.plan_agent(
        config,
        context=_context(),
        message="Change the example.",
    )

    normalized = plan.task_template.casefold()
    assert "do not write the proposed japanese" in normalized
    assert "output is a plan request, never\nexecution" in normalized
    assert "only by Janki's existing\n`extract`, `enrich --ai`, or `revise`" in (
        plan.task_template
    )


def test_schema_allows_at_most_one_closed_action_intent() -> None:
    import pydantic

    adapter = pydantic.TypeAdapter(ai_schema.assistant_agent_schema())
    intent = {
        "kind": "build_deck",
        "resource_ids": ["resource_deck_01"],
        "record_ids": [],
        "instruction": "Build this deck.",
    }

    parsed = adapter.validate_python(
        {"answer": "I can plan that.", "action_intents": [intent]}
    )
    assert parsed.action_intents[0].kind == "build_deck"
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(
            {"answer": "Too many.", "action_intents": [intent, intent]}
        )
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(
            {
                "answer": "Unknown.",
                "action_intents": [{**intent, "kind": "run_shell"}],
            }
        )


def test_schema_exposes_bare_card_enrichment_as_a_closed_writing_path() -> None:
    import pydantic

    adapter = pydantic.TypeAdapter(ai_schema.assistant_agent_schema())
    parsed = adapter.validate_python(
        {
            "answer": "I can prepare that exact enrichment batch.",
            "action_intents": [
                {
                    "kind": "enrich_cards",
                    "resource_ids": ["resource_card_01"],
                    "record_ids": ["word:話す:はなす"],
                    "instruction": "Fill this incomplete bare card.",
                }
            ],
        }
    )

    assert parsed.action_intents[0].kind == "enrich_cards"


def test_schema_exposes_captured_result_recovery_as_a_closed_operation_action() -> None:
    import pydantic

    adapter = pydantic.TypeAdapter(ai_schema.assistant_agent_schema())
    parsed = adapter.validate_python(
        {
            "answer": "I can recover that already-paid result without another call.",
            "action_intents": [
                {
                    "kind": "manage_operation",
                    "resource_ids": ["resource_operations"],
                    "record_ids": [],
                    "instruction": "Recover the captured result.",
                    "options": {
                        "operation_id": "11111111-1111-4111-8111-111111111111",
                        "operation_action": "recover",
                    },
                }
            ],
        }
    )

    [intent] = parsed.action_intents
    assert intent.options.operation_action == "recover"
    assert intent.options.operation_id == "11111111-1111-4111-8111-111111111111"


def test_schema_carries_only_closed_action_specific_options() -> None:
    import pydantic

    adapter = pydantic.TypeAdapter(ai_schema.assistant_agent_schema())
    value = {
        "kind": "create_deck",
        "resource_ids": [],
        "record_ids": [],
        "instruction": "Create Lesson 14 with recognition and reading cards.",
        "options": {
            "deck_name": "Lesson 14",
            "card_directions": ["recognition", "reading"],
            "audio_words": False,
            "audio_examples": True,
            "audio_force": True,
            "audio_prune": True,
            "new_expression": "写真を撮る",
            "new_reading": "しゃしんをとる",
            "coverage_reason": "I checked both source rows.",
        },
    }

    parsed = adapter.validate_python(
        {"answer": "I can prepare that deck.", "action_intents": [value]}
    )

    assert parsed.action_intents[0].options.deck_name == "Lesson 14"
    assert parsed.action_intents[0].options.card_directions == [
        "recognition",
        "reading",
    ]
    assert parsed.action_intents[0].options.audio_words is False
    assert parsed.action_intents[0].options.audio_examples is True
    assert parsed.action_intents[0].options.audio_force is True
    assert parsed.action_intents[0].options.audio_prune is True
    assert parsed.action_intents[0].options.new_expression == "写真を撮る"
    assert parsed.action_intents[0].options.new_reading == "しゃしんをとる"
    assert (
        parsed.action_intents[0].options.coverage_reason
        == "I checked both source rows."
    )
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(
            {
                "answer": "No arbitrary options.",
                "action_intents": [
                    {**value, "options": {"shell_command": "rm -rf anything"}}
                ],
            }
        )
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(
            {
                "answer": "Audio choices must be booleans.",
                "action_intents": [
                    {
                        **value,
                        "options": {
                            "audio_words": "yes",
                            "audio_examples": True,
                        },
                    }
                ],
            }
        )


def test_run_preserves_exact_typed_action_options(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Create a recognition-only Lesson 14 deck.",
    )
    intent = {
        "kind": "create_deck",
        "resource_ids": [],
        "record_ids": [],
        "instruction": "Create a recognition-only Lesson 14 deck.",
        "options": {
            "deck_name": "Lesson 14",
            "card_directions": ["recognition"],
        },
    }

    result = assistant_agent.run_agent(
        config,
        plan,
        context_loader=lambda: _context(focused=False, editable=False),
        client=object(),
        api_call=_api_call(_api_response("I prepared one plan.", intents=[intent])),
    )

    [action] = result.action_intents
    assert action.options == {
        "card_directions": ["recognition"],
        "deck_name": "Lesson 14",
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["action_intents"][0]["options"] == dict(action.options)


def test_run_without_focus_journals_exact_request_before_dispatch(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )
    events: list[str] = []

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        [entry] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        assert entry.kind == "assistant_agent"
        assert entry.state == "dispatching"
        assert entry.source_file == "janki-project"
        assert entry.source_sha256 == context.fingerprint
        assert entry.request_fp == plan.request_fingerprint
        manifest_path = config.assistant_dir / f"{entry.operation_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "request"
        assert manifest["context"] == {
            "wire": context.wire,
            "fingerprint": context.fingerprint,
            "resource_ids": list(context.resource_ids),
            "focus_resource_id": None,
            "editable_records": [],
        }
        events.append("request durable")
        assert capture is not None
        capture(_api_response("There are several configured decks."))
        [captured] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        assert captured.state == "result_captured"
        events.append("reply captured")
        return CallResult(None, "end_turn", None)

    result = assistant_agent.run_agent(
        config,
        plan,
        context_loader=lambda: _context(focused=False, editable=False),
        client=object(),
        api_call=call,
    )

    assert events == ["request durable", "reply captured"]
    assert result.answer == "There are several configured decks."
    assert result.action_intents == ()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["answer"] == result.answer
    assert manifest["action_intents"] == []
    [committed] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert committed.state == "committed"


def test_run_accepts_only_targets_in_exact_disclosed_context(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context()
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Add one polite example.",
    )
    intent = {
        "kind": "revise_cards",
        "resource_ids": ["resource_deck_01"],
        "record_ids": [_record().id],
        "instruction": "Add one polite example through revise.",
    }

    result = assistant_agent.run_agent(
        config,
        plan,
        context_loader=lambda: _context(),
        client=object(),
        api_call=_api_call(_api_response("I prepared one plan.", intents=[intent])),
    )

    assert result.action_intents == (
        assistant_agent.AgentActionIntent(
            kind="revise_cards",
            resource_ids=("resource_deck_01",),
            record_ids=(_record().id,),
            instruction="Add one polite example through revise.",
        ),
    )


def test_captured_provider_failure_is_committed_as_a_durable_failed_turn(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )
    response = _unanswered_api_response()
    fail_after_capture = _api_call(response)

    with pytest.raises(
        assistant_agent.AgentRunError,
        match="returned no complete Assistant answer",
    ) as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=fail_after_capture,
        )

    operation_id = raised.value.operation_id
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "committed"
    assert held.blocks_spending is False
    raw_reply = operations.serialize_response(response)
    manifest = json.loads(
        (config.assistant_dir / f"{operation_id}.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "failed"
    assert manifest["failure"] == {
        "message": (
            "claude-opus-5 returned no complete Assistant answer (max_tokens). "
            "Its exact reply was captured."
        ),
        "provider_reply_base64": base64.b64encode(raw_reply).decode("ascii"),
        "provider_reply_bytes": len(raw_reply),
        "provider_reply_sha256": hashlib.sha256(raw_reply).hexdigest(),
    }
    assert manifest["provenance"] == {
        "provider": plan.provider,
        "billing_class": plan.billing_class,
        "model": plan.model,
        "request_fingerprint": plan.request_fingerprint,
    }

    retry = assistant_agent.plan_agent(
        config,
        context=context,
        message="Try a different question.",
    )
    result = assistant_agent.run_agent(
        config,
        retry,
        context_loader=lambda: context,
        client=object(),
        api_call=_api_call(_api_response("This answer succeeded.")),
    )

    assert result.answer == "This answer succeeded."


def test_recovery_finishes_a_failed_manifest_without_calling_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )
    response = _unanswered_api_response()
    fail_after_capture = _api_call(response)

    real_commit = operations.OperationJournal.commit_result

    def persist_then_fail(
        self: operations.OperationJournal,
        operation_id: str,
        persist: Any,
    ) -> Any:
        persist()
        raise operations.OperationError("simulated journal commit failure")

    monkeypatch.setattr(
        operations.OperationJournal,
        "commit_result",
        persist_then_fail,
    )
    with pytest.raises(assistant_agent.AgentRunError) as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=fail_after_capture,
        )
    monkeypatch.setattr(
        operations.OperationJournal,
        "commit_result",
        real_commit,
    )
    monkeypatch.setattr(
        assistant_agent.revision_provider,
        "provider_for",
        lambda _name: pytest.fail("failed-manifest recovery must not call a provider"),
    )

    operation_id = raised.value.operation_id
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )
    result = assistant_agent.recover_agent(config, operation_id)

    assert result.answer == (
        "The earlier Assistant turn failed: claude-opus-5 returned no complete "
        "Assistant answer (max_tokens). Its exact reply was captured."
    )
    assert result.action_intents == ()
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "committed"
    )


def test_recovery_persists_a_captured_provider_error_from_a_request_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )
    response = _unanswered_api_response()
    fail_after_capture = _api_call(response)

    real_write = assistant_agent.atomic_write_text_bound

    def fail_failed_manifest(path: Path, text: str, **kwargs: Any) -> Any:
        if json.loads(text).get("state") == "failed":
            raise OSError("simulated failed-manifest write failure")
        return real_write(path, text, **kwargs)

    monkeypatch.setattr(
        assistant_agent,
        "atomic_write_text_bound",
        fail_failed_manifest,
    )
    with pytest.raises(assistant_agent.AgentRunError) as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=fail_after_capture,
        )
    monkeypatch.setattr(
        assistant_agent,
        "atomic_write_text_bound",
        real_write,
    )

    operation_id = raised.value.operation_id
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["state"] == "request"

    result = assistant_agent.recover_agent(config, operation_id)

    assert "returned no complete Assistant answer" in result.answer
    assert result.action_intents == ()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["state"] == "failed"
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "committed"
    )


def test_invented_resource_target_is_refused_after_exact_reply_capture(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    context = _context()
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Change one card.",
    )
    intent = {
        "kind": "revise_cards",
        "resource_ids": ["resource_deck_01"],
        "record_ids": [_record().id],
        "instruction": "Change the exact selected card.",
    }
    intent["resource_ids"] = ["resource_invented"]

    with pytest.raises(assistant_agent.AgentRunError, match="absent.*context") as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: _context(),
            client=object(),
            api_call=_api_call(_api_response("I prepared it.", intents=[intent])),
        )

    held = operations.OperationJournal.load(config.operations_file).operations[
        raised.value.operation_id
    ]
    assert held.state == "result_captured"
    assert held.artifact is not None
    manifest = json.loads(
        (config.assistant_dir / f"{raised.value.operation_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["state"] == "request"
    assert "answer" not in manifest


def test_a_failing_progress_callback_still_settles_the_turn(tmp_path: Path) -> None:
    """A caller that goes away mid-turn must not strand the operation.

    The progress callback runs after the dispatch boundary is recorded, so if
    it escapes uncaught the entry stays mid-flight and keeps blocking spending
    with no typed error to act on. A disconnecting client is exactly when this
    happens, so the failure has to settle like any other.
    """

    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )

    def failing_progress(label: str) -> None:
        if label == "Writing answer":
            raise RuntimeError("the caller's event loop is closed")

    with pytest.raises(assistant_agent.AgentRunError) as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=_api_call(_api_response("This answer never arrives.")),
            progress=failing_progress,
        )

    assert raised.value.operation_id
    held = operations.OperationJournal.load(config.operations_file).operations[
        raised.value.operation_id
    ]
    assert held.state not in {"authorized", "dispatching"}


def test_unreadable_provider_reply_is_left_for_the_owner_to_settle(
    tmp_path: Path,
) -> None:
    """A reply janki cannot read is not a reply that said nothing.

    Structured output that finishes normally and only fails schema validation
    still holds whatever the model wrote. Filing it away as a failure would
    retire the operation and spend the owner's decision for them, so only a
    genuinely unanswered reply may settle itself.
    """

    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )

    with pytest.raises(assistant_agent.AgentRunError, match="schema") as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=_api_call(_unreadable_api_response()),
        )

    operation_id = raised.value.operation_id
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "result_captured"
    manifest = json.loads(
        (config.assistant_dir / f"{operation_id}.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "request"
    assert "failure" not in manifest


def test_recovery_leaves_a_refused_answer_for_the_owner_to_settle(
    tmp_path: Path,
) -> None:
    """A refused answer is not a missing one, and janki must not file it away.

    The failed-manifest path exists for a reply that holds no answer at all.
    This reply holds one: janki refused it only because an intent named a
    resource the turn never disclosed. Writing that off as a failure would
    retire the operation and discard the owner's chance to read what the model
    actually said.
    """

    config = _project(tmp_path)
    context = _context()
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Change one card.",
    )
    intent = {
        "kind": "revise_cards",
        "resource_ids": ["resource_invented"],
        "record_ids": [_record().id],
        "instruction": "Change the exact selected card.",
    }

    with pytest.raises(assistant_agent.AgentRunError, match="absent.*context") as raised:
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=_api_call(_api_response("I prepared it.", intents=[intent])),
        )

    operation_id = raised.value.operation_id
    manifest_path = config.assistant_dir / f"{operation_id}.json"

    with pytest.raises(
        assistant_agent.AgentApplicationError,
        match="could not be decoded",
    ):
        assistant_agent.recover_agent(config, operation_id)

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["state"] == "request"
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )


def test_record_ids_are_forwarded_to_the_resource_specific_planner(
    tmp_path: Path,
) -> None:
    """Proposal rows learned through a prior read are not focused-deck records.

    The decoder validates opaque resources. The local action planner validates
    record membership against that freshly resolved resource before it can
    render authority, so rejecting every non-focus id here made real staging
    review and assignment impossible without adding safety.
    """

    config = _project(tmp_path)
    context = _context(editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Review the proposal row I inspected.",
    )
    intent = {
        "kind": "review_staging",
        "resource_ids": ["resource_deck_01"],
        "record_ids": ["word:proposal-only:proposal-only"],
        "instruction": "Review this exact proposal row.",
        "options": {"review_patterns": False},
    }

    result = assistant_agent.run_agent(
        config,
        plan,
        context_loader=lambda: _context(editable=False),
        client=object(),
        api_call=_api_call(_api_response("I prepared it.", intents=[intent])),
    )

    assert result.action_intents[0].record_ids == (
        "word:proposal-only:proposal-only",
    )


def test_run_replans_and_refuses_context_drift_before_authority_or_dispatch(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    expected_context = _context(marker="planned")
    changed_context = _context(marker="changed")
    plan = assistant_agent.plan_agent(
        config,
        context=expected_context,
        message="What changed?",
    )
    calls = 0

    def loader() -> assistant_agent.AgentContext:
        nonlocal calls
        calls += 1
        return expected_context if calls == 1 else changed_context

    with pytest.raises(assistant_agent.AgentApplicationError, match="context.*changed"):
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=loader,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("must not dispatch"),
        )

    assert calls == 2
    assert not config.operations_file.exists()
    assert not config.assistant_dir.exists()


def test_one_live_operation_gate_refuses_under_journal_authority(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context(focused=False, editable=False)
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Summarize this library.",
    )
    operations.OperationJournal.load(config.operations_file).authorize(
        "already-running",
        kind="extract",
        source_file="source.pdf",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )

    with pytest.raises(assistant_agent.AgentRunError, match="will not start another"):
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: context,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("must not dispatch"),
        )

    assert set(
        operations.OperationJournal.load(config.operations_file).operations
    ) == {"already-running"}


def test_recover_commits_captured_reply_without_planning_preparing_or_dispatching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = assistant_agent.plan_agent(
        config,
        context=_context(),
        message="Add one polite example.",
    )
    operation_id = _capture_without_commit(config, plan, monkeypatch)
    (tmp_path / "prompts" / "assistant-agent.md").unlink()
    monkeypatch.setattr(
        assistant_agent.revision_provider,
        "plan_provider",
        lambda *_args, **_kwargs: pytest.fail("recovery must not plan a new call"),
    )
    real_provider = assistant_agent.revision_provider.provider_for(plan.provider)

    class RecoveryOnlyProvider:
        def recover(self, provider_plan: Any, reply: bytes) -> CallResult:
            return real_provider.recover(provider_plan, reply)

        def prepare(self, *_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("recovery must not prepare provider credentials")

        def dispatch(self, *_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("recovery must not redispatch")

    monkeypatch.setattr(
        assistant_agent.revision_provider,
        "provider_for",
        lambda _name: RecoveryOnlyProvider(),
    )
    progress: list[str] = []

    result = assistant_agent.recover_agent(
        config,
        operation_id,
        progress=progress.append,
    )

    assert result.answer == "The captured answer."
    assert result.action_intents[0].record_ids == (_record().id,)
    assert progress == ["Preparing answer", "Writing answer", "Saving answer"]
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "committed"
    )
    with pytest.raises(assistant_agent.AgentApplicationError, match="no captured"):
        assistant_agent.recover_agent(config, operation_id)


def test_recover_preserves_exact_complete_manifest_after_journal_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = assistant_agent.plan_agent(
        config,
        context=_context(),
        message="Add one polite example.",
    )
    operation_id = _capture_without_commit(
        config,
        plan,
        monkeypatch,
        complete_manifest_landed=True,
    )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    before = manifest_path.read_bytes()
    assert json.loads(before)["state"] == "complete"

    result = assistant_agent.recover_agent(config, operation_id)

    assert result.answer == "The captured answer."
    assert manifest_path.read_bytes() == before
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "committed"
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "top-level-extra",
        "context-extra",
        "request-extra",
        "editable-record-extra",
        "message",
        "completed-answer",
        "completed-intent",
    ],
)
def test_recover_refuses_tampered_manifest_and_preserves_captured_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = _project(tmp_path)
    plan = assistant_agent.plan_agent(
        config,
        context=_context(),
        message="Add one polite example.",
    )
    complete = tamper.startswith("completed-")
    operation_id = _capture_without_commit(
        config,
        plan,
        monkeypatch,
        complete_manifest_landed=complete,
    )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "top-level-extra":
        manifest["unexpected"] = True
    elif tamper == "context-extra":
        manifest["context"]["unexpected"] = True
    elif tamper == "request-extra":
        manifest["request"]["unexpected"] = True
    elif tamper == "editable-record-extra":
        manifest["context"]["editable_records"][0]["unexpected"] = True
    elif tamper == "message":
        manifest["request"]["message"] = "A substituted request."
    elif tamper == "completed-answer":
        manifest["answer"] = "A substituted answer."
    else:
        manifest["action_intents"][0]["instruction"] = "A substituted intent."
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(assistant_agent.AgentApplicationError):
        assistant_agent.recover_agent(config, operation_id)

    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )


def test_recover_refuses_noncanonical_operation_id_before_path_resolution(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    with pytest.raises(assistant_agent.AgentApplicationError, match="ID.*invalid"):
        assistant_agent.recover_agent(config, "../../outside")

    assert not config.assistant_dir.exists()


def test_fresh_plan_binds_hidden_editable_record_snapshot_too(tmp_path: Path) -> None:
    config = _project(tmp_path)
    context = _context()
    plan = assistant_agent.plan_agent(
        config,
        context=context,
        message="Change this card.",
    )
    changed = _context()
    changed_record = changed.editable_records[0]
    changed_record.meanings = ["a changed current meaning"]
    changed = replace(changed, editable_records=(changed_record,))

    with pytest.raises(assistant_agent.AgentApplicationError, match="context.*changed"):
        assistant_agent.run_agent(
            config,
            plan,
            context_loader=lambda: changed,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("must not dispatch"),
        )

    assert not config.operations_file.exists()
