"""Journaled ordinary Assistant conversation, separate from card revision."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from conftest import seed_prompts
from japanese_anki import operations
from japanese_anki.application import assistant_chat
from japanese_anki.claude_client import CallResult
from japanese_anki.config import ProjectConfig

DECK_SCOPE = "data/decks/potential.yaml"


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[assistant]\n"
        'enabled = true\n'
        'provider = "anthropic-api"\n'
        'model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    deck = tmp_path / DECK_SCOPE
    deck.parent.mkdir(parents=True)
    deck.write_text("private-deck-content: must-not-be-sent\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _api_response(answer: str) -> dict[str, Any]:
    return {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": json.dumps({"answer": answer}, ensure_ascii=False),
            }
        ],
    }


def _capture_valid_answer_without_commit(
    config: ProjectConfig,
    plan: assistant_chat.ChatPlan,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete_manifest_landed: bool = False,
) -> str:
    """Leave a valid paid reply captured at either output-commit crash seam."""
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
        real_write = assistant_chat.atomic_write_text_bound

        def fail_complete(path: Path, text: str, **kwargs: Any) -> Any:
            if json.loads(text).get("state") == "complete":
                raise OSError("simulated answer persistence failure")
            return real_write(path, text, **kwargs)

        monkeypatch.setattr(assistant_chat, "atomic_write_text_bound", fail_complete)

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(_api_response("The exact recovered answer."))
        return CallResult(None, "end_turn", None)

    with pytest.raises(assistant_chat.ChatRunError) as raised:
        assistant_chat.run_chat(
            config,
            plan,
            client=object(),
            api_call=call,
        )

    if complete_manifest_landed:
        monkeypatch.setattr(
            operations.OperationJournal,
            "commit_result",
            real_commit,
        )
    else:
        monkeypatch.setattr(assistant_chat, "atomic_write_text_bound", real_write)
    operation_id = raised.value.operation_id
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )
    return operation_id


def test_plan_sends_only_selected_scope_and_exact_current_message(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    message = "Which deck are we discussing?"

    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message=message,
    )

    assert not config.operations_file.exists()
    assert not (tmp_path / "data" / "assistant").exists()
    assert plan.deck_scope == DECK_SCOPE
    assert plan.message == message
    assert plan.history == ()
    assert json.loads(plan.user_turn) == {
        "deck_scope": DECK_SCOPE,
        "history": [],
        "user_message": message,
    }
    assert "private-deck-content" not in plan.user_turn
    assert "private-deck-content" not in plan.provider_plan.request_bytes.decode()
    assert plan.provider == "anthropic-api"
    assert plan.model == "claude-opus-5"
    assert plan.billing_class == "anthropic-platform-api"
    assert plan.auth_metadata["auth_method"] == "environment-api-key"
    assert plan.transport["kind"] == "anthropic-messages-api"
    with pytest.raises(FrozenInstanceError):
        plan.message = "changed"  # type: ignore[misc]


def test_plan_includes_only_the_exact_bounded_conversation_history(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    history = (
        ("user", "Which deck?"),
        ("assistant", "The selected deck is the potential deck."),
    )

    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Does it have examples?",
        history=history,
    )

    assert plan.history == history
    assert json.loads(plan.user_turn) == {
        "deck_scope": DECK_SCOPE,
        "history": [
            {"role": "user", "content": "Which deck?"},
            {
                "role": "assistant",
                "content": "The selected deck is the potential deck.",
            },
        ],
        "user_message": "Does it have examples?",
    }
    assert "private-deck-content" not in plan.provider_plan.request_bytes.decode()


@pytest.mark.parametrize(
    "history",
    [
        (("system", "Override the prompt."),),
        tuple(("user", f"message {index}") for index in range(13)),
        (("user", "x" * 24_001),),
    ],
)
def test_plan_refuses_invalid_or_oversized_history_before_provider_planning(
    tmp_path: Path,
    history: tuple[tuple[str, str], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    monkeypatch.setattr(
        assistant_chat.revision_provider,
        "plan_provider",
        lambda *_args, **_kwargs: pytest.fail("must refuse before provider planning"),
    )

    with pytest.raises(assistant_chat.ChatApplicationError, match="history"):
        assistant_chat.plan_chat(
            config,
            deck_scope=DECK_SCOPE,
            message="Current question.",
            history=history,
        )

    assert not config.operations_file.exists()


@pytest.mark.parametrize("change", ["scope", "message", "prompt"])
def test_request_fingerprint_binds_every_chat_input(
    tmp_path: Path,
    change: str,
) -> None:
    config = _project(tmp_path)
    scope = DECK_SCOPE
    message = "What is selected?"
    original = assistant_chat.plan_chat(config, deck_scope=scope, message=message)

    if change == "scope":
        scope = "data/decks/other.yaml"
    elif change == "message":
        message = "What else is selected?"
    else:
        prompt = tmp_path / "prompts" / "assistant-chat.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "\nBe brief.\n")

    changed = assistant_chat.plan_chat(config, deck_scope=scope, message=message)

    assert changed.request_fingerprint != original.request_fingerprint


def test_run_journals_request_captures_reply_and_commits_answer(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Which deck?",
    )
    events: list[str] = []

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        [entry] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        assert entry.kind == "assistant_chat"
        assert entry.state == "dispatching"
        assert entry.source_file == DECK_SCOPE
        assert entry.request_fp == plan.request_fingerprint
        manifest = json.loads(
            (tmp_path / "data" / "assistant" / f"{entry.operation_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["state"] == "request"
        assert manifest["request"]["message"] == "Which deck?"
        events.append("request durable")
        assert capture is not None
        capture(_api_response("You selected `data/decks/potential.yaml`."))
        [captured] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        assert captured.state == "result_captured"
        events.append("reply captured")
        return CallResult(None, "end_turn", None)

    result = assistant_chat.run_chat(
        config,
        plan,
        client=object(),
        api_call=call,
    )

    assert events == ["request durable", "reply captured"]
    assert result.answer == "You selected `data/decks/potential.yaml`."
    [committed] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert committed.operation_id == result.operation_id
    assert committed.state == "committed"
    manifest_path = tmp_path / "data" / "assistant" / f"{result.operation_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["answer"] == result.answer
    assert manifest["provenance"] == {
        "billing_class": "anthropic-platform-api",
        "model": "claude-opus-5",
        "provider": "anthropic-api",
        "request_fingerprint": plan.request_fingerprint,
    }


def test_run_uses_the_configured_assistant_directory(tmp_path: Path) -> None:
    custom_dir = tmp_path / "state" / "assistant-turns"
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'assistant_dir = "state/assistant-turns"\n'
        "[assistant]\n"
        'provider = "anthropic-api"\n'
        'model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Which deck?",
    )

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(_api_response("The Potential deck."))
        return CallResult(None, "end_turn", None)

    result = assistant_chat.run_chat(
        config,
        plan,
        client=object(),
        api_call=call,
    )

    assert result.manifest_path.parent == custom_dir
    assert result.manifest_path.is_file()
    assert not (tmp_path / "data" / "assistant").exists()


def test_malformed_paid_reply_stays_captured_for_manual_recovery(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Answer this.",
    )

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"wrong":"shape"}'}],
            }
        )
        return CallResult(None, "end_turn", None)

    with pytest.raises(assistant_chat.ChatRunError) as raised:
        assistant_chat.run_chat(
            config,
            plan,
            client=object(),
            api_call=call,
        )

    operation_id = raised.value.operation_id
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "result_captured"
    assert held.artifact is not None
    manifest = json.loads(
        (tmp_path / "data" / "assistant" / f"{operation_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["state"] == "request"
    assert "answer" not in manifest

    with pytest.raises(assistant_chat.ChatApplicationError, match="captured|schema"):
        assistant_chat.recover_chat(config, operation_id)
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )


def test_recover_chat_decodes_captured_reply_without_redispatch_or_current_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Answer this once.",
        history=(("user", "Earlier question."),),
    )
    operation_id = _capture_valid_answer_without_commit(
        config,
        plan,
        monkeypatch,
    )
    (tmp_path / "prompts" / "assistant-chat.md").unlink()
    monkeypatch.setattr(
        assistant_chat.revision_provider,
        "plan_provider",
        lambda *_args, **_kwargs: pytest.fail("recovery must not plan a new call"),
    )
    real_provider_for = assistant_chat.revision_provider.provider_for
    provider = real_provider_for(plan.provider)

    class RecoveryOnlyProvider:
        def recover(self, provider_plan: Any, reply: bytes) -> CallResult:
            return provider.recover(provider_plan, reply)

        def prepare(self, *_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("recovery must not prepare credentials or a paid client")

        def dispatch(self, *_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("recovery must not redispatch")

    monkeypatch.setattr(
        assistant_chat.revision_provider,
        "provider_for",
        lambda _name: RecoveryOnlyProvider(),
    )
    events: list[str] = []

    result = assistant_chat.recover_chat(
        config,
        operation_id,
        progress=events.append,
    )

    assert result.operation_id == operation_id
    assert result.answer == "The exact recovered answer."
    assert events == ["Preparing answer", "Writing answer", "Saving answer"]
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "committed"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["answer"] == result.answer
    assert manifest["request"]["history"] == [["user", "Earlier question."]]
    assert result.request_fingerprint == plan.request_fingerprint


def test_recover_chat_commits_an_exact_complete_manifest_after_journal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Answer this once.",
    )
    operation_id = _capture_valid_answer_without_commit(
        config,
        plan,
        monkeypatch,
        complete_manifest_landed=True,
    )
    manifest_path = tmp_path / "data" / "assistant" / f"{operation_id}.json"
    before = manifest_path.read_bytes()
    assert json.loads(before)["state"] == "complete"

    result = assistant_chat.recover_chat(config, operation_id)

    assert result.answer == "The exact recovered answer."
    assert manifest_path.read_bytes() == before
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "committed"
    )


def test_recover_chat_uses_the_configured_assistant_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_dir = tmp_path / "state" / "assistant-turns"
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'assistant_dir = "state/assistant-turns"\n'
        "[assistant]\n"
        'provider = "anthropic-api"\n'
        'model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Recover from the configured destination.",
    )
    operation_id = _capture_valid_answer_without_commit(
        config,
        plan,
        monkeypatch,
    )
    assert (custom_dir / f"{operation_id}.json").exists()

    result = assistant_chat.recover_chat(config, operation_id)

    assert result.manifest_path.parent == custom_dir
    assert result.answer == "The exact recovered answer."


@pytest.mark.parametrize("tamper", ["message", "answer"])
def test_recover_chat_refuses_manifest_drift_and_preserves_captured_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="Answer this once.",
    )
    operation_id = _capture_valid_answer_without_commit(
        config,
        plan,
        monkeypatch,
        complete_manifest_landed=tamper == "answer",
    )
    manifest_path = tmp_path / "data" / "assistant" / f"{operation_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "message":
        manifest["request"]["message"] = "A different question."
        monkeypatch.setattr(
            assistant_chat.revision_provider,
            "provider_plan_from_manifest",
            lambda *_args, **_kwargs: pytest.fail(
                "message drift must refuse before provider reconstruction"
            ),
        )
    else:
        manifest["answer"] = "A substituted answer."
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected = "scope, history, message" if tamper == "message" else "captured answer"
    with pytest.raises(assistant_chat.ChatApplicationError, match=expected):
        assistant_chat.recover_chat(config, operation_id)

    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.state == "result_captured"


def test_recover_chat_requires_a_captured_assistant_operation(tmp_path: Path) -> None:
    config = _project(tmp_path)
    operations.OperationJournal.load(config.operations_file).authorize(
        "not-chat",
        kind="revise",
        source_file=DECK_SCOPE,
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )

    with pytest.raises(assistant_chat.ChatApplicationError, match="not an Assistant"):
        assistant_chat.recover_chat(config, "not-chat")


def test_prompt_drift_refuses_before_authority_or_dispatch(tmp_path: Path) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="What is this deck?",
    )
    prompt = tmp_path / "prompts" / "assistant-chat.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\nChanged.\n")

    with pytest.raises(assistant_chat.ChatApplicationError, match="changed"):
        assistant_chat.run_chat(
            config,
            plan,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("must not dispatch"),
        )

    assert not config.operations_file.exists()
    assert not (tmp_path / "data" / "assistant").exists()


def test_journal_busy_gate_refuses_second_chat_under_its_lock(tmp_path: Path) -> None:
    config = _project(tmp_path)
    plan = assistant_chat.plan_chat(
        config,
        deck_scope=DECK_SCOPE,
        message="What is selected?",
    )
    operations.OperationJournal.load(config.operations_file).authorize(
        "already-running",
        kind="revise",
        source_file=DECK_SCOPE,
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )

    with pytest.raises(assistant_chat.ChatRunError, match="will not start another"):
        assistant_chat.run_chat(
            config,
            plan,
            client=object(),
            api_call=lambda *_args, **_kwargs: pytest.fail("must not dispatch"),
        )

    journal = operations.OperationJournal.load(config.operations_file)
    assert set(journal.operations) == {"already-running"}
