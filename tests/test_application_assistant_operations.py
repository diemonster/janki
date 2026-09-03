"""Plan-bound Assistant access to paid-operation recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import operations, prompts
from japanese_anki.application import assistant_operations
from japanese_anki.config import ProjectConfig


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        '[project]\nname = "Operation Assistant fixture"\n'
        '[paths]\noperations_file = "data/operations.json"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _authorize(config: ProjectConfig, operation_id: str = "op-1") -> operations.Operation:
    return operations.OperationJournal.load(config.operations_file).authorize(
        operation_id,
        kind="assistant_chat",
        source_file="data/assistant/request.json",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )


def _capture(config: ProjectConfig, payload: bytes) -> operations.Operation:
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance("op-1", "dispatching")
    return journal.capture_result(
        "op-1",
        lambda: operations.capture_artifact(
            config.operations_file,
            "op-1",
            payload,
        ),
    )


def _captured_operation(
    config: ProjectConfig,
    *,
    operation_id: str,
    kind: str,
    source_file: str,
    request_fp: str = "b" * 64,
    payload: bytes = b'RAW-PRIVATE-REPLY-{"answer":"captured"}',
) -> operations.Operation:
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        operation_id,
        kind=kind,
        source_file=source_file,
        source_sha256="a" * 64,
        request_fp=request_fp,
        model="claude-opus-5",
    )
    journal.advance(operation_id, "dispatching")
    return journal.capture_result(
        operation_id,
        lambda: operations.capture_artifact(
            config.operations_file,
            operation_id,
            payload,
        ),
    )


def _manifest(path: Path, *, kind: str, state: str, operation_id: str) -> bytes:
    wire = (
        json.dumps(
            {
                "schema_version": 1 if kind != "conjugation_deck_revision" else 2,
                "kind": kind,
                "state": state,
                "operation_id": operation_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wire)
    return wire


def test_show_reply_plan_reads_the_exact_bound_private_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _authorize(config)
    payload = '{"answer":"日本語"}\n'.encode()
    captured = _capture(config, payload)

    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="show_reply",
    )
    result = assistant_operations.execute_operation_action(config, plan)

    assert plan.operation == captured
    assert plan.projection["operation"]["state"] == "result_captured"
    assert plan.projection["action"] == "show_reply"
    assert result.reply == payload
    assert result.reply_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.private is True
    assert operations.OperationJournal.load(config.operations_file).operations["op-1"] == captured


def test_operation_plan_cannot_widen_ordinary_forget_into_forced_loss(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _authorize(config)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance("op-1", "dispatching")
    journal.end("op-1")
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="forget",
    )

    with pytest.raises(ValueError, match="projection does not bind"):
        replace(plan, force=True)


def test_show_reply_refuses_if_the_operation_was_committed_after_render(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _authorize(config)
    _capture(config, b'{"answer":"paid"}')
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="show_reply",
    )
    operations.OperationJournal.load(config.operations_file).advance(
        "op-1",
        "committed",
    )

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="changed after the action was rendered",
    ):
        assistant_operations.execute_operation_action(config, plan)


def test_end_refuses_if_authorized_became_dispatching_after_render(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _authorize(config)
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="end",
    )
    operations.OperationJournal.load(config.operations_file).advance(
        "op-1",
        "dispatching",
    )

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="changed after the action was rendered",
    ):
        assistant_operations.execute_operation_action(config, plan)
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations["op-1"]
        .state
        == "dispatching"
    )


def test_end_authorized_operation_records_that_nothing_was_sent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _authorize(config)
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="end",
    )

    result = assistant_operations.execute_operation_action(config, plan)

    assert result.operation is not None
    assert result.operation.state == "canceled_before_send"
    assert result.reply is None
    assert result.private is False


def test_forced_forget_requires_explicit_paid_output_loss_acceptance(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _authorize(config)
    _capture(config, b'{"answer":"paid"}')

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="explicitly accept losing",
    ):
        assistant_operations.plan_operation_action(
            config,
            operation_id="op-1",
            action="forget",
        )


def test_forced_forget_refuses_if_reply_became_committed_after_render(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _authorize(config)
    _capture(config, b'{"answer":"paid"}')
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="forget",
        accept_paid_output_loss=True,
    )
    operations.OperationJournal.load(config.operations_file).advance(
        "op-1",
        "committed",
    )

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="changed after the action was rendered",
    ):
        assistant_operations.execute_operation_action(config, plan)
    assert "op-1" in operations.OperationJournal.load(config.operations_file).operations


def test_explicit_forced_forget_retires_the_exact_paid_reply(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _authorize(config)
    captured = _capture(config, b'{"answer":"paid"}')
    assert captured.artifact is not None
    artifact = config.operations_file.parent / captured.artifact.relative_name
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="forget",
        accept_paid_output_loss=True,
    )

    result = assistant_operations.execute_operation_action(config, plan)

    assert result.operation is None
    assert result.forgotten == 1
    assert not artifact.exists()
    assert "op-1" not in operations.OperationJournal.load(config.operations_file).operations


def test_operation_choices_put_the_current_spending_blocker_first(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _authorize(config, "a-finished")
    journal = operations.OperationJournal.load(config.operations_file)
    journal.advance("a-finished", "dispatching")
    journal.capture_result(
        "a-finished",
        lambda: operations.capture_artifact(
            config.operations_file,
            "a-finished",
            b'{"answer":"already durable"}',
        ),
    )
    journal.advance("a-finished", "committed")
    _authorize(config, "z-blocking")

    choices = assistant_operations.list_operation_choices(config)

    assert [choice.operation_id for choice in choices] == ["z-blocking", "a-finished"]
    assert choices[0].blocks_spending is True


def test_captured_assistant_agent_can_recover_its_exact_manifest_without_reply_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    operation_id = "11111111-1111-4111-8111-111111111111"
    payload = b"RAW-PRIVATE-ASSISTANT-REPLY"
    captured = _captured_operation(
        config,
        operation_id=operation_id,
        kind="assistant_agent",
        source_file="resource_deck",
        payload=payload,
    )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    manifest = _manifest(
        manifest_path,
        kind="assistant_agent",
        state="request",
        operation_id=operation_id,
    )
    called: list[str] = []

    def recover_agent(
        fresh_config: ProjectConfig,
        selected_operation_id: str,
        **_kwargs: object,
    ) -> object:
        called.append(selected_operation_id)
        operations.OperationJournal.load(fresh_config.operations_file).advance(
            selected_operation_id,
            "committed",
        )
        manifest_path.write_bytes(manifest + b"recovered\n")
        return SimpleNamespace(
            manifest_path=manifest_path,
            answer="Recovered ordinary answer",
            action_intents=(SimpleNamespace(kind="inspect"),),
        )

    monkeypatch.setattr(assistant_operations.assistant_agent, "recover_agent", recover_agent)

    choices = assistant_operations.list_operation_choices(config)
    choice = next(item for item in choices if item.operation_id == operation_id)
    assert [action.action for action in choice.actions][:2] == ["recover", "show_reply"]
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id=operation_id,
        action="recover",
    )
    recovery = plan.projection["recovery"]
    assert recovery["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert recovery["captured_reply_sha256"] == captured.artifact.content_sha256
    assert recovery["consequence"] == "commit the captured Assistant turn"

    result = assistant_operations.execute_operation_action(config, plan)

    assert called == [operation_id]
    assert result.reply is None
    assert result.private is False
    assert result.recovery_kind == "assistant_agent"
    assert result.assistant_answer == "Recovered ordinary answer"
    assert result.assistant_action_intent_count == 1
    assert result.result_names == (manifest_path.name,)
    assert result.result_sha256 == (hashlib.sha256(manifest + b"recovered\n").hexdigest(),)
    assert payload not in b"\n".join(name.encode() for name in result.result_names)


@pytest.mark.parametrize("manifest_state", ["request", "complete", "failed"])
def test_every_durable_agent_manifest_state_can_still_be_recovered(
    tmp_path: Path,
    manifest_state: str,
) -> None:
    """A durable manifest state the owner cannot act on is a dead end.

    `recover_agent` settles a `failed` manifest without another model call, but
    that branch is only reachable if the recovery binding accepts the state.
    Leaving one out silently reduces the owner's choices to discarding the
    reply, which is the opposite of what the failed state is for.
    """

    config = _config(tmp_path)
    operation_id = "22222222-2222-4222-8222-222222222222"
    _captured_operation(
        config,
        operation_id=operation_id,
        kind="assistant_agent",
        source_file="resource_deck",
    )
    _manifest(
        config.assistant_dir / f"{operation_id}.json",
        kind="assistant_agent",
        state=manifest_state,
        operation_id=operation_id,
    )

    choices = assistant_operations.list_operation_choices(config)
    choice = next(item for item in choices if item.operation_id == operation_id)

    assert "recover" in [action.action for action in choice.actions]


def test_captured_generic_card_revision_routes_to_its_existing_recovery_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    operation_id = "22222222-2222-4222-8222-222222222222"
    captured = _captured_operation(
        config,
        operation_id=operation_id,
        kind="revise",
        source_file="data/decks/lesson.yaml",
    )
    manifest_path = config.staging_dir / f"card-revision-{operation_id}.request.json"
    manifest = _manifest(
        manifest_path,
        kind="canonical_card_revision",
        state="request",
        operation_id=operation_id,
    )
    staging_path = config.staging_dir / f"card-revision-{operation_id}.yaml"
    called: list[str] = []

    def recover_card_revision(
        fresh_config: ProjectConfig,
        selected_operation_id: str,
        **_kwargs: object,
    ) -> object:
        called.append(selected_operation_id)
        staging_path.write_text("records: []\n", encoding="utf-8")
        operations.OperationJournal.load(fresh_config.operations_file).advance(
            selected_operation_id,
            "committed",
        )
        return SimpleNamespace(
            request_manifest_path=manifest_path,
            staging_path=staging_path,
        )

    monkeypatch.setattr(
        assistant_operations.card_revision,
        "recover_card_revision",
        recover_card_revision,
    )

    plan = assistant_operations.plan_operation_action(
        config,
        operation_id=operation_id,
        action="recover",
    )
    assert plan.projection["recovery"] == {
        "kind": "card_revision",
        "manifest_name": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "captured_reply_sha256": captured.artifact.content_sha256,
        "result_names": [manifest_path.name, staging_path.name],
        "consequence": "stage the captured generic card revision for owner review",
    }

    result = assistant_operations.execute_operation_action(config, plan)

    assert called == [operation_id]
    assert result.recovery_kind == "card_revision"
    assert result.result_names == (manifest_path.name, staging_path.name)
    assert result.result_sha256 == (
        hashlib.sha256(manifest).hexdigest(),
        hashlib.sha256(b"records: []\n").hexdigest(),
    )
    assert result.reply is None


def test_captured_rich_revision_routes_to_its_existing_recovery_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    operation_id = "33333333-3333-4333-8333-333333333333"
    request_fp = "c" * 64
    source_file = "data/decks/potential.yaml"
    captured = _captured_operation(
        config,
        operation_id=operation_id,
        kind="revise",
        source_file=source_file,
        request_fp=request_fp,
    )
    manifest_path = config.staging_dir / (
        f"revise-{prompts.fingerprint(source_file)[:12]}-{request_fp[:16]}.json"
    )
    manifest = _manifest(
        manifest_path,
        kind="conjugation_deck_revision",
        state="request",
        operation_id=operation_id,
    )
    called: list[str] = []

    def recover_revision(
        fresh_config: ProjectConfig,
        selected_operation_id: str,
        **_kwargs: object,
    ) -> object:
        called.append(selected_operation_id)
        operations.OperationJournal.load(fresh_config.operations_file).advance(
            selected_operation_id,
            "committed",
        )
        return SimpleNamespace(staging_path=manifest_path)

    monkeypatch.setattr(assistant_operations.revision, "recover_revision", recover_revision)

    plan = assistant_operations.plan_operation_action(
        config,
        operation_id=operation_id,
        action="recover",
    )
    recovery = plan.projection["recovery"]
    assert recovery["kind"] == "conjugation_revision"
    assert recovery["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert recovery["captured_reply_sha256"] == captured.artifact.content_sha256
    assert recovery["result_names"] == [manifest_path.name]

    result = assistant_operations.execute_operation_action(config, plan)

    assert called == [operation_id]
    assert result.recovery_kind == "conjugation_revision"
    assert result.result_names == (manifest_path.name,)
    assert result.result_sha256 == (hashlib.sha256(manifest).hexdigest(),)
    assert result.reply is None


def test_recover_refuses_when_the_exact_manifest_changes_after_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    operation_id = "44444444-4444-4444-8444-444444444444"
    _captured_operation(
        config,
        operation_id=operation_id,
        kind="assistant_agent",
        source_file="resource_deck",
    )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    _manifest(
        manifest_path,
        kind="assistant_agent",
        state="request",
        operation_id=operation_id,
    )
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id=operation_id,
        action="recover",
    )
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        assistant_operations.assistant_agent,
        "recover_agent",
        lambda *_args, **_kwargs: pytest.fail("stale recovery reached its service"),
    )

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="changed after the action was rendered",
    ):
        assistant_operations.execute_operation_action(config, plan)


def test_recover_refuses_when_the_operation_changes_after_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    operation_id = "66666666-6666-4666-8666-666666666666"
    _captured_operation(
        config,
        operation_id=operation_id,
        kind="assistant_agent",
        source_file="resource_deck",
    )
    _manifest(
        config.assistant_dir / f"{operation_id}.json",
        kind="assistant_agent",
        state="request",
        operation_id=operation_id,
    )
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id=operation_id,
        action="recover",
    )
    operations.OperationJournal.load(config.operations_file).advance(
        operation_id,
        "committed",
    )
    monkeypatch.setattr(
        assistant_operations.assistant_agent,
        "recover_agent",
        lambda *_args, **_kwargs: pytest.fail("stale recovery reached its service"),
    )

    with pytest.raises(
        assistant_operations.AssistantOperationError,
        match="changed after the action was rendered",
    ):
        assistant_operations.execute_operation_action(config, plan)


@pytest.mark.parametrize("state", ["committed", "outcome_unknown"])
def test_finished_operation_can_be_forgotten_without_loss_acceptance(
    tmp_path: Path,
    state: str,
) -> None:
    config = _config(tmp_path)
    _authorize(config)
    journal = operations.OperationJournal.load(config.operations_file)
    if state == "committed":
        journal.advance("op-1", "dispatching")
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                config.operations_file,
                "op-1",
                b'{"answer":"durable elsewhere"}',
            ),
        )
        journal.advance("op-1", "committed")
    else:
        journal.advance("op-1", "dispatching")
        journal.end("op-1")
    plan = assistant_operations.plan_operation_action(
        config,
        operation_id="op-1",
        action="forget",
    )

    result = assistant_operations.execute_operation_action(config, plan)

    assert result.forgotten == 1
    assert "op-1" not in operations.OperationJournal.load(config.operations_file).operations
