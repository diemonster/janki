"""Assistant staging-owner intents use one exact local confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant as assistant_controller
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.assistant import RevisionConfirmation, RevisionRefusal


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _adapter(config: ProjectConfig) -> assistant_adapter.RevisionAssistantAdapter:
    return assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )


def _intent(
    kind: str,
    *,
    resource_ids: tuple[str, ...] = ("proposal-resource",),
    record_ids: tuple[str, ...] = (),
    instruction: str = "Apply this exact staged owner action.",
    options: dict[str, Any] | None = None,
) -> Any:
    return assistant_adapter.assistant_agent.AgentActionIntent(
        kind=kind,
        resource_ids=resource_ids,
        record_ids=record_ids,
        instruction=instruction,
        options_json=json.dumps(
            options or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _result(intent: Any) -> Any:
    return SimpleNamespace(
        answer="Here is the exact local action.",
        action_intents=(intent,),
    )


class _FakeDeletionPlan:
    def __init__(self, *, fingerprint: str = "a" * 64) -> None:
        self.fingerprint = fingerprint
        self.proposal_path = Path("/tmp/janki-staging-actions/lesson.pdf.yaml")
        self.projection = {
            "target": {
                "proposal_kind": "source_extraction",
                "resource_id": "proposal-resource",
                "staging_proposal": "data/staging/lesson.pdf.yaml",
            },
            "selection": {
                "record_ids": ["word:撮る:とる"],
                "records": [
                    {
                        "record_id": "word:撮る:とる",
                        "expression": "撮る",
                        "reading": "とる",
                        "meanings": ["to take a photo"],
                        "examples": [
                            {
                                "japanese": "写真を撮ります。",
                                "english": "I will take a photo.",
                            }
                        ],
                    }
                ],
            },
            "snapshots": {
                "before_sha256": "1" * 64,
                "after_sha256": "2" * 64,
            },
            "effects": {
                "rows_before": 2,
                "rows_removed": 1,
                "rows_after": 1,
                "canonical_cards_changed": False,
                "paid_provider_call": False,
            },
        }


class _FakeCanonicalDeletionPlan:
    def __init__(self, *, fingerprint: str = "d" * 64) -> None:
        self.fingerprint = fingerprint
        self.projection = {
            "selection": {
                "record_ids": ["word:撮る:とる"],
                "cards": [
                    {
                        "resource_id": "card-resource",
                        "record_id": "word:撮る:とる",
                        "canonical_sha256": "6" * 64,
                        "record": {
                            "expression": "撮る",
                            "reading": "とる",
                            "meanings": ["to take a photo"],
                        },
                    }
                ],
            },
            "canonical_collection": {
                "configured_file": "data/normalized/vocabulary.json",
                "rows_before": 2,
                "rows_removed": 1,
                "rows_after": 1,
                "before_sha256": "7" * 64,
                "after_sha256": "8" * 64,
            },
            "affected_decks": [
                {
                    "name": "Lesson deck",
                    "configured_file": "data/decks/lesson.yaml",
                    "selected_before": ["word:撮る:とる"],
                    "selected_after_rebuild": [],
                    "notes_before": 2,
                    "notes_after_rebuild": 1,
                    "declared_package": "dist/lesson.apkg",
                }
            ],
            "retained": {
                "configured_deck_definitions": True,
                "generated_packages": ["dist/lesson.apkg"],
                "ledger_history_for_record_ids": ["word:撮る:とる"],
                "media_references": ["word-toru.wav"],
                "staging_proposals": True,
            },
            "writes": {
                "replace_canonical_collection": "data/normalized/vocabulary.json",
                "remove_record_ids": ["word:撮る:とる"],
            },
            "inputs": {
                "repository_files": [
                    {
                        "repository_file": "data/normalized/custom-input.json",
                        "exists": True,
                        "sha256": "f" * 64,
                    }
                ],
                "ledger_sha256": "c" * 64,
            },
        }


class _FakeDeckDeletionPlan:
    def __init__(self, *, fingerprint: str = "e" * 64) -> None:
        self.fingerprint = fingerprint
        self.projection = {
            "target": {
                "resource_id": "deck-resource",
                "name": "Lesson deck",
                "kind": "vocabulary",
                "configured_file": "data/decks/lesson.yaml",
                "sha256": "9" * 64,
            },
            "consequences": {
                "currently_selected_record_ids": ["word:撮る:とる"],
                "canonical_cards_removed": [],
                "inline_only_record_ids_removed_from_library": ["word:inline"],
                "configured_deck_count_before": 3,
                "configured_deck_count_after": 2,
            },
            "retained": {
                "canonical_collection": "data/normalized/vocabulary.json",
                "generated_package": "dist/lesson.apkg",
                "ledger_history": True,
                "media": True,
                "staging_proposals": True,
            },
            "removals": ["data/decks/lesson.yaml"],
            "inputs": {
                "repository_files": [
                    {
                        "repository_file": "data/decks/custom-source.json",
                        "exists": True,
                        "sha256": "b" * 64,
                    }
                ],
                "ledger_sha256": "c" * 64,
            },
        }


class _FakeReidentificationPlan:
    def __init__(self, *, fingerprint: str = "b" * 64) -> None:
        self.fingerprint = fingerprint
        self.proposal_path = Path("/tmp/janki-staging-actions/lesson.pdf.yaml")
        self.projection = {
            "target": {
                "proposal_kind": "source_extraction",
                "resource_id": "proposal-resource",
                "staging_proposal": "data/staging/lesson.pdf.yaml",
            },
            "identity": {
                "old": {
                    "record_id": "word:撮る:とる",
                    "expression": "撮る",
                    "reading": "とる",
                },
                "new": {
                    "record_id": "word:写真を撮る:しゃしんをとる",
                    "expression": "写真を撮る",
                    "reading": "しゃしんをとる",
                },
                "neighbours": [
                    {
                        "record_id": "word:写真:しゃしん",
                        "expression": "写真",
                        "reading": "しゃしん",
                        "relation": "same-spelling",
                        "where": "your collection",
                    }
                ],
                "consequences": [
                    "This card has not been built into a deck yet.",
                    "The Japanese sentences on this card are not changed.",
                ],
                "was_exported": False,
            },
            "snapshots": {
                "before_sha256": "3" * 64,
                "after_sha256": "4" * 64,
            },
            "effects": {
                "japanese_examples_changed": False,
                "canonical_cards_changed": False,
                "paid_provider_call": False,
            },
        }


class _FakeCoveragePlan:
    def __init__(self, *, fingerprint: str = "c" * 64) -> None:
        self.fingerprint = fingerprint
        self.proposal_path = Path("/tmp/janki-staging-actions/lesson.pdf.yaml")
        self.projection = {
            "target": {
                "proposal_kind": "source_extraction",
                "resource_id": "proposal-resource",
                "source_name": "lesson.pdf",
                "staging_proposal": "data/staging/lesson.pdf.yaml",
            },
            "coverage": {
                "account": "row 1: card word:撮る:とる\nrow 2: unusable heading",
                "candidate_units": 1,
                "source_units": 2,
                "staging_sha256": "5" * 64,
            },
            "owner_decision": {
                "authority": "repository-owner",
                "reason": "I compared both source rows with the proposal.",
            },
            "effects": {
                "writes_coverage_approval_to_staging": True,
                "promotes_cards": False,
                "paid_provider_call": False,
            },
        }


def _target(plan: Any) -> str:
    return str(plan.projection["target"]["staging_proposal"])


def _prepared(kind: str, plan: Any, instruction: str) -> Any:
    return assistant_adapter._PreparedAgentAction(
        kind=kind,
        focus_scope="",
        instruction=instruction,
        target=_target(plan),
        plan=plan,
    )


def _confirmation(plan: Any, instruction: str, *, fingerprint: str | None = None) -> Any:
    return RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=fingerprint or plan.fingerprint,
        target=_target(plan),
    )


def test_delete_staged_cards_renders_every_destroyed_row_and_exact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeDeletionPlan()
    instruction = "Remove the selected 撮る proposal."
    calls: list[tuple[str, tuple[str, ...], str]] = []

    def plan_deletion(
        fresh: ProjectConfig,
        *,
        proposal_resource_id: str,
        record_ids: tuple[str, ...],
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        calls.append((proposal_resource_id, record_ids, instruction))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        "plan_staged_deletion",
        plan_deletion,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_result(
            _intent(
                "delete_content",
                record_ids=("word:撮る:とる",),
                instruction=instruction,
                options={"deletion_kind": "staged_cards"},
            )
        ),
        deck_scope="",
    )

    assert calls == [("proposal-resource", ("word:撮る:とる",), instruction)]
    assert reply.action is not None
    assistant_controller._validate_plan(reply.action)
    assert reply.action.target == "data/staging/lesson.pdf.yaml"
    assert reply.action.confirm_label == "Delete these exact staged cards"
    assert reply.action.progress_label == "Removing staged cards"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "word:撮る:とる" in rendered
    assert "写真を撮ります。" in rendered
    assert "to take a photo" in rendered
    assert "from 2 to 1 by removing 1 row(s)" in rendered
    assert "1" * 64 in rendered
    assert "2" * 64 in rendered
    assert '"canonical_cards_changed": false' in rendered
    assert "permanently removes only the exact displayed rows" in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan


def test_reidentify_staged_card_renders_old_new_neighbours_and_consequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeReidentificationPlan()
    instruction = "This proposal is the full expression 写真を撮る."
    calls: list[tuple[str, str, str, str, str]] = []

    def plan_reidentification(
        fresh: ProjectConfig,
        *,
        proposal_resource_id: str,
        record_id: str,
        new_expression: str,
        new_reading: str,
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        calls.append(
            (
                proposal_resource_id,
                record_id,
                new_expression,
                new_reading,
                instruction,
            )
        )
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        "plan_reidentification",
        plan_reidentification,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_result(
            _intent(
                "reidentify_staged_card",
                record_ids=("word:撮る:とる",),
                instruction=instruction,
                options={
                    "new_expression": "写真を撮る",
                    "new_reading": "しゃしんをとる",
                },
            )
        ),
        deck_scope="",
        owner_message="Change it to 写真を撮る with reading しゃしんをとる.",
    )

    assert calls == [
        (
            "proposal-resource",
            "word:撮る:とる",
            "写真を撮る",
            "しゃしんをとる",
            instruction,
        )
    ]
    assert reply.action is not None
    assistant_controller._validate_plan(reply.action)
    assert reply.action.target == "data/staging/lesson.pdf.yaml"
    assert reply.action.confirm_label == "Change this exact staged identity"
    assert reply.action.progress_label == "Saving staged identity"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert '"record_id": "word:撮る:とる"' in rendered
    assert '"record_id": "word:写真を撮る:しゃしんをとる"' in rendered
    assert '"where": "your collection"' in rendered
    assert "has not been built into a deck yet" in rendered
    assert "Previously exported under the old identity: False" in rendered
    assert "3" * 64 in rendered
    assert "4" * 64 in rendered
    assert '"japanese_examples_changed": false' in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan


@pytest.mark.parametrize(
    ("owner_message", "message"),
    [
        (
            "Use reading しゃしんをとる.",
            "new expression must appear verbatim",
        ),
        (
            "Change it to 写真を撮る.",
            "new reading must appear verbatim",
        ),
    ],
    ids=("model-invented-expression", "model-invented-reading"),
)
def test_reidentify_refuses_owner_only_identity_invented_by_the_model_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_message: str,
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        "plan_reidentification",
        lambda *_args, **_kwargs: pytest.fail(
            "model-invented identity text must refuse before reidentification planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_result(
                _intent(
                    "reidentify_staged_card",
                    record_ids=("word:撮る:とる",),
                    options={
                        "new_expression": "写真を撮る",
                        "new_reading": "しゃしんをとる",
                    },
                )
            ),
            deck_scope="",
            owner_message=owner_message,
        )

    assert adapter._agent_plans == {}


def test_approve_coverage_renders_exact_account_and_owner_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeCoveragePlan()
    instruction = "Approve this coverage account."
    reason = "I compared both source rows with the proposal."
    calls: list[tuple[str, str, str]] = []

    def plan_coverage(
        fresh: ProjectConfig,
        *,
        proposal_resource_id: str,
        reason: str,
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        calls.append((proposal_resource_id, reason, instruction))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        "plan_coverage_approval",
        plan_coverage,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_result(
            _intent(
                "approve_coverage",
                instruction=instruction,
                options={"coverage_reason": reason},
            )
        ),
        deck_scope="",
        owner_message=f"Approve coverage because: {reason}",
    )

    assert calls == [("proposal-resource", reason, instruction)]
    assert reply.action is not None
    assistant_controller._validate_plan(reply.action)
    assert reply.action.target == "data/staging/lesson.pdf.yaml"
    assert reply.action.confirm_label == "Approve this exact coverage account"
    assert reply.action.progress_label == "Saving coverage approval"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "row 1: card word:撮る:とる" in rendered
    assert "row 2: unusable heading" in rendered
    assert "Accounted candidate/source units: 1 / 2" in rendered
    assert f"Owner reason: {reason}" in rendered
    assert "5" * 64 in rendered
    assert '"writes_coverage_approval_to_staging": true' in rendered
    assert "not a Japanese-content review" in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan


def test_coverage_approval_refuses_reason_invented_by_the_model_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    reason = "I compared both source rows with the proposal."
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        "plan_coverage_approval",
        lambda *_args, **_kwargs: pytest.fail(
            "a model-invented owner reason must refuse before coverage planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match="coverage reason must appear verbatim"):
        adapter._prepare_agent_intent(
            config,
            result=_result(
                _intent(
                    "approve_coverage",
                    options={"coverage_reason": reason},
                )
            ),
            deck_scope="",
            owner_message="Approve this coverage account.",
        )

    assert adapter._agent_plans == {}


def test_delete_canonical_cards_renders_exact_cards_and_retained_consequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeCanonicalDeletionPlan()
    instruction = "Permanently delete this exact canonical card."
    calls: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []

    def plan_deletion(
        fresh: ProjectConfig,
        *,
        card_resource_ids: tuple[str, ...],
        record_ids: tuple[str, ...],
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        calls.append((card_resource_ids, record_ids, instruction))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_deletion,
        "plan_canonical_deletion",
        plan_deletion,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_result(
            _intent(
                "delete_content",
                resource_ids=("card-resource",),
                record_ids=("word:撮る:とる",),
                instruction=instruction,
                options={"deletion_kind": "canonical_cards"},
            )
        ),
        deck_scope="",
    )

    assert calls == [
        (("card-resource",), ("word:撮る:とる",), instruction),
    ]
    assert reply.action is not None
    assistant_controller._validate_plan(reply.action)
    assert reply.action.target == "data/normalized/vocabulary.json"
    assert reply.action.confirm_label == "Delete these exact canonical cards"
    assert reply.action.progress_label == "Deleting canonical cards"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "撮る" in rendered
    assert "word:撮る:とる" in rendered
    assert "from 2 to 1" in rendered
    assert "Lesson deck" in rendered
    assert "dist/lesson.apkg" in rendered
    assert "retained" in rendered.lower()
    assert "ledger" in rendered.lower()
    assert "Bound repository inputs" in rendered
    assert "data/normalized/custom-input.json" in rendered
    assert "f" * 64 in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan


def test_delete_configured_deck_renders_definition_inline_loss_and_retained_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeDeckDeletionPlan()
    instruction = "Permanently delete this exact configured deck."
    calls: list[tuple[str, str]] = []

    def plan_deletion(
        fresh: ProjectConfig,
        *,
        deck_resource_id: str,
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        calls.append((deck_resource_id, instruction))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_deletion,
        "plan_deck_deletion",
        plan_deletion,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_result(
            _intent(
                "delete_content",
                resource_ids=("deck-resource",),
                instruction=instruction,
                options={"deletion_kind": "deck"},
            )
        ),
        deck_scope="",
    )

    assert calls == [("deck-resource", instruction)]
    assert reply.action is not None
    assistant_controller._validate_plan(reply.action)
    assert reply.action.target == "data/decks/lesson.yaml"
    assert reply.action.confirm_label == "Delete this exact deck definition"
    assert reply.action.progress_label == "Deleting deck definition"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "word:inline" in rendered
    assert "from 3 to 2" in rendered
    assert "dist/lesson.apkg" in rendered
    assert "generated package" in rendered.lower()
    assert "retained" in rendered.lower()
    assert "canonical" in rendered.lower()
    assert "Bound repository inputs" in rendered
    assert "data/decks/custom-source.json" in rendered
    assert "b" * 64 in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan


@pytest.mark.parametrize(
    ("intent", "message"),
    [
        (_intent("delete_content", options={}), "exactly one explicit deletion_kind"),
        (
            _intent(
                "delete_content",
                record_ids=("word:first",),
                options={"deletion_kind": "staged_cards", "coverage_reason": "x"},
            ),
            "exactly one explicit deletion_kind",
        ),
        (
            _intent(
                "delete_content",
                resource_ids=(),
                record_ids=("word:first",),
                options={"deletion_kind": "staged_cards"},
            ),
            "exactly one staging proposal",
        ),
        (
            _intent(
                "delete_content",
                options={"deletion_kind": "staged_cards"},
            ),
            "one or more explicit staged card ids",
        ),
        (
            _intent(
                "delete_content",
                resource_ids=("card-resource",),
                options={"deletion_kind": "canonical_cards"},
            ),
            "same number of explicit canonical card ids",
        ),
        (
            _intent(
                "delete_content",
                resource_ids=("deck-resource",),
                record_ids=("word:first",),
                options={"deletion_kind": "deck"},
            ),
            "one deck resource and no card ids",
        ),
        (
            _intent(
                "reidentify_staged_card",
                record_ids=("word:first",),
                options={"new_expression": "新しい"},
            ),
            "explicit nonblank new_expression and new_reading",
        ),
        (
            _intent(
                "reidentify_staged_card",
                record_ids=("word:first", "word:second"),
                options={"new_expression": "新しい", "new_reading": "あたらしい"},
            ),
            "exactly one source-extraction proposal",
        ),
        (
            _intent(
                "reidentify_staged_card",
                record_ids=("word:first",),
                options={"new_expression": " ", "new_reading": "あたらしい"},
            ),
            "explicit nonblank",
        ),
        (
            _intent(
                "reidentify_staged_card",
                record_ids=("word:first",),
                options={
                    "new_expression": "新しい",
                    "new_reading": "あたらしい",
                    "review_patterns": False,
                },
            ),
            "explicit nonblank new_expression and new_reading",
        ),
        (
            _intent("approve_coverage", options={}),
            "explicit nonblank coverage_reason",
        ),
        (
            _intent(
                "approve_coverage",
                record_ids=("word:first",),
                options={"coverage_reason": "I checked it."},
            ),
            "no card ids",
        ),
        (
            _intent(
                "approve_coverage",
                options={"coverage_reason": " ", "review_patterns": False},
            ),
            "explicit nonblank coverage_reason",
        ),
        (
            _intent(
                "approve_coverage",
                options={
                    "coverage_reason": "I checked it.",
                    "review_patterns": False,
                },
            ),
            "explicit nonblank coverage_reason",
        ),
    ],
)
def test_staging_owner_actions_refuse_incomplete_or_extra_option_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    for planner in (
        "plan_staged_deletion",
        "plan_reidentification",
        "plan_coverage_approval",
    ):
        monkeypatch.setattr(
            assistant_adapter.assistant_staging_actions,
            planner,
            lambda *_args, **_kwargs: pytest.fail(
                "an invalid staged owner action must refuse before planning"
            ),
        )

    with pytest.raises(RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_result(intent),
            deck_scope="",
        )


@pytest.mark.parametrize(
    ("kind", "plan_type", "executor_name", "result_factory", "message_parts"),
    [
        (
            "delete_content",
            _FakeDeletionPlan,
            "execute_staged_deletion",
            lambda plan: SimpleNamespace(
                plan=plan,
                removed_record_ids=("word:撮る:とる",),
            ),
            ("Removed staged card(s) word:撮る:とる", "Canonical cards"),
        ),
        (
            "reidentify_staged_card",
            _FakeReidentificationPlan,
            "execute_reidentification",
            lambda plan: SimpleNamespace(
                plan=plan,
                old_record_id="word:撮る:とる",
                new_record_id="word:写真を撮る:しゃしんをとる",
            ),
            ("word:撮る:とる", "word:写真を撮る:しゃしんをとる"),
        ),
        (
            "approve_coverage",
            _FakeCoveragePlan,
            "execute_coverage_approval",
            lambda plan: SimpleNamespace(
                plan=plan,
                staging_path=plan.proposal_path,
            ),
            ("Recorded repository-owner coverage approval", "No cards were promoted"),
        ),
    ],
)
def test_one_staging_owner_confirmation_delegates_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    plan_type: type[Any],
    executor_name: str,
    result_factory: Any,
    message_parts: tuple[str, str],
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = plan_type()
    instruction = "Apply this exact staged owner action."
    adapter._agent_plans[plan.fingerprint] = _prepared(kind, plan, instruction)
    class_name = {
        "delete_content": "AssistantStagedDeletionPlan",
        "reidentify_staged_card": "AssistantReidentificationPlan",
        "approve_coverage": "AssistantCoverageApprovalPlan",
    }[kind]
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        class_name,
        plan_type,
    )
    delegated: list[Any] = []

    def execute(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        return result_factory(received)

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        executor_name,
        execute,
    )
    confirmation = _confirmation(plan, instruction)
    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=lambda _label: pytest.fail(
            "local staging-owner actions report no provider progress"
        ),
    )

    assert delegated == [plan]
    assert execution.complete is True
    assert all(part in execution.message for part in message_parts)
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


@pytest.mark.parametrize(
    (
        "plan_type",
        "target",
        "class_name",
        "executor_name",
        "execution_result",
        "message_parts",
    ),
    [
        (
            _FakeCanonicalDeletionPlan,
            "data/normalized/vocabulary.json",
            "AssistantCanonicalDeletionPlan",
            "execute_canonical_deletion",
            lambda plan: SimpleNamespace(
                plan=plan,
                removed_record_ids=("word:撮る:とる",),
            ),
            ("Removed canonical card(s) word:撮る:とる", "packages remain"),
        ),
        (
            _FakeDeckDeletionPlan,
            "data/decks/lesson.yaml",
            "AssistantDeckDeletionPlan",
            "execute_deck_deletion",
            lambda plan: SimpleNamespace(
                plan=plan,
                removed_deck_path=Path("/tmp/janki-staging-actions/lesson.yaml"),
            ),
            ("Removed configured deck definition", "package remains"),
        ),
    ],
)
def test_one_canonical_or_deck_deletion_confirmation_delegates_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_type: type[Any],
    target: str,
    class_name: str,
    executor_name: str,
    execution_result: Any,
    message_parts: tuple[str, str],
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = plan_type()
    instruction = "Apply this exact destructive action."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="delete_content",
        focus_scope="",
        instruction=instruction,
        target=target,
        plan=plan,
    )
    monkeypatch.setattr(assistant_adapter.assistant_deletion, class_name, plan_type)
    delegated: list[Any] = []

    def execute(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        return execution_result(received)

    monkeypatch.setattr(assistant_adapter.assistant_deletion, executor_name, execute)
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target=target,
    )

    result = adapter.consume_replan_and_execute(
        confirmation,
        progress=lambda _label: pytest.fail(
            "local deletion reports no provider progress"
        ),
    )

    assert delegated == [plan]
    assert result.complete is True
    assert all(part in result.message for part in message_parts)
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


@pytest.mark.parametrize(
    ("kind", "plan_type", "class_name", "executor_name"),
    [
        (
            "delete_content",
            _FakeDeletionPlan,
            "AssistantStagedDeletionPlan",
            "execute_staged_deletion",
        ),
        (
            "reidentify_staged_card",
            _FakeReidentificationPlan,
            "AssistantReidentificationPlan",
            "execute_reidentification",
        ),
        (
            "approve_coverage",
            _FakeCoveragePlan,
            "AssistantCoverageApprovalPlan",
            "execute_coverage_approval",
        ),
    ],
)
def test_each_staging_owner_executor_independently_rechecks_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    plan_type: type[Any],
    class_name: str,
    executor_name: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = plan_type()
    instruction = "Apply this exact staged owner action."
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        class_name,
        plan_type,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_actions,
        executor_name,
        lambda *_args, **_kwargs: pytest.fail(
            "a mismatched staging-owner fingerprint must not execute"
        ),
    )

    with pytest.raises(RevisionRefusal, match="no longer matches"):
        adapter._consume_agent_action(
            _confirmation(plan, instruction, fingerprint="0" * 64),
            _prepared(kind, plan, instruction),
            progress=lambda _label: None,
        )


@pytest.mark.parametrize(
    ("kind", "plan_type"),
    [
        ("delete_content", _FakeDeletionPlan),
        ("reidentify_staged_card", _FakeReidentificationPlan),
        ("approve_coverage", _FakeCoveragePlan),
    ],
)
def test_each_staging_owner_confirmation_binds_exact_rendered_target(
    tmp_path: Path,
    kind: str,
    plan_type: type[Any],
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = plan_type()
    instruction = "Apply this exact staged owner action."
    adapter._agent_plans[plan.fingerprint] = _prepared(kind, plan, instruction)
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target="data/staging/another.pdf.yaml",
    )

    with pytest.raises(RevisionRefusal, match="different focus or target"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
