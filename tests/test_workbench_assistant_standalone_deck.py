"""The Assistant's shared/standalone choice when it creates a vocabulary deck.

The fixtures come from ``test_workbench_assistant_integration``: they build a
configured project, a deck-less adapter and the fake typed agent result the
broker validates.  Nothing here reaches a model — ``_agent_result`` *is* the
answer, so every assertion is about janki's own local gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_workbench_assistant_integration import (
    _adapter,
    _agent_intent,
    _agent_result,
    _config,
)

from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter

#: The owner named the deck and its direction here, and nothing else. A
#: structured standalone choice carried from an earlier turn must survive this
#: message without being restated in it.
OWNER_MESSAGE = "Create Weekly Review with recognition."


def _real_project(tmp_path: Path) -> ProjectConfig:
    """A configured project whose empty canonical word collection really exists."""
    config = _config(tmp_path)
    config.normalized_file.parent.mkdir(parents=True, exist_ok=True)
    config.normalized_file.write_text("[]\n", encoding="utf-8")
    return config


def _create_deck_intent(**options: Any) -> Any:
    return _agent_intent(
        kind="create_deck",
        resource_ids=(),
        record_ids=(),
        instruction="Create a deck for this week's review.",
        options={
            "deck_name": "Weekly Review",
            "card_directions": ["recognition"],
            **options,
        },
    )


def test_a_structured_standalone_choice_needs_no_literal_in_the_current_message(
    tmp_path: Path,
) -> None:
    config = _real_project(tmp_path)
    adapter = _adapter(config)
    assert "standalone" not in OWNER_MESSAGE.casefold()

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(deck_scope="standalone"),)),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )

    assert reply.action is not None
    assert len(adapter._agent_plans) == 1
    prepared = adapter._agent_plans[reply.action.request_fingerprint]
    assert prepared.plan.request.deck_scope == "standalone"
    scope_id = prepared.plan.service_plan.scope_id
    assert len(scope_id) == 64
    assert reply.action.effects == (
        "Create the study deck 'Weekly Review'",
        "Enable card directions: recognition",
        "Standalone: independent copies and separate review progress",
    )
    written = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "This local action makes no model or audio-provider call." in written
    # The scope is machine bookkeeping. It is bound in the plan the confirmation
    # consumes, and it has no business in what the learner reads.
    assert scope_id not in written
    assert scope_id in prepared.plan.projection["definition"]["yaml"]
    assert prepared.plan.projection["target"]["scope_id"] == scope_id
    assert not list(config.deck_dir.glob("*.yaml"))


def test_an_omitted_scope_option_still_renders_the_unchanged_shared_confirmation(
    tmp_path: Path,
) -> None:
    config = _real_project(tmp_path)
    adapter = _adapter(config)

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(),)),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )

    assert reply.action is not None
    prepared = adapter._agent_plans[reply.action.request_fingerprint]
    assert prepared.plan.request.deck_scope == "shared"
    assert prepared.plan.service_plan.standalone is False
    assert prepared.plan.service_plan.scope_id == ""
    assert reply.action.effects == (
        "Create the study deck 'Weekly Review'",
        "Enable card directions: recognition",
        "Shared: it reuses your existing cards and their review progress",
    )
    written = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "standalone" not in written.lower()
    assert "scope_id" not in prepared.plan.projection["definition"]["yaml"]
    assert not list(config.deck_dir.glob("*.yaml"))


def test_an_explicit_shared_choice_is_the_same_confirmation(tmp_path: Path) -> None:
    config = _real_project(tmp_path)
    adapter = _adapter(config)

    chosen = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(deck_scope="shared"),)),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )
    omitted = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(),)),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )

    assert chosen.action is not None and omitted.action is not None
    assert chosen.action.effects == omitted.action.effects
    assert chosen.action.disclosures == omitted.action.disclosures
    assert chosen.action.request_fingerprint == omitted.action.request_fingerprint


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"deck_scope": "private"}, "'shared' or 'standalone'"),
        ({"deck_scope": True}, "'shared' or 'standalone'"),
        ({"deck_scope": ""}, "'shared' or 'standalone'"),
        ({"study_type": "vocabulary"}, "exactly one deck name"),
    ],
    ids=("unknown-scope", "nonstring-scope", "blank-scope", "unrelated-option"),
)
def test_create_deck_refuses_an_invalid_scope_or_extra_option_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    message: str,
) -> None:
    config = _real_project(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "plan_deck_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid deck scope must refuse before deck planning"
        ),
    )

    with pytest.raises(assistant_adapter.RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(_create_deck_intent(**options),)),
            deck_scope="",
            owner_message=OWNER_MESSAGE,
        )

    assert adapter._agent_plans == {}
    assert not list(config.deck_dir.glob("*.yaml"))


def test_agreeing_to_a_setup_discussed_earlier_plans_it_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The owner settled the name, directions and scope in an earlier turn.

    Making them retype those words in the message that accepts the setup is a
    second confirmation of something they already said. The typed intent carries
    the choices forward; the one plan-bound confirmation still owns the write.
    """
    config = _real_project(tmp_path)
    adapter = _adapter(config)

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(deck_scope="standalone"),)),
        deck_scope="",
        owner_message="yes, use that setup",
    )

    assert reply.action is not None
    assert reply.action.confirm_label == "Create this exact deck"
    prepared = adapter._agent_plans[reply.action.request_fingerprint]
    assert prepared.plan.request.name == "Weekly Review"
    assert prepared.plan.request.recognition is True
    assert prepared.plan.request.deck_scope == "standalone"
    assert reply.action.effects == (
        "Create the study deck 'Weekly Review'",
        "Enable card directions: recognition",
        "Standalone: independent copies and separate review progress",
    )
    assert not list(config.deck_dir.glob("*.yaml"))
