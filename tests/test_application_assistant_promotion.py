"""Typed Assistant promotion over reviewed opaque proposal resources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import staging
from japanese_anki.application import assistant_promotion
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    ProposalContext,
)
from japanese_anki.application.journey import (
    EXAMPLES_NEED_REVIEW,
    READY_TO_ADD,
    SourceJourney,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.models import VocabularyRecord


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[project]\nname = \"Assistant promotion fixture\"\n"
        "[paths]\n"
        "scan_inbox = \"inbox\"\n"
        "staging_dir = \"staging\"\n"
        "normalized_file = \"normalized.json\"\n"
        "deck_dir = \"decks\"\n"
        "ledger_file = \"ledger.json\"\n"
        "patterns_file = \"patterns.json\"\n",
        encoding="utf-8",
    )
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox" / "lesson.pdf").write_bytes(b"preserved source")
    (tmp_path / "staging").mkdir()
    (tmp_path / "staging" / "lesson.pdf.yaml").write_text(
        "source_file: lesson.pdf\nrecords: []\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _ready_landing_project(tmp_path: Path) -> ProjectConfig:
    config = _project(tmp_path)
    config.normalized_file.write_text("[]\n", encoding="utf-8")
    config.deck_dir.mkdir()
    (config.deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  source: ../normalized.json\n"
        "  include_tags: [lesson]\n"
        "  intake_tag: lesson\n",
        encoding="utf-8",
    )
    (config.staging_dir / "lesson.pdf.yaml").write_text(
        json.dumps(
            {
                "source_file": "lesson.pdf",
                "records": [
                    {
                        "id": "word:食べる:たべる",
                        "expression": "食べる",
                        "reading": "たべる",
                        "meanings": ["to eat"],
                        "tags": ["lesson"],
                        "source": {"type": "manual"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config


def _ready_ai_enrichment_project(tmp_path: Path) -> ProjectConfig:
    config = _project(tmp_path)
    original = VocabularyRecord.from_dict(
        {
            "id": "word:話す:はなす",
            "expression": "話す",
            "reading": "はなす",
            "meanings": ["to speak"],
            "tags": ["lesson"],
            "source": {"type": "manual"},
        }
    )
    proposal = replace(original, meanings=["to speak; to converse"])
    config.normalized_file.write_text(
        json.dumps([original.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    config.deck_dir.mkdir()
    (config.deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  source: ../normalized.json\n"
        "  include_tags: [lesson]\n"
        "  intake_tag: lesson\n",
        encoding="utf-8",
    )
    (config.staging_dir / "lesson.pdf.yaml").unlink()
    changes = {
        original.id: {
            "meanings": (list(original.meanings), list(proposal.meanings))
        }
    }
    staging.write_staging(
        config.staging_dir / "ai-enrichment.yaml",
        [proposal],
        {
            "source_file": config.normalized_file.name,
            "model": "claude-opus-5",
            "provider": "anthropic",
            "ai_enrichment": {
                "version": 1,
                "model": "claude-opus-5",
                "provider": "anthropic",
                "request_fingerprints": {original.id: "a" * 64},
                "input_fingerprints": {original.id: "b" * 64},
                "fields": {original.id: ["meanings"]},
            },
            "field_replacements": staging.field_replacement_block(
                [original], changes
            ),
        },
    )
    return config


def _catalog(config: ProjectConfig) -> list[dict[str, object]]:
    return json.loads(AssistantContextBroker(config).catalog().wire)["data"][
        "resources"
    ]


def _resource_id(config: ProjectConfig, kind: str) -> str:
    return next(
        str(item["resource_id"])
        for item in _catalog(config)
        if item["kind"] == kind
    )


def test_local_resolvers_return_only_current_opaque_proposal_and_source_targets(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    source_id = _resource_id(config, "source")
    broker = AssistantContextBroker(config)

    proposal = broker.proposal_context(proposal_id)
    source = broker.source_path(source_id)

    assert proposal == ProposalContext(
        resource_id=proposal_id,
        proposal_kind="source_extraction",
        path=config.staging_dir / "lesson.pdf.yaml",
        proposal_sha256=hashlib.sha256(
            (config.staging_dir / "lesson.pdf.yaml").read_bytes()
        ).hexdigest(),
    )
    assert source == config.scan_inbox / "lesson.pdf"
    with pytest.raises(AssistantContextError, match="proposal resource"):
        broker.proposal_context(source_id)
    with pytest.raises(AssistantContextError, match="source resource"):
        broker.source_path(proposal_id)


def test_proposal_resolver_rechecks_the_regular_file_after_catalog_creation(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    broker = AssistantContextBroker(config)
    proposal_id = _resource_id(config, "proposal")
    outside_proposal = tmp_path.parent / f"{tmp_path.name}-outside-proposal.yaml"
    outside_proposal.write_text("source_file: secret\nrecords: []\n", encoding="utf-8")
    proposal_path = config.staging_dir / "lesson.pdf.yaml"
    proposal_path.unlink()
    proposal_path.symlink_to(outside_proposal)

    with pytest.raises(AssistantContextError, match="symlink"):
        broker.proposal_context(proposal_id)


def test_source_resolver_rechecks_the_regular_file_after_catalog_creation(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    broker = AssistantContextBroker(config)
    source_id = _resource_id(config, "source")
    outside_source = tmp_path.parent / f"{tmp_path.name}-outside-source.pdf"
    outside_source.write_bytes(b"secret")
    source_path = config.scan_inbox / "lesson.pdf"
    source_path.unlink()
    source_path.symlink_to(outside_source)

    with pytest.raises(AssistantContextError, match="symlink"):
        broker.source_path(source_id)


def test_source_resolver_refuses_an_inbox_outside_the_repository(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-inbox"
    outside.mkdir()
    (outside / "lesson.pdf").write_bytes(b"outside")
    (tmp_path / "janki.toml").write_text(
        "[paths]\nscan_inbox = \"../"
        + outside.name
        + "\"\nstaging_dir = \"staging\"\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    source_id = _resource_id(config, "source")

    with pytest.raises(AssistantContextError, match="escapes the project"):
        AssistantContextBroker(config).source_path(source_id)


def test_plan_is_canonical_and_binds_instruction_review_and_service(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")

    plan = assistant_promotion.plan_promotion_action(
        config,
        proposal_id,
        "Promote this reviewed proposal.",
    )
    changed_instruction = assistant_promotion.plan_promotion_action(
        config,
        proposal_id,
        "Archive this reviewed empty proposal.",
    )

    projection = plan.projection
    assert plan.proposal_kind == "source_extraction"
    assert plan.decision.state == "nothing"
    assert projection["kind"] == "promote_staging"
    assert projection["instruction"] == "Promote this reviewed proposal."
    target = dict(projection["target"])
    assert target == {
        "proposal": "staging/lesson.pdf.yaml",
        "proposal_kind": "source_extraction",
        "proposal_sha256": hashlib.sha256(plan.proposal_path.read_bytes()).hexdigest(),
        "resource_id": proposal_id,
        "source_name": "lesson.pdf",
    }
    assert projection["authority"] == {
        "coverage_acceptance_requested": False,
        "owner_decisions_remaining_before_reading_check": [],
        "reading_conflicts_are_never_resolved_by_the_assistant": True,
        "review_and_deck_choices_are_already_durable": True,
    }
    assert projection["service_fingerprint"] == plan.service_fingerprint
    assert projection["decision"]["readings_checked_at_execution"] is False
    assert projection["decision"]["reading_conflicts_stay_in_the_live_review"] is True
    assert plan.projection_wire == json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert changed_instruction.service_fingerprint == plan.service_fingerprint
    assert changed_instruction.fingerprint != plan.fingerprint


def test_ready_landing_projection_names_the_exact_card_owner_and_writes(
    tmp_path: Path,
) -> None:
    config = _ready_landing_project(tmp_path)
    proposal_id = _resource_id(config, "proposal")

    plan = assistant_promotion.plan_promotion_action(
        config, proposal_id, "Add this reviewed card."
    )

    decision = plan.projection["decision"]
    assert decision["state"] == "lands"
    assert decision["readings_checked_at_execution"] is True
    [landing] = decision["landing"]
    assert {
        key: landing[key]
        for key in (
            "current",
            "keeps_existing_meanings",
            "landing_record_id",
            "proposed_record_id",
            "reminted_from",
            "result",
        )
    } == {
        "current": None,
        "keeps_existing_meanings": False,
        "landing_record_id": "word:食べる:たべる",
        "proposed_record_id": "word:食べる:たべる",
        "reminted_from": None,
        "result": "add",
    }
    assert landing["proposed"] == landing["landing"]
    assert landing["landing"]["expression"] == "食べる"
    assert landing["landing"]["reading"] == "たべる"
    assert landing["landing"]["meanings"] == ["to eat"]
    assert landing["landing"]["source"] == {
        "name": "",
        "row": None,
        "type": "manual",
    }
    assert decision["deck_ownership"] == [
        {
            "decks": [
                {"name": "Lesson", "refusal": None, "selected": True}
            ],
            "record_id": "word:食べる:たべる",
            "state": "exactly_one",
        }
    ]
    assert plan.projection["writes"] == {
        "archive": "staging/done/lesson.pdf.yaml",
        "collection": "normalized.json",
        "ledger": "ledger.json",
        "live_review": "staging/lesson.pdf.yaml",
    }


def test_plan_routes_reviewed_revision_proposals_through_shared_promotion_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    target = ProposalContext(
        resource_id="resource_revision",
        proposal_kind="card_revision",
        path=config.staging_dir / "lesson.pdf.yaml",
    )
    monkeypatch.setattr(assistant_promotion, "_resolve_proposal", lambda *_: target)
    called = False

    def deciding(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise assistant_promotion.AssistantPromotionError("shared decision called")

    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        deciding,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="shared decision called",
    ):
        assistant_promotion.plan_promotion_action(
            config, target.resource_id, "Apply the revision."
        )
    assert called


def test_plan_refuses_unreviewed_ai_enrichment_in_favor_of_apply_and_finish(
    tmp_path: Path,
) -> None:
    config = _ready_ai_enrichment_project(tmp_path)
    proposal_id = _resource_id(config, "proposal")

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="Apply and finish.*exact owner review",
    ):
        assistant_promotion.plan_promotion_action(
            config,
            proposal_id,
            "Promote the enrichment proposal.",
        )


def test_plan_accepts_exactly_reviewed_ai_enrichment(
    tmp_path: Path,
) -> None:
    config = _ready_ai_enrichment_project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    proposal = config.staging_dir / "ai-enrichment.yaml"
    proposed, _meta = staging.read_staging(proposal)
    staging.record_ai_enrichment_review(
        proposal,
        [record.id for record in proposed],
        expected_revision=hashlib.sha256(proposal.read_bytes()).hexdigest(),
    )

    plan = assistant_promotion.plan_promotion_action(
        config,
        proposal_id,
        "Promote the reviewed enrichment proposal.",
    )

    assert plan.proposal_kind == "ai_enrichment"
    assert plan.projection["target"]["proposal_kind"] == "ai_enrichment"
    [change] = plan.projection["decision"]["landing"]
    assert change["result"] == "merge"
    assert change["current"]["meanings"] == ["to speak"]
    assert change["proposed"]["meanings"] == ["to speak; to converse"]
    assert change["landing"]["meanings"] == ["to speak; to converse"]


def test_plan_refuses_a_proposal_still_waiting_for_an_owner_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    waiting = SourceJourney(
        source="lesson.pdf",
        state=EXAMPLES_NEED_REVIEW,
        next_action="Review the Japanese examples on 1 card",
        staging_path=config.staging_dir / "lesson.pdf.yaml",
        example_review_count=1,
    )
    monkeypatch.setattr(
        assistant_promotion,
        "source_journeys",
        lambda _config: ([waiting], []),
    )
    called = False

    def deciding(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("an undecided review must not be planned")

    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        deciding,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="Examples need review.*will not supply that owner decision",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )
    assert called is False


def test_plan_refuses_a_malformed_example_authority_that_journey_calls_ready(
    tmp_path: Path,
) -> None:
    config = _ready_landing_project(tmp_path)
    proposal = config.staging_dir / "lesson.pdf.yaml"
    value = json.loads(proposal.read_text(encoding="utf-8"))
    [record] = value["records"]
    record["examples"] = [
        {
            "japanese": "パンが食べられます。",
            "furigana": "パンが食[た]べられます。",
            "romaji": "pan ga taberaremasu.",
            "english": "I can eat bread.",
            "register": "polite",
            "audio": "",
        }
    ]
    record["source"] = {
        "type": "extract",
        "imported_from": "lesson.pdf",
        "raw_fields": {"example_authority": "reviewd"},
    }
    proposal.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    proposal_id = _resource_id(config, "proposal")

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="exact durable example review authority.*invalid.*will not infer approval",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_plan_refuses_a_proposal_swapped_after_its_bound_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    proposal = config.staging_dir / "lesson.pdf.yaml"
    alternate = tmp_path / "protected-proposal.yaml"
    alternate.write_text(
        "source_file: lesson.pdf\nreview_notes: private alternate\nrecords: []\n",
        encoding="utf-8",
    )
    original = assistant_promotion.promotion_application.decide_promotion

    def swapped(*args: object, **kwargs: object) -> object:
        proposal.unlink()
        proposal.symlink_to(alternate)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        swapped,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="proposal path changed after its safe resolution",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_plan_refuses_regular_proposal_bytes_changed_after_bound_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    proposal = config.staging_dir / "lesson.pdf.yaml"
    original = assistant_promotion.promotion_application.decide_promotion

    def changed(*args: object, **kwargs: object) -> object:
        proposal.write_text(
            "source_file: lesson.pdf\nreview_notes: changed bytes\nrecords: []\n",
            encoding="utf-8",
        )
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        changed,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="proposal bytes changed after their safe resolution",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_plan_never_treats_a_path_shaped_source_as_repository_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    current = SourceJourney(
        source="../private/lesson.pdf",
        state=READY_TO_ADD,
        next_action="Add 1 card to your collection",
        staging_path=config.staging_dir / "lesson.pdf.yaml",
    )
    monkeypatch.setattr(
        assistant_promotion,
        "source_journeys",
        lambda _config: ([current], []),
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="not a basename-shaped repository identity",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_plan_refuses_a_structural_or_coverage_gate_even_if_journey_looks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    original = assistant_promotion.promotion_application.decide_promotion
    blocked = original(
        config,
        config.staging_dir / "lesson.pdf.yaml",
        source="lesson.pdf",
        skip_reading_check=None,
    )
    blocked = replace(
        blocked,
        state="blocked",
        gate="coverage",
        error=assistant_promotion.AssistantPromotionError(
            "coverage still needs the owner's decision"
        ),
    )
    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        lambda *args, **kwargs: blocked,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="not ready for ordinary promotion.*no coverage.*was inferred",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_plan_binds_the_fresh_decision_to_the_same_source_journey(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    original_decide = assistant_promotion.promotion_application.decide_promotion
    decision = original_decide(
        config,
        config.staging_dir / "lesson.pdf.yaml",
        source="lesson.pdf",
        skip_reading_check=None,
    )
    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "decide_promotion",
        lambda *args, **kwargs: replace(
            decision, meta={**decision.meta, "source_file": "other.pdf"}
        ),
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="source changed while its promotion was being planned",
    ):
        assistant_promotion.plan_promotion_action(
            config, proposal_id, "Promote this proposal."
        )


def test_execution_replans_compares_then_calls_the_existing_resolver_and_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    expected = assistant_promotion.plan_promotion_action(
        config, proposal_id, "Promote this reviewed proposal."
    )
    original_resolve = assistant_promotion.promotion_action.resolve_promotion_for_execution
    original_execute = assistant_promotion.promotion_application.execute_promotion
    calls: list[object] = []

    def resolve(*args: object, **kwargs: object) -> object:
        calls.append(("resolve", args[1], kwargs["expected_preview_fingerprint"]))
        return original_resolve(*args, **kwargs)  # type: ignore[arg-type]

    def execute(*args: object, **kwargs: object) -> object:
        calls.append(("execute", args[1]))
        return original_execute(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        assistant_promotion.promotion_action,
        "resolve_promotion_for_execution",
        resolve,
    )
    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "execute_promotion",
        execute,
    )
    progress: list[str] = []
    execution = assistant_promotion.execute_promotion_action(
        config,
        expected,
        client_factory=lambda: (_ for _ in ()).throw(
            AssertionError("empty promotion must not construct jpdb")
        ),
        progress=progress.append,
    )

    assert execution.plan is not expected
    assert execution.plan.fingerprint == expected.fingerprint
    assert execution.result.state == "nothing"
    assert calls[0][0] == "resolve"
    assert calls[0][1] is execution.plan.decision
    assert calls[0][2] == expected.service_fingerprint
    assert calls[1][0] == "execute"
    assert progress == ["Checking the reviewed proposal", "Saving reviewed cards"]


def test_execution_refuses_a_changed_proposal_before_reading_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    expected = assistant_promotion.plan_promotion_action(
        config, proposal_id, "Promote this reviewed proposal."
    )
    with (config.staging_dir / "lesson.pdf.yaml").open("a", encoding="utf-8") as handle:
        handle.write("review_notes: changed after confirmation\n")
    resolved = False
    executed = False

    def resolve(*args: object, **kwargs: object) -> object:
        nonlocal resolved
        resolved = True
        return SimpleNamespace(is_blocked=False)

    def execute(*args: object, **kwargs: object) -> object:
        nonlocal executed
        executed = True
        return SimpleNamespace(state="nothing")

    monkeypatch.setattr(
        assistant_promotion.promotion_action,
        "resolve_promotion_for_execution",
        resolve,
    )
    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "execute_promotion",
        execute,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="changed after it was displayed",
    ):
        assistant_promotion.execute_promotion_action(config, expected)
    assert resolved is False
    assert executed is False


def test_execution_refuses_a_fresh_post_reading_gate_before_the_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    expected = assistant_promotion.plan_promotion_action(
        config, proposal_id, "Promote this reviewed proposal."
    )
    executed = False
    monkeypatch.setattr(
        assistant_promotion.promotion_action,
        "resolve_promotion_for_execution",
        lambda *args, **kwargs: SimpleNamespace(
            is_blocked=True,
            error=assistant_promotion.AssistantPromotionError(
                "deck ownership changed"
            ),
        ),
    )

    def execute(*args: object, **kwargs: object) -> object:
        nonlocal executed
        executed = True
        return SimpleNamespace(state="nothing")

    monkeypatch.setattr(
        assistant_promotion.promotion_application,
        "execute_promotion",
        execute,
    )

    with pytest.raises(
        assistant_promotion.AssistantPromotionError,
        match="no longer promotable.*Nothing was promoted",
    ):
        assistant_promotion.execute_promotion_action(config, expected)
    assert executed is False


def test_execution_uses_the_shared_writer_for_a_ready_landing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _ready_landing_project(tmp_path)
    proposal_id = _resource_id(config, "proposal")
    expected = assistant_promotion.plan_promotion_action(
        config, proposal_id, "Add this reviewed card."
    )
    resolver_calls = 0

    def resolve(
        config: ProjectConfig,
        decision: object,
        **kwargs: object,
    ) -> object:
        nonlocal resolver_calls
        resolver_calls += 1
        assert kwargs["expected_preview_fingerprint"] == expected.service_fingerprint
        return replace(decision, reading_check="consulted", consulted=True)

    monkeypatch.setattr(
        assistant_promotion.promotion_action,
        "resolve_promotion_for_execution",
        resolve,
    )
    progress: list[str] = []

    execution = assistant_promotion.execute_promotion_action(
        config,
        expected,
        progress=progress.append,
    )

    stored = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    assert execution.result.state == "landed"
    assert execution.result.promoted_ids == ("word:食べる:たべる",)
    assert stored[0]["expression"] == "食べる"
    assert not expected.proposal_path.exists()
    assert (config.staging_dir / "done" / expected.proposal_path.name).is_file()
    assert resolver_calls == 1
    assert progress == [
        "Checking the reviewed proposal",
        "Checking readings",
        "Saving reviewed cards",
    ]
