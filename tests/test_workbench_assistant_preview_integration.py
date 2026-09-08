"""Previews drawn by the real renderer from the real services' own proposals.

These are the end-to-end cases: an actual staged proposal, an actual finish
plan, the actual renderer, and the actual Assistant reply or review the owner
sees. They exist because a recording double proves the arguments Janki passes
and nothing about whether the cards it claims to draw are the *proposed* ones.

No paid provider is reachable from here. Audio providers are recording fakes,
and a preview dispatches none of them.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_application_ai_enrichment_finish import SELECTED as _ENRICH_SELECTED
from test_application_ai_enrichment_finish import _staged_enrichment
from test_application_assistant_staging_review import _project
from test_application_audio import Provider
from test_application_card_revision_finish import SELECTED as _REVISE_SELECTED
from test_application_card_revision_finish import RecordingProvider, _staged_revision
from test_application_revision_finish import _fixture, _RealtimeTransport

from japanese_anki import staging
from japanese_anki.application import card_revision_finish, revision_finish
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.card_preview import preview_unavailable
from japanese_anki.config import ProjectConfig
from japanese_anki.tts import openai_realtime
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.assistant import (
    _confirmation_body,
    _finish_review_body,
    _validate_finish_review,
)

_PREFIX = "http://127.0.0.1:9931/session/previews/"

pytestmark = pytest.mark.skipif(
    preview_unavailable() is not None,
    reason=preview_unavailable() or "",
)


def _adapter(config: ProjectConfig) -> assistant_adapter.RevisionAssistantAdapter:
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )
    adapter.bind_preview_links(_PREFIX)
    return adapter


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rendered(adapter: Any, url: str) -> str:
    store = adapter._preview_store
    assert store is not None
    return store.read(url.rsplit("/", 1)[-1]).html.decode("utf-8")


def _proposal_resource(config: ProjectConfig) -> str:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)
    proposals = [
        item for item in catalog["data"]["resources"] if item["kind"] == "proposal"
    ]
    assert len(proposals) == 1
    return str(proposals[0]["resource_id"])


def _intent(kind: str, **values: Any) -> Any:
    options = values.pop("options", None)
    return assistant_adapter.assistant_agent.AgentActionIntent(
        kind=kind,
        resource_ids=tuple(values.pop("resource_ids", ())),
        record_ids=tuple(values.pop("record_ids", ())),
        instruction=values.pop("instruction", "Review these cards."),
        options_json=json.dumps(options or {}, sort_keys=True, separators=(",", ":")),
    )


def _reply(
    adapter: assistant_adapter.RevisionAssistantAdapter,
    config: ProjectConfig,
    intent: Any,
) -> Any:
    return adapter._prepare_agent_intent(
        config,
        result=SimpleNamespace(answer="Here is that review.", action_intents=(intent,)),
        deck_scope="",
    )


def _widget_text(action: Any) -> str:
    return repr(
        _confirmation_body(
            title="Confirm this exact action",
            target=action.target,
            effects=action.effects,
            disclosures=action.disclosures,
            fingerprint_label="Request fingerprint",
            fingerprint=action.request_fingerprint,
            one_use_note="one use",
            preview_url=action.preview_url,
            preview_label=action.preview_label,
        )
    )


def _assigned_project(
    tmp_path: Path,
) -> tuple[ProjectConfig, Path]:
    config, proposal, records, _patterns = _project(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    _staged, metadata = staging.read_staging_text(
        proposal.read_text(encoding="utf-8"), source=str(proposal)
    )
    staging.write_staging(
        proposal,
        [replace(record, tags=["lesson"]) for record in records],
        metadata,
        force=True,
    )
    (config.deck_dir / "lesson.yaml").write_text(
        json.dumps(
            {
                "deck": {
                    "name": "Lesson",
                    "source": os.path.relpath(config.normalized_file, config.deck_dir),
                    "include_tags": ["lesson"],
                    "intake_tag": "lesson",
                }
            }
        ),
        encoding="utf-8",
    )
    return config, proposal


def test_an_assigned_source_extraction_review_previews_its_proposed_cards(
    tmp_path: Path,
) -> None:
    """The selected staged rows are drawn in the deck that would take them."""

    config, _proposal = _assigned_project(tmp_path)
    adapter = _adapter(config)
    before = _tree(tmp_path)

    reply = _reply(
        adapter,
        config,
        _intent(
            "review_staging",
            resource_ids=(_proposal_resource(config),),
            record_ids=("word:食べる:たべる",),
            options={"review_patterns": False},
        ),
    )

    action = reply.action
    assert action is not None
    assert action.preview_url is not None, action.disclosures
    assert action.preview_url.startswith(_PREFIX)
    assert action.preview_url in _widget_text(action)

    # The proposal, not the collection: the canonical store is still empty.
    html = _rendered(adapter, action.preview_url)
    assert "食べる" in html
    snapshot = adapter._preview_store.read(action.preview_url.rsplit("/", 1)[-1])
    assert snapshot.card_count == 2
    assert config.normalized_file.read_text(encoding="utf-8").strip() == "[]"
    assert _tree(tmp_path) == before


def test_a_source_preview_refuses_proposal_bytes_changed_after_review(
    tmp_path: Path,
) -> None:
    config, proposal = _assigned_project(tmp_path)
    adapter = _adapter(config)
    reply = _reply(
        adapter,
        config,
        _intent(
            "review_staging",
            resource_ids=(_proposal_resource(config),),
            record_ids=("word:食べる:たべる",),
            options={"review_patterns": False},
        ),
    )
    action = reply.action
    assert action is not None and action.preview_url is not None
    original_html = _rendered(adapter, action.preview_url)
    plan = adapter._agent_plans[action.request_fingerprint].plan
    proposal.write_bytes(proposal.read_bytes() + b"\n")
    before = _tree(tmp_path)

    offer, problem = adapter._offer_staging_review_preview(config, plan)

    assert offer is None
    assert problem is not None and "changed after this review" in problem
    assert _rendered(adapter, action.preview_url) == original_html
    assert _tree(tmp_path) == before


def test_an_unassigned_source_extraction_review_names_the_missing_destination(
    tmp_path: Path,
) -> None:
    config, _staging, records, _patterns = _project(tmp_path)
    adapter = _adapter(config)
    before = _tree(tmp_path)

    reply = _reply(
        adapter,
        config,
        _intent(
            "review_staging",
            resource_ids=(_proposal_resource(config),),
            record_ids=(records[0].id,),
            options={"review_patterns": False},
        ),
    )

    action = reply.action
    assert action is not None
    # The review stays complete and usable; it just says what is missing.
    assert action.preview_url is None
    assert action.effects
    assert any(
        "no configured word deck" in disclosure for disclosure in action.disclosures
    )
    assert _tree(tmp_path) == before


def _seed_missing_media(config: ProjectConfig) -> None:
    """Write the clip the revision fixture deliberately leaves absent.

    A fictional silent WAV: the point is that the packaging step has bytes to
    carry, not that anything was voiced. No audio provider is called.
    """

    target = config.media_dir / "audio" / "old.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )


def _conjugation_plan(config: ProjectConfig, staging: Path) -> Any:
    """Plan the finish with fake providers; planning dispatches neither."""

    transport = _RealtimeTransport()
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=Provider("voicevox", 7),
        sentence_provider=openai_realtime.OpenAiRealtimePool(
            api_key="test-key",
            transport=transport,
            operations_path=config.operations_file,
        ),
    )
    assert transport.calls == []
    return plan


def test_a_conjugation_revision_finish_review_previews_the_intended_deck(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _seed_missing_media(config)
    adapter = _adapter(config)
    plan = _conjugation_plan(config, staging)
    # The proposal's own teaching note: present in the intended deck text and
    # absent from the deck as it stands.
    proposed = plan.revision.form_note
    assert proposed and proposed != plan.revision.current_form_note
    assert proposed in plan.revision.intended_deck_text
    assert proposed not in deck.read_text(encoding="utf-8")
    before = _tree(tmp_path)

    review = adapter._finish_review(config, plan, preparation_id="p" * 43)

    assert review.preview_unavailable_message is None
    assert review.preview_url is not None
    _validate_finish_review(review)
    assert review.preview_url in repr(_finish_review_body(review))
    assert proposed in _rendered(adapter, review.preview_url)
    assert _tree(tmp_path) == before


def test_a_revision_finish_review_names_why_it_could_not_draw_the_cards(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    adapter = _adapter(config)
    plan = _conjugation_plan(config, staging)

    review = adapter._finish_review(config, plan, preparation_id="p" * 43)

    # The reason survives instead of being dropped, and the review still works.
    assert review.preview_url is None
    assert review.preview_unavailable_message is not None
    assert "audio/old.wav" in review.preview_unavailable_message
    _validate_finish_review(review)
    assert review.preview_unavailable_message in repr(_finish_review_body(review))
    assert review.card_count > 0


def _changed_value(config: ProjectConfig, projected_text: str) -> str:
    """One exact string the proposal changes, taken from the projection."""

    canonical = {
        record["id"]: record
        for record in json.loads(config.normalized_file.read_text(encoding="utf-8"))
    }
    for record in json.loads(projected_text):
        before = canonical.get(record["id"], {})
        for key, value in record.items():
            if before.get(key) == value:
                continue
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item
    raise AssertionError("the projection changes no renderable text")


@pytest.mark.parametrize("kind", ["ai_enrichment", "card_revision"])
def test_a_reviewed_card_finish_previews_the_proposed_fields(
    tmp_path: Path,
    kind: str,
) -> None:
    provider = RecordingProvider("voicevox", 7)
    if kind == "ai_enrichment":
        config, _deck, resource_id = _staged_enrichment(tmp_path)
        plan = card_revision_finish.plan_ai_enrichment_finish(
            config,
            resource_id,
            "Fill the gaps.",
            record_ids=[_ENRICH_SELECTED],
            word_provider=provider,
            sentence_provider=provider,
        )
    else:
        config, _deck, resource_id = _staged_revision(tmp_path)
        plan = card_revision_finish.plan_card_revision_finish(
            config,
            resource_id,
            "Make the reviewed change.",
            record_ids=[_REVISE_SELECTED],
            word_provider=provider,
            sentence_provider=provider,
        )
    adapter = _adapter(config)
    proposed = _changed_value(config, plan.projected_canonical_text)
    before = _tree(tmp_path)

    offer, problem = adapter._offer_card_finish_preview(config, plan)

    assert problem is None
    assert offer is not None
    assert proposed in _rendered(adapter, offer.url)
    assert _tree(tmp_path) == before
    assert provider.said == []
