"""What a model may ask a study job for, and what only the owner may do.

Every provider here is faked. These tests are about janki's own gates:

- the action schema is closed, and the fields it deliberately does **not**
  have are the ones a model could otherwise use to mint an owner decision;
- a job snapshot discloses ids, hashes, states and counts, and no reply bytes,
  candidate content or source bytes;
- a new destination deck and its job are one owner action: the job write rides
  inside the existing deck-creation confirmation, after the deck exists, and a
  refused deck creation leaves no job behind;
- a job-opened region editor records its backlink before publishing, and the
  receipt it writes is still job-independent;
- assignment's path-keyed plan re-prepares by the key it planned with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_assistant_integration import (
    _adapter,
    _agent_intent,
    _agent_result,
    _config,
)
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import ai_schema
from japanese_anki.application import (
    assistant_context,
    extraction_batch,
    source_parts,
    study_job,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.source_part_editor import LocalSourcePartEditorStore


def _stream() -> bytes:
    """One faked subscription reply carrying a valid extraction answer."""

    answer = json.loads(
        (RESPONSES / "table_exhaustive.json").read_text(encoding="utf-8")
    )
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)

OWNER_MESSAGE = "Turn verbs.pdf into a new standalone verb deck."


def _project(tmp_path: Path) -> ProjectConfig:
    config = _config(tmp_path)
    config.normalized_file.parent.mkdir(parents=True, exist_ok=True)
    config.normalized_file.write_text("[]\n", encoding="utf-8")
    return config


def _source(config: ProjectConfig, name: str = "verbs.pdf") -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 verbs")
    return path


def _deck(config: ProjectConfig, stem: str = "201-verbs") -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": stem,
                    "deck_id": 1_500_000_001,
                    "source": "../normalized.json",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


# --- the closed model schema --------------------------------------------------


def _fields() -> tuple[dict[str, Any], dict[str, Any]]:
    schema = ai_schema.assistant_agent_schema().model_json_schema()
    definitions = schema["$defs"]
    return (
        definitions["AssistantActionIntent"],
        definitions["AssistantActionOptions"],
    )


def test_the_action_schema_gains_only_the_study_intents_s4_implements() -> None:
    intent, _options = _fields()
    kinds = set(intent["properties"]["kind"]["enum"])

    assert {
        "study_job_status",
        "inspect_capture_proposals",
        "extract_study_parts",
        "retry_study_parts",
        "stage_capture_proposal",
        "resume_study_job",
    } <= kinds
    # Owner-only actions and later milestones' editors are not emittable at
    # all: a kind that is absent from this closed Literal cannot be sent.
    assert not kinds & {
        "create_study_job",
        "record_study_choice",
        "open_layout_editor",
        "open_curation_editor",
        "open_review_editor",
        "open_disposition_editor",
        "curate_study_job",
        "review_study_job",
        "study_job_coverage",
        "study_job_disposition",
        "finish_study_job",
    }


def test_action_options_gain_retry_indices_and_no_owner_decision_field() -> None:
    _intent, options = _fields()
    names = set(options["properties"])

    assert "retry_child_indices" in names
    assert options["additionalProperties"] is False
    # Every component of the capture location tuple, the job's own owner
    # decisions and the audio control are absent by construction: anything on
    # this model is model-emittable.
    assert not names & {
        "capture_proposal_sha256",
        "capture_sha256",
        "response_schema_fingerprint",
        "frame_index",
        "block_index",
        "tool_use_id",
        "json_pointer",
        "proposal_sha256",
        "include_example_audio",
        "review_flags",
        "study_coverage_reason",
        "disposition",
    }
    # The separate `approve_coverage` relay keeps its existing owner-literal
    # field; this plan neither removes nor widens it.
    assert "coverage_reason" in names


def test_a_model_cannot_emit_a_capture_location_or_an_audio_opt_out() -> None:
    answer = ai_schema.assistant_agent_schema()
    with pytest.raises(ValidationError):
        answer.model_validate(
            {
                "answer": "ok",
                "action_intents": [
                    {
                        "kind": "stage_capture_proposal",
                        "resource_ids": ["resource_op"],
                        "record_ids": [],
                        "instruction": "Recover it.",
                        "options": {"json_pointer": "/content/0/input"},
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        answer.model_validate(
            {
                "answer": "ok",
                "action_intents": [
                    {
                        "kind": "extract_study_parts",
                        "resource_ids": ["resource_job"],
                        "record_ids": [],
                        "instruction": "Send them.",
                        "options": {"include_example_audio": False},
                    }
                ],
            }
        )


# --- what a job snapshot may disclose -----------------------------------------


def test_a_study_job_is_discoverable_and_its_snapshot_carries_no_bytes(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    source = _source(config)
    deck = _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=source,
        deck_path=deck,
    )

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    entry = next(
        item for item in catalog["resources"] if item["kind"] == "study_job"
    )
    assert entry["available"] is True
    assert entry["title"] == "verbs.pdf → 201-verbs.yaml"
    # The opaque id is what the model sees; the job id is never a path.
    assert entry["resource_id"].startswith("resource_")
    assert job.header.job_id not in entry["resource_id"]

    disclosure = broker.snapshot(entry["resource_id"])
    wire = json.loads(disclosure.wire)["data"]["study_job"]
    assert disclosure.kind == "study_job"
    assert wire["job_id"] == job.header.job_id
    assert wire["parent_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert wire["deck_current"] is True
    assert wire["parts"] == [] and wire["batches"] == []
    # Hashes, ids, states and counts only. The source's own bytes and the
    # deck's own YAML are absent, and so is any proposal content.
    text = disclosure.wire
    assert "%PDF" not in text
    assert "deck_id" not in text
    assert broker.study_job_id(entry["resource_id"]) == job.header.job_id


def test_an_unreadable_job_stays_listed_and_refuses_only_when_selected(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    _source(config)
    _deck(config)
    good = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    broken_id = "22222222-2222-4222-8222-222222222222"
    study_job.study_job_path(config, broken_id).write_text("{", encoding="utf-8")

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    jobs = {
        item["title"]: item
        for item in catalog["resources"]
        if item["kind"] == "study_job"
    }
    assert len(jobs) == 2
    broken = next(item for item in jobs.values() if item["available"] is False)
    with pytest.raises(assistant_context.AssistantContextError):
        broker.study_job_id(broken["resource_id"])
    # Its sibling is unaffected.
    assert broker.study_job_id(
        next(
            item["resource_id"]
            for item in jobs.values()
            if item["available"] is True
        )
    ) == good.header.job_id


# --- one owner action: the new deck and its job -------------------------------


def _create_deck_intent(*, resource_ids: tuple[str, ...] = ()) -> Any:
    return _agent_intent(
        kind="create_deck",
        resource_ids=resource_ids,
        record_ids=(),
        instruction="Create a standalone deck for these verbs.",
        options={
            "deck_name": "201 Verbs",
            "card_directions": ["recognition", "production"],
            "deck_scope": "standalone",
        },
    )


def _source_resource_id(config: ProjectConfig, name: str) -> str:
    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    return next(
        item["resource_id"]
        for item in catalog["resources"]
        if item["kind"] == "source" and item["title"] == name
    )


def _confirm(adapter: Any, reply: Any, *, deck_scope: str = "") -> Any:
    confirmation = assistant_adapter.RevisionConfirmation(
        capability="capability-token",
        deck_scope=deck_scope,
        instruction=reply.action_instruction,
        target=reply.action.target,
        expected_fingerprint=reply.action.request_fingerprint,
    )
    return adapter.consume_replan_and_execute(
        confirmation, progress=lambda _label: None
    )


def test_one_confirmation_creates_the_deck_and_opens_the_job_over_it(
    tmp_path: Path,
) -> None:
    """§1 step 3: the local job write rides inside the deck confirmation."""

    config = _project(tmp_path)
    source = _source(config)
    adapter = _adapter(config)
    resource_id = _source_resource_id(config, "verbs.pdf")

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            intents=(_create_deck_intent(resource_ids=(resource_id,)),)
        ),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )
    assert reply.action is not None
    assert any("Open a study job over verbs.pdf" in effect for effect in reply.action.effects)
    assert study_job.list_study_jobs(config) == (), "planning writes nothing"

    execution = _confirm(adapter, reply)

    job_ids = study_job.list_study_jobs(config)
    assert len(job_ids) == 1
    job = study_job.load_study_job(config, job_ids[0])
    deck = config.root / job.header.deck_path
    assert deck.is_file()
    assert job.header.deck_sha256 == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert job.header.parent_source_name == "verbs.pdf"
    assert job.header.parent_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert job.layouts == {} and job.intents == () and job.outcomes == ()
    assert job.header.job_id in execution.message

    status = study_job.study_job_status(config, job.header.job_id)
    assert status.batches == () and status.parts == ()

    # One-use: replaying the same confirmation creates no second job.
    with pytest.raises(assistant_adapter.RevisionRefusal):
        _confirm(adapter, reply)
    assert study_job.list_study_jobs(config) == job_ids


def test_a_refused_deck_creation_leaves_no_job_document(tmp_path: Path) -> None:
    """No orphan binding a deck that does not exist."""

    config = _project(tmp_path)
    _source(config)
    adapter = _adapter(config)
    resource_id = _source_resource_id(config, "verbs.pdf")

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            intents=(_create_deck_intent(resource_ids=(resource_id,)),)
        ),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )
    assert reply.action is not None
    prepared = adapter._agent_plans[reply.action.request_fingerprint]
    # Somebody else wrote that exact deck file first, so the service refuses.
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    prepared.plan.service_plan.path.write_text("deck: {}\n", encoding="utf-8")

    with pytest.raises(assistant_adapter.RevisionRefusal):
        _confirm(adapter, reply)
    assert study_job.list_study_jobs(config) == ()


def test_a_created_deck_whose_job_write_failed_names_a_reachable_recovery(
    tmp_path: Path,
) -> None:
    """The deck is real, so the way back to it has to be one the owner has.

    A workbench computes its deck choices once, when it starts, so a deck
    created seconds ago by this very action cannot be focused in this session —
    "focus that deck and say Manage study jobs" is not a route yet. The exact
    `janki study new` command is, and the Assistant one is after a restart.
    """

    config = _project(tmp_path)
    source = _source(config)
    adapter = _adapter(config)
    resource_id = _source_resource_id(config, "verbs.pdf")

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            intents=(_create_deck_intent(resource_ids=(resource_id,)),)
        ),
        deck_scope="",
        owner_message=OWNER_MESSAGE,
    )
    assert reply.action is not None
    stem = adapter._agent_plans[reply.action.request_fingerprint].plan.service_plan.stem
    # The real failure this branch is written for: the source is gone between
    # the plan and the execution, so the deck lands and the job cannot open.
    source.unlink()

    with pytest.raises(assistant_adapter.RevisionRefusal) as refused:
        _confirm(adapter, reply)

    told = str(refused.value)
    assert (config.deck_dir / f"{stem}.yaml").is_file(), "the deck really was created"
    assert study_job.list_study_jobs(config) == ()
    assert f"janki study new --source verbs.pdf --deck {stem}" in told
    # The Assistant route is named as what it is: available after a restart.
    assert "restart the workbench" in told
    assert told.index("restart the workbench") < told.index("Manage study jobs")


def test_a_plain_deck_creation_still_opens_no_job(tmp_path: Path) -> None:
    """The existing action is unchanged when no source is named."""

    config = _project(tmp_path)
    adapter = _adapter(config)

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_create_deck_intent(),)),
        deck_scope="",
        owner_message="Create 201 Verbs with recognition and production.",
    )
    assert reply.action is not None
    assert all("study job" not in effect for effect in reply.action.effects)

    execution = _confirm(adapter, reply)
    assert "ready to receive explicitly assigned cards" in execution.message
    assert study_job.list_study_jobs(config) == ()


# --- the job-opened region editor ---------------------------------------------


def _sheet() -> Any:
    """One real contact sheet value, with a page the editor can draw."""

    return source_parts.ContactSheet(
        parent_name="verbs.pdf",
        parent_sha256="9f2c",
        render_dpi=110,
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        pages=(
            source_parts.ContactSheetPage(
                page_index=0,
                page_rotate=0,
                page_size_pt=(612.0, 792.0),
                width=100,
                height=140,
                png_base64="",
            ),
        ),
    )


def test_the_editor_carries_its_opening_job_and_never_a_posted_one() -> None:
    """The session decides which job a publication belongs to, not the browser."""

    published: list[dict[str, Any]] = []

    def publish(source_name: str, fingerprint: str, *, job_id: str = "") -> dict[str, Any]:
        published.append(
            {"source": source_name, "fingerprint": fingerprint, "job_id": job_id}
        )
        return {"ok": True}

    store = LocalSourcePartEditorStore(
        editor_prefix="/editor/",
        plan_recipe=lambda _name, _payload: {"ok": True},
        publish_plan=publish,
    )
    offer = store.open(
        _sheet(), recipe_id="3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908", job_id="job-1"
    )
    assert offer.job_id == "job-1"

    store.act(
        offer.token,
        {"action": "publish", "plan_fingerprint": "f" * 64, "job_id": "job-9"},
    )
    assert published == [
        {"source": "verbs.pdf", "fingerprint": "f" * 64, "job_id": "job-1"}
    ]


def test_a_jobless_editor_publishes_exactly_as_it_did_before() -> None:
    published: list[str] = []

    store = LocalSourcePartEditorStore(
        editor_prefix="/editor/",
        plan_recipe=lambda _name, _payload: {"ok": True},
        publish_plan=lambda name, fingerprint, *, job_id="": published.append(job_id)
        or {"ok": True},
    )
    offer = store.open(_sheet(), recipe_id="3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908")
    assert offer.job_id == ""
    store.act(offer.token, {"action": "publish", "plan_fingerprint": "f" * 64})
    assert published == [""]


# --- the publication backlink, attach and initiate ----------------------------


class _FakePart:
    def __init__(self, name: str, data: bytes) -> None:
        self.ordinal = 1
        self.target_name = name
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.byte_length = len(data)
        self.page_index = 0
        self.page_rotate = 0
        self.page_size_pt = (612.0, 792.0)
        self.pixel_rect = (0, 0, 100, 100)
        self.regions: tuple[Any, ...] = ()
        self.thumbnail_png_base64 = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "page_size_pt": list(self.page_size_pt),
            "pixel_rect": list(self.pixel_rect),
            "regions": [],
        }


def _fake_plan(recipe_id: str, data: bytes) -> Any:
    part = _FakePart("verbs-p1.png", data)
    return source_parts.SourcePartsPlan(
        recipe_id=recipe_id,
        parent_name="verbs.pdf",
        parent_sha256="9f2c",
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        render_dpi=200,
        plan_fingerprint="f" * 64,
        parts=(part,),
        recipe_sha256="a" * 64,
        payloads=(data,),
    )


def test_publishing_for_a_job_binds_the_hash_that_lands_before_it_lands(
    tmp_path: Path,
) -> None:
    """The intent precedes the effect, and the receipt is job-independent."""

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    plan = _fake_plan(recipe_id, b"rendered part bytes")

    receipt = study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )

    saved = study_job.load_study_job(config, job.header.job_id)
    assert [intent.kind for intent in saved.intents] == ["source_parts"]
    intent = saved.intents[0]
    assert intent.reserves == {
        "recipe_id": recipe_id,
        "receipt_sha256": receipt.receipt_sha256,
    }
    # The hash the intent bound is the hash on disk, byte for byte.
    assert (
        hashlib.sha256(receipt.path.read_bytes()).hexdigest()
        == receipt.receipt_sha256
    )
    assert [outcome.state for outcome in saved.outcomes] == ["applied"]
    assert saved.outcomes[0].observed == (
        (str(receipt.path), receipt.receipt_sha256),
    )
    # The receipt records no job at all: it belongs to the parts.
    stored = json.loads(receipt.path.read_text(encoding="utf-8"))
    assert "job_id" not in stored
    assert job.header.job_id not in receipt.path.read_text(encoding="utf-8")

    # A second job attaches to the same receipt and republishes nothing.
    other = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    before = receipt.path.read_bytes()
    reused = study_job.publish_job_source_parts(
        config, other.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    assert reused.receipt_sha256 == receipt.receipt_sha256
    assert receipt.path.read_bytes() == before
    assert reused.published == () and len(reused.reused) == 1


def _job_over_existing_deck(config: ProjectConfig) -> Any:
    return study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )


def test_the_job_route_opens_the_editor_bound_to_that_job_and_publishing_backlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner route that makes a published part reachable from its job.

    End to end through the real editor store: the adapter's own job control
    opens the session, the session — not the browser — decides which job the
    Publish inside it belongs to, and that publication records this job's
    source-part intent. Without the binding the parts land jobless and the
    job's own extraction finds nothing.
    """

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    job = _job_over_existing_deck(config)
    adapter = _adapter(config)
    store = adapter.bind_source_part_editors("/editor/")
    monkeypatch.setattr(source_parts, "sources_unavailable", lambda: None)
    monkeypatch.setattr(
        source_parts, "render_contact_sheet", lambda _config, _name: _sheet()
    )
    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    plan = _fake_plan(recipe_id, b"rendered part bytes")
    # What the owner's own Render inside the editor leaves behind, which the
    # Publish below is bound to by its exact fingerprint.
    with adapter._plan_lock:
        adapter._source_part_plans[plan.plan_fingerprint] = plan

    told = adapter.open_study_job_source_part_editor(job_id=job.header.job_id)

    assert "/editor/" in told
    token = told.split("/editor/", 1)[1].split(")", 1)[0]
    assert store.read(f"/editor/{token}".removeprefix("/editor/")) is not None

    # The browser posts a different job; the session's own binding wins.
    result = store.act(
        token,
        {
            "action": "publish",
            "plan_fingerprint": plan.plan_fingerprint,
            "job_id": "22222222-2222-4222-8222-222222222222",
        },
    )

    assert result["ok"] is True
    saved = study_job.load_study_job(config, job.header.job_id)
    assert [intent.kind for intent in saved.intents] == ["source_parts"]
    assert saved.intents[0].reserves["recipe_id"] == recipe_id
    assert [outcome.state for outcome in saved.outcomes] == ["applied"]
    # And the parts are now what this job would send.
    assert [
        path.name
        for path, _lineage in study_job.job_part_sources(config, job.header.job_id)
    ] == ["verbs-p1.png"]
    # The receipt still belongs to the parts, not to this job.
    stored = json.loads(
        source_parts.receipt_path(config, recipe_id).read_text(encoding="utf-8")
    )
    assert "job_id" not in stored


def test_an_owner_control_opens_a_job_over_an_existing_deck_without_a_ladder(
    tmp_path: Path,
) -> None:
    """§1 step 2 and step 3: a deck is a selection, Create/Open is the owner's.

    No deck is created, nothing is confirmed twice and nothing is sent: the
    control binds the source the owner clicked and the deck the conversation
    was focused on, and writes the local job document.
    """

    config = _project(tmp_path)
    source = _source(config)
    deck = _deck(config)
    adapter = _adapter(config, deck)
    scope = deck.absolute().relative_to(config.root.absolute()).as_posix()

    offered = adapter.list_study_job_starts(deck_scope=scope)
    assert [start.source_name for start in offered] == ["verbs.pdf"]
    assert scope in offered[0].summary

    told = adapter.open_study_job_over_deck(source_name="verbs.pdf", deck_scope=scope)

    job_ids = study_job.list_study_jobs(config)
    assert len(job_ids) == 1
    job = study_job.load_study_job(config, job_ids[0])
    assert job.header.parent_source_name == "verbs.pdf"
    assert job.header.deck_path == scope
    assert job.header.parent_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert job.header.deck_sha256 == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert job.header.job_id in told
    # A local record: no deck was created and nothing was journalled.
    assert sorted(path.name for path in config.deck_dir.iterdir()) == [
        "201-verbs.yaml"
    ]
    assert not config.operations_file.exists()

    # An unfocused conversation offers no start, because a job binds a deck.
    assert adapter.list_study_job_starts(deck_scope="") == ()
    with pytest.raises(assistant_adapter.RevisionRefusal):
        adapter.open_study_job_over_deck(source_name="verbs.pdf", deck_scope="")
    with pytest.raises(assistant_adapter.RevisionRefusal):
        adapter.open_study_job_over_deck(source_name="not-in-the-corpus.pdf", deck_scope=scope)
    assert study_job.list_study_jobs(config) == job_ids


def test_the_owner_saves_which_published_parts_the_next_batch_covers(
    tmp_path: Path,
) -> None:
    """The one S4 job choice with a writer, saved through its own CAS service.

    A reversible local preference: it publishes nothing, sends nothing and
    changes no deck definition, and `job_part_sources` already honours it.
    """

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    job = _job_over_existing_deck(config)
    plan = source_parts.SourcePartsPlan(
        recipe_id="3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908",
        parent_name="verbs.pdf",
        parent_sha256="9f2c",
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        render_dpi=200,
        plan_fingerprint="f" * 64,
        parts=(_FakePart("verbs-p1.png", b"one"), _FakePart("verbs-p2.png", b"two")),
        recipe_sha256="a" * 64,
        payloads=(b"one", b"two"),
    )
    object.__setattr__(plan.parts[1], "ordinal", 2)
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    adapter = _adapter(config)

    # Rendered from saved state: with no choice recorded every part is covered.
    controls = {
        action.target: action.action
        for action in next(
            entry
            for entry in adapter.list_study_job_choices()
            if entry.job_id == job.header.job_id
        ).actions
        if action.target
    }
    assert controls == {
        "verbs-p1.png": "exclude-part",
        "verbs-p2.png": "exclude-part",
    }

    told = adapter.save_study_job_part_selection(
        job_id=job.header.job_id, part_name="verbs-p2.png", include=False
    )

    assert "verbs-p1.png" in told
    saved = study_job.load_study_job(config, job.header.job_id)
    assert saved.choices == {"part_selections": ["verbs-p1.png"]}
    assert [
        path.name
        for path, _lineage in study_job.job_part_sources(config, job.header.job_id)
    ] == ["verbs-p1.png"]
    # The excluded part is still offered, so the owner can put it back.
    after = {
        action.target: action.action
        for action in next(
            entry
            for entry in adapter.list_study_job_choices()
            if entry.job_id == job.header.job_id
        ).actions
        if action.target
    }
    assert after == {
        "verbs-p1.png": "exclude-part",
        "verbs-p2.png": "include-part",
    }
    adapter.save_study_job_part_selection(
        job_id=job.header.job_id, part_name="verbs-p2.png", include=True
    )
    assert study_job.load_study_job(config, job.header.job_id).choices == {
        "part_selections": ["verbs-p1.png", "verbs-p2.png"]
    }
    # A part this job never published, and emptying the batch, both refuse.
    with pytest.raises(assistant_adapter.RevisionRefusal):
        adapter.save_study_job_part_selection(
            job_id=job.header.job_id, part_name="verbs-p9.png", include=True
        )
    adapter.save_study_job_part_selection(
        job_id=job.header.job_id, part_name="verbs-p2.png", include=False
    )
    with pytest.raises(assistant_adapter.RevisionRefusal) as empty:
        adapter.save_study_job_part_selection(
            job_id=job.header.job_id, part_name="verbs-p1.png", include=False
        )
    assert "at least one part" in str(empty.value)
    assert study_job.load_study_job(config, job.header.job_id).choices == {
        "part_selections": ["verbs-p1.png"]
    }
    # A local preference only: no deck definition moved and nothing journalled.
    assert not config.operations_file.exists()
    assert hashlib.sha256(
        (config.deck_dir / "201-verbs.yaml").read_bytes()
    ).hexdigest() == job.header.deck_sha256


def test_the_desk_offers_resume_for_an_interrupted_publication_and_finishes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control the desk offers and what resuming does must be one answer.

    The receipt and its part landed and the outcome write did not, which is the
    window §2.4 exists for. The desk offers "Continue what this job already
    bought" off the open intent, so resuming it has to close that intent from
    the exact receipt it reserved rather than reporting there is nothing to do.
    """

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    plan = _fake_plan(recipe_id, b"rendered part bytes")

    real_append = study_job.append_outcome

    def die(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("the process died before the outcome write landed")

    monkeypatch.setattr(study_job, "append_outcome", die)
    with pytest.raises(KeyboardInterrupt):
        study_job.publish_job_source_parts(
            config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
        )
    monkeypatch.setattr(study_job, "append_outcome", real_append)

    adapter = _adapter(config)
    choice = next(
        entry
        for entry in adapter.list_study_job_choices()
        if entry.job_id == job.header.job_id
    )
    assert "resume" in {action.action for action in choice.actions}
    assert "recorded action(s) have no outcome" in adapter.study_job_status_text(
        job_id=job.header.job_id
    )

    told = adapter.resume_study_job(job_id=job.header.job_id, progress=lambda _l: None)

    assert "Nothing in this job is waiting" not in told
    assert "verbs-p1.png" in told
    saved = study_job.load_study_job(config, job.header.job_id)
    assert [outcome.state for outcome in saved.outcomes] == ["applied"]
    assert "recorded action(s) have no outcome" not in adapter.study_job_status_text(
        job_id=job.header.job_id
    )


def test_a_diverging_receipt_is_refused_and_recorded_never_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another run wrote the receipt first: replan as an attach, never adopt."""

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    plan = _fake_plan(recipe_id, b"rendered part bytes")

    real = source_parts.execute_source_parts

    def stamp_something_else(
        conf: Any, value: Any, *, publish_token: str, created_at: str = ""
    ) -> Any:
        # Exactly the mutant this guard exists for: the writer ignores the
        # frozen stamp, so the receipt that lands is not the one bound.
        return real(conf, value, publish_token=publish_token, created_at="")

    monkeypatch.setattr(source_parts, "execute_source_parts", stamp_something_else)
    monkeypatch.setattr(study_job.source_parts, "execute_source_parts", stamp_something_else)

    with pytest.raises(JankiError) as error:
        study_job.publish_job_source_parts(
            config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
        )
    assert "will not adopt" in str(error.value)

    saved = study_job.load_study_job(config, job.header.job_id)
    assert [outcome.state for outcome in saved.outcomes] == ["refused"]
    assert saved.outcomes[0].observed[0][1] != saved.intents[0].reserves[
        "receipt_sha256"
    ]


# --- the retry a model may ask for, through the real decode ------------------


def _subscription_project(tmp_path: Path) -> ProjectConfig:
    """A project whose extraction transport is the owner's subscription.

    A batch — and therefore a retry of one — refuses on the metered API, so the
    surfaces fixture's default provider cannot exercise this route at all.
    """

    _project(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[assistant]\nenabled = true\n"
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    fresh = ProjectConfig.load(tmp_path)
    seed_prompts(fresh.root)
    return fresh


def _failed_job_batch(
    config: ProjectConfig, job_id: str, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """One published part, sent once, refused by the provider. Retryable."""

    plan = _fake_plan("3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908", b"rendered part bytes")
    study_job.publish_job_source_parts(
        config, job_id, plan, publish_token=plan.plan_fingerprint
    )
    batch = study_job.plan_job_extraction_batch(
        config,
        job_id,
        provider_env={},
        provider_runner=FakeClaudeRunner(reply=_stream()),
        provider_which=_which,
    )
    refusing = FakeClaudeRunner(reply=b'{"type":"result","is_error":true}\n')
    outcome = study_job.dispatch_job_batch(
        config,
        job_id,
        batch,
        provider_env={},
        provider_runner=refusing,
        provider_which=_which,
        provider_spawn=refusing.spawn,
    )
    assert outcome.committed_count == 0, "the fixture needs a retryable child"
    return batch


def _decoded(config: ProjectConfig, resource_ids: tuple[str, ...], options: Any) -> Any:
    """One model answer through the real allowlist the turn actually enforces."""

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]["resources"]
    disclosed = tuple(item["resource_id"] for item in catalog)
    context = assistant_adapter.assistant_agent.AgentContext(
        wire=broker.catalog().wire,
        fingerprint=hashlib.sha256(
            broker.catalog().wire.encode("utf-8")
        ).hexdigest(),
        resource_ids=disclosed,
    )
    parsed = ai_schema.assistant_agent_schema().model_validate(
        {
            "answer": "Sending that page again.",
            "action_intents": [
                {
                    "kind": "retry_study_parts",
                    "resource_ids": list(resource_ids),
                    "record_ids": [],
                    "instruction": "Send the refused page again.",
                    "options": options,
                }
            ],
        }
    )
    return assistant_adapter.assistant_agent._decode_answer(
        SimpleNamespace(parsed=parsed),
        model="claude-opus-5",
        captured=False,
        context=context,
    )


def test_a_model_retry_reaches_the_service_through_a_disclosed_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry a model may ask for has to survive the turn's own allowlist.

    Every `resource_ids` entry is checked against the exact opaque catalogue
    the turn disclosed, so a bare batch id — which is not a resource at all —
    refuses before any study-job code runs. The batch is named by one of its
    own model calls instead, which is what the job's snapshot discloses and
    what janki resolves back through the manifest that reserved it. The owner
    still confirms the retry.
    """

    config = _subscription_project(tmp_path)
    _source(config)
    _deck(config)
    job = _job_over_existing_deck(config)
    batch = _failed_job_batch(config, job.header.job_id, monkeypatch)
    adapter = _adapter(config)
    job_resource = _job_resource_id(config)
    operation_id = batch.children[0].operation_id

    # What the model could otherwise name: the batch id its own snapshot
    # disclosed. It is not an opaque resource, so the turn refuses it.
    with pytest.raises(assistant_adapter.assistant_agent.AgentApplicationError) as blocked:
        _decoded(
            config,
            (job_resource, batch.batch_id),
            {"retry_child_indices": [1]},
        )
    assert batch.batch_id in str(blocked.value)

    _answer, intents = _decoded(
        config,
        (job_resource,),
        {"retry_child_indices": [1], "operation_id": operation_id},
    )
    assert len(intents) == 1

    real_retry = extraction_batch.plan_extraction_batch_retry

    def plan_retry(conf: Any, batch_id: str, indices: Any, **options: Any) -> Any:
        options.setdefault("provider_env", {})
        options.setdefault("provider_runner", FakeClaudeRunner(reply=_stream()))
        options.setdefault("provider_which", _which)
        return real_retry(conf, batch_id, indices, **options)

    monkeypatch.setattr(
        study_job.extraction_batch, "plan_extraction_batch_retry", plan_retry
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=intents),
        deck_scope="",
        owner_message="Send that page again.",
    )

    assert reply.action is not None
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "verbs-p1.png" in rendered
    with adapter._plan_lock:
        stored = adapter._batch_plans[reply.action.request_fingerprint]
    # A fresh identity for a fresh authority, and the old attempt named as the
    # discard this confirmation buys.
    assert stored.plan.batch_id != batch.batch_id
    assert stored.plan.retry_of == batch.batch_id
    assert stored.plan.children[0].operation_id != operation_id
    assert stored.plan.job_id == job.header.job_id

    runner = FakeClaudeRunner(reply=_stream())
    real_dispatch = extraction_batch.dispatch_extraction_batch

    def dispatch(conf: Any, plan: Any, **options: Any) -> Any:
        options.setdefault("provider_env", {})
        options.setdefault("provider_runner", runner)
        options.setdefault("provider_which", _which)
        options.setdefault("provider_spawn", runner.spawn)
        return real_dispatch(conf, plan, **options)

    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", dispatch
    )

    execution = _confirm(adapter, reply)

    assert "verbs-p1.png" in execution.message
    saved = study_job.load_study_job(config, job.header.job_id)
    assert [intent.kind for intent in saved.intents] == [
        "source_parts",
        "extract_batch",
        "retry",
    ]
    assert saved.intents[-1].reserves["batch_id"] == stored.plan.batch_id


def test_the_desk_and_the_status_text_agree_that_a_paused_batch_can_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One milestone, one answer about the same batch.

    A child whose transport preflight refused is reserved, unsent and unspent,
    which the batch's own durable eligibility says. The desk therefore offers
    the control, the status text says the same thing, and continuing the job
    really finishes that child under the authority already recorded.
    """

    config = _subscription_project(tmp_path)
    _source(config)
    _deck(config)
    job = _job_over_existing_deck(config)
    plan = _fake_plan("3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908", b"rendered part bytes")
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    runner = FakeClaudeRunner(reply=_stream())
    batch = study_job.plan_job_extraction_batch(
        config,
        job.header.job_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    real_prepare = extraction_batch.prepare_extraction_transport
    monkeypatch.setattr(
        extraction_batch,
        "prepare_extraction_transport",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            JankiError("temporary login refusal")
        ),
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        batch,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert [child.state for child in outcome.children] == ["authorized"]

    adapter = _adapter(config)
    told = adapter.study_job_status_text(job_id=job.header.job_id)
    assert "continuing this job finishes it under the authority it already " in told
    choice = next(
        entry
        for entry in adapter.list_study_job_choices()
        if entry.job_id == job.header.job_id
    )
    assert "resume" in {action.action for action in choice.actions}

    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", real_prepare
    )
    real_resume = extraction_batch.resume_extraction_batch

    def resume(conf: Any, batch_id: str, **options: Any) -> Any:
        options.setdefault("provider_env", {})
        options.setdefault("provider_runner", runner)
        options.setdefault("provider_which", _which)
        options.setdefault("provider_spawn", runner.spawn)
        return real_resume(conf, batch_id, **options)

    monkeypatch.setattr(
        study_job.extraction_batch, "resume_extraction_batch", resume
    )

    continued = adapter.resume_study_job(
        job_id=job.header.job_id, progress=lambda _label: None
    )

    assert "Nothing in this job is waiting" not in continued
    assert "verbs-p1.png" in continued
    saved = study_job.load_study_job(config, job.header.job_id)
    assert [entry.state for entry in saved.outcomes] == ["applied", "applied"]
    assert "recorded action(s) have no outcome" not in adapter.study_job_status_text(
        job_id=job.header.job_id
    )


def test_the_status_text_and_a_resume_agree_that_a_call_may_be_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same rule: a batch nobody may continue yet.

    The workbench was killed between the child's dispatch claim and its
    settlement, so its money may already have gone and the batch's own
    eligibility refuses to start another call. The status text says that;
    continuing the job has to say the same thing and name the decision that
    actually unblocks it, rather than promising reserved authority.
    """

    config = _subscription_project(tmp_path)
    _source(config)
    _deck(config)
    job = _job_over_existing_deck(config)
    plan = _fake_plan("3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908", b"rendered part bytes")
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    runner = FakeClaudeRunner(reply=_stream())
    batch = study_job.plan_job_extraction_batch(
        config,
        job.header.job_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def killed(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("killed with a call in flight")

    with pytest.raises(KeyboardInterrupt):
        study_job.dispatch_job_batch(
            config,
            job.header.job_id,
            batch,
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
            provider_spawn=killed,
        )
    assert [
        child.state
        for child in extraction_batch.extraction_batch_status(
            config, batch.batch_id
        ).children
    ] == ["dispatching"]

    adapter = _adapter(config)
    refusal = "a call may be in flight right now, so janki will not start another"
    told = adapter.study_job_status_text(job_id=job.header.job_id)
    assert refusal in told
    assert "continuing this job finishes it under the authority" not in told

    journalled = config.operations_file.read_bytes()
    continued = adapter.resume_study_job(
        job_id=job.header.job_id, progress=lambda _label: None
    )

    assert refusal in continued
    assert "still hold reserved authority" not in continued
    assert "janki operations --end" in continued
    # Reading where it stands ends nothing and sends nothing.
    assert config.operations_file.read_bytes() == journalled
    saved = study_job.load_study_job(config, job.header.job_id)
    assert [entry.state for entry in saved.outcomes] == ["applied"]
    assert "recorded action(s) have no outcome" in adapter.study_job_status_text(
        job_id=job.header.job_id
    )


def test_a_model_retry_refuses_a_model_call_from_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a call is not naming somebody else's batch."""

    config = _subscription_project(tmp_path)
    _source(config)
    _deck(config)
    mine = _job_over_existing_deck(config)
    theirs = "22222222-2222-4222-8222-222222222222"
    batch = _failed_job_batch(config, mine.header.job_id, monkeypatch)
    adapter = _adapter(config)

    class _OtherPlan:
        job_id = theirs

    monkeypatch.setattr(
        assistant_adapter.extraction_batch,
        "find_capture_child",
        lambda _config, _operation_id: (_OtherPlan(), object()),
    )
    _answer, intents = _decoded(
        config,
        (_job_resource_id(config),),
        {
            "retry_child_indices": [1],
            "operation_id": batch.children[0].operation_id,
        },
    )

    with pytest.raises(assistant_adapter.RevisionRefusal) as error:
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=intents),
            deck_scope="",
            owner_message="Send that page again.",
        )

    assert theirs in str(error.value)
    assert mine.header.job_id in str(error.value)


# --- the two capture actions a model may take ---------------------------------


def _capture_intent(kind: str, resource_id: str, **options: Any) -> Any:
    return _agent_intent(
        kind=kind,
        resource_ids=(resource_id,),
        record_ids=(),
        instruction="Read what that reply holds.",
        options=options,
    )


def _job_resource_id(config: ProjectConfig) -> str:
    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    return next(
        item["resource_id"]
        for item in catalog["resources"]
        if item["kind"] == "study_job"
    )


def test_a_capture_action_names_the_job_and_the_call_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The model supplies the operation; no part of a location tuple exists."""

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    adapter = _adapter(config)
    resource_id = _job_resource_id(config)

    with pytest.raises(assistant_adapter.RevisionRefusal) as missing:
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(
                intents=(_capture_intent("stage_capture_proposal", resource_id),)
            ),
            deck_scope="",
            owner_message="Recover that reply.",
        )
    assert "names the exact model call" in str(missing.value)

    with pytest.raises(assistant_adapter.RevisionRefusal) as extra:
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(
                intents=(
                    _capture_intent(
                        "inspect_capture_proposals",
                        resource_id,
                        operation_id="op-1",
                        retry_child_indices=[1],
                    ),
                )
            ),
            deck_scope="",
            owner_message="Read that reply.",
        )
    assert "and nothing else" in str(extra.value)


def test_a_capture_action_refuses_a_model_call_from_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job never reads or stages a stranger's already-paid reply."""

    config = _project(tmp_path)
    _source(config)
    _deck(config)
    mine = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    adapter = _adapter(config)
    resource_id = _job_resource_id(config)
    theirs = "22222222-2222-4222-8222-222222222222"

    class _OtherPlan:
        job_id = theirs

    monkeypatch.setattr(
        assistant_adapter.extraction_batch,
        "find_capture_child",
        lambda _config, _operation_id: (_OtherPlan(), object()),
    )
    staged: list[str] = []
    monkeypatch.setattr(
        assistant_adapter.capture_recovery,
        "stage_capture_proposal",
        lambda *args, **kwargs: staged.append("staged"),
    )

    with pytest.raises(assistant_adapter.RevisionRefusal) as error:
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(
                intents=(
                    _capture_intent(
                        "stage_capture_proposal", resource_id, operation_id="op-1"
                    ),
                )
            ),
            deck_scope="",
            owner_message="Recover that reply.",
        )

    message = str(error.value)
    assert theirs in message and mine.header.job_id in message
    assert "Nothing was read or staged." in message
    assert staged == []
