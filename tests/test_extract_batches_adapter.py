"""The real adapter over the real batch core, with only the provider faked.

These are the regressions the DTO-level surface tests cannot give: that one
typed ``extract_batch`` intent produces exactly one confirmation, that
confirming dispatches *that stored plan* rather than a freshly minted one, and
that a plan whose identity moved underneath the confirmation is refused before
anything is sent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_extraction_batch import _runner
from test_revision_provider import _which
from test_workbench_assistant_integration import (
    _adapter,
    _agent_intent,
    _agent_result,
    _config,
)

from conftest import seed_prompts
from japanese_anki import card_preview
from japanese_anki.application import extraction_batch
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter, assistant_batch_surface
from japanese_anki.workbench.assistant import RevisionConfirmation, RevisionRefusal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _InjectedCore:
    """The real core, reached through the surface's own seam.

    Only the subscription transport is faked: planning still probes a provider,
    and this test may not spend the owner's allowance to do it.
    """

    def __init__(self) -> None:
        self.runner = _runner()

    def __getattr__(self, name: str) -> Any:
        return getattr(extraction_batch, name)

    def plan_extraction_batch(self, config: Any, sources: Any, **options: Any) -> Any:
        return extraction_batch.plan_extraction_batch(
            config,
            sources,
            provider_env={},
            provider_runner=self.runner,
            provider_which=_which,
            **options,
        )


class _TwoSourceBroker:
    """Two disclosed, already-preserved sources and nothing inferred."""

    def __init__(self, config: ProjectConfig) -> None:
        self._inbox = config.scan_inbox

    def source_path(self, resource_id: str) -> Path:
        names = {
            "resource_source_a": "lesson-a.pdf",
            "resource_source_b": "lesson-b.pdf",
        }
        if resource_id not in names:
            raise AssertionError(f"unexpected source resource {resource_id}")
        return self._inbox / names[resource_id]


def _project(tmp_path: Path) -> ProjectConfig:
    _config(tmp_path)
    # Batches run on the subscription transport; the core refuses to batch on
    # the metered API. No call is made here — planning is all this needs.
    (tmp_path / "janki.toml").write_text(
        "[assistant]\nenabled = true\n[ai]\nextract_provider = \"claude-code\"\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    seed_prompts(config.root)
    # The repository's real card templates, so a preview is a real render.
    shutil.copytree(
        PROJECT_ROOT / "templates", config.root / "templates", dirs_exist_ok=True
    )
    config.scan_inbox.mkdir(parents=True, exist_ok=True)
    for name in ("lesson-a.pdf", "lesson-b.pdf"):
        (config.scan_inbox / name).write_bytes(b"%PDF-1.4\n" + name.encode("utf-8"))
    return config


def _batch_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    options: dict[str, Any] | None = None,
    resource_ids: tuple[str, ...] = ("resource_source_a", "resource_source_b"),
    record_ids: tuple[str, ...] = (),
) -> tuple[Any, Any, ProjectConfig]:
    config = _project(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _TwoSourceBroker,
    )
    injected = _InjectedCore()
    monkeypatch.setattr(assistant_batch_surface, "core", lambda: injected)
    result = _agent_result(
        answer="Two sources, one batch.",
        intents=(
            _agent_intent(
                kind="extract_batch",
                resource_ids=resource_ids,
                record_ids=record_ids,
                instruction="Read both lesson pages.",
                options=options,
            ),
        ),
    )
    reply = adapter._prepare_agent_intent(config, result=result, deck_scope="")
    return adapter, reply, config, injected


def _receipts(config: ProjectConfig) -> list[Path]:
    directory = config.operations_file.parent / extraction_batch.BATCH_DIR_NAME
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.execution.json"))


def test_one_batch_intent_renders_one_confirmation_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, reply, config, _core = _batch_reply(tmp_path, monkeypatch)

    plan = reply.action
    assert plan is not None
    assert reply.action_instruction == "Read both lesson pages."
    rendered = "\n".join((*plan.effects, *plan.disclosures))
    assert "1. lesson-a.pdf" in rendered, rendered
    assert "2. lesson-b.pdf" in rendered, rendered
    assert "2 at a time" in rendered, rendered
    # The confirmation is bound to the stored plan's exact identity.
    with adapter._plan_lock:
        stored = adapter._batch_plans[plan.request_fingerprint]
    assert stored.plan.fingerprint == plan.request_fingerprint
    assert len(stored.plan.children) == 2
    # Planning reserves and sends nothing: no dispatch receipt, no staging.
    assert _receipts(config) == []
    assert not list(config.staging_dir.glob("*.yaml"))


def test_extract_batch_refuses_record_targets_and_a_single_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RevisionRefusal):
        _batch_reply(tmp_path, monkeypatch, record_ids=("word:one",))
    with pytest.raises(RevisionRefusal):
        _batch_reply(tmp_path, monkeypatch, resource_ids=("resource_source_a",))


def test_extract_batch_refuses_concurrency_outside_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RevisionRefusal):
        _batch_reply(tmp_path, monkeypatch, options={"concurrency_limit": 9})


def test_confirming_dispatches_the_stored_plan_through_the_real_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real core really runs. Only the CLI process behind it is faked."""

    adapter, reply, config, injected = _batch_reply(tmp_path, monkeypatch)
    fingerprint = reply.action.request_fingerprint
    with adapter._plan_lock:
        stored = adapter._batch_plans[fingerprint]
    expected_batch_id = stored.plan.batch_id
    expected_operations = tuple(child.operation_id for child in stored.plan.children)

    dispatched: list[Any] = []
    real_dispatch = extraction_batch.dispatch_extraction_batch

    def record(config_arg: Any, plan: Any, **kwargs: Any) -> Any:
        dispatched.append(plan)
        # The same transport planning used, so the request fingerprints the
        # journal recorded are the ones the children are sent under.
        return real_dispatch(
            config_arg,
            plan,
            provider_env={},
            provider_runner=injected.runner,
            provider_which=_which,
            provider_spawn=injected.runner.spawn,
            **kwargs,
        )

    monkeypatch.setattr(extraction_batch, "dispatch_extraction_batch", record)
    reported: list[str] = []

    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="capability-token",
            deck_scope="",
            instruction="Read both lesson pages.",
            expected_fingerprint=fingerprint,
            target=reply.action.target,
        ),
        progress=reported.append,
    )

    assert len(dispatched) == 1, "one confirmation is one dispatch"
    sent = dispatched[0]
    assert sent.batch_id == expected_batch_id, "no new batch identity was minted"
    assert tuple(child.operation_id for child in sent.children) == expected_operations
    assert sent.fingerprint == fingerprint

    # The core really wrote: one staging file per source.
    staged = sorted(path.name for path in config.staging_dir.glob("*.yaml"))
    assert staged == ["lesson-a.pdf.yaml", "lesson-b.pdf.yaml"], staged

    # And the journal really settled those exact children.
    status = extraction_batch.extraction_batch_status(config, expected_batch_id)
    assert tuple(child.operation_id for child in status.children) == expected_operations
    assert [child.state for child in status.children] == ["committed", "committed"]
    journal_text = config.operations_file.read_text(encoding="utf-8")
    for operation_id in expected_operations:
        assert operation_id in journal_text

    assert execution.complete is True
    assert "lesson-a.pdf" in execution.message
    # Live narration names which numbered source reached which state.
    assert reported, "the confirmed batch reported no progress at all"
    assert all(label.startswith("Source ") and " of 2: " in label for label in reported), (
        reported
    )
    assert any(label.endswith(": committed") for label in reported), reported

    with pytest.raises(RevisionRefusal):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="capability-token",
                deck_scope="",
                instruction="Read both lesson pages.",
                expected_fingerprint=fingerprint,
                target=reply.action.target,
            ),
            progress=reported.append,
        )
    assert len(dispatched) == 1


def test_a_plan_whose_concurrency_moved_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered fingerprint covers the whole manifest, concurrency included."""

    adapter, reply, config, _core = _batch_reply(tmp_path, monkeypatch)
    fingerprint = reply.action.request_fingerprint

    dispatched: list[Any] = []
    monkeypatch.setattr(
        extraction_batch,
        "dispatch_extraction_batch",
        lambda *args, **kwargs: dispatched.append(args) or None,
    )
    with adapter._plan_lock:
        stored = adapter._batch_plans[fingerprint]
        object.__setattr__(stored.plan, "concurrency_limit", 4)

    with pytest.raises(RevisionRefusal, match="no longer matches"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="capability-token",
                deck_scope="",
                instruction="Read both lesson pages.",
                expected_fingerprint=fingerprint,
                target=reply.action.target,
            ),
            progress=lambda _label: None,
        )
    assert dispatched == []
    assert _receipts(config) == []


def test_an_unknown_fingerprint_is_refused_without_replanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, reply, _config_out, _core = _batch_reply(tmp_path, monkeypatch)
    planned: list[Any] = []
    monkeypatch.setattr(
        extraction_batch,
        "plan_extraction_batch",
        lambda *args, **kwargs: planned.append(args),
    )

    with pytest.raises(RevisionRefusal):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="capability-token",
                deck_scope="",
                instruction="Read both lesson pages.",
                expected_fingerprint="0" * 64,
                target=reply.action.target,
            ),
            progress=lambda _label: None,
        )
    assert planned == [], "a confirmation never re-plans a batch"


def test_the_local_desk_reads_durable_batches_through_the_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planned-but-unconfirmed batch is not durable work, and the desk agrees."""

    adapter, _reply, config, _core = _batch_reply(tmp_path, monkeypatch)

    assert extraction_batch.list_extraction_batches(config) == ()
    assert adapter.list_extraction_batch_choices() == ()
    assert _receipts(config) == []


def test_the_desk_reads_resume_eligibility_from_the_core_outcome() -> None:
    """Continue is offered only when the core says so, with its own reason."""

    outcome = extraction_batch.ExtractionBatchOutcome
    names = {field.name for field in dataclasses.fields(outcome)} | {
        name for name, value in vars(outcome).items() if isinstance(value, property)
    }

    assert {"resume_available", "resume_refusal"} <= names


def _confirm(adapter: Any, reply: Any, config: Any, injected: Any) -> Any:
    """Drive the real confirm through the real core, faking only the process."""

    real_dispatch = extraction_batch.dispatch_extraction_batch

    def record(config_arg: Any, plan: Any, **kwargs: Any) -> Any:
        return real_dispatch(
            config_arg,
            plan,
            provider_env={},
            provider_runner=injected.runner,
            provider_which=_which,
            provider_spawn=injected.runner.spawn,
            **kwargs,
        )

    extraction_batch.dispatch_extraction_batch = record
    try:
        return adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="capability-token",
                deck_scope="",
                instruction="Read both lesson pages.",
                expected_fingerprint=reply.action.request_fingerprint,
                target=reply.action.target,
            ),
            progress=lambda _label: None,
        )
    finally:
        extraction_batch.dispatch_extraction_batch = real_dispatch


@pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)
def test_a_finished_batch_offers_the_real_combined_cards_to_look_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Anki render, real templates, real staging — served from the store."""

    adapter, reply, config, injected = _batch_reply(tmp_path, monkeypatch)
    store = adapter.bind_preview_links("/assistant/preview")

    execution = _confirm(adapter, reply, config, injected)

    assert execution.complete is True
    found = re.findall(r"\(/assistant/preview[^)\s]+\)", execution.message)
    assert found, execution.message
    url = found[0].strip("()")
    # Disagreements are counted for the owner; the provenance behind them stays
    # beside the cards rather than in the chat reply.
    assert "raw_fields" not in execution.message
    # Says where each thing actually is: the preview draws one proposal per
    # word, and the differences live in the batch desk.
    assert "in more than one source" in execution.message
    assert "first source's proposal" in execution.message
    assert "Manage extraction batches lists what the sources differ on" in (
        execution.message
    )

    snapshots = list(store._snapshots.items())
    assert len(snapshots) == 1
    token, snapshot = snapshots[0]
    assert url.endswith(token)
    assert snapshot.card_count > 0
    assert snapshot.html
    assert hashlib.sha256(snapshot.html).hexdigest() == snapshot.sha256

    # Looking is not accepting: no combined staging document, and nothing
    # canonical was written.
    staged = sorted(path.name for path in config.staging_dir.glob("*"))
    assert staged == ["lesson-a.pdf.yaml", "lesson-b.pdf.yaml"], staged
    assert not config.normalized_file.exists()


def test_a_batch_that_saved_nothing_claims_no_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch is stubbed here on purpose: the guard is about zero proposals."""

    adapter, reply, config, _injected = _batch_reply(tmp_path, monkeypatch)
    adapter.bind_preview_links("/assistant/preview")
    rendered: list[str] = []
    def refuse(*args: Any, **kwargs: Any) -> Any:
        rendered.append(kwargs.get("batch_id", args))
        raise AssertionError("no proposals were saved, so nothing may be drawn")

    monkeypatch.setattr(assistant_batch_surface, "preview_offer", refuse)
    monkeypatch.setattr(
        extraction_batch,
        "dispatch_extraction_batch",
        lambda config_arg, plan, **kwargs: extraction_batch.ExtractionBatchOutcome(
            batch_id=plan.batch_id,
            children=tuple(
                extraction_batch.ExtractionBatchChildOutcome(
                    index=child.index,
                    operation_id=child.operation_id,
                    source=child.source,
                    state="failed_before_send",
                    staging_path=None,
                    error="the provider refused",
                    bookkeeping_complete=True,
                    records=None,
                )
                for child in plan.children
            ),
            concurrency_limit=plan.concurrency_limit,
        ),
    )

    execution = _confirm(adapter, reply, config, _injected)

    assert rendered == [], "nothing was saved, so nothing may be offered to look at"
    assert "/assistant/preview" not in execution.message
    assert "failed_before_send" in execution.message
