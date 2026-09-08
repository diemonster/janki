"""Local extraction-batch management inside the Janki Assistant.

Status, the combined preview, resume, and selected-child retry are all local
controls: none of them needs a model turn, and the read-only two must keep
working while the thread is busy with a paid one. Every callback is a fake.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_workbench_assistant import (  # noqa: E402
    SESSION_TOKEN,
    _action_request,
    _assistant_deck_choices,
    _confirmation_request,
    _events,
    _FakeRevisions,
    _followup_message_request,
    _message_request,
    _post,
    _widget_items,
    create_assistant_sidecar,
)
from test_workbench_assistant_integration import _adapter, _config

from japanese_anki.application import extraction_batch
from japanese_anki.errors import JankiError
from japanese_anki.workbench import assistant as assistant_module
from japanese_anki.workbench import assistant_batch_surface
from japanese_anki.workbench.assistant import (
    ChatReply,
    ExtractionBatchActionChoice,
    ExtractionBatchChildStatus,
    ExtractionBatchChoice,
    ExtractionBatchPreviewOffer,
    ExtractionBatchResumption,
    RevisionExecution,
)
from japanese_anki.workbench.assistant_http import _application_javascript

BATCH_ID = "6f1c2b7a-0d3e-4a55-9c71-2f8e1b4d5a60"
OTHER_BATCH_ID = "1b2c3d4e-5f60-4718-8293-a4b5c6d7e8f9"


def batch_choice(
    *,
    batch_id: str = BATCH_ID,
    label: str = "Lesson pack (3 sources)",
    with_retry: bool = True,
) -> ExtractionBatchChoice:
    actions = [
        ExtractionBatchActionChoice(
            action="preview",
            label="Show the combined cards",
            child_indices=(),
        ),
        ExtractionBatchActionChoice(
            action="resume",
            label="Continue this batch",
            child_indices=(),
        ),
    ]
    if with_retry:
        actions.append(
            ExtractionBatchActionChoice(
                action="retry",
                label="Retry source 3 (lesson-c.pdf)",
                child_indices=(3,),
            )
        )
    return ExtractionBatchChoice(
        batch_id=batch_id,
        label=label,
        summary="2 saved, 1 failed, 0 unknown, 0 waiting",
        concurrency_limit=2,
        children=(
            ExtractionBatchChildStatus(1, "lesson-a.pdf", "committed"),
            ExtractionBatchChildStatus(2, "lesson-b.pdf", "committed"),
            ExtractionBatchChildStatus(
                3, "lesson-c.pdf", "failed_before_send", "the provider refused"
            ),
        ),
        actions=tuple(actions),
    )


@dataclass
class _FakeBatchRevisions(_FakeRevisions):
    """The existing fake plus exactly the four local batch callbacks."""

    batch_choices: tuple[ExtractionBatchChoice, ...] = ()
    batch_listed: int = 0
    batch_resumed: list[str] = field(default_factory=list)
    batch_retry_prepared: list[tuple[str, tuple[int, ...], str]] = field(
        default_factory=list
    )
    batch_previewed: list[tuple[str, str]] = field(default_factory=list)
    batch_progress: tuple[str, ...] = (
        "Source 1 of 3: running",
        "Source 1 of 3: committed",
    )
    batch_resumption: ExtractionBatchResumption | None = None
    batch_preview_offer: ExtractionBatchPreviewOffer | None = None
    batch_retry_refusal: str | None = None

    def list_extraction_batch_choices(self) -> tuple[ExtractionBatchChoice, ...]:
        self.batch_listed += 1
        return self.batch_choices

    def resume_extraction_batch(
        self, *, batch_id: str, progress: Any
    ) -> ExtractionBatchResumption:
        self.batch_resumed.append(batch_id)
        for label in self.batch_progress:
            progress(label)
        if self.batch_resumption is not None:
            return self.batch_resumption
        return ExtractionBatchResumption(
            message="Batch continued under its recorded authority: 3 saved.",
            state="committed",
            complete=True,
        )

    def prepare_extraction_batch_retry(
        self, *, batch_id: str, child_indices: tuple[int, ...], deck_scope: str
    ) -> ChatReply:
        self.batch_retry_prepared.append((batch_id, tuple(child_indices), deck_scope))
        if self.batch_retry_refusal is not None:
            raise assistant_module.RevisionRefusal(self.batch_retry_refusal)
        listed = ", ".join(str(index) for index in child_indices)
        return ChatReply(
            text=f"Prepared a retry of source {listed}.",
            action=assistant_module.RevisionPlan(
                request_fingerprint="9" * 64,
                target=f"Extraction batch {batch_id}",
                effects=(
                    f"Send source {listed} again as one new paid model call",
                    "Discard the old failed model call op-3 for lesson-c.pdf; "
                    "its recorded evidence is lost",
                ),
                disclosures=(
                    "Nothing already saved by this batch is resent or replaced.",
                ),
                confirm_label=f"Retry source {listed} — paid model call",
                progress_label="Preparing pages",
            ),
            action_instruction=f"retry batch {batch_id} sources {listed}",
        )

    def render_extraction_batch_preview(
        self, *, batch_id: str, deck_scope: str
    ) -> ExtractionBatchPreviewOffer:
        self.batch_previewed.append((batch_id, deck_scope))
        if self.batch_preview_offer is not None:
            return self.batch_preview_offer
        return ExtractionBatchPreviewOffer(
            message="These are the 7 cards this batch has saved so far.",
            preview_url="/janki-preview/combined-token",
            conflicts=("食べる appears in sources 1 and 2 with different readings",),
        )


def _sidecar(revisions: _FakeBatchRevisions) -> Any:
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    return sidecar


def _manage(sidecar: Any, thread_id: str = "") -> tuple[str, dict[str, Any], Any]:
    request = (
        _followup_message_request(
            thread_id, assistant_module.MANAGE_EXTRACTION_BATCHES_MESSAGE
        )
        if thread_id
        else _message_request(assistant_module.MANAGE_EXTRACTION_BATCHES_MESSAGE)
    )
    status, _headers, body = _post(sidecar, request)
    assert status == 200
    events = _events(body)
    if not thread_id:
        thread_id = next(
            event["thread"]["id"] for event in events if event["type"] == "thread.created"
        )
    widgets = _widget_items(events)
    return thread_id, (widgets[0] if widgets else {}), events


def _batch_action(widget: dict[str, Any], action: str) -> dict[str, Any]:
    clicks = [
        child.get("onClickAction")
        for child in widget["widget"]["children"]
        if child["type"] == "ListViewItem"
    ]
    return next(
        click
        for click in clicks
        if click is not None and click["payload"].get("action") == action
    )


def test_local_batch_desk_lists_numbered_children_without_a_model_turn() -> None:
    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(),))
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, _events_out = _manage(sidecar)
        wire = json.dumps(widget["widget"], ensure_ascii=False)

        assert revisions.batch_listed == 1
        assert revisions.chatted == [], "the batch desk must not make a paid call"
        assert "Lesson pack (3 sources)" in wire
        for numbered in ("1. lesson-a.pdf", "2. lesson-b.pdf", "3. lesson-c.pdf"):
            assert numbered in wire, wire
        assert "committed" in wire
        assert "failed_before_send" in wire
        assert "2 at a time" in wire
        assert "Retry source 3 (lesson-c.pdf)" in wire
    finally:
        sidecar.close()


def test_no_batch_says_so_locally_rather_than_asking_the_model() -> None:
    revisions = _FakeBatchRevisions(batch_choices=())
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, events = _manage(sidecar)
        assert widget == {}
        assert revisions.chatted == []
        assert "No extraction batch" in json.dumps(events, ensure_ascii=False)
    finally:
        sidecar.close()


def test_status_and_preview_stay_available_while_the_thread_is_busy() -> None:
    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(),))
    revisions.chat_release = threading.Event()
    revisions.chat_release.set()
    sidecar = _sidecar(revisions)
    try:
        status, _headers, body = _post(sidecar, _message_request("Tell me about は"))
        assert status == 200
        thread_id = next(
            event["thread"]["id"]
            for event in _events(body)
            if event["type"] == "thread.created"
        )

        entered = threading.Event()
        release = threading.Event()
        revisions.chat_entered = entered
        revisions.chat_release = release
        blocked: list[int] = []

        def occupy() -> None:
            blocked.append(
                _post(sidecar, _followup_message_request(thread_id, "And が?"))[0]
            )

        worker = threading.Thread(target=occupy)
        worker.start()
        try:
            assert entered.wait(3), "the fake chat never started"

            _same, widget, _seen = _manage(sidecar, thread_id)
            assert revisions.batch_listed == 1, "status works during a busy thread"

            preview_events = _events(
                _post(
                    sidecar,
                    _action_request(
                        thread_id, widget, _batch_action(widget, "preview")
                    ),
                )[2]
            )
            preview_wire = json.dumps(preview_events, ensure_ascii=False)
            assert revisions.batch_previewed == [(BATCH_ID, "")]
            assert "/janki-preview/combined-token" in preview_wire
            assert "食べる appears in sources 1 and 2" in preview_wire

            resume_wire = json.dumps(
                _events(
                    _post(
                        sidecar,
                        _action_request(
                            thread_id, widget, _batch_action(widget, "resume")
                        ),
                    )[2]
                ),
                ensure_ascii=False,
            )
            assert "Wait for this thread's current operation to finish" in resume_wire
            assert revisions.batch_resumed == [], "a busy thread may not dispatch"

            ordinary = json.dumps(
                _events(
                    _post(sidecar, _followup_message_request(thread_id, "And を?"))[2]
                ),
                ensure_ascii=False,
            )
            assert "Wait for this thread's current operation to finish" in ordinary
        finally:
            release.set()
            worker.join(5)
        assert blocked == [200]
        assert len(revisions.chatted) == 2, "no extra paid chat happened"
    finally:
        sidecar.close()


def test_resume_uses_the_recorded_authority_and_reports_numbered_progress() -> None:
    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        events = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _batch_action(widget, "resume")),
            )[2]
        )
        wire = json.dumps(events, ensure_ascii=False)

        assert revisions.batch_resumed == [BATCH_ID]
        assert revisions.chatted == [], "resume must not buy a new assistant turn"
        assert revisions.extraction_prepared == [], "resume must not replan"
        assert "Source 1 of 3: running" in wire, wire
        assert "Batch continued under its recorded authority" in wire
    finally:
        sidecar.close()


def test_a_resume_capability_cannot_be_replayed_or_reused_cross_thread() -> None:
    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        resume = _batch_action(widget, "resume")
        assert _post(sidecar, _action_request(thread_id, widget, resume))[0] == 200
        assert revisions.batch_resumed == [BATCH_ID]

        replayed = json.dumps(
            _events(_post(sidecar, _action_request(thread_id, widget, resume))[2]),
            ensure_ascii=False,
        )
        assert "missing, stale, already used" in replayed
        assert revisions.batch_resumed == [BATCH_ID]

        other_id, other_widget, _more = _manage(sidecar)
        assert other_id != thread_id
        stolen = json.dumps(
            _events(
                _post(
                    sidecar,
                    _action_request(
                        thread_id, other_widget, _batch_action(other_widget, "resume")
                    ),
                )[2]
            ),
            ensure_ascii=False,
        )
        # Refused before the handler even sees it: the item store will not hand
        # one thread a widget that belongs to another.
        assert '"type": "error"' in stolen
        assert revisions.batch_resumed == [BATCH_ID]
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    "tamper",
    [
        {"child_indices": [1]},
        {"child_indices": [3, 4]},
        {"batch_id": OTHER_BATCH_ID},
        {"action": "forget"},
    ],
)
def test_tampered_membership_or_action_is_refused_before_any_core_call(
    tamper: dict[str, Any],
) -> None:
    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        retry = _batch_action(widget, "retry")
        payload = dict(retry["payload"])
        payload.update(tamper)
        refused = json.dumps(
            _events(
                _post(
                    sidecar,
                    _action_request(
                        thread_id,
                        widget,
                        {"type": retry["type"], "payload": payload},
                    ),
                )[2]
            ),
            ensure_ascii=False,
        )

        assert "missing, stale, already used" in refused
        assert revisions.batch_retry_prepared == []
        assert revisions.batch_resumed == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_selected_retry_prepares_one_fresh_confirmation_naming_the_discards() -> None:
    revisions = _FakeBatchRevisions(
        batch_choices=(batch_choice(),),
        execution_result=RevisionExecution(
            message="Retry saved 5 proposals from lesson-c.pdf.",
            finish=None,
            complete=True,
        ),
        execution_progress=(),
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        prepared = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _batch_action(widget, "retry")),
            )[2]
        )
        [confirmation] = _widget_items(prepared)
        confirm_wire = json.dumps(confirmation["widget"], ensure_ascii=False)

        assert revisions.batch_retry_prepared == [(BATCH_ID, (3,), "")]
        assert revisions.executed == [], "preparing a retry sends nothing"
        assert "Discard the old failed model call op-3" in confirm_wire
        assert "recorded evidence is lost" in confirm_wire
        assert "Send source 3 again as one new paid model call" in confirm_wire

        done = json.dumps(
            _events(
                _post(sidecar, _confirmation_request(thread_id, confirmation))[2]
            ),
            ensure_ascii=False,
        )
        assert len(revisions.executed) == 1
        assert "Retry saved 5 proposals" in done
    finally:
        sidecar.close()


def test_a_batch_whose_children_are_not_retryable_offers_no_retry_button() -> None:
    """Retry eligibility is the core's; the desk only shows what it was given."""

    revisions = _FakeBatchRevisions(batch_choices=(batch_choice(with_retry=False),))
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, _seen = _manage(sidecar)
        payloads = [
            child.get("onClickAction", {}).get("payload", {}).get("action")
            for child in widget["widget"]["children"]
            if child["type"] == "ListViewItem"
        ]
        assert "retry" not in payloads
        assert "preview" in payloads and "resume" in payloads
    finally:
        sidecar.close()


def test_preview_without_a_store_says_so_instead_of_claiming_a_review() -> None:
    revisions = _FakeBatchRevisions(
        batch_choices=(batch_choice(),),
        batch_preview_offer=ExtractionBatchPreviewOffer(
            message=(
                "The combined cards were rendered, but this workbench has no "
                "local preview address to serve them from."
            ),
            preview_url=None,
            conflicts=(),
        ),
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        wire = json.dumps(
            _events(
                _post(
                    sidecar,
                    _action_request(
                        thread_id, widget, _batch_action(widget, "preview")
                    ),
                )[2]
            ),
            ensure_ascii=False,
        )
        assert "no local preview address" in wire
        assert "reviewed" not in wire.casefold()
    finally:
        sidecar.close()


def test_the_shell_offers_local_batch_management_as_a_starter() -> None:
    revisions = _FakeBatchRevisions()
    sidecar = _sidecar(revisions)
    try:
        script = _application_javascript(sidecar.server)
        assert json.dumps(assistant_module.MANAGE_EXTRACTION_BATCHES_MESSAGE) in script
        assert "Manage extraction batches" in script
    finally:
        sidecar.close()


def test_batch_progress_labels_name_a_numbered_child_and_its_state() -> None:
    accepted = assistant_module._batch_progress_label
    for label in ("Source 1 of 3: running", "Source 12 of 12: outcome_unknown"):
        assert accepted(label) == label
    for label in ("Reading the source", "Source one of three: running", "running"):
        with pytest.raises(ValueError):
            accepted(label)


def _outcome(*states: str, resume_available: bool = False, resume_refusal: str = ""):
    """A core-shaped outcome double for the helper's control decisions."""

    from types import SimpleNamespace

    children = tuple(
        SimpleNamespace(
            index=index,
            operation_id=f"op-{index}",
            source=Path(f"data/inbox/lesson-{index}.pdf"),
            state=state,
            staging_path=None,
            error="",
            bookkeeping_complete=True,
            records=None,
        )
        for index, state in enumerate(states, start=1)
    )
    return SimpleNamespace(
        batch_id=BATCH_ID,
        children=children,
        concurrency_limit=2,
        committed_count=sum(1 for state in states if state == "committed"),
        failed_count=0,
        unknown_count=0,
        pending_count=0,
        resume_available=resume_available,
        resume_refusal=resume_refusal,
    )


@pytest.mark.parametrize(
    "state",
    [
        "failed_before_send",
        "canceled_before_send",
        "expired",
        "result_captured",
        "outcome_unknown",
    ],
)
def test_every_retry_eligible_state_gets_a_retry_control(state: str) -> None:
    choice = assistant_batch_surface._choice(_outcome(state))

    retries = [option for option in choice.actions if option.action == "retry"]
    assert [option.child_indices for option in retries] == [(1,)]


@pytest.mark.parametrize(
    "state",
    ["authorized", "dispatching", "running", "committed", "retired", "unreserved"],
)
def test_no_retry_control_for_a_live_settled_or_unreserved_child(state: str) -> None:
    choice = assistant_batch_surface._choice(_outcome(state))

    assert [option for option in choice.actions if option.action == "retry"] == []


def test_a_saved_reply_offers_recovery_before_the_discarding_retry() -> None:
    choice = assistant_batch_surface._choice(
        _outcome("result_captured", resume_available=True)
    )

    assert choice.actions[0].action == "resume"
    [retry] = [option for option in choice.actions if option.action == "retry"]
    assert "discard" in retry.label.casefold()


def test_an_unknown_outcome_names_that_it_is_unknown() -> None:
    choice = assistant_batch_surface._choice(_outcome("outcome_unknown"))

    [retry] = [option for option in choice.actions if option.action == "retry"]
    assert "unknown" in retry.label.casefold()


def test_the_core_refusal_to_continue_is_shown_not_swallowed() -> None:
    choice = assistant_batch_surface._choice(
        _outcome("running", resume_refusal="a child is still in flight")
    )

    assert "a child is still in flight" in choice.summary
    assert [option for option in choice.actions if option.action == "resume"] == []


def _retry_plan_disclosure(monkeypatch: pytest.MonkeyPatch, *states: str):
    from types import SimpleNamespace

    plan = SimpleNamespace(
        batch_id=BATCH_ID,
        children=(
            SimpleNamespace(index=1, source=Path("data/inbox/lesson-1.pdf")),
        ),
        concurrency_limit=2,
        model="claude-opus-5",
        provider="claude-code",
        fingerprint="9" * 64,
        retry_of=BATCH_ID,
        discards=tuple(
            SimpleNamespace(
                operation_id=f"op-{index}",
                source_file=f"lesson-{index}.pdf",
                state=state,
                request_fp="a" * 64,
                model="claude-opus-5",
                detail="",
            )
            for index, state in enumerate(states, start=1)
        ),
    )
    monkeypatch.setattr(
        assistant_batch_surface,
        "core",
        lambda: SimpleNamespace(plan_extraction_batch_retry=lambda *a, **k: plan),
    )
    prepared = assistant_batch_surface.plan_batch_retry(object(), BATCH_ID, (1,))
    return "\n".join((*prepared.effects, *prepared.disclosures))


def test_discarding_a_saved_reply_is_disclosed(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = _retry_plan_disclosure(monkeypatch, "result_captured")

    assert "op-1" in rendered
    assert "saved" in rendered.casefold()


def test_an_unknown_outcome_retry_discloses_unknown_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _retry_plan_disclosure(monkeypatch, "outcome_unknown")

    assert "op-1" in rendered
    assert "unknown" in rendered.casefold()
    assert "cost" in rendered.casefold()


def test_a_confirmed_batch_streams_numbered_child_progress() -> None:
    revisions = _FakeBatchRevisions(
        batch_choices=(batch_choice(),),
        execution_progress=("Source 1 of 2: running", "Source 2 of 2: committed"),
        execution_result=RevisionExecution(
            message="Retry saved 5 proposals.", finish=None, complete=True
        ),
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        prepared = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _batch_action(widget, "retry")),
            )[2]
        )
        [confirmation] = _widget_items(prepared)
        wire = json.dumps(
            _events(_post(sidecar, _confirmation_request(thread_id, confirmation))[2]),
            ensure_ascii=False,
        )

        assert "Source 1 of 2: running" in wire, wire
        assert "Source 2 of 2: committed" in wire, wire
    finally:
        sidecar.close()


def test_a_malformed_batch_progress_label_is_still_refused() -> None:
    revisions = _FakeBatchRevisions(
        batch_choices=(batch_choice(),),
        execution_progress=("Source one: nope",),
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        prepared = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _batch_action(widget, "retry")),
            )[2]
        )
        [confirmation] = _widget_items(prepared)
        wire = json.dumps(
            _events(_post(sidecar, _confirmation_request(thread_id, confirmation))[2]),
            ensure_ascii=False,
        )

        assert "Source one: nope" not in wire
    finally:
        sidecar.close()


@dataclass
class _RealResumeRevisions(_FakeBatchRevisions):
    """Fake listing, real resume: the translation under test is the adapter's."""

    adapter: Any = None

    def resume_extraction_batch(
        self, *, batch_id: str, progress: Any
    ) -> ExtractionBatchResumption:
        self.batch_resumed.append(batch_id)
        return self.adapter.resume_extraction_batch(
            batch_id=batch_id, progress=progress
        )


def _strings(value: Any) -> Any:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


@pytest.mark.parametrize("stage", ["status", "resume"])
def test_a_core_resume_refusal_reaches_the_owner_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """A real core refusal must arrive as its own words, not a stream failure."""

    from types import SimpleNamespace

    config = _config(tmp_path)
    real = _adapter(config)
    if stage == "resume":
        # Let the read-only status succeed so the refusal comes from the core's
        # own resume, which finds no manifest for this batch.
        monkeypatch.setattr(
            extraction_batch,
            "extraction_batch_status",
            lambda _config, _batch_id: SimpleNamespace(children=(1, 2)),
        )
        with pytest.raises(JankiError) as caught:
            extraction_batch.resume_extraction_batch(config, BATCH_ID)
    else:
        with pytest.raises(JankiError) as caught:
            extraction_batch.extraction_batch_status(config, BATCH_ID)
    reason = str(caught.value)
    assert reason

    revisions = _RealResumeRevisions(batch_choices=(batch_choice(),), adapter=real)
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        events = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _batch_action(widget, "resume")),
            )[2]
        )
        shown = list(_strings(events))

        assert any(reason in text for text in shown), shown
        assert not any("The assistant stream failed" in text for text in shown), shown
        # A refusal is not a dispatch: nothing was written or sent.
        assert not list(config.staging_dir.glob("*")) if config.staging_dir.exists() else True
        assert not config.operations_file.exists()
    finally:
        sidecar.close()
