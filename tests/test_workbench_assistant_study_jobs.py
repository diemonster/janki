"""The study-job desk, and what a running batch is allowed to do to chat.

Three rules, all local and all faked at the callback boundary:

1. **Reads answer while a batch runs.** A job's status, its saved cards and
   what one captured reply holds are exactly what an owner needs while their
   own confirmed batch is in flight, and none of them spends anything. They are
   routed ahead of the journal and busy guards, like the batch desk already is.
2. **Ordinary prose still gets the deterministic busy reply and dispatches
   nothing.** A blocking operation stops a paid turn; nothing is queued.
3. **Only the batch-blocked case loses the generic warning.** A running
   confirmed batch gets a reply naming the job, the batch, k of n settled,
   what is in flight and the local controls. Every unrelated unsettled call
   keeps the recovery desk it already had.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from test_study_job_surfaces import _deck, _project, _source
from test_workbench_assistant import (
    SESSION_TOKEN,
    _action_request,
    _assistant_deck_choices,
    _deck_selector_action,
    _events,
    _FakeRevisions,
    _followup_message_request,
    _message_request,
    _post,
    _start_deck_selector,
    _widget_items,
    create_assistant_sidecar,
)
from test_workbench_assistant_integration import _adapter

from japanese_anki.application import study_job
from japanese_anki.workbench import assistant as assistant_module
from japanese_anki.workbench.assistant import (
    ExtractionBatchPreviewOffer,
    OperationActionChoice,
    OperationChoice,
    StudyJobActionChoice,
    StudyJobChoice,
    StudyJobStartChoice,
)

JOB_ID = "11111111-1111-4111-8111-111111111111"
OPERATION_ID = "op-captured-1"


def job_choice(
    *,
    with_capture: bool = True,
    with_resume: bool = True,
    with_parts: bool = True,
) -> StudyJobChoice:
    actions = [
        StudyJobActionChoice(action="status", label="Where this job stands"),
        StudyJobActionChoice(action="preview", label="Look at the saved cards"),
    ]
    if with_parts:
        actions.append(
            StudyJobActionChoice(
                action="parts", label="Choose the parts of verbs.pdf"
            )
        )
        actions.append(
            StudyJobActionChoice(
                action="exclude-part",
                label="Leave verbs-p3.png out of this job's next batch",
                target="verbs-p3.png",
            )
        )
        actions.append(
            StudyJobActionChoice(
                action="include-part",
                label="Put verbs-p4.png back in this job's next batch",
                target="verbs-p4.png",
            )
        )
    if with_capture:
        actions.append(
            StudyJobActionChoice(
                action="inspect-capture",
                label="Read the saved reply for verbs-p3.png",
                operation_id=OPERATION_ID,
            )
        )
    if with_resume:
        actions.append(
            StudyJobActionChoice(
                action="resume", label="Continue what this job already bought"
            )
        )
    return StudyJobChoice(
        job_id=JOB_ID,
        label="verbs.pdf → data/decks/201-verbs.yaml",
        summary="18 part(s) published, 16 of 18 source(s) settled",
        actions=tuple(actions),
    )


def start_choice(source_name: str = "verbs.pdf") -> StudyJobStartChoice:
    return StudyJobStartChoice(
        source_name=source_name,
        label=f"Start a study job over {source_name}",
        summary=f"{source_name} → data/decks/potential-practice.yaml",
    )


@dataclass
class _FakeStudyJobRevisions(_FakeRevisions):
    """The existing fake plus exactly the local study-job callbacks."""

    job_choices: tuple[StudyJobChoice, ...] = ()
    start_choices: tuple[StudyJobStartChoice, ...] = ()
    job_listed: int = 0
    starts_listed: list[str] = field(default_factory=list)
    jobs_opened: list[tuple[str, str]] = field(default_factory=list)
    editors_opened: list[str] = field(default_factory=list)
    parts_selected: list[tuple[str, str, bool]] = field(default_factory=list)
    job_status_read: list[str] = field(default_factory=list)
    job_previewed: list[tuple[str, str]] = field(default_factory=list)
    captures_inspected: list[str] = field(default_factory=list)
    jobs_resumed: list[str] = field(default_factory=list)

    def list_study_job_choices(self) -> tuple[StudyJobChoice, ...]:
        self.job_listed += 1
        return self.job_choices

    def list_study_job_starts(
        self, *, deck_scope: str
    ) -> tuple[StudyJobStartChoice, ...]:
        self.starts_listed.append(deck_scope)
        return self.start_choices if deck_scope else ()

    def open_study_job_over_deck(self, *, source_name: str, deck_scope: str) -> str:
        self.jobs_opened.append((source_name, deck_scope))
        return f"Opened study job over {source_name}, writing into {deck_scope}."

    def open_study_job_source_part_editor(self, *, job_id: str) -> str:
        self.editors_opened.append(job_id)
        return f"[Open the region editor](/editor/{job_id[:8]})"

    def save_study_job_part_selection(
        self, *, job_id: str, part_name: str, include: bool
    ) -> str:
        self.parts_selected.append((job_id, part_name, include))
        return (
            f"Study job {job_id[:8]}'s next batch "
            + ("covers" if include else "leaves out")
            + f" {part_name}."
        )

    def study_job_status_text(self, *, job_id: str) -> str:
        self.job_status_read.append(job_id)
        return f"Study job {job_id}: 16 of 18 settled."

    def render_study_job_preview(
        self, *, job_id: str, deck_scope: str
    ) -> ExtractionBatchPreviewOffer:
        self.job_previewed.append((job_id, deck_scope))
        return ExtractionBatchPreviewOffer(
            message="These are the 47 cards this job has saved.",
            preview_url="/janki-preview/job-token",
            conflicts=(),
        )

    def inspect_capture_proposals_text(self, *, operation_id: str) -> str:
        self.captures_inspected.append(operation_id)
        return f"Captured reply for model call {operation_id}: 1 valid proposal."

    def resume_study_job(self, *, job_id: str, progress: Any) -> str:
        self.jobs_resumed.append(job_id)
        progress("Preparing pages")
        return f"Study job {job_id} continued under its recorded authority."


def _sidecar(revisions: _FakeStudyJobRevisions) -> Any:
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
            thread_id, assistant_module.MANAGE_STUDY_JOBS_MESSAGE
        )
        if thread_id
        else _message_request(assistant_module.MANAGE_STUDY_JOBS_MESSAGE)
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


def _job_action(
    widget: dict[str, Any], action: str, target: str = ""
) -> dict[str, Any]:
    clicks = [
        child.get("onClickAction")
        for child in widget["widget"]["children"]
        if child["type"] == "ListViewItem"
    ]
    return next(
        click
        for click in clicks
        if click is not None
        and click["payload"].get("action") == action
        and click["payload"].get("target", "") == target
    )


def _start_action(widget: dict[str, Any], source_name: str) -> dict[str, Any]:
    clicks = [
        child.get("onClickAction")
        for child in widget["widget"]["children"]
        if child["type"] == "ListViewItem"
    ]
    return next(
        click
        for click in clicks
        if click is not None and click["payload"].get("source_name") == source_name
    )


def _blocking_batch_operation() -> OperationChoice:
    return OperationChoice(
        operation_id="op-running-1",
        kind="extract",
        state="running",
        source_name="verbs-p3.png",
        model="claude-opus-5",
        authorized_at="2026-09-09T10:07:31+00:00",
        blocks_spending=True,
        money_may_have_been_spent=True,
        has_captured_reply=False,
        has_response_spool=False,
        cleanup_pending=False,
        actions=(OperationActionChoice(action="end", label="End this call"),),
    )


def test_the_study_job_desk_answers_without_a_model_turn() -> None:
    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, _seen = _manage(sidecar)
        wire = json.dumps(widget["widget"], ensure_ascii=False)

        assert revisions.job_listed == 1
        assert revisions.chatted == [], "the study job desk must not make a paid call"
        assert "verbs.pdf → data/decks/201-verbs.yaml" in wire
        assert "16 of 18 source(s) settled" in wire
        assert "Where this job stands" in wire
    finally:
        sidecar.close()


def test_no_study_job_says_so_locally_rather_than_asking_the_model() -> None:
    revisions = _FakeStudyJobRevisions(job_choices=())
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, events = _manage(sidecar)
        assert widget == {}
        assert revisions.chatted == []
        assert "No study job" in json.dumps(events, ensure_ascii=False)
    finally:
        sidecar.close()


def test_status_preview_and_capture_inspection_answer_while_a_batch_runs() -> None:
    """The three reads §9.4 names, all routed ahead of the busy guard."""

    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
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
            assert revisions.job_listed == 1, "the desk works during a busy thread"

            for action, recorded in (
                ("status", revisions.job_status_read),
                ("preview", revisions.job_previewed),
                ("inspect-capture", revisions.captures_inspected),
            ):
                seen = _events(
                    _post(
                        sidecar,
                        _action_request(thread_id, widget, _job_action(widget, action)),
                    )[2]
                )
                assert recorded, f"{action} was refused while the thread was busy"
                assert "Wait for this thread" not in json.dumps(seen), action

            # Resume is not a read: it stays blocked while that thread's worker
            # is alive, exactly as the batch desk's resume does.
            refused = _events(
                _post(
                    sidecar,
                    _action_request(thread_id, widget, _job_action(widget, "resume")),
                )[2]
            )
            assert "Wait for this thread" in json.dumps(refused)
            assert revisions.jobs_resumed == []
        finally:
            release.set()
            worker.join(5)
        assert blocked == [200]
    finally:
        sidecar.close()


def test_a_running_confirmed_batch_replaces_the_generic_attention_warning() -> None:
    """The one branch S4 takes over, and only for this case."""

    revisions = _FakeStudyJobRevisions(
        job_choices=(job_choice(),),
        operation_choices=(_blocking_batch_operation(),),
        blocking_batch_text=(
            "Janki did not make a new model call: a batch you confirmed is "
            "running.\n\nStudy job 11111111: batch 22222222 — 16 of 18 "
            "source(s) settled, 2 in flight or unsent.\n\nThese stay available "
            "right now and cost nothing: **Manage study jobs**."
        ),
    )
    sidecar = _sidecar(revisions)
    try:
        seen = _events(_post(sidecar, _message_request("Write me some cards"))[2])
        wire = json.dumps(seen, ensure_ascii=False)

        assert revisions.blocking_batch_asked == [("op-running-1",)]
        assert "Earlier model call needs attention" not in wire
        assert "A confirmed extraction batch is running" in wire
        assert "16 of 18" in wire
        assert "in flight" in wire
        assert "Manage study jobs" in wire
        # Nothing was dispatched and no recovery desk was offered for work
        # that is proceeding normally.
        assert revisions.chatted == []
        assert _widget_items(seen) == []
    finally:
        sidecar.close()


def test_an_unrelated_unsettled_call_keeps_its_recovery_desk() -> None:
    """The adapter answers "" for anything that is not a running batch child."""

    revisions = _FakeStudyJobRevisions(
        job_choices=(job_choice(),),
        operation_choices=(_blocking_batch_operation(),),
        blocking_batch_text="",
    )
    sidecar = _sidecar(revisions)
    try:
        seen = _events(_post(sidecar, _message_request("Write me some cards"))[2])
        wire = json.dumps(seen, ensure_ascii=False)

        assert revisions.blocking_batch_asked == [("op-running-1",)]
        assert "Earlier model call needs attention" in wire
        assert revisions.chatted == []
        assert _widget_items(seen), "the operation recovery desk must still render"
    finally:
        sidecar.close()


def test_the_desk_opens_the_region_editor_for_the_job_whose_row_was_clicked() -> None:
    """The owner route that binds a publication to a job.

    Nothing else in the Assistant carries a job into the region editor, so the
    control has to reach the callback with this job's own id — a publication
    made inside that editor is what records this job's source-part intent.
    """

    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        assert "Choose the parts of verbs.pdf" in json.dumps(widget, ensure_ascii=False)

        seen = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _job_action(widget, "parts")),
            )[2]
        )

        assert revisions.editors_opened == [JOB_ID]
        assert "/editor/" in json.dumps(seen, ensure_ascii=False)
        assert revisions.chatted == [], "opening an editor is not a model call"
    finally:
        sidecar.close()


def test_the_desk_saves_a_part_selection_and_stays_usable_for_the_next_part() -> None:
    """Which parts a job covers is the owner's reversible local preference.

    Choosing a subset takes several clicks, so this control is deliberately not
    consumed by the first one, and each click names the exact part it was
    rendered for.
    """

    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)

        excluded = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    widget,
                    _job_action(widget, "exclude-part", "verbs-p3.png"),
                ),
            )[2]
        )
        included = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    widget,
                    _job_action(widget, "include-part", "verbs-p4.png"),
                ),
            )[2]
        )

        assert revisions.parts_selected == [
            (JOB_ID, "verbs-p3.png", False),
            (JOB_ID, "verbs-p4.png", True),
        ]
        assert "leaves out verbs-p3.png" in json.dumps(excluded, ensure_ascii=False)
        assert "covers verbs-p4.png" in json.dumps(included, ensure_ascii=False)
        assert revisions.chatted == []
    finally:
        sidecar.close()


def test_a_part_control_pointed_at_another_part_is_refused() -> None:
    """The control binds the exact part it was rendered for, not any part."""

    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        click = _job_action(widget, "exclude-part", "verbs-p3.png")
        tampered = json.loads(json.dumps(click))
        tampered["payload"]["target"] = "verbs-p9.png"

        seen = _events(_post(sidecar, _action_request(thread_id, widget, tampered))[2])

        assert "This study job control is missing, stale" in json.dumps(seen)
        assert revisions.parts_selected == []
    finally:
        sidecar.close()


def test_the_desk_offers_to_start_a_job_over_a_source_against_the_focused_deck() -> None:
    """§1 step 2: reusing a deck is a selection, and Create/Open is the owner's.

    The destination is the deck this thread already focuses on, bound into the
    control when it is rendered, so the browser cannot post a different one.
    """

    revisions = _FakeStudyJobRevisions(
        job_choices=(), start_choices=(start_choice(),)
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id, selector, _deck_selector_action(selector, "potential")
            ),
        )
        _same, widget, _seen = _manage(sidecar, thread_id)
        assert revisions.starts_listed[-1] == "data/decks/potential-practice.yaml"

        seen = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _start_action(widget, "verbs.pdf")),
            )[2]
        )

        assert revisions.jobs_opened == [
            ("verbs.pdf", "data/decks/potential-practice.yaml")
        ]
        assert "Opened study job over verbs.pdf" in json.dumps(seen, ensure_ascii=False)
        assert revisions.chatted == [], "opening a job is a local write, not a turn"
    finally:
        sidecar.close()


def test_every_preserved_source_gets_a_start_control_however_many_there_are(
    tmp_path: Any,
) -> None:
    """A project's parts must not push its own documents off the desk.

    Published parts are ordinary intake, so a prepared 25-page PDF puts 25
    more sources in the catalogue — and each one sorts *before* the document
    they came from, because `-` precedes `.`. The desk lists what the project
    holds, so the parent is still there to start a job over.

    The real adapter answers here, and the control it produces is rendered and
    clicked through the real widget: a start that is not rendered is not a
    route, however long the returned list is.
    """

    config = _project(tmp_path)
    _source(config)
    for page in range(1, 26):
        (config.scan_inbox / f"verbs--p{page:03d}-r5e5e5e5e-01.png").write_bytes(
            b"\x89PNG published part"
        )
    deck = _deck(config)
    adapter = _adapter(config, deck)
    scope = deck.absolute().relative_to(config.root.absolute()).as_posix()

    offered = adapter.list_study_job_starts(deck_scope=scope)
    assert len(offered) == 26
    assert "verbs.pdf" in {start.source_name for start in offered}

    revisions = _FakeStudyJobRevisions(job_choices=())
    revisions.list_study_job_starts = adapter.list_study_job_starts
    revisions.open_study_job_over_deck = adapter.open_study_job_over_deck
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=adapter.deck_choices,
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _chosen = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id, selector, _deck_selector_action(selector, "test-deck-id")
            ),
        )
        _same, widget, _listed = _manage(sidecar, thread_id)

        seen = _events(
            _post(
                sidecar,
                _action_request(thread_id, widget, _start_action(widget, "verbs.pdf")),
            )[2]
        )

        assert "Opened study job" in json.dumps(seen, ensure_ascii=False)
        job_ids = study_job.list_study_jobs(config)
        assert len(job_ids) == 1
        job = study_job.load_study_job(config, job_ids[0])
        assert job.header.parent_source_name == "verbs.pdf"
        assert job.header.deck_path == scope
        assert revisions.chatted == []
    finally:
        sidecar.close()


def test_a_start_control_pointed_at_another_source_is_refused() -> None:
    """A rendered start binds its exact source; a posted one is not authority."""

    revisions = _FakeStudyJobRevisions(
        job_choices=(), start_choices=(start_choice(),)
    )
    sidecar = _sidecar(revisions)
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id, selector, _deck_selector_action(selector, "potential")
            ),
        )
        _same, widget, _seen = _manage(sidecar, thread_id)
        click = _start_action(widget, "verbs.pdf")
        tampered = json.loads(json.dumps(click))
        tampered["payload"]["source_name"] = "somebody-elses.pdf"

        seen = _events(_post(sidecar, _action_request(thread_id, widget, tampered))[2])

        assert "This study job control is missing, stale" in json.dumps(seen)
        assert revisions.jobs_opened == []
    finally:
        sidecar.close()


def test_a_project_with_no_job_and_no_focused_deck_says_how_to_start_one() -> None:
    """The empty desk names the one thing that has to happen first."""

    revisions = _FakeStudyJobRevisions(job_choices=(), start_choices=(start_choice(),))
    sidecar = _sidecar(revisions)
    try:
        _thread_id, widget, events = _manage(sidecar)
        wire = json.dumps(events, ensure_ascii=False)

        assert widget == {}
        assert revisions.starts_listed == [""]
        assert "Focus the destination deck" in wire
        assert revisions.chatted == []
    finally:
        sidecar.close()


def test_a_study_job_control_from_another_widget_is_refused() -> None:
    """An anti-replay guard, not authority: a stale control acts on nothing."""

    revisions = _FakeStudyJobRevisions(job_choices=(job_choice(),))
    sidecar = _sidecar(revisions)
    try:
        thread_id, widget, _seen = _manage(sidecar)
        click = _job_action(widget, "status")
        tampered = json.loads(json.dumps(click))
        tampered["payload"]["job_id"] = "22222222-2222-4222-8222-222222222222"

        seen = _events(_post(sidecar, _action_request(thread_id, widget, tampered))[2])
        assert "This study job control is missing, stale" in json.dumps(seen)
        assert revisions.job_status_read == []
    finally:
        sidecar.close()
