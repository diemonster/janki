"""The isolated conversational ChatKit workbench controller.

Every callback is fake. These tests exercise the ChatKit wire protocol and the
loopback HTTP boundary without contacting OpenAI or another provider.
"""

from __future__ import annotations

import ast
import asyncio
import http.client
import io
import json
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from chatkit.icons import IconName
from pydantic import TypeAdapter

from japanese_anki.application import assistant_agent
from japanese_anki.workbench import assistant as assistant_module
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.assistant import (
    AssistantDeckChoice,
    AssistantRequestContext,
    ChatReply,
    OperationActionChoice,
    OperationChoice,
    RevisionConfirmation,
    RevisionExampleReview,
    RevisionExecution,
    RevisionFinishConfirmation,
    RevisionFinishExecution,
    RevisionFinishReview,
    RevisionPlan,
    RevisionRecordReview,
    RevisionRefusal,
    ScopedMemoryStore,
    SourceExtractionConfirmation,
    SourceExtractionExecution,
    SourceExtractionPlan,
    StagedContentFinishReview,
    create_assistant_core,
)
from japanese_anki.workbench.assistant_http import (
    create_assistant_sidecar as _create_assistant_sidecar,
)

SESSION_TOKEN = "assistant-session-token-000000000000"
REQUEST_FINGERPRINT = "0123456789abcdef" * 4
FINISH_FINGERPRINT = "fedcba9876543210" * 4
PACKAGE_FINGERPRINT = "abcdef0123456789" * 4


def _action_plan(
    *,
    target: str = "data/decks/potential-practice.yaml",
    progress_label: str = "Preparing revision",
) -> RevisionPlan:
    return RevisionPlan(
        request_fingerprint=REQUEST_FINGERPRINT,
        target=target,
        effects=(
            "revise the selected cards",
            "stage the proposal without changing the deck",
        ),
        disclosures=("The confirmed action calls Claude through Claude Code.",),
        confirm_label="Confirm exact revision",
        progress_label=progress_label,
    )


def _finish_review(*, target: str = "data/decks/potential.yaml") -> RevisionFinishReview:
    current = (
        RevisionExampleReview(
            register="polite",
            japanese="今は遊べません。",
            furigana="今[いま]は 遊[あそ]べません。",
            english="I cannot play now.",
        ),
        RevisionExampleReview(
            register="casual",
            japanese="今日は遊べない。",
            furigana="今日[きょう]は 遊[あそ]べない。",
            english="I can't play today.",
        ),
    )
    proposed = (
        RevisionExampleReview(
            register="polite",
            japanese="明日は遊べます。",
            furigana="明日[あした]は 遊[あそ]べます。",
            english="I can play tomorrow.",
        ),
        RevisionExampleReview(
            register="casual",
            japanese="今日は遊べる。",
            furigana="今日[きょう]は 遊[あそ]べる。",
            english="I can play today.",
        ),
    )
    return RevisionFinishReview(
        preparation_id="finish-preparation-1",
        request_fingerprint=FINISH_FINGERPRINT,
        target=target,
        current_form_note="Old potential note.",
        proposed_form_note="Potential expresses ability or possibility.",
        records=(
            RevisionRecordReview(
                record_id="word:遊ぶ:あそぶ",
                current_examples=current,
                proposed_examples=proposed,
            ),
        ),
        audio_provider="openai-realtime",
        audio_model="gpt-realtime-1.5",
        audio_access="paid-network",
        audio_total=2,
        audio_current=0,
        audio_recoverable=1,
        audio_provider_required=1,
        output_path="dist/potential.apkg",
        card_count=16,
    )


@dataclass
class _FakeRevisions:
    resolved: list[str] = field(default_factory=list)
    chatted: list[tuple[str, str]] = field(default_factory=list)
    chat_histories: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    executed: list[RevisionConfirmation] = field(default_factory=list)
    finish_executed: list[RevisionFinishConfirmation] = field(default_factory=list)
    extraction_prepared: list[Path] = field(default_factory=list)
    extracted: list[SourceExtractionConfirmation] = field(default_factory=list)
    entered: threading.Event | None = None
    release: threading.Event | None = None
    finished: threading.Event | None = None
    chat_entered: threading.Event | None = None
    chat_release: threading.Event | None = None
    chat_finished: threading.Event | None = None
    operation_entered: threading.Event | None = None
    operation_release: threading.Event | None = None
    finish_entered: threading.Event | None = None
    finish_release: threading.Event | None = None
    finish_finished: threading.Event | None = None
    resolve_block_for: str | None = None
    resolve_entered: threading.Event | None = None
    resolve_release: threading.Event | None = None
    resolve_refusal: str | None = None
    deck_choices: tuple[AssistantDeckChoice, ...] = ()
    chat_reply: ChatReply | None = None
    chat_preview: tuple[str, ...] = ()
    chat_refusal: str | None = None
    execution_result: RevisionExecution | None = None
    execution_progress: tuple[str, ...] = (
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    )
    operation_choices: tuple[OperationChoice, ...] = ()
    operation_prepared: list[tuple[str, str, bool, str]] = field(default_factory=list)
    finish_progress: tuple[str, ...] = (
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    )
    finish_result: RevisionFinishExecution | None = None

    def list_operation_choices(self) -> tuple[OperationChoice, ...]:
        if self.operation_entered is not None:
            self.operation_entered.set()
        if self.operation_release is not None and not self.operation_release.wait(3):
            raise AssertionError("the test never released operation status")
        return self.operation_choices

    def prepare_operation_action(
        self,
        *,
        operation_id: str,
        action: str,
        accept_paid_output_loss: bool,
        deck_scope: str,
    ) -> ChatReply:
        self.operation_prepared.append(
            (operation_id, action, accept_paid_output_loss, deck_scope)
        )
        return ChatReply(
            text=f"Prepared {action} for {operation_id}.",
            action=_action_plan(
                target=f"Paid operation {operation_id}",
                progress_label="Checking an earlier model call",
            ),
            action_instruction=f"{action} paid operation {operation_id}",
        )

    def resolve_deck_selection(self, deck_id: str) -> AssistantDeckChoice:
        self.resolved.append(deck_id)
        if deck_id == self.resolve_block_for:
            if self.resolve_entered is not None:
                self.resolve_entered.set()
            if self.resolve_release is not None and not self.resolve_release.wait(3):
                raise AssertionError("the test never released deck resolution")
        if self.resolve_refusal is not None:
            raise RevisionRefusal(self.resolve_refusal)
        return next(choice for choice in self.deck_choices if choice.deck_id == deck_id)

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Any,
        preview: Any,
    ) -> ChatReply:
        self.chatted.append((deck_scope, message))
        self.chat_histories.append(history)
        for delta in self.chat_preview:
            preview(delta)
        if self.chat_entered is not None:
            self.chat_entered.set()
        if self.chat_release is not None and not self.chat_release.wait(3):
            raise AssertionError("the test never released the fake chat")
        progress("Preparing answer")
        progress("Writing answer")
        progress("Staging proposed changes")
        progress("Saving answer")
        if self.chat_refusal is not None:
            raise RevisionRefusal(self.chat_refusal)
        if self.chat_finished is not None:
            self.chat_finished.set()
        if self.chat_reply is not None:
            return self.chat_reply
        return ChatReply(text=f"Answer about {deck_scope}: {message}")

    def consume_replan_and_execute(
        self,
        confirmation: RevisionConfirmation,
        *,
        progress: Any,
    ) -> RevisionExecution:
        self.executed.append(confirmation)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(3):
            raise AssertionError("the test never released the fake revision")
        for label in self.execution_progress:
            progress(label)
        if self.finished is not None:
            self.finished.set()
        if self.execution_result is not None:
            return self.execution_result
        return RevisionExecution(
            message=("The revision proposal is staged. Nothing has been accepted or applied."),
            finish=_finish_review(target=confirmation.deck_scope),
        )

    def consume_replan_and_finish(
        self,
        confirmation: RevisionFinishConfirmation,
        *,
        progress: Any,
    ) -> RevisionFinishExecution:
        self.finish_executed.append(confirmation)
        if self.finish_entered is not None:
            self.finish_entered.set()
        if self.finish_release is not None and not self.finish_release.wait(3):
            raise AssertionError("the test never released the fake finish")
        for label in self.finish_progress:
            progress(label)
        if self.finish_finished is not None:
            self.finish_finished.set()
        if self.finish_result is not None:
            return self.finish_result
        return RevisionFinishExecution(
            message=("The reviewed revision, example audio, and Anki package are complete."),
            receipt_id=FINISH_FINGERPRINT,
            state="complete",
            target="data/decks/potential.yaml",
            output_path="dist/potential.apkg",
            package_sha256=PACKAGE_FINGERPRINT,
            card_count=16,
        )

    def prepare_source_extraction(
        self,
        *,
        source_path: Path,
        deck_scope: str = "",
    ) -> SourceExtractionPlan:
        self.extraction_prepared.append(source_path)
        return SourceExtractionPlan(
            preparation_id="prepared-extraction-1",
            source_name=source_path.name,
            request_fingerprint=REQUEST_FINGERPRINT,
            target=f"data/staging/{source_path.stem}.yaml",
            effects=(
                f"Send the whole {source_path.name} source to Claude Opus 5",
                "Propose vocabulary cards and grammar in staging for owner review",
            ),
            disclosures=(
                "This is one paid Anthropic API call; Claude Max does not pay for it.",
                "No canonical deck, audio, or build is created by extraction.",
            ),
            confirm_label=f"Send {source_path.name} — paid API call",
        )

    def consume_replan_and_extract(
        self,
        confirmation: SourceExtractionConfirmation,
        *,
        progress: Any,
    ) -> SourceExtractionExecution:
        self.extracted.append(confirmation)
        progress("Preparing pages")
        progress("Reading the source")
        progress("Checking the answer's shape")
        progress("Saving proposals")
        return SourceExtractionExecution(
            message="Extraction staged 12 card proposals for owner review."
        )


def create_assistant_sidecar(
    callbacks: _FakeRevisions,
    *,
    deck_choices: tuple[AssistantDeckChoice, ...] | None = None,
    deck_scope: str | None = None,
    deck_display_name: str | None = None,
    conversation_available: bool = True,
    **kwargs: Any,
) -> Any:
    """Compact test fixture adapter for the selector's project-scoped API."""

    if deck_choices is None:
        if conversation_available:
            assert deck_scope is not None
            deck_choices = (
                AssistantDeckChoice(
                    deck_id="fixture-deck",
                    label=deck_display_name or deck_scope,
                    scope=deck_scope,
                    chat_supported=True,
                    revision_supported=True,
                ),
            )
        else:
            deck_choices = ()
    callbacks.deck_choices = deck_choices
    sidecar = _create_assistant_sidecar(
        callbacks,
        deck_choices=deck_choices,
        **kwargs,
    )
    sidecar.server.test_callbacks = callbacks
    return sidecar


def _message_request(message: str = "Add polite and casual examples") -> dict[str, Any]:
    return {
        "type": "threads.create",
        "params": {
            "input": {
                "content": [{"type": "input_text", "text": message}],
                "attachments": [],
                "quoted_text": None,
                "inference_options": {},
            }
        },
    }


def _followup_message_request(thread_id: str, message: str) -> dict[str, Any]:
    request = _message_request(message)
    request["type"] = "threads.add_user_message"
    request["params"]["thread_id"] = thread_id
    return request


def _attachment_request(
    *,
    name: str = "lesson.pdf",
    size: int,
    mime_type: str = "application/pdf",
) -> dict[str, Any]:
    return {
        "type": "attachments.create",
        "params": {"name": name, "size": size, "mime_type": mime_type},
    }


def _events(payload: bytes) -> list[dict[str, Any]]:
    events = []
    for block in payload.split(b"\n\n"):
        if block:
            assert block.startswith(b"data: "), block
            events.append(json.loads(block.removeprefix(b"data: ")))
    return events


def _request(
    sidecar: Any,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        sidecar.server.server_address[1],
        timeout=3,
    )
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    response_headers = {key.casefold(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, payload


def _post(sidecar: Any, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode("utf-8")
    return _request(
        sidecar,
        "POST",
        sidecar.server.api_path,
        body=body,
        headers={"Content-Type": "application/json", "Origin": sidecar.origin},
    )


def _action_request(
    thread_id: str,
    sender: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "threads.custom_action",
        "params": {
            "thread_id": thread_id,
            "item_id": sender["id"],
            "action": {"type": action["type"], "payload": action["payload"]},
        },
    }


def _widget_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event["item"]
        for event in events
        if event["type"] == "thread.item.done" and event["item"]["type"] == "widget"
    ]


def _confirmation_request(thread_id: str, widget: dict[str, Any]) -> dict[str, Any]:
    action = widget["widget"]["confirm"]["action"]
    return _action_request(thread_id, widget, action)


def _create_chat(
    sidecar: Any,
    message: str = "Add polite and casual examples",
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    thread_id, selector, _selector_events = _start_deck_selector(sidecar)
    selectable = next(
        choice for choice in sidecar.server.deck_choices if choice.revision_supported
    )
    select_status, _select_headers, _select_body = _post(
        sidecar,
        _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, selectable.deck_id),
        ),
    )
    assert select_status == 200
    status, _headers, body = _post(
        sidecar,
        _followup_message_request(thread_id, message),
    )
    assert status == 200
    events = _events(body)
    assistant = next(
        event["item"]
        for event in events
        if event["type"] == "thread.item.done" and event["item"]["type"] == "assistant_message"
    )
    assert _widget_items(events) == []
    return thread_id, assistant, assistant, events


def _create_automatic_action(
    sidecar: Any,
    *,
    focused: bool,
    message: str = "Revise these cards",
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if focused:
        thread_id, selector, _selector_events = _start_deck_selector(sidecar)
        selection = _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, "potential"),
        )
        assert _post(sidecar, selection)[0] == 200
        request = _followup_message_request(thread_id, message)
    else:
        thread_id = ""
        request = _message_request(message)

    status, _headers, body = _post(sidecar, request)

    assert status == 200
    events = _events(body)
    if not focused:
        thread_id = next(
            event["thread"]["id"]
            for event in events
            if event["type"] == "thread.created"
        )
    widgets = _widget_items(events)
    assert len(widgets) == 1
    return thread_id, widgets[0], events


def _create_plan(
    sidecar: Any,
    message: str = "Add polite and casual examples",
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    callbacks = sidecar.server.test_callbacks
    previous = callbacks.chat_reply
    thread_id, selector, _selector_events = _start_deck_selector(sidecar)
    selected = next(
        choice for choice in sidecar.server.deck_choices if choice.revision_supported
    )
    assert _post(
        sidecar,
        _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, selected.deck_id),
        ),
    )[0] == 200
    callbacks.chat_reply = ChatReply(
        text="I prepared the exact requested change.",
        action=RevisionPlan(
            request_fingerprint=REQUEST_FINGERPRINT,
            target=selected.scope,
            effects=(
                "propose one polite and one casual example per selected card",
                "stage the proposal without changing the deck",
            ),
            disclosures=("The confirmed revision is a paid OpenAI API call.",),
        ),
        action_instruction=message,
    )
    try:
        events = _events(
            _post(sidecar, _followup_message_request(thread_id, message))[2]
        )
    finally:
        callbacks.chat_reply = previous
    widgets = _widget_items(events)
    assert len(widgets) == 1
    return thread_id, widgets[0], events


def _create_finish_review(
    sidecar: Any,
    message: str = "Add polite and casual examples",
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    thread_id, revision_widget, _prepare_events = _create_plan(sidecar, message)
    finish_events = _events(
        _post(
            sidecar,
            _confirmation_request(thread_id, revision_widget),
        )[2]
    )
    return thread_id, _widget_items(finish_events)[0], finish_events


def _create_source_extraction_plan(
    sidecar: Any,
    source: bytes,
    *,
    caption: str = "Create card proposals from this source",
    thread_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    status, _headers, body = _post(
        sidecar,
        _attachment_request(size=len(source)),
    )
    assert status == 200
    attachment = json.loads(body)
    upload_path = urlsplit(attachment["upload_descriptor"]["url"]).path
    assert (
        _request(
            sidecar,
            "PUT",
            upload_path,
            body=source,
            headers={"Origin": sidecar.origin},
        )[0]
        == 204
    )
    message = (
        _message_request(caption)
        if thread_id is None
        else _followup_message_request(thread_id, caption)
    )
    message["params"]["input"]["attachments"] = [attachment["id"]]
    send_status, _send_headers, send_body = _post(sidecar, message)
    assert send_status == 200
    events = _events(send_body)
    if thread_id is None:
        thread_id = next(
            event["thread"]["id"]
            for event in events
            if event["type"] == "thread.created"
        )
    return thread_id, _widget_items(events)[0]


def _component_with_id(component: dict[str, Any], component_id: str) -> dict[str, Any]:
    if component.get("id") == component_id:
        return component
    children = component.get("children", [])
    if isinstance(children, dict):
        children = [children]
    for child in children:
        found = _component_with_id(child, component_id)
        if found:
            return found
    return {}


def _assistant_deck_choices() -> tuple[AssistantDeckChoice, ...]:
    return (
        AssistantDeckChoice(
            deck_id="potential",
            label="Brandon Japanese::Genki II::Lesson 13::Potential Practice",
            scope="data/decks/potential-practice.yaml",
            chat_supported=True,
            revision_supported=True,
        ),
        AssistantDeckChoice(
            deck_id="te-form",
            label="Brandon Japanese::Te-form Practice",
            scope="data/decks/teform-drill.yaml",
            chat_supported=True,
            revision_supported=True,
        ),
        AssistantDeckChoice(
            deck_id="lesson-vocabulary",
            label="Brandon Japanese::Genki II::Lesson 13::Vocabulary",
            scope="data/decks/lesson-13-vocabulary.yaml",
            chat_supported=True,
            revision_supported=False,
            unavailable_reason=(
                "Deck revision currently requires a rich conjugation practice deck."
            ),
        ),
    )


def _blocking_operation_choice() -> OperationChoice:
    return OperationChoice(
        operation_id="captured-op",
        kind="assistant_chat",
        state="result_captured",
        source_name="request.json",
        model="claude-opus-5",
        authorized_at="2026-09-02T00:00:00+00:00",
        blocks_spending=True,
        money_may_have_been_spent=True,
        has_captured_reply=True,
        has_response_spool=False,
        cleanup_pending=False,
        actions=(
            OperationActionChoice(
                action="show_reply",
                label="Show exact recovery reply",
            ),
            OperationActionChoice(
                action="forget",
                label="Discard captured reply and forget",
                accept_paid_output_loss=True,
            ),
        ),
    )


def _start_deck_selector(
    sidecar: Any,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    status, _headers, body = _post(
        sidecar,
        _message_request("Choose a deck to focus on"),
    )
    assert status == 200
    events = _events(body)
    thread_id = next(event["thread"]["id"] for event in events if event["type"] == "thread.created")
    widgets = _widget_items(events)
    assert len(widgets) == 1
    return thread_id, widgets[0], events


def _deck_selector_action(widget: dict[str, Any], deck_id: str) -> dict[str, Any]:
    actions = [
        child.get("onClickAction")
        for child in widget["widget"]["children"]
        if child["type"] == "ListViewItem"
    ]
    return next(
        action
        for action in actions
        if action is not None and action["payload"]["deck_id"] == deck_id
    )


def test_unreadable_deck_choice_must_explain_why_focus_is_unavailable() -> None:
    with pytest.raises(
        ValueError,
        match="Every unreadable deck choice must explain why focus is unavailable",
    ):
        create_assistant_core(
            _FakeRevisions(),
            deck_choices=(
                AssistantDeckChoice(
                    deck_id="invalid",
                    label="Invalid",
                    scope="data/decks/invalid.yaml",
                    chat_supported=False,
                    revision_supported=True,
                ),
            ),
        )


def test_local_deck_starter_renders_one_use_selector_without_chat_callback() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, selector, events = _start_deck_selector(sidecar)

        assert revisions.chatted == []
        assert revisions.chat_histories == []
        assert revisions.executed == []
        assert [event for event in events if event["type"] == "progress_update"] == []
        assert [event for event in events if event["type"] == "notice"] == []

        root = selector["widget"]
        assert root["type"] == "ListView"
        assert root["status"]["text"] == "Optional deck focus"
        assert [child["type"] for child in root["children"]] == [
            "ListViewItem",
            "ListViewItem",
            "ListViewItem",
            "ListViewItem",
        ]
        wire = json.dumps(root, ensure_ascii=False)
        assert "All library" in wire
        assert "No deck focus; use the whole Japanese library" in wire
        for choice in _assistant_deck_choices():
            assert choice.label in wire
            assert choice.scope not in wire
        assert _assistant_deck_choices()[-1].unavailable_reason not in wire

        actions = []
        for choice, row in zip(
            _assistant_deck_choices(), root["children"][1:], strict=True
        ):
            if choice.chat_supported:
                actions.append(_deck_selector_action(selector, choice.deck_id))
            else:
                assert "onClickAction" not in row
        assert "Chat + changes" not in wire
        assert "Chat only" not in wire
        assert "Unavailable" not in wire
        assert {action["type"] for action in actions} == {"janki.deck.select"}
        assert {action["handler"] for action in actions} == {"server"}
        assert {action["loadingBehavior"] for action in actions} == {"container"}
        assert {action["streaming"] for action in actions} == {True}
        assert all(set(action["payload"]) == {"capability", "deck_id"} for action in actions)
        capabilities = {action["payload"]["capability"] for action in actions}
        assert len(capabilities) == 1
        assert len(capabilities.pop()) >= 32
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    "message",
    [
        "let's pick new study content",
        "Let’s pick new study content!",
        "switch to a different deck",
        "Choose an active deck.",
    ],
)
def test_natural_deck_switch_request_renders_local_selector_without_paid_chat(
    message: str,
) -> None:
    revisions = _FakeRevisions(operation_choices=(_blocking_operation_choice(),))
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, first_selector, _events_before = _start_deck_selector(sidecar)
        assert _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )[0] == 200

        events = _events(
            _post(sidecar, _followup_message_request(thread_id, message))[2]
        )

        [selector] = _widget_items(events)
        assert selector["widget"]["status"]["text"] == "Optional deck focus"
        assert revisions.chatted == []
        assert [event for event in events if event["type"] == "progress_update"] == []
        assert not any(
            event["type"] == "notice" and event.get("level") == "danger"
            for event in events
        )
    finally:
        sidecar.close()


def test_non_navigation_study_content_question_still_uses_chat() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        message = "What new study content is in this deck?"
        events = _events(_post(sidecar, _message_request(message))[2])

        assert revisions.chatted == [("", message)]
        assert _widget_items(events) == []
    finally:
        sidecar.close()


def test_every_adapter_progress_label_is_one_the_validator_accepts() -> None:
    """A progress label the validator rejects wedges the action that emits it.

    `_validate_plan` refuses an unknown label, so a literal here that drifts
    from `_PROGRESS_LABELS` does not degrade — it fails every use of that
    action. Renaming one of these is exactly when the two fall out of step, so
    the pairing is checked rather than remembered.
    """

    source = Path(assistant_adapter.__file__).read_text(encoding="utf-8")
    emitted = {
        keyword.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "progress_label"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }

    assert emitted, "no literal progress labels found; the scan stopped working"
    accepted = assistant_module._PROGRESS_LABELS
    assert emitted <= accepted, sorted(emitted - accepted)


def test_every_agent_progress_label_is_accepted_on_both_routes() -> None:
    """One agent label set reaches two validators, so both must accept it.

    `run_agent` reports these on the chat route and `recover_agent` reports the
    same ones on the action route, where an unknown label is refused outright —
    so a label accepted by only one of the two wedges the other route.
    """

    source = Path(assistant_agent.__file__).read_text(encoding="utf-8")
    emitted = {
        node.args[1].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_report_progress"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }

    assert emitted, "no literal progress labels found; the scan stopped working"
    for accepted in (
        assistant_module._CHAT_PROGRESS_LABELS,
        assistant_module._PROGRESS_LABELS,
    ):
        assert emitted <= accepted, sorted(emitted - accepted)


def test_blocking_paid_operation_renders_local_recovery_without_chat() -> None:
    revisions = _FakeRevisions(operation_choices=(_blocking_operation_choice(),))
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        events = _events(
            _post(sidecar, _message_request("Tell me about my Japanese library"))[2]
        )

        [manager] = _widget_items(events)
        wire = json.dumps(manager["widget"], ensure_ascii=False)
        assert manager["widget"]["status"]["text"] == "Model-call recovery"
        assert "captured-op" in wire
        assert "Show exact recovery reply" in wire
        assert "Discard captured reply and forget" in wire
        assert revisions.chatted == []
        assert [event for event in events if event["type"] == "progress_update"] == []
        assert any(
            event["type"] == "notice"
            and event.get("title") == "Earlier model call needs attention"
            and "not a confirmation for each question" in event["message"]
            for event in events
        )
    finally:
        sidecar.close()


def test_recovery_only_paid_operation_does_not_preempt_chat() -> None:
    recovery_only = replace(_blocking_operation_choice(), blocks_spending=False)
    revisions = _FakeRevisions(operation_choices=(recovery_only,))
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        message = "Tell me about my Japanese library"
        events = _events(_post(sidecar, _message_request(message))[2])

        assert revisions.chatted == [("", message)]
        assert _widget_items(events) == []
    finally:
        sidecar.close()


def test_unavailable_paid_operation_status_refuses_before_chat() -> None:
    revisions = _FakeRevisions()

    def refuse_status() -> tuple[OperationChoice, ...]:
        raise RevisionRefusal("Operation status is unreadable.")

    revisions.list_operation_choices = refuse_status  # type: ignore[method-assign]
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        events = _events(
            _post(sidecar, _message_request("Tell me about my Japanese library"))[2]
        )

        assert revisions.chatted == []
        assert any(
            event["type"] == "notice"
            and event.get("title") == "Model-call recovery unavailable"
            and "did not make a model call" in event["message"]
            for event in events
        )
    finally:
        sidecar.close()


def test_chat_authorization_race_refreshes_blocking_operation_actions() -> None:
    revisions = _FakeRevisions()
    operation_checks = iter(((), (_blocking_operation_choice(),)))
    revisions.list_operation_choices = lambda: next(operation_checks)  # type: ignore[method-assign]

    def refuse_chat(
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Any,
        preview: Any,
    ) -> ChatReply:
        del history, progress, preview
        revisions.chatted.append((deck_scope, message))
        raise RevisionRefusal("A paid operation began before authorization.")

    revisions.chat = refuse_chat  # type: ignore[method-assign]
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        message = "Tell me about my Japanese library"
        events = _events(_post(sidecar, _message_request(message))[2])

        [manager] = _widget_items(events)
        assert manager["widget"]["status"]["text"] == "Model-call recovery"
        assert revisions.chatted == [("", message)]
        assert any(
            event["type"] == "notice"
            and event.get("title") == "Answer refused"
            for event in events
        )
        assert any(
            event["type"] == "notice"
            and event.get("title") == "Earlier model call needs attention"
            and "blocks further model calls" in event["message"]
            for event in events
        )
    finally:
        sidecar.close()


def test_local_operation_manager_prepares_exact_actions_without_a_model_turn() -> None:
    revisions = _FakeRevisions(
        operation_choices=(
            OperationChoice(
                operation_id="captured-op",
                kind="assistant_chat",
                state="result_captured",
                source_name="request.json",
                model="claude-opus-5",
                authorized_at="2026-09-02T00:00:00+00:00",
                blocks_spending=True,
                money_may_have_been_spent=True,
                has_captured_reply=True,
                has_response_spool=False,
                cleanup_pending=False,
                actions=(
                    OperationActionChoice(
                        action="recover",
                        label="Recover captured result",
                    ),
                    OperationActionChoice(
                        action="show_reply",
                        label="Show exact recovery reply",
                    ),
                    OperationActionChoice(
                        action="forget",
                        label="Discard captured reply and forget",
                        accept_paid_output_loss=True,
                    ),
                ),
            ),
        ),
        execution_result=RevisionExecution(
            message="Private paid reply shown here.",
            finish=None,
            complete=True,
            remember_in_chat_context=False,
        ),
        execution_progress=(),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        status, _headers, body = _post(
            sidecar,
            _message_request("Manage model calls"),
        )
        assert status == 200
        events = _events(body)
        thread_id = next(
            event["thread"]["id"]
            for event in events
            if event["type"] == "thread.created"
        )
        [manager] = _widget_items(events)
        wire = json.dumps(manager["widget"], ensure_ascii=False)
        assert "captured-op" in wire
        assert "result_captured" in wire
        assert "Recover captured result" in wire
        assert "Show exact recovery reply" in wire
        assert "Discard captured reply and forget" in wire
        assert revisions.chatted == []

        recover_action = next(
            child["onClickAction"]
            for child in manager["widget"]["children"]
            if child.get("onClickAction", {}).get("payload", {}).get("action")
            == "recover"
        )
        prepared_events = _events(
            _post(
                sidecar,
                _action_request(thread_id, manager, recover_action),
            )[2]
        )
        [confirmation] = _widget_items(prepared_events)
        assert revisions.operation_prepared == [
            ("captured-op", "recover", False, "")
        ]
        assert "Confirm this exact action" in json.dumps(
            confirmation["widget"], ensure_ascii=False
        )

        completed = _events(
            _post(
                sidecar,
                _confirmation_request(thread_id, confirmation),
            )[2]
        )
        assert "Private paid reply shown here" in json.dumps(
            completed, ensure_ascii=False
        )

        revisions.operation_choices = ()
        _post(sidecar, _followup_message_request(thread_id, "What is available?"))
        assert revisions.chat_histories[-1] == ()
    finally:
        sidecar.close()


def test_paid_operation_selector_refuses_a_widened_action_as_one_use() -> None:
    revisions = _FakeRevisions(
        operation_choices=(
            OperationChoice(
                operation_id="authorized-op",
                kind="assistant_chat",
                state="authorized",
                source_name="request.json",
                model="claude-opus-5",
                authorized_at="2026-09-02T00:00:00+00:00",
                blocks_spending=True,
                money_may_have_been_spent=False,
                has_captured_reply=False,
                has_response_spool=False,
                cleanup_pending=False,
                actions=(
                    OperationActionChoice(
                        action="end",
                        label="End this operation",
                    ),
                ),
            ),
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        events = _events(
            _post(sidecar, _message_request("Manage model calls"))[2]
        )
        thread_id = next(
            event["thread"]["id"]
            for event in events
            if event["type"] == "thread.created"
        )
        [manager] = _widget_items(events)
        original = next(
            child["onClickAction"]
            for child in manager["widget"]["children"]
            if "onClickAction" in child
        )
        widened = json.loads(json.dumps(original))
        widened["payload"]["action"] = "forget"
        widened["payload"]["accept_paid_output_loss"] = True

        refused = _events(
            _post(
                sidecar,
                _action_request(thread_id, manager, widened),
            )[2]
        )
        replayed = _events(
            _post(
                sidecar,
                _action_request(thread_id, manager, original),
            )[2]
        )

        assert revisions.operation_prepared == []
        assert any(event["type"] == "error" for event in refused)
        assert any(event["type"] == "error" for event in replayed)
    finally:
        sidecar.close()


def test_unfocused_new_thread_calls_chat_with_empty_deck_scope() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        status, _headers, body = _post(
            sidecar,
            _message_request("What does this deck teach?"),
        )

        assert status == 200
        assert revisions.chatted == [("", "What does this deck teach?")]
        assert revisions.chat_histories == [()]
        events = _events(body)
        assert _widget_items(events) == []
        assert "Answer about : What does this deck teach?" in json.dumps(
            events, ensure_ascii=False
        )
    finally:
        sidecar.close()


def test_first_chat_turn_preserves_thread_metadata_store_contract() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        events = _events(
            _post(sidecar, _message_request("Tell me about my Japanese library"))[2]
        )
        thread_id = next(
            event["thread"]["id"]
            for event in events
            if event["type"] == "thread.created"
        )

        status, _headers, body = _post(
            sidecar,
            {
                "type": "threads.get_by_id",
                "params": {"thread_id": thread_id},
            },
        )

        assert status == 200
        assert json.loads(body)["id"] == thread_id
    finally:
        sidecar.close()


def test_supported_deck_selection_routes_next_message_with_empty_history() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        selection = _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, "potential"),
        )

        selected_events = _events(_post(sidecar, selection)[2])

        assert revisions.chatted == []
        assert revisions.resolved == ["potential"]
        assert "Brandon Japanese::Genki II::Lesson 13::Potential Practice" in json.dumps(
            selected_events,
            ensure_ascii=False,
        )

        replay = _events(_post(sidecar, selection)[2])
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "already used" in event["message"])
            for event in replay
        )

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(thread_id, "How can I improve this deck?"),
        )

        assert status == 200
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "How can I improve this deck?")
        ]
        assert revisions.chat_histories == [()]
        assert _widget_items(_events(body)) == []
    finally:
        sidecar.close()


def test_every_readable_deck_focus_routes_exact_scope_without_capability_labels() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        selection = _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, "lesson-vocabulary"),
        )

        selected_events = _events(_post(sidecar, selection)[2])

        assert revisions.resolved == ["lesson-vocabulary"]
        assert not any(event["type"] == "error" for event in selected_events)

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(thread_id, "What does this deck teach?"),
        )

        assert status == 200
        assert revisions.chatted == [
            (
                "data/decks/lesson-13-vocabulary.yaml",
                "What does this deck teach?",
            )
        ]
        assert revisions.chat_histories == [()]
        events = _events(body)
        assert any(
            event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
            for event in events
        )
        assert _widget_items(events) == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_switching_deck_focus_clears_focus_specific_history() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, first_selector, _events_before = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        first_chat_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Question about the first deck"),
            )[2]
        )
        assert _widget_items(first_chat_events) == []

        second_selector_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Choose a deck to focus on"),
            )[2]
        )
        second_selector = _widget_items(second_selector_events)[0]
        _post(
            sidecar,
            _action_request(
                thread_id,
                second_selector,
                _deck_selector_action(second_selector, "te-form"),
            ),
        )

        second_chat_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Question about the second deck"),
            )[2]
        )
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Question about the first deck"),
            ("data/decks/teform-drill.yaml", "Question about the second deck"),
        ]
        assert revisions.chat_histories == [(), ()]
        assert _widget_items(second_chat_events) == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_all_library_selector_clears_focus_history_and_action_bindings() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The focused action is ready.",
            action=_action_plan(),
            action_instruction="Revise the focused cards",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        focused_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Revise the focused cards"),
            )[2]
        )
        [focused_action] = _widget_items(focused_events)
        assert thread_id in server._histories
        assert server._plans

        [selector] = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )
        all_library_row = next(
            row
            for row in selector["widget"]["children"]
            if "All library" in json.dumps(row, ensure_ascii=False)
        )
        clear_action = all_library_row["onClickAction"]

        cleared = _events(
            _post(
                sidecar,
                _action_request(thread_id, selector, clear_action),
            )[2]
        )

        assert "Continuing with all library" in json.dumps(cleared, ensure_ascii=False)
        assert thread_id not in server._histories
        assert server._plans == {}
        stale = _events(
            _post(
                sidecar,
                _confirmation_request(thread_id, focused_action),
            )[2]
        )
        assert any(event["type"] == "error" for event in stale)

        revisions.chat_reply = None
        _post(sidecar, _followup_message_request(thread_id, "What is available?"))
        assert revisions.chatted[-1] == ("", "What is available?")
        assert revisions.chat_histories[-1] == ()
    finally:
        sidecar.close()


def test_selector_refuses_tampering_and_renders_chat_unavailable_decks_inert() -> None:
    revisions = _FakeRevisions()
    choices = (
        *_assistant_deck_choices(),
        AssistantDeckChoice(
            deck_id="broken-deck",
            label="Broken Deck",
            scope="data/decks/broken.yaml",
            chat_supported=False,
            revision_supported=False,
            unavailable_reason="This configured deck could not be read safely.",
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=choices,
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_selector, _events_before = _start_deck_selector(sidecar)
        tampered_action = json.loads(json.dumps(_deck_selector_action(first_selector, "potential")))
        tampered_action["payload"]["deck_id"] = "not-a-configured-deck"
        tampered = _events(
            _post(
                sidecar,
                _action_request(first_thread, first_selector, tampered_action),
            )[2]
        )
        assert any(event["type"] == "error" for event in tampered)

        second_thread, second_selector, _events_before = _start_deck_selector(sidecar)
        unavailable_row = next(
            row
            for row in second_selector["widget"]["children"]
            if "Broken Deck" in json.dumps(row, ensure_ascii=False)
        )
        assert "onClickAction" not in unavailable_row
        unavailable_badge = next(
            child
            for child in unavailable_row["children"][0]["children"]
            if child["type"] == "Badge"
        )
        assert unavailable_badge["label"] == "Unavailable"
        assert "could not be read safely" in json.dumps(
            unavailable_row,
            ensure_ascii=False,
        )
        selected = _events(
            _post(
                sidecar,
                _action_request(
                    second_thread,
                    second_selector,
                    _deck_selector_action(second_selector, "potential"),
                ),
            )[2]
        )
        assert not any(event["type"] == "error" for event in selected)

        third_thread, third_selector, _events_before = _start_deck_selector(sidecar)
        forged_unavailable = json.loads(
            json.dumps(_deck_selector_action(third_selector, "potential"))
        )
        forged_unavailable["payload"]["deck_id"] = "broken-deck"
        refused_unavailable = _events(
            _post(
                sidecar,
                _action_request(
                    third_thread,
                    third_selector,
                    forged_unavailable,
                ),
            )[2]
        )
        assert any(
            event["type"] == "error"
            and ("tampered" in event["message"] or "stale" in event["message"])
            for event in refused_unavailable
        )
        assert revisions.resolved == ["potential"]
        assert revisions.chatted == []
        assert revisions.chat_histories == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_deck_selection_revalidation_refusal_preserves_the_active_deck() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        second_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]
        revisions.resolve_refusal = "the deck changed on disk"

        refused = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    second_selector,
                    _deck_selector_action(second_selector, "te-form"),
                ),
            )[2]
        )

        assert revisions.resolved == ["potential", "te-form"]
        assert any(
            event["type"] == "error"
            and "no longer available" in event["message"]
            and "changed on disk" in event["message"]
            for event in refused
        )

        _post(sidecar, _followup_message_request(thread_id, "Which deck is active?"))
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Which deck is active?")
        ]
    finally:
        sidecar.close()


def test_stale_deck_focus_is_cleared_and_chat_continues_unfocused() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        old_chat_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Old request for deck A"),
            )[2]
        )
        assert _widget_items(old_chat_events) == []
        saved_thread = sidecar.server.assistant_core.store._threads[thread_id]
        saved_thread.metadata["janki_active_deck_id"] = "removed-deck"

        recovery_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "What does this deck teach?"),
            )[2]
        )

        assert not any(event["type"] == "error" for event in recovery_events)
        assert any(
            event["type"] == "notice"
            and event.get("title") == "Deck focus cleared"
            and "no longer available" in event["message"]
            for event in recovery_events
        )
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Old request for deck A"),
            ("", "What does this deck teach?"),
        ]
        assert revisions.chat_histories == [(), ()]
        assert _widget_items(recovery_events) == []

        selector_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Choose a deck to focus on"),
            )[2]
        )
        recovery_selector = _widget_items(selector_events)[0]
        selected = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    recovery_selector,
                    _deck_selector_action(recovery_selector, "potential"),
                ),
            )[2]
        )
        assert not any(event["type"] == "error" for event in selected)
        assert revisions.resolved == ["potential", "potential"]

        _post(sidecar, _followup_message_request(thread_id, "Question for recovered deck"))
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Old request for deck A"),
            ("", "What does this deck teach?"),
            ("data/decks/potential-practice.yaml", "Question for recovered deck"),
        ]
        assert revisions.chat_histories == [(), (), ()]
    finally:
        sidecar.close()


def test_switching_a_b_a_does_not_revive_an_old_finish_capability() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, finish_widget, _events_before = _create_finish_review(sidecar)
        finish_action = finish_widget["widget"]["confirm"]["action"]
        capability = finish_action["payload"]["capability"]
        old_binding = sidecar.server.assistant_core.server._finishes[capability]

        for deck_id in ("te-form", "potential"):
            selector_events = _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
            selector = _widget_items(selector_events)[0]
            selected = _events(
                _post(
                    sidecar,
                    _action_request(
                        thread_id,
                        selector,
                        _deck_selector_action(selector, deck_id),
                    ),
                )[2]
            )
            assert not any(event["type"] == "error" for event in selected)

        # Reinsert the old binding to prove the monotonically increasing epoch,
        # independently from the switch's ordinary capability cleanup.
        sidecar.server.assistant_core.server._finishes[capability] = old_binding

        stale = _events(
            _post(sidecar, _confirmation_request(thread_id, finish_widget))[2]
        )

        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "already used" in event["message"])
            for event in stale
        )
    finally:
        sidecar.close()


def test_two_threads_keep_independent_active_decks_and_histories() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_selector, _ = _start_deck_selector(sidecar)
        second_thread, second_selector, _ = _start_deck_selector(sidecar)
        for thread_id, selector, deck_id in (
            (first_thread, first_selector, "potential"),
            (second_thread, second_selector, "te-form"),
        ):
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    selector,
                    _deck_selector_action(selector, deck_id),
                ),
            )

        _post(sidecar, _followup_message_request(first_thread, "First question"))
        _post(sidecar, _followup_message_request(second_thread, "Second question"))
        _post(sidecar, _followup_message_request(first_thread, "First follow-up"))

        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "First question"),
            ("data/decks/teform-drill.yaml", "Second question"),
            ("data/decks/potential-practice.yaml", "First follow-up"),
        ]
        assert revisions.chat_histories == [
            (),
            (),
            (
                ("user", "First question"),
                (
                    "assistant",
                    "Answer about data/decks/potential-practice.yaml: First question",
                ),
            ),
        ]
    finally:
        sidecar.close()


def test_active_deck_cannot_switch_during_an_in_flight_chat_turn() -> None:
    entered = threading.Event()
    release = threading.Event()
    revisions = _FakeRevisions(chat_entered=entered, chat_release=release)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    sender: threading.Thread | None = None
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        switch_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]
        response: list[tuple[int, dict[str, str], bytes]] = []
        sender = threading.Thread(
            target=lambda: response.append(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "A question for deck A"),
                )
            )
        )
        sender.start()
        assert entered.wait(3)

        refused = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    switch_selector,
                    _deck_selector_action(switch_selector, "te-form"),
                ),
            )[2]
        )

        assert any(
            event["type"] == "error" and "still running" in event["message"]
            for event in refused
        )
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "A question for deck A")
        ]
    finally:
        release.set()
        if sender is not None:
            sender.join(3)
        sidecar.close()
    assert response and response[0][0] == 200


def test_chat_reads_fresh_deck_focus_after_paid_operation_preflight() -> None:
    operation_entered = threading.Event()
    operation_release = threading.Event()
    revisions = _FakeRevisions(
        operation_entered=operation_entered,
        operation_release=operation_release,
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    chatter: threading.Thread | None = None
    chat_response: list[tuple[int, dict[str, str], bytes]] = []
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        assert _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )[0] == 200
        switch_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]

        chatter = threading.Thread(
            target=lambda: chat_response.append(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Which deck is in focus?"),
                )
            )
        )
        chatter.start()
        assert operation_entered.wait(3)

        switched = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    switch_selector,
                    _deck_selector_action(switch_selector, "te-form"),
                ),
            )[2]
        )
        assert not any(event["type"] == "error" for event in switched)

        operation_release.set()
        chatter.join(3)
        assert chat_response and chat_response[0][0] == 200
        assert revisions.chatted == [
            ("data/decks/teform-drill.yaml", "Which deck is in focus?")
        ]
    finally:
        operation_release.set()
        if chatter is not None:
            chatter.join(3)
        sidecar.close()


def test_chat_waits_for_in_progress_deck_focus_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_entered = threading.Event()
    save_release = threading.Event()
    chat_entered = threading.Event()
    revisions = _FakeRevisions(chat_entered=chat_entered)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    store = sidecar.server.assistant_core.store
    original_save_thread = store.save_thread

    async def delayed_save_thread(thread: Any, context: Any) -> None:
        if thread.metadata.get("janki_active_deck_id") == "te-form":
            save_entered.set()
            released = await asyncio.to_thread(save_release.wait, 3)
            if not released:
                raise AssertionError("the test never released deck-focus persistence")
        await original_save_thread(thread, context)

    monkeypatch.setattr(store, "save_thread", delayed_save_thread)
    sidecar.start()
    switcher: threading.Thread | None = None
    chatter: threading.Thread | None = None
    switch_response: list[tuple[int, dict[str, str], bytes]] = []
    chat_response: list[tuple[int, dict[str, str], bytes]] = []
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        assert _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )[0] == 200
        switch_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]

        switcher = threading.Thread(
            target=lambda: switch_response.append(
                _post(
                    sidecar,
                    _action_request(
                        thread_id,
                        switch_selector,
                        _deck_selector_action(switch_selector, "te-form"),
                    ),
                )
            )
        )
        switcher.start()
        assert save_entered.wait(3)

        chatter = threading.Thread(
            target=lambda: chat_response.append(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Which deck is in focus?"),
                )
            )
        )
        chatter.start()
        assert not chat_entered.wait(0.2)

        save_release.set()
        switcher.join(3)
        chatter.join(3)
        assert switch_response and switch_response[0][0] == 200
        assert chat_response and chat_response[0][0] == 200
        assert revisions.chatted == [
            ("data/decks/teform-drill.yaml", "Which deck is in focus?")
        ]
    finally:
        save_release.set()
        if switcher is not None:
            switcher.join(3)
        if chatter is not None:
            chatter.join(3)
        sidecar.close()


def test_chat_waits_for_deck_resolution_and_uses_resolved_focus() -> None:
    resolve_entered = threading.Event()
    resolve_release = threading.Event()
    chat_entered = threading.Event()
    chat_release = threading.Event()
    revisions = _FakeRevisions(
        resolve_block_for="te-form",
        resolve_entered=resolve_entered,
        resolve_release=resolve_release,
        chat_entered=chat_entered,
        chat_release=chat_release,
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    switcher: threading.Thread | None = None
    chatter: threading.Thread | None = None
    switch_response: list[tuple[int, dict[str, str], bytes]] = []
    chat_response: list[tuple[int, dict[str, str], bytes]] = []
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        switch_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]

        switcher = threading.Thread(
            target=lambda: switch_response.append(
                _post(
                    sidecar,
                    _action_request(
                        thread_id,
                        switch_selector,
                        _deck_selector_action(switch_selector, "te-form"),
                    ),
                )
            )
        )
        switcher.start()
        assert resolve_entered.wait(3)

        chatter = threading.Thread(
            target=lambda: chat_response.append(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "A question for deck A"),
                )
            )
        )
        chatter.start()
        assert not chat_entered.wait(0.2)

        resolve_release.set()
        switcher.join(3)
        assert switch_response and switch_response[0][0] == 200
        assert not any(
            event["type"] == "error"
            for event in _events(switch_response[0][2])
        )
        assert chat_entered.wait(3)
        assert revisions.resolved == ["potential", "te-form"]
        assert revisions.chatted == [
            ("data/decks/teform-drill.yaml", "A question for deck A")
        ]

        chat_release.set()
        chatter.join(3)
        assert chat_response and chat_response[0][0] == 200
        _post(sidecar, _followup_message_request(thread_id, "Follow up on deck B"))
        assert revisions.chatted == [
            ("data/decks/teform-drill.yaml", "A question for deck A"),
            ("data/decks/teform-drill.yaml", "Follow up on deck B"),
        ]
        assert revisions.chat_histories[-1] == (
            ("user", "A question for deck A"),
            (
                "assistant",
                "Answer about data/decks/teform-drill.yaml: A question for deck A",
            ),
        )
    finally:
        resolve_release.set()
        chat_release.set()
        if switcher is not None:
            switcher.join(3)
        if chatter is not None:
            chatter.join(3)
        sidecar.close()
    assert chat_response and chat_response[0][0] == 200


def test_thread_stays_busy_until_typed_reply_and_action_bindings_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact change is ready for confirmation.",
            action=_action_plan(target="data/decks/potential-practice.yaml"),
            action_instruction="Prepare a change",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        original_message_event = type(server)._message_event
        chat_binding_checks: list[bool] = []

        def guarded_message_event(self: Any, thread: Any, text: str) -> Any:
            if text == "The exact change is ready for confirmation.":
                chat_binding_checks.append(thread.id in self._busy_threads)
            return original_message_event(self, thread, text)

        monkeypatch.setattr(
            type(server),
            "_message_event",
            guarded_message_event,
        )
        original_plan_widget = type(server)._plan_widget
        plan_binding_checks: list[bool] = []

        def guarded_plan_widget(
            plan: RevisionPlan,
            *,
            instruction: str,
            capability: str,
        ) -> Any:
            plan_binding_checks.append(bool(server._busy_threads))
            return original_plan_widget(
                plan,
                instruction=instruction,
                capability=capability,
            )

        monkeypatch.setattr(
            type(server),
            "_plan_widget",
            staticmethod(guarded_plan_widget),
        )
        thread_id, plan_widget, _plan_events = _create_automatic_action(
            sidecar,
            focused=True,
            message="Prepare a change",
        )
        assert chat_binding_checks == [True]
        assert plan_binding_checks == [True]
        assert thread_id not in server._busy_threads

        original_finish_widget = type(server)._finish_widget
        finish_binding_checks: list[bool] = []

        def guarded_finish_widget(
            review: RevisionFinishReview,
            *,
            capability: str,
        ) -> Any:
            finish_binding_checks.append(thread_id in server._busy_threads)
            return original_finish_widget(review, capability=capability)

        monkeypatch.setattr(
            type(server),
            "_finish_widget",
            staticmethod(guarded_finish_widget),
        )
        finish_events = _events(
            _post(sidecar, _confirmation_request(thread_id, plan_widget))[2]
        )
        assert len(_widget_items(finish_events)) == 1
        assert finish_binding_checks == [True]
        assert thread_id not in server._busy_threads
    finally:
        sidecar.close()


def test_shell_is_a_separate_tokenized_origin_with_only_the_chatkit_cdn() -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="data/decks/potential-practice.yaml",
        deck_display_name="Brandon Japanese::Potential Practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        status, headers, body = _request(sidecar, "GET", sidecar.server.shell_path)

        assert status == 200
        assert headers["referrer-policy"] == "no-referrer"
        assert "access-control-allow-origin" not in headers
        csp = headers["content-security-policy"]
        assert "script-src 'self' https://cdn.platform.openai.com" in csp
        assert "frame-src https://cdn.platform.openai.com" in csp
        assert ("style-src 'self' 'sha256-G2shiuZXM1qoGNHm7OQ6u7Ye45SO8f8LKO17q0kfGvw='") in csp
        assert "'unsafe-inline'" not in csp
        assert b"https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" in body
        assert b"<title>Janki</title>" in body
        assert b"<h1>Janki</h1>" in body
        assert b'id="janki-chat-error"' in body
        assert b'role="alert" hidden' in body
        assert b"Chat interface did not load" in body
        assert b"The local workbench is still running" in body
        assert b"Ask janki" not in body
        assert b"Ask about your Japanese library" in body
        assert b"optionally focus this conversation on one deck" in body
        assert b"does not limit what Janki can do" in body
        assert b"Questions are read-only" not in body
        assert b"explicit deck-change action" not in body
        assert b"Selected deck for changes:" not in body
        assert b"Brandon Japanese::Potential Practice" not in body
        assert b"data/decks/potential-practice.yaml" not in body
        assert b"main-workbench-secret" not in body

        script_status, _script_headers, script = _request(
            sidecar,
            "GET",
            sidecar.server.script_path,
        )
        style_status, _style_headers, stylesheet = _request(
            sidecar,
            "GET",
            sidecar.server.style_path,
        )
        assert script_status == 200
        assert style_status == 200
        assert sidecar.server.api_path.encode() in script
        assert b'domainKey: "domain_pk_localhost_dev"' in script
        assert b"const isRequest = input instanceof Request;" in script
        assert b"return window.fetch(isRequest ? input : target, {" in script
        assert b'credentials: "omit"' in script
        assert b'referrerPolicy: "no-referrer"' in script
        assert b"attachments: { enabled: false }" in script
        assert b'placeholder: "Ask Janki or attach a source"' in script
        assert b'greeting: "What would you like to do?"' in script
        assert b'label: "Explore my library"' in script
        assert (
            b'prompt: "Show me what I can do with my Japanese library"' in script
        )
        assert b'label: "Focus on a deck"' in script
        assert b'prompt: "Choose a deck to focus on"' in script
        assert b'label: "Add study material"' in script
        assert b'prompt: "How do I add study material?"' in script
        assert b'label: "Manage model calls"' in script
        assert b'prompt: "Manage model calls"' in script
        assert b'label: "How changes work"' not in script
        assert b"janki.revision.prepare" not in script
        assert b"Summarize what this deck is designed to teach." not in script
        assert b'icon: "book-open"' in script
        assert b'icon: "document"' in script
        starter_icons = {
            value.decode() for value in re.findall(rb'icon:\s*"([^"]+)"', script)
        }
        assert len(starter_icons) == 4
        icon_adapter = TypeAdapter(IconName)
        for icon in starter_icons:
            icon_adapter.validate_python(icon)
        assert b"header { width: min(100%, 48rem); margin: 0 auto 1rem; }" in stylesheet
        assert (
            b"threadItemActions: {\n      feedback: false,\n      retry: false,\n    }," in script
        )
        assert b"onClientTool" not in script
        assert b'window.setTimeout(() => reject(new Error("ChatKit load timed out"))' in script
        assert b'customElements.whenDefined("openai-chatkit")' in script
        assert b"failure.hidden = false" in script
        assert b'[data-chatkit-failed="true"] openai-chatkit' in stylesheet
        assert b".chat-error" in stylesheet
    finally:
        sidecar.close()


def test_shell_does_not_embed_deck_catalog_names_or_scopes() -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="data/decks/<script>alert(2)</script>.yaml",
        deck_display_name='<img src=x onerror="alert(1)">',
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        status, _headers, body = _request(sidecar, "GET", sidecar.server.shell_path)

        assert status == 200
        assert b'<img src=x onerror="alert(1)">' not in body
        assert b"&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" not in body
        assert b"<script>alert(2)</script>" not in body
        assert b"data/decks/&lt;script&gt;alert(2)&lt;/script&gt;.yaml" not in body
    finally:
        sidecar.close()


def test_shell_keeps_repository_conversation_available_without_decks(
    tmp_path: Path,
) -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="janki-project",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
        conversation_available=False,
    )
    sidecar.start()
    try:
        status, _headers, body = _request(
            sidecar,
            "GET",
            sidecar.server.shell_path,
        )
        script_status, _script_headers, script = _request(
            sidecar,
            "GET",
            sidecar.server.script_path,
        )

        assert status == 200
        assert script_status == 200
        assert b"Ask about your Japanese library" in body
        assert b"optionally focus this conversation on one deck" in body
        assert b"until a readable deck is configured" not in body
        assert b'placeholder: "Ask Janki or attach a source"' in script
        assert b'greeting: "What would you like to do?"' in script
        assert b'label: "Explore my library"' in script
        assert b'label: "Focus on a deck"' in script
        assert b'label: "Add study material"' in script
        assert b'label: "Manage model calls"' in script
        assert b'label: "How changes work"' not in script
    finally:
        sidecar.close()


def test_project_without_a_focusable_deck_still_supports_repository_chat() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        question_events = _events(_post(sidecar, _message_request("Can you help?"))[2])
        answer = next(
            event["item"]["content"][0]["text"]
            for event in question_events
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert answer == "Answer about : Can you help?"
        assert not any(event["type"] == "notice" for event in question_events)
        assert _widget_items(question_events) == []

        help_events = _events(
            _post(
                sidecar,
                _message_request("Show me what I can do with my Japanese library"),
            )[2]
        )
        help_answer = next(
            event["item"]["content"][0]["text"]
            for event in help_events
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert "whole Japanese library" in help_answer
        assert "only gives the conversation a convenient focus" in help_answer
        assert "revise selected canonical cards in any readable" in help_answer
        assert _widget_items(help_events) == []

        choose_events = _events(
            _post(sidecar, _message_request("Choose a deck to focus on"))[2]
        )
        assert any(
            event["type"] == "notice"
            and "No readable deck is available to focus" in event["message"]
            for event in choose_events
        )
        assert _widget_items(choose_events) == []

        assert revisions.chatted == [("", "Can you help?")]
    finally:
        sidecar.close()


def test_shell_does_not_classify_a_readable_deck_by_revision_support() -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_choices=(
            AssistantDeckChoice(
                deck_id="vocabulary",
                label="Vocabulary",
                scope="data/decks/vocabulary.yaml",
                chat_supported=True,
                revision_supported=False,
                unavailable_reason=(
                    "Deck changes currently support rich conjugation practice "
                    "decks only."
                ),
            ),
        ),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        status, _headers, body = _request(
            sidecar,
            "GET",
            sidecar.server.shell_path,
        )
        script_status, _script_headers, script = _request(
            sidecar,
            "GET",
            sidecar.server.script_path,
        )

        assert status == 200
        assert script_status == 200
        assert b"Ask about your Japanese library" in body
        assert b"A deck focus narrows context" in body
        assert b"chat only" not in body.lower()
        assert b"Chat + changes" not in body
        assert b'label: "Explore my library"' in script
        assert b'label: "Focus on a deck"' in script
        assert b'label: "Add study material"' in script
        assert b'label: "Manage model calls"' in script
        assert b'label: "How changes work"' not in script
        assert b"Chat only" not in script
        assert b"Chat + changes" not in script
    finally:
        sidecar.close()


def test_revision_support_flag_does_not_intercept_an_ordinary_focused_message() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=(
            AssistantDeckChoice(
                deck_id="vocabulary",
                label="Vocabulary",
                scope="data/decks/vocabulary.yaml",
                chat_supported=True,
                revision_supported=False,
                unavailable_reason=(
                    "Deck changes currently support rich conjugation practice "
                    "decks only."
                ),
            ),
        ),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        selection = _action_request(
            thread_id,
            selector,
            _deck_selector_action(selector, "vocabulary"),
        )
        assert _post(sidecar, selection)[0] == 200

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(
                thread_id,
                "Explain how I can prepare and confirm a deck change.",
            ),
        )

        assert status == 200
        answer = next(
            event["item"]["content"][0]["text"]
            for event in _events(body)
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert answer == (
            "Answer about data/decks/vocabulary.yaml: "
            "Explain how I can prepare and confirm a deck change."
        )
        assert revisions.chatted == [
            (
                "data/decks/vocabulary.yaml",
                "Explain how I can prepare and confirm a deck change.",
            )
        ]
        assert _widget_items(_events(body)) == []
    finally:
        sidecar.close()


def test_capabilities_help_is_repository_wide_and_does_not_select_a_deck() -> None:
    revisions = _FakeRevisions()

    def unexpected_operation_status() -> tuple[OperationChoice, ...]:
        raise AssertionError("local help must not inspect paid operations")

    revisions.list_operation_choices = (  # type: ignore[method-assign]
        unexpected_operation_status
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=(
            AssistantDeckChoice(
                deck_id="vocabulary",
                label="Vocabulary",
                scope="data/decks/vocabulary.yaml",
                chat_supported=True,
                revision_supported=False,
                unavailable_reason=(
                    "Deck changes currently support rich conjugation practice "
                    "decks only."
                ),
            ),
        ),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _selector, _events_before = _start_deck_selector(sidecar)

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(
                thread_id,
                "Show me what I can do with my Japanese library",
            ),
        )

        assert status == 200
        answer = next(
            event["item"]["content"][0]["text"]
            for event in _events(body)
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert "whole Japanese library" in answer
        assert "only gives the conversation a convenient focus" in answer
        assert "attach one PDF or photo" in answer
        assert "revise selected canonical cards in any readable" in answer
        assert "raw shell" in answer
        assert "Chat + changes" not in answer
        assert _widget_items(_events(body)) == []
        assert revisions.chatted == []
    finally:
        sidecar.close()


def test_old_revision_help_phrase_is_an_ordinary_unfocused_model_message() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="data/decks/potential.yaml",
        deck_display_name="Potential Practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _selector, _events_before = _start_deck_selector(sidecar)

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(
                thread_id,
                "Explain how I can prepare and confirm a deck change.",
            ),
        )

        assert status == 200
        answer = next(
            event["item"]["content"][0]["text"]
            for event in _events(body)
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert answer == (
            "Answer about : Explain how I can prepare and confirm a deck change."
        )
        assert revisions.chatted == [
            ("", "Explain how I can prepare and confirm a deck change.")
        ]
        assert _widget_items(_events(body)) == []
    finally:
        sidecar.close()


def test_api_requires_exact_host_origin_content_type_and_capability_shape() -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        body = json.dumps(_message_request()).encode()
        missing_origin = _request(
            sidecar,
            "POST",
            sidecar.server.api_path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert missing_origin[0] == 403

        wrong_origin = _request(
            sidecar,
            "POST",
            sidecar.server.api_path,
            body=body,
            headers={"Content-Type": "application/json", "Origin": "http://evil.test"},
        )
        assert wrong_origin[0] == 403

        wrong_type = _request(
            sidecar,
            "POST",
            sidecar.server.api_path,
            body=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Origin": sidecar.origin,
            },
        )
        assert wrong_type[0] == 415

        with_attachment = _message_request()
        with_attachment["params"]["input"]["attachments"] = ["atc_private"]
        assert _post(sidecar, with_attachment)[0] == 400

        client_tool = {
            "type": "threads.add_client_tool_output",
            "params": {"thread_id": "thr_x", "result": {}},
        }
        assert _post(sidecar, client_tool)[0] == 400
    finally:
        sidecar.close()


def test_attachment_send_saves_the_exact_source_locally_without_calling_claude(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    inbox = tmp_path / "data" / "inbox"
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=inbox,
    )
    sidecar.start()
    source = b"%PDF-1.7\nexact owner bytes\n"
    try:
        script_status, _script_headers, script = _request(
            sidecar,
            "GET",
            sidecar.server.script_path,
        )
        assert script_status == 200
        assert b"enabled: true" in script
        assert (
            b'api: {\n      url: apiURL,\n      domainKey: "domain_pk_localhost_dev",'
            b'\n      fetch: localFetch,\n      uploadStrategy: { type: "two_phase" },'
            b"\n    }" in script
        )
        assert b'attachments: {\n        enabled: true,\n        uploadStrategy:' not in script
        assert b"maxSize: 128 * 1024 * 1024" in script
        assert b"maxCount: 1" in script
        assert b'"application/pdf": [".pdf"]' in script
        assert b'"image/heic": [".heic"]' in script

        status, _headers, body = _post(
            sidecar,
            _attachment_request(size=len(source)),
        )
        assert status == 200
        attachment = json.loads(body)
        assert attachment["name"] == "lesson.pdf"
        assert attachment["type"] == "file"
        descriptor = attachment["upload_descriptor"]
        assert descriptor["method"] == "PUT"
        upload_url = urlsplit(descriptor["url"])
        assert upload_url.scheme == "http"
        assert upload_url.netloc == sidecar.server.expected_host

        upload = _request(
            sidecar,
            "PUT",
            upload_url.path,
            body=source,
            headers={"Origin": sidecar.origin},
        )
        assert upload[0] == 204

        message = _message_request("Make a deck from this source")
        message["params"]["input"]["attachments"] = [attachment["id"]]
        response_status, _response_headers, response_body = _post(sidecar, message)

        assert response_status == 200
        events = _events(response_body)
        answer = next(
            event["item"]["content"][0]["text"]
            for event in events
            if event["type"] == "thread.item.done" and event["item"]["type"] == "assistant_message"
        )
        assert "lesson.pdf" in answer
        assert "saved" in answer.casefold()
        assert "not sent" in answer.casefold()
        assert "text accompanying this upload was not used" in answer.casefold()
        assert "review the exact extraction plan below" in answer.casefold()
        widgets = _widget_items(events)
        assert len(widgets) == 1
        widget_wire = json.dumps(widgets[0], ensure_ascii=False)
        assert "lesson.pdf" in widget_wire
        assert "whole lesson.pdf" in widget_wire
        assert "paid Anthropic API call" in widget_wire
        assert "Claude Max does not pay" in widget_wire
        assert "No canonical deck, audio, or build" in widget_wire
        extract_action = widgets[0]["widget"]["confirm"]["action"]
        assert extract_action["type"] == "janki.extraction.confirm"
        assert extract_action["handler"] == "server"
        assert revisions.chatted == []
        assert revisions.executed == []
        assert revisions.extraction_prepared == [inbox / "lesson.pdf"]
        assert revisions.extracted == []
        assert (inbox / "lesson.pdf").read_bytes() == source
    finally:
        sidecar.close()


def test_localhost_page_receives_a_same_origin_attachment_upload_url(
    tmp_path: Path,
) -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    localhost_authority = f"localhost:{sidecar.server.server_address[1]}"
    localhost_origin = f"http://{localhost_authority}"
    body = json.dumps(_attachment_request(size=4)).encode("utf-8")
    try:
        status, _headers, response = _request(
            sidecar,
            "POST",
            sidecar.server.api_path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Host": localhost_authority,
                "Origin": localhost_origin,
            },
        )

        assert status == 200
        descriptor = json.loads(response)["upload_descriptor"]
        upload_url = urlsplit(descriptor["url"])
        assert upload_url.scheme == "http"
        assert upload_url.netloc == localhost_authority
        assert upload_url.netloc != sidecar.server.expected_host

        upload_status, _upload_headers, _upload_body = _request(
            sidecar,
            "PUT",
            upload_url.path,
            body=b"%PDF",
            headers={"Host": localhost_authority, "Origin": localhost_origin},
        )
        assert upload_status == 204
    finally:
        sidecar.close()


def test_attachment_can_be_sent_without_a_caption(tmp_path: Path) -> None:
    revisions = _FakeRevisions()
    inbox = tmp_path / "data" / "inbox"
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=inbox,
    )
    sidecar.start()
    source = b"%PDF-1.7\nno caption needed\n"
    try:
        _thread_id, widget = _create_source_extraction_plan(
            sidecar,
            source,
            caption="",
        )

        assert widget["widget"]["confirm"]["action"]["type"] == ("janki.extraction.confirm")
        assert revisions.chatted == []
        assert revisions.extraction_prepared == [inbox / "lesson.pdf"]
        assert (inbox / "lesson.pdf").read_bytes() == source
    finally:
        sidecar.close()


def test_switching_decks_preserves_project_scoped_extraction_confirmation(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    try:
        thread_id, first_selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                first_selector,
                _deck_selector_action(first_selector, "potential"),
            ),
        )
        thread_id, extraction_widget = _create_source_extraction_plan(
            sidecar,
            b"%PDF-1.7\nproject-scoped source\n",
            thread_id=thread_id,
        )
        selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]
        selected = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    selector,
                    _deck_selector_action(selector, "te-form"),
                ),
            )[2]
        )
        assert not any(event["type"] == "error" for event in selected)

        confirmed = _events(
            _post(sidecar, _confirmation_request(thread_id, extraction_widget))[2]
        )

        assert not any(event["type"] == "error" for event in confirmed)
        assert revisions.extracted == [
            SourceExtractionConfirmation(
                preparation_id="prepared-extraction-1",
                source_name="lesson.pdf",
                expected_fingerprint=REQUEST_FINGERPRINT,
            )
        ]
    finally:
        sidecar.close()


def test_one_attachment_cannot_be_reused_in_a_second_thread(tmp_path: Path) -> None:
    revisions = _FakeRevisions()
    inbox = tmp_path / "data" / "inbox"
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=inbox,
    )
    sidecar.start()
    source = b"%PDF-1.7\none thread only\n"
    try:
        create_status, _headers, create_body = _post(
            sidecar,
            _attachment_request(size=len(source)),
        )
        assert create_status == 200
        attachment = json.loads(create_body)
        upload_path = urlsplit(attachment["upload_descriptor"]["url"]).path
        assert (
            _request(
                sidecar,
                "PUT",
                upload_path,
                body=source,
                headers={"Origin": sidecar.origin},
            )[0]
            == 204
        )
        first = _message_request("")
        first["params"]["input"]["attachments"] = [attachment["id"]]
        assert _post(sidecar, first)[0] == 200
        assert revisions.extraction_prepared == [inbox / "lesson.pdf"]

        second = _message_request("")
        second["params"]["input"]["attachments"] = [attachment["id"]]
        refused = _post(sidecar, second)

        assert refused[0] == 200
        assert any(event["type"] == "error" for event in _events(refused[2]))
        assert revisions.extraction_prepared == [inbox / "lesson.pdf"]
        assert (inbox / "lesson.pdf").read_bytes() == source
    finally:
        sidecar.close()


def test_attachment_must_finish_its_one_use_upload_before_message_send(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    inbox = tmp_path / "data" / "inbox"
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=inbox,
    )
    sidecar.start()
    try:
        status, _headers, body = _post(
            sidecar,
            _attachment_request(size=12),
        )
        assert status == 200
        attachment = json.loads(body)
        message = _message_request("Use this source")
        message["params"]["input"]["attachments"] = [attachment["id"]]

        refused = _post(sidecar, message)

        assert refused[0] == 400
        assert revisions.chatted == []
        assert not inbox.exists()
    finally:
        sidecar.close()


def test_attachment_upload_capability_is_exact_size_and_one_use(
    tmp_path: Path,
) -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    source = b"%PDF-1.7\n"
    try:
        status, _headers, body = _post(
            sidecar,
            _attachment_request(size=len(source)),
        )
        assert status == 200
        upload_path = urlsplit(json.loads(body)["upload_descriptor"]["url"]).path

        missing_origin = _request(
            sidecar,
            "PUT",
            upload_path,
            body=source,
        )
        assert missing_origin[0] == 403

        attachment_id, capability = upload_path.rstrip("/").rsplit("/", 2)[-2:]
        wrong_capability_path = upload_path[: -len(capability)] + "wrong-capability"
        wrong_capability = _request(
            sidecar,
            "PUT",
            wrong_capability_path,
            body=source,
            headers={"Origin": sidecar.origin},
        )
        assert wrong_capability[0] == 409
        assert attachment_id.startswith("atc_")

        too_large = _request(
            sidecar,
            "PUT",
            upload_path,
            body=b"",
            headers={
                "Origin": sidecar.origin,
                "Content-Length": str(128 * 1024 * 1024 + 1),
            },
        )
        assert too_large[0] == 413
        assert b"128 MiB limit" in too_large[2]

        too_short = _request(
            sidecar,
            "PUT",
            upload_path,
            body=source[:-1],
            headers={"Origin": sidecar.origin},
        )
        assert too_short[0] == 409

        accepted = _request(
            sidecar,
            "PUT",
            upload_path,
            body=source,
            headers={"Origin": sidecar.origin},
        )
        assert accepted[0] == 204

        replay = _request(
            sidecar,
            "PUT",
            upload_path,
            body=source,
            headers={"Origin": sidecar.origin},
        )
        assert replay[0] in {404, 409}
    finally:
        sidecar.close()


def test_put_streams_request_body_directly_into_attachment_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    source = b"%PDF-1.7\nstream me\n"
    observed_streams: list[Any] = []
    store = sidecar.server.attachment_store
    assert store is not None
    original = store.accept_upload_stream

    def capture_stream(
        attachment_id: str,
        capability: str,
        stream: Any,
        *,
        declared_content_length: int,
    ) -> None:
        observed_streams.append(stream)
        original(
            attachment_id,
            capability,
            stream,
            declared_content_length=declared_content_length,
        )

    monkeypatch.setattr(store, "accept_upload_stream", capture_stream)
    try:
        status, _headers, body = _post(
            sidecar,
            _attachment_request(size=len(source)),
        )
        assert status == 200
        upload_path = urlsplit(json.loads(body)["upload_descriptor"]["url"]).path

        uploaded = _request(
            sidecar,
            "PUT",
            upload_path,
            body=source,
            headers={"Origin": sidecar.origin},
        )

        assert uploaded[0] == 204
        assert len(observed_streams) == 1
        assert not isinstance(observed_streams[0], (bytes, bytearray, io.BytesIO))
    finally:
        sidecar.close()


def test_attachment_registration_refusal_explains_supported_sources(
    tmp_path: Path,
) -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    try:
        status, _headers, body = _post(
            sidecar,
            _attachment_request(name="notes.txt", size=4),
        )

        assert status == 409
        assert b"Supported:" in body
        assert b"ChatKit request was refused" not in body
    finally:
        sidecar.close()


def test_one_extraction_confirmation_survives_attachment_cleanup_and_runs_once(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    inbox = tmp_path / "data" / "inbox"
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=inbox,
    )
    sidecar.start()
    source = b"%PDF-1.7\nowner source\n"
    try:
        create_status, _headers, create_body = _post(
            sidecar,
            _attachment_request(size=len(source)),
        )
        assert create_status == 200
        attachment = json.loads(create_body)
        upload_path = urlsplit(attachment["upload_descriptor"]["url"]).path
        assert (
            _request(
                sidecar,
                "PUT",
                upload_path,
                body=source,
                headers={"Origin": sidecar.origin},
            )[0]
            == 204
        )
        message = _message_request("Create card proposals from this source")
        message["params"]["input"]["attachments"] = [attachment["id"]]
        send_status, _send_headers, send_body = _post(sidecar, message)
        assert send_status == 200
        send_events = _events(send_body)
        thread_id = next(
            event["thread"]["id"] for event in send_events if event["type"] == "thread.created"
        )
        widget = _widget_items(send_events)[0]

        delete_status, _delete_headers, _delete_body = _post(
            sidecar,
            {
                "type": "attachments.delete",
                "params": {"attachment_id": attachment["id"]},
            },
        )
        assert delete_status == 200
        assert (inbox / "lesson.pdf").read_bytes() == source

        request = _confirmation_request(thread_id, widget)
        status, _response_headers, body = _post(sidecar, request)

        assert status == 200
        events = _events(body)
        progress = [event["text"] for event in events if event["type"] == "progress_update"]
        assert progress == [
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ]
        assert all("%" not in label and "percent" not in label for label in progress)
        assert revisions.extracted == [
            SourceExtractionConfirmation(
                preparation_id="prepared-extraction-1",
                source_name="lesson.pdf",
                expected_fingerprint=REQUEST_FINGERPRINT,
            )
        ]
        assert b"staged 12 card proposals" in body
        assert _widget_items(events) == []

        replay = _events(_post(sidecar, request)[2])
        assert len(revisions.extracted) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_busy_refusal_preserves_valid_extraction_confirmation_for_retry(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        thread_id, widget = _create_source_extraction_plan(
            sidecar,
            b"%PDF-1.7\nowner source\n",
        )
        request = _confirmation_request(thread_id, widget)
        capability = request["params"]["action"]["payload"]["capability"]
        server._busy_threads.add(thread_id)

        refused = _events(_post(sidecar, request)[2])

        assert any(
            event["type"] == "error" and "Wait for" in event["message"]
            for event in refused
        )
        assert capability in server._extractions
        assert revisions.extracted == []

        server._busy_threads.remove(thread_id)
        retried = _events(_post(sidecar, request)[2])
        assert "Extraction staged" in json.dumps(retried)
        assert len(revisions.extracted) == 1
        assert capability not in server._extractions
    finally:
        sidecar.close()


def test_tampered_extraction_fingerprint_consumes_confirmation_without_dispatch(
    tmp_path: Path,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    try:
        thread_id, widget = _create_source_extraction_plan(
            sidecar,
            b"%PDF-1.7\nowner source\n",
        )
        request = _confirmation_request(thread_id, widget)
        request["params"]["action"]["payload"]["request_fingerprint"] = "tampered"

        refused = _events(_post(sidecar, request)[2])

        assert revisions.extracted == []
        assert any(event["type"] == "error" and "stale" in event["message"] for event in refused)

        request["params"]["action"]["payload"]["request_fingerprint"] = REQUEST_FINGERPRINT
        replay = _events(_post(sidecar, request)[2])
        assert revisions.extracted == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    "message",
    [
        "which deck?",
        "is there anything else needed?",
        "Add polite and casual examples to every card",
    ],
)
def test_every_composer_message_is_forwarded_without_legacy_revision_actions(
    message: str,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, assistant, _legacy_widget, events = _create_chat(sidecar, message)

        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", message)
        ]
        assert revisions.chat_histories == [()]
        assert revisions.executed == []
        assert (
            f"Answer about data/decks/potential-practice.yaml: {message}"
            in json.dumps(
            assistant, ensure_ascii=False
            )
        )
        assert any(event["type"] == "stream_options" for event in events)
        assert next(event for event in events if event["type"] == "stream_options")[
            "stream_options"
        ] == {"allow_cancel": False}
        assert [event["text"] for event in events if event["type"] == "progress_update"] == [
            "Preparing answer",
            "Writing answer",
            "Staging proposed changes",
            "Saving answer",
        ]
        assert _widget_items(events) == []
    finally:
        sidecar.close()


def test_second_turn_receives_only_the_prior_user_and_assistant_text() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _assistant, _widget, _events_before = _create_chat(sidecar, "which deck?")

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(thread_id, "what did I just ask?"),
        )

        assert status == 200
        assert revisions.chatted == [
            ("potential-practice", "which deck?"),
            ("potential-practice", "what did I just ask?"),
        ]
        assert revisions.chat_histories == [
            (),
            (
                ("user", "which deck?"),
                ("assistant", "Answer about potential-practice: which deck?"),
            ),
        ]
        assert _widget_items(_events(body)) == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_history_drops_oldest_entries_before_the_thirteenth_prior_message() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _assistant, _widget, _ = _create_chat(sidecar, "message 1")
        for number in range(2, 9):
            assert (
                _post(
                    sidecar,
                    _followup_message_request(thread_id, f"message {number}"),
                )[0]
                == 200
            )

        eighth_history = revisions.chat_histories[-1]
        assert len(eighth_history) == 12
        assert eighth_history[0] == ("user", "message 2")
        assert eighth_history[-1] == (
            "assistant",
            "Answer about potential-practice: message 7",
        )
    finally:
        sidecar.close()


def test_history_drops_an_oversized_prior_reply_without_truncating_it() -> None:
    @dataclass
    class LargeReplyRevisions(_FakeRevisions):
        def chat(
            self,
            *,
            deck_scope: str,
            history: tuple[tuple[str, str], ...],
            message: str,
            progress: Any,
            preview: Any,
        ) -> ChatReply:
            del preview
            self.chatted.append((deck_scope, message))
            self.chat_histories.append(history)
            progress("Preparing answer")
            progress("Writing answer")
            progress("Saving answer")
            if not history:
                return ChatReply(text="大" * 24_000)
            return ChatReply(text="short answer")

    revisions = LargeReplyRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _assistant, _widget, _ = _create_chat(sidecar, "first")

        assert (
            _post(
                sidecar,
                _followup_message_request(thread_id, "second"),
            )[0]
            == 200
        )

        assert revisions.chat_histories == [(), (("user", "first"),)]
    finally:
        sidecar.close()


@pytest.mark.parametrize("focused", [True, False], ids=["focused", "unfocused"])
def test_planned_chat_reply_renders_one_action_card_directly(
    focused: bool,
) -> None:
    instruction = "Add usage notes to the selected cards"
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="I prepared one exact card revision for your review.",
            action=_action_plan(),
            action_instruction=instruction,
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, widget, events = _create_automatic_action(
            sidecar,
            focused=focused,
            message=instruction,
        )

        assert _widget_items(events) == [widget]
        wire = json.dumps(events, ensure_ascii=False)
        assert "I prepared one exact card revision for your review." in wire
        assert "Confirm this exact action" in wire
        assert "Prepare this message as a deck change" not in wire
        assert "janki.revision.prepare" not in wire
        assert widget["widget"]["confirm"]["action"]["type"] == (
            "janki.revision.confirm"
        )
        assert len(sidecar.server.assistant_core.server._plans) == 1
    finally:
        sidecar.close()


def test_automatic_confirmation_uses_the_validated_plan_progress_label() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact build is ready to confirm.",
            action=_action_plan(progress_label="Preparing build"),
            action_instruction="Build the selected deck",
        ),
        execution_result=RevisionExecution(
            message="The selected deck was built.",
            finish=None,
            complete=True,
        ),
        execution_progress=(),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_automatic_action(
            sidecar,
            focused=False,
        )

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        assert [
            event["text"]
            for event in events
            if event["type"] == "progress_update"
        ] == ["Preparing build"]
    finally:
        sidecar.close()


def test_automatic_audio_action_accepts_only_shared_named_progress() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact audio generation is ready to confirm.",
            action=_action_plan(progress_label="Preparing audio"),
            action_instruction="Generate audio for the selected cards",
        ),
        execution_result=RevisionExecution(
            message="The selected cards now have audio.",
            finish=None,
            complete=True,
        ),
        execution_progress=(
            "Creating audio",
            "Saving audio",
            "Cleaning up audio",
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_automatic_action(
            sidecar,
            focused=True,
        )

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        progress = [
            event["text"]
            for event in events
            if event["type"] == "progress_update"
        ]
        assert progress == [
            "Preparing audio",
            "Creating audio",
            "Saving audio",
            "Cleaning up audio",
        ]
        assert all("%" not in label and "percent" not in label for label in progress)
    finally:
        sidecar.close()


def test_automatic_generic_finish_accepts_only_its_named_progress() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact reviewed-card finish is ready to confirm.",
            action=_action_plan(progress_label="Preparing finish"),
            action_instruction="Apply and finish the reviewed cards",
        ),
        execution_result=RevisionExecution(
            message="The reviewed cards, audio, and package are complete.",
            finish=None,
            complete=True,
        ),
        execution_progress=(
            "Applying reviewed cards",
            "Saving finish receipt",
            "Creating card audio",
            "Saving finish receipt",
            "Building Anki package",
            "Saving finish receipt",
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_automatic_action(
            sidecar,
            focused=True,
        )

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        progress = [
            event["text"]
            for event in events
            if event["type"] == "progress_update"
        ]
        assert progress == [
            "Preparing finish",
            "Applying reviewed cards",
            "Saving finish receipt",
            "Creating card audio",
            "Saving finish receipt",
            "Building Anki package",
            "Saving finish receipt",
        ]
        assert all("%" not in label and "percent" not in label for label in progress)
    finally:
        sidecar.close()


def test_action_plan_refuses_an_unknown_progress_label() -> None:
    with pytest.raises(ValueError, match="unknown progress state"):
        assistant_module._validate_plan(
            _action_plan(progress_label="Generating audio: 50%")
        )


@pytest.mark.parametrize(
    "progress_label",
    ["Deleting canonical cards", "Deleting deck definition"],
)
def test_action_plan_accepts_exact_deletion_progress_labels(
    progress_label: str,
) -> None:
    assert (
        assistant_module._validate_plan(
            _action_plan(progress_label=progress_label)
        ).progress_label
        == progress_label
    )


@pytest.mark.parametrize(
    "progress_label",
    [
        "Preparing finish",
        "Applying reviewed cards",
        "Creating card audio",
        "Building Anki package",
        "Saving finish receipt",
    ],
    ids=["prepare", "apply", "audio", "build", "receipt"],
)
def test_action_plan_accepts_exact_generic_finish_progress_labels(
    progress_label: str,
) -> None:
    assert (
        assistant_module._validate_plan(
            _action_plan(progress_label=progress_label)
        ).progress_label
        == progress_label
    )


def test_automatic_confirmation_is_one_use() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise the focused cards",
        ),
        execution_result=RevisionExecution(
            message="The exact revision is complete.",
            finish=None,
            complete=True,
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_automatic_action(
            sidecar,
            focused=True,
        )
        request = _confirmation_request(thread_id, widget)

        first = _events(_post(sidecar, request)[2])
        replay = _events(_post(sidecar, request)[2])

        assert len(revisions.executed) == 1
        assert "exact revision is complete" in json.dumps(first)
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in replay
        )
    finally:
        sidecar.close()


def test_busy_refusal_preserves_valid_generic_confirmation_for_retry() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise the focused cards",
        ),
        execution_result=RevisionExecution(
            message="The exact revision is complete.",
            finish=None,
            complete=True,
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=True)
        request = _confirmation_request(thread_id, widget)
        capability = request["params"]["action"]["payload"]["capability"]
        server._busy_threads.add(thread_id)

        refused = _events(_post(sidecar, request)[2])

        assert any(
            event["type"] == "error" and "Wait for" in event["message"]
            for event in refused
        )
        assert capability in server._plans
        assert revisions.executed == []

        server._busy_threads.remove(thread_id)
        retried = _events(_post(sidecar, request)[2])
        assert "exact revision is complete" in json.dumps(retried)
        assert len(revisions.executed) == 1
        assert capability not in server._plans
    finally:
        sidecar.close()


def test_completed_automatic_action_enters_the_next_turn_history() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact build is ready to confirm.",
            action=_action_plan(progress_label="Preparing build"),
            action_instruction="Build the selected deck",
        ),
        execution_result=RevisionExecution(
            message="The selected deck was built.",
            finish=None,
            complete=True,
        ),
        execution_progress=(),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_automatic_action(
            sidecar,
            focused=False,
            message="Build the selected deck",
        )
        assert _post(sidecar, _confirmation_request(thread_id, widget))[0] == 200
        revisions.chat_reply = None

        assert (
            _post(
                sidecar,
                _followup_message_request(thread_id, "What happened?"),
            )[0]
            == 200
        )

        assert revisions.chat_histories[-1] == (
            ("user", "Build the selected deck"),
            ("assistant", "The exact build is ready to confirm."),
            ("assistant", "The selected deck was built."),
        )
    finally:
        sidecar.close()


def test_completed_extraction_enters_the_next_turn_history(tmp_path: Path) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
        inbox_root=tmp_path / "data" / "inbox",
    )
    sidecar.start()
    try:
        thread_id, widget = _create_source_extraction_plan(
            sidecar,
            b"%PDF-1.7\nowner source\n",
        )
        assert _post(sidecar, _confirmation_request(thread_id, widget))[0] == 200

        assert (
            _post(
                sidecar,
                _followup_message_request(thread_id, "What happened?"),
            )[0]
            == 200
        )

        assert revisions.chat_histories[-1] == (
            ("assistant", "Extraction staged 12 card proposals for owner review."),
        )
    finally:
        sidecar.close()


def test_completed_finish_enters_the_next_turn_history() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _review_events = _create_finish_review(sidecar)
        assert _post(sidecar, _confirmation_request(thread_id, widget))[0] == 200

        assert (
            _post(
                sidecar,
                _followup_message_request(thread_id, "What happened?"),
            )[0]
            == 200
        )

        assert revisions.chat_histories[-1][-2:] == (
            (
                "assistant",
                "The revision proposal is staged. Nothing has been accepted or applied.",
            ),
            (
                "assistant",
                "The reviewed revision, example audio, and Anki package are complete.",
            ),
        )
    finally:
        sidecar.close()


def test_automatic_confirmation_is_bound_to_its_thread() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise these cards",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_widget, _ = _create_automatic_action(
            sidecar,
            focused=False,
        )
        second_thread, second_widget, _ = _create_automatic_action(
            sidecar,
            focused=False,
        )
        first_action = first_widget["widget"]["confirm"]["action"]
        capability = first_action["payload"]["capability"]
        bindings = sidecar.server.assistant_core.server._plans
        first_binding = bindings[capability]
        bindings[capability] = type(first_binding)(
            confirmation=first_binding.confirmation,
            thread_id=first_binding.thread_id,
            widget_item_id=second_widget["id"],
            deck_id=first_binding.deck_id,
            selection_epoch=first_binding.selection_epoch,
            progress_label=first_binding.progress_label,
        )

        refused = _events(
            _post(
                sidecar,
                _action_request(second_thread, second_widget, first_action),
            )[2]
        )
        replay = _events(
            _post(
                sidecar,
                _action_request(first_thread, first_widget, first_action),
            )[2]
        )

        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "belongs elsewhere" in event["message"]
            for event in refused
        )
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in replay
        )
    finally:
        sidecar.close()


def test_automatic_confirmation_is_bound_to_its_sender() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise these cards",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=False)
        selector_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Choose a deck to focus on"),
            )[2]
        )
        other_sender = _widget_items(selector_events)[0]
        action = widget["widget"]["confirm"]["action"]

        refused = _events(
            _post(sidecar, _action_request(thread_id, other_sender, action))[2]
        )
        replay = _events(
            _post(sidecar, _action_request(thread_id, widget, action))[2]
        )

        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "belongs elsewhere" in event["message"]
            for event in refused
        )
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in replay
        )
    finally:
        sidecar.close()


def test_automatic_confirmation_is_bound_to_the_rendered_fingerprint() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise these cards",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=False)
        request = _confirmation_request(thread_id, widget)
        request["params"]["action"]["payload"]["request_fingerprint"] = "tampered"

        refused = _events(_post(sidecar, request)[2])
        request["params"]["action"]["payload"]["request_fingerprint"] = (
            REQUEST_FINGERPRINT
        )
        replay = _events(_post(sidecar, request)[2])

        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "stale" in event["message"]
            for event in refused
        )
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in replay
        )
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    ("focused", "next_deck"),
    [(False, "potential"), (True, "te-form")],
    ids=["adding-focus", "switching-focus"],
)
def test_focus_change_invalidates_an_automatic_action(
    focused: bool,
    next_deck: str,
) -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise these cards",
        )
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=focused)
        selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose a deck to focus on"),
                )[2]
            )
        )[0]
        selected = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    selector,
                    _deck_selector_action(selector, next_deck),
                ),
            )[2]
        )

        refused = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        assert not any(event["type"] == "error" for event in selected)
        assert revisions.executed == []
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "already used" in event["message"])
            for event in refused
        )
    finally:
        sidecar.close()


def test_unfocused_automatic_action_can_target_another_deck_and_complete() -> None:
    target = "data/decks/lesson-13-vocabulary.yaml"
    instruction = "Add a usage note to the vocabulary cards"
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact cross-library revision is ready to confirm.",
            action=_action_plan(target=target),
            action_instruction=instruction,
        ),
        execution_result=RevisionExecution(
            message="The requested card changes were staged.",
            finish=None,
            complete=True,
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=False)

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        assert len(revisions.executed) == 1
        confirmation = revisions.executed[0]
        assert confirmation.deck_scope == ""
        assert confirmation.target == target
        assert confirmation.instruction == instruction
        assert confirmation.expected_fingerprint == REQUEST_FINGERPRINT
        assert "requested card changes were staged" in json.dumps(events)
        assert _widget_items(events) == []
        assert not any(event["type"] == "notice" for event in events)
        assert "unavailable" not in json.dumps(events).casefold()
    finally:
        sidecar.close()


def test_automatic_action_review_required_is_an_informational_notice() -> None:
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact revision is ready to confirm.",
            action=_action_plan(),
            action_instruction="Revise these cards",
        ),
        execution_result=RevisionExecution(
            message="The proposed card changes are staged.",
            finish=None,
            review_required="Review the exact proposed Japanese in staging.",
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_automatic_action(sidecar, focused=True)

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        notices = [event for event in events if event["type"] == "notice"]
        assert len(notices) == 1
        assert notices[0]["level"] == "info"
        assert notices[0]["title"] == "Review the proposed card changes"
        assert notices[0]["message"] == (
            "Review the exact proposed Japanese in staging."
        )
        assert "unavailable" not in json.dumps(events).casefold()
        assert _widget_items(events) == []
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    "reply",
    [
        ChatReply(
            text="An answer without an action.",
            action_instruction="This instruction has no action.",
        ),
        ChatReply(
            text="An action without its bound instruction.",
            action=_action_plan(),
        ),
        ChatReply(
            text="An action with a blank instruction.",
            action=_action_plan(),
            action_instruction="  ",
        ),
    ],
    ids=["instruction-without-action", "action-without-instruction", "blank-instruction"],
)
def test_invalid_planned_chat_reply_combinations_are_refused(reply: ChatReply) -> None:
    with pytest.raises(ValueError):
        assistant_module._validate_chat_reply(reply)


def test_confirm_stages_one_proposal_and_renders_one_exact_finish_review() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)
        request = _confirmation_request(thread_id, widget)

        status, _headers, body = _post(sidecar, request)
        assert status == 200
        events = _events(body)
        progress = [event["text"] for event in events if event["type"] == "progress_update"]
        assert progress == [
            "Preparing revision",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ]
        assert all("%" not in label and "percent" not in label for label in progress)
        assert len(revisions.executed) == 1
        confirmation = revisions.executed[0]
        assert confirmation.deck_scope == "potential-practice"
        assert confirmation.instruction == "Add polite and casual examples"
        assert confirmation.expected_fingerprint == REQUEST_FINGERPRINT
        assert b"Nothing has been accepted or applied" in body
        assert b"http://127.0.0.1:" not in body
        finish_widgets = _widget_items(events)
        assert len(finish_widgets) == 1
        finish_widget = finish_widgets[0]
        wire = json.dumps(finish_widget, ensure_ascii=False)
        assert "Old potential note." in wire
        assert "Potential expresses ability or possibility." in wire
        assert "今は遊べません。" in wire
        assert "明日は遊べます。" in wire
        assert "Today" not in wire
        assert "openai-realtime · gpt-realtime-1.5 · paid-network" in wire
        assert "Total clips: 2" in wire
        assert "Recoverable exact clips: 1" in wire
        assert "Provider calls required: 1" in wire
        assert "dist/potential.apkg · cards: 16" in wire
        assert FINISH_FINGERPRINT in wire
        finish_action = finish_widget["widget"]["confirm"]
        assert finish_action["label"] == "Apply and finish"
        assert finish_action["action"]["type"] == "janki.revision.finish"
        assert finish_action["action"]["handler"] == "server"

        repeated = _events(_post(sidecar, request)[2])
        assert len(revisions.executed) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in repeated
        )
    finally:
        sidecar.close()


def test_blank_current_furigana_keeps_the_staged_revision_reviewable() -> None:
    @dataclass
    class BlankCurrentFurigana(_FakeRevisions):
        def consume_replan_and_execute(
            self,
            confirmation: RevisionConfirmation,
            *,
            progress: Any,
        ) -> RevisionExecution:
            result = super().consume_replan_and_execute(
                confirmation,
                progress=progress,
            )
            assert result.finish is not None
            record = result.finish.records[0]
            current = (
                replace(record.current_examples[0], furigana=""),
                record.current_examples[1],
            )
            return replace(
                result,
                finish=replace(
                    result.finish,
                    records=(replace(record, current_examples=current),),
                ),
            )

    revisions = BlankCurrentFurigana()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)

        events = _events(_post(sidecar, _confirmation_request(thread_id, widget))[2])

        wire = json.dumps(events, ensure_ascii=False)
        assert "revision proposal is staged" in wire
        assert "今は遊べません。" in wire
        finish_widgets = _widget_items(events)
        assert len(finish_widgets) == 1
        assert finish_widgets[0]["widget"]["confirm"]["label"] == "Apply and finish"
        assert len(sidecar.server.assistant_core.server._finishes) == 1
    finally:
        sidecar.close()


def test_invalid_proposed_review_preserves_the_paid_staged_deliverable() -> None:
    @dataclass
    class BlankProposedFurigana(_FakeRevisions):
        def consume_replan_and_execute(
            self,
            confirmation: RevisionConfirmation,
            *,
            progress: Any,
        ) -> RevisionExecution:
            result = super().consume_replan_and_execute(
                confirmation,
                progress=progress,
            )
            assert result.finish is not None
            record = result.finish.records[0]
            proposed = (
                replace(record.proposed_examples[0], furigana=""),
                record.proposed_examples[1],
            )
            return replace(
                result,
                finish=replace(
                    result.finish,
                    records=(replace(record, proposed_examples=proposed),),
                ),
            )

    revisions = BlankProposedFurigana()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)
        request = _confirmation_request(thread_id, widget)

        events = _events(_post(sidecar, request)[2])

        wire = json.dumps(events)
        assert "revision proposal is staged" in wire
        assert "staged proposal remains the durable deliverable" in wire
        assert "do not repeat the paid revision call" in wire
        assert _widget_items(events) == []
        assert sidecar.server.assistant_core.server._finishes == {}
        assert len(revisions.executed) == 1

        replay = _events(_post(sidecar, request)[2])
        assert len(revisions.executed) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_finish_widget_render_failure_preserves_staging_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server

    def fail_render(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("widget schema unavailable")

    monkeypatch.setattr(type(server), "_finish_widget", staticmethod(fail_render))
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)

        events = _events(_post(sidecar, _confirmation_request(thread_id, widget))[2])

        wire = json.dumps(events)
        assert "revision proposal is staged" in wire
        assert "widget schema unavailable" in wire
        assert "staged proposal remains the durable deliverable" in wire
        assert _widget_items(events) == []
        assert server._finishes == {}
        assert len(revisions.executed) == 1
    finally:
        sidecar.close()


def test_apply_and_finish_is_one_action_with_only_shared_named_progress() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _review_events = _create_finish_review(sidecar)
        request = _confirmation_request(thread_id, widget)

        status, _headers, body = _post(sidecar, request)

        assert status == 200
        events = _events(body)
        assert [event["text"] for event in events if event["type"] == "progress_update"] == [
            "Preparing finish",
            "Applying reviewed revision",
            "Creating example audio",
            "Building Anki package",
            "Saving finish receipt",
        ]
        assert b"reviewed revision, example audio, and Anki package are complete" in body
        assert _widget_items(events) == []
        assert revisions.finish_executed == [
            RevisionFinishConfirmation(
                preparation_id="finish-preparation-1",
                expected_fingerprint=FINISH_FINGERPRINT,
                target="potential-practice",
            )
        ]

        replay = _events(_post(sidecar, request)[2])
        assert len(revisions.finish_executed) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_generic_apply_and_finish_renders_exact_effects_and_named_progress() -> None:
    target = "data/decks/lesson-13-vocabulary.yaml"
    effects = (
        "Review word:遊ぶ:あそぶ meanings: [to play] → [to play; to hang out]",
        "Create the selected card's word and example audio",
        "Build dist/lesson-13.apkg from data/decks/lesson-13.yaml",
    )
    revisions = _FakeRevisions(
        execution_result=RevisionExecution(
            message="The paid proposal is staged and ready for exact review.",
            finish=StagedContentFinishReview(
                preparation_id="generic-finish-preparation-1",
                request_fingerprint=FINISH_FINGERPRINT,
                target=target,
                effects=effects,
                disclosures=(
                    "One provider-backed audio clip is still required.",
                    "The finish receipt prevents repeating paid audio on resume.",
                ),
            ),
        ),
        finish_progress=(
            "Preparing finish",
            "Applying reviewed cards",
            "Creating card audio",
            "Building Anki package",
            "Saving finish receipt",
        ),
        finish_result=RevisionFinishExecution(
            message="The reviewed cards, audio, and Anki package are complete.",
            receipt_id=FINISH_FINGERPRINT,
            state="complete",
            target=target,
            output_path="dist/lesson-13.apkg",
            package_sha256=PACKAGE_FINGERPRINT,
            card_count=4,
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, review_events = _create_finish_review(sidecar)

        review_wire = json.dumps(review_events, ensure_ascii=False)
        assert "Review this exact content and finish" in review_wire
        assert all(effect in review_wire for effect in effects)
        assert widget["widget"]["confirm"]["label"] == "Apply and finish"

        events = _events(
            _post(sidecar, _confirmation_request(thread_id, widget))[2]
        )

        assert [
            event["text"]
            for event in events
            if event["type"] == "progress_update"
        ] == list(revisions.finish_progress)
        assert b"reviewed cards, audio, and Anki package are complete" in json.dumps(
            events
        ).encode()
        assert revisions.finish_executed == [
            RevisionFinishConfirmation(
                preparation_id="generic-finish-preparation-1",
                expected_fingerprint=FINISH_FINGERPRINT,
                target=target,
            )
        ]
    finally:
        sidecar.close()


def test_unfocused_whole_deck_revision_can_apply_and_finish() -> None:
    target = "data/decks/potential-practice.yaml"
    instruction = "Revise every drill example in the potential deck"
    revisions = _FakeRevisions(
        chat_reply=ChatReply(
            text="The exact whole-deck revision is ready to confirm.",
            action=_action_plan(target=target),
            action_instruction=instruction,
        ),
        execution_result=RevisionExecution(
            message="The revision proposal is staged for review.",
            finish=_finish_review(target=target),
        ),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, revision_widget, _events_before = _create_automatic_action(
            sidecar,
            focused=False,
            message=instruction,
        )
        review_events = _events(
            _post(
                sidecar,
                _confirmation_request(thread_id, revision_widget),
            )[2]
        )
        finish_widget = _widget_items(review_events)[0]

        finish_events = _events(
            _post(sidecar, _confirmation_request(thread_id, finish_widget))[2]
        )

        assert not any(event["type"] == "error" for event in finish_events)
        assert revisions.finish_executed == [
            RevisionFinishConfirmation(
                preparation_id="finish-preparation-1",
                expected_fingerprint=FINISH_FINGERPRINT,
                target=target,
            )
        ]
    finally:
        sidecar.close()


def test_busy_refusal_preserves_valid_finish_confirmation_for_retry() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        thread_id, widget, _ = _create_finish_review(sidecar)
        request = _confirmation_request(thread_id, widget)
        capability = request["params"]["action"]["payload"]["capability"]
        server._busy_threads.add(thread_id)

        refused = _events(_post(sidecar, request)[2])

        assert any(
            event["type"] == "error" and "Wait for" in event["message"]
            for event in refused
        )
        assert capability in server._finishes
        assert revisions.finish_executed == []

        server._busy_threads.remove(thread_id)
        retried = _events(_post(sidecar, request)[2])
        assert "reviewed revision" in json.dumps(retried)
        assert len(revisions.finish_executed) == 1
        assert capability not in server._finishes
    finally:
        sidecar.close()


def test_tampered_finish_fingerprint_consumes_exact_capability() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _review_events = _create_finish_review(sidecar)
        request = _confirmation_request(thread_id, widget)
        request["params"]["action"]["payload"]["request_fingerprint"] = "0" * 64

        refused = _events(_post(sidecar, request)[2])

        assert revisions.finish_executed == []
        assert any(event["type"] == "error" and "stale" in event["message"] for event in refused)
        request["params"]["action"]["payload"]["request_fingerprint"] = FINISH_FINGERPRINT
        replay = _events(_post(sidecar, request)[2])
        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_finish_capability_is_consumed_on_cross_thread_use() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_widget, _ = _create_finish_review(
            sidecar,
            "first revision",
        )
        second_thread, second_widget, _ = _create_finish_review(
            sidecar,
            "second revision",
        )
        first_action = first_widget["widget"]["confirm"]["action"]
        capability = first_action["payload"]["capability"]
        finish_bindings = sidecar.server.assistant_core.server._finishes
        first_binding = finish_bindings[capability]
        # Isolate the thread binding from the independently tested sender
        # binding: make the action's sender valid for the second thread while
        # preserving the capability's original thread id.
        finish_bindings[capability] = type(first_binding)(
            confirmation=first_binding.confirmation,
            thread_id=first_binding.thread_id,
            widget_item_id=second_widget["id"],
            deck_id=first_binding.deck_id,
            selection_epoch=first_binding.selection_epoch,
        )

        refused = _events(
            _post(
                sidecar,
                _action_request(second_thread, second_widget, first_action),
            )[2]
        )

        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error" and "belongs elsewhere" in event["message"]
            for event in refused
        )
        replay = _events(
            _post(
                sidecar,
                _action_request(first_thread, first_widget, first_action),
            )[2]
        )
        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_finish_capability_is_consumed_on_same_thread_sender_mismatch() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, finish_widget, _ = _create_finish_review(sidecar)
        followup_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Choose a deck to focus on"),
            )[2]
        )
        [other_sender] = _widget_items(followup_events)
        finish_action = finish_widget["widget"]["confirm"]["action"]

        refused = _events(
            _post(
                sidecar,
                _action_request(thread_id, other_sender, finish_action),
            )[2]
        )

        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error" and "belongs elsewhere" in event["message"]
            for event in refused
        )
        replay = _events(
            _post(
                sidecar,
                _action_request(thread_id, finish_widget, finish_action),
            )[2]
        )
        assert revisions.finish_executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )
    finally:
        sidecar.close()


def test_staged_proposal_without_finish_plan_shows_truth_and_no_action() -> None:
    @dataclass
    class FinishUnavailable(_FakeRevisions):
        def consume_replan_and_execute(
            self,
            confirmation: RevisionConfirmation,
            *,
            progress: Any,
        ) -> RevisionExecution:
            self.executed.append(confirmation)
            progress("Reading the source")
            progress("Checking the answer's shape")
            progress("Saving proposals")
            return RevisionExecution(
                message=("The paid proposal is staged. Do not repeat the revision call."),
                finish=None,
                finish_unavailable=(
                    "Apply and finish cannot be reviewed because a template is missing."
                ),
            )

    revisions = FinishUnavailable()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)

        events = _events(_post(sidecar, _confirmation_request(thread_id, widget))[2])

        wire = json.dumps(events)
        assert "paid proposal is staged" in wire
        assert "Do not repeat the revision call" in wire
        assert "template is missing" in wire
        assert _widget_items(events) == []
        assert revisions.finish_executed == []
    finally:
        sidecar.close()


def test_tampered_fingerprint_consumes_the_capability_without_execution() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)
        request = _confirmation_request(thread_id, widget)
        request["params"]["action"]["payload"]["request_fingerprint"] = "tampered"

        first = _events(_post(sidecar, request)[2])
        assert revisions.executed == []
        assert any(event["type"] == "error" and "stale" in event["message"] for event in first)

        request["params"]["action"]["payload"]["request_fingerprint"] = REQUEST_FINGERPRINT
        second = _events(_post(sidecar, request)[2])
        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in second
        )
    finally:
        sidecar.close()


@pytest.mark.parametrize(
    "old_action",
    [
        "janki.revision.prepare",
        "janki.revision.apply.prepare",
        "janki.revision.apply.confirm",
        "janki.revision.audio.prepare",
        "janki.revision.audio.confirm",
        "janki.revision.build.prepare",
        "janki.revision.build.confirm",
    ],
)
def test_invented_legacy_followup_actions_are_refused(old_action: str) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _ = _create_plan(sidecar)
        invented = _action_request(
            thread_id,
            widget,
            {"type": old_action, "payload": {"capability": "invented"}},
        )

        events = _events(_post(sidecar, invented)[2])

        assert revisions.executed == []
        assert any(
            event["type"] == "error" and event["message"] == "That assistant action was refused."
            for event in events
        )
    finally:
        sidecar.close()


def _preview_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["type"]
        in {"thread.item.added", "thread.item.updated", "thread.item.removed"}
    ]


def _assistant_done_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event["item"]
        for event in events
        if event["type"] == "thread.item.done"
        and event["item"]["type"] == "assistant_message"
    ]


def test_chat_streams_preview_then_replaces_it_with_the_validated_answer() -> None:
    """One message item is streamed into and then finished, not two.

    The preview and the answer are the same turn. Finishing under a new id
    would leave the streamed text in the transcript beside the validated one.
    """

    revisions = _FakeRevisions(
        chat_preview=("Partial ", "answer"),
        chat_reply=ChatReply(text="The validated answer."),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, _assistant, _item, events = _create_chat(sidecar, "which deck?")

        added = [event for event in events if event["type"] == "thread.item.added"]
        assert len(added) == 1
        preview_id = added[0]["item"]["id"]
        assert added[0]["item"]["type"] == "assistant_message"
        assert added[0]["item"]["content"] == []
        updates = [
            event["update"]
            for event in events
            if event["type"] == "thread.item.updated"
            and event["item_id"] == preview_id
        ]
        assert updates[0]["type"] == "assistant_message.content_part.added"
        assert updates[0]["content"]["text"] == ""
        deltas = [
            update["delta"]
            for update in updates
            if update["type"] == "assistant_message.content_part.text_delta"
        ]
        assert "".join(deltas) == "Partial answer"
        assert updates[-1]["type"] == "assistant_message.content_part.done"
        assert updates[-1]["content"]["text"] == "The validated answer."
        done = _assistant_done_items(events)
        assert len(done) == 1
        assert done[0]["id"] == preview_id
        assert done[0]["content"] == [
            {
                "annotations": [],
                "text": "The validated answer.",
                "type": "output_text",
            }
        ]
    finally:
        sidecar.close()


def test_chat_preview_that_diverges_is_replaced_by_the_validated_answer() -> None:
    """The streamed prose is a draft, so the finished part is always rewritten.

    The real transport streams one pass and returns another; treating the
    preview as a prefix would leave a sentence nobody wrote in the transcript.
    """

    revisions = _FakeRevisions(
        chat_preview=("Wrong start",),
        chat_reply=ChatReply(text="Right answer"),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, _assistant, _item, events = _create_chat(sidecar, "which deck?")

        done_parts = [
            event["update"]
            for event in events
            if event["type"] == "thread.item.updated"
            and event["update"]["type"] == "assistant_message.content_part.done"
        ]
        assert [part["content"]["text"] for part in done_parts] == ["Right answer"]
        assert _assistant_done_items(events)[0]["content"][0]["text"] == "Right answer"
    finally:
        sidecar.close()


def test_chat_without_preview_keeps_the_single_done_message() -> None:
    """A turn whose provider streams nothing still emits exactly one message."""
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, _assistant, _item, events = _create_chat(sidecar, "which deck?")

        assert _preview_events(events) == []
        assert len(_assistant_done_items(events)) == 1
    finally:
        sidecar.close()


def test_a_refused_turn_removes_its_preview() -> None:
    """A refusal keeps no half-said answer on screen.

    Streamed prose is not an answer janki validated or journaled; leaving it
    beside the refusal would read as though part of it still stood.
    """

    revisions = _FakeRevisions(
        chat_preview=("Half an ", "answer"),
        chat_refusal="The Assistant context changed before the call.",
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _selector_events = _start_deck_selector(sidecar)
        selectable = next(
            choice for choice in sidecar.server.deck_choices if choice.revision_supported
        )
        _post(
            sidecar,
            _action_request(
                thread_id,
                selector,
                _deck_selector_action(selector, selectable.deck_id),
            ),
        )
        status, _headers, body = _post(
            sidecar,
            _followup_message_request(thread_id, "which deck?"),
        )

        assert status == 200
        events = _events(body)
        added = [event for event in events if event["type"] == "thread.item.added"]
        removed = [event for event in events if event["type"] == "thread.item.removed"]
        assert len(added) == 1
        assert [event["item_id"] for event in removed] == [added[0]["item"]["id"]]
        notices = [event for event in events if event["type"] == "notice"]
        assert notices[-1]["title"] == "Answer refused"
        assert events.index(removed[0]) < events.index(notices[-1])
        assert [event for event in events if event["type"] == "error"] == []
        assert _assistant_done_items(events) == []
    finally:
        sidecar.close()


def test_any_failure_after_a_preview_removes_it() -> None:
    """A crash is not gentler than a refusal, so it takes the preview too.

    A refusal is the failure this turn expects; every other one leaves the
    same unvalidated prose on screen, where the generic error that follows
    would read as an aside beside half an answer that still stands.
    """

    @dataclass
    class CrashingRevisions(_FakeRevisions):
        def chat(
            self,
            *,
            deck_scope: str,
            history: tuple[tuple[str, str], ...],
            message: str,
            progress: Any,
            preview: Any,
        ) -> ChatReply:
            preview("Half an ")
            preview("answer")
            raise RuntimeError("the chat service has a bug")

    sidecar = create_assistant_sidecar(
        CrashingRevisions(),
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _selector_events = _start_deck_selector(sidecar)
        selectable = next(
            choice for choice in sidecar.server.deck_choices if choice.revision_supported
        )
        _post(
            sidecar,
            _action_request(
                thread_id,
                selector,
                _deck_selector_action(selector, selectable.deck_id),
            ),
        )
        status, _headers, body = _post(
            sidecar,
            _followup_message_request(thread_id, "which deck?"),
        )

        assert status == 200
        events = _events(body)
        added = [event for event in events if event["type"] == "thread.item.added"]
        removed = [event for event in events if event["type"] == "thread.item.removed"]
        errors = [event for event in events if event["type"] == "error"]
        assert len(added) == 1
        assert [event["item_id"] for event in removed] == [added[0]["item"]["id"]]
        assert errors
        assert events.index(removed[0]) < events.index(errors[0])
        assert _assistant_done_items(events) == []
    finally:
        sidecar.close()


def test_progress_and_preview_keep_one_order() -> None:
    """Narration and prose share one queue, so the transcript keeps their order.

    Two queues drained one after the other would show every progress state
    before every delta, whatever the provider actually reported when.
    """

    @dataclass
    class InterleavingRevisions(_FakeRevisions):
        def chat(
            self,
            *,
            deck_scope: str,
            history: tuple[tuple[str, str], ...],
            message: str,
            progress: Any,
            preview: Any,
        ) -> ChatReply:
            self.chatted.append((deck_scope, message))
            self.chat_histories.append(history)
            progress("Preparing answer")
            preview("first ")
            progress("Writing answer")
            preview("second")
            progress("Saving answer")
            return ChatReply(text="The validated answer.")

    sidecar = create_assistant_sidecar(
        InterleavingRevisions(),
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, _assistant, _item, events = _create_chat(sidecar, "which deck?")

        narration = [
            (
                event["text"]
                if event["type"] == "progress_update"
                else event["update"]["delta"]
            )
            for event in events
            if event["type"] == "progress_update"
            or (
                event["type"] == "thread.item.updated"
                and event["update"]["type"]
                == "assistant_message.content_part.text_delta"
            )
        ]
        assert narration == [
            "Preparing answer",
            "first ",
            "Writing answer",
            "second",
            "Saving answer",
        ]
    finally:
        sidecar.close()


def test_stream_cancellation_does_not_persist_an_unvalidated_preview() -> None:
    """A canceled stream leaves nothing behind, not a half-streamed message.

    ChatKit's default saves unfinished assistant messages; here the unfinished
    item is a preview of an answer no Assistant manifest ever recorded.
    """

    from chatkit.types import (
        AssistantMessageContent,
        AssistantMessageItem,
        ThreadMetadata,
    )

    core = create_assistant_core(
        _FakeRevisions(),
        deck_choices=_assistant_deck_choices(),
    )

    async def exercise() -> list[Any]:
        thread = ThreadMetadata(
            id=core.store.generate_thread_id(core.context),
            created_at=datetime.now(),
        )
        await core.store.save_thread(thread, core.context)
        pending = AssistantMessageItem(
            id=core.store.generate_item_id("message", thread, core.context),
            thread_id=thread.id,
            created_at=datetime.now(),
            content=[AssistantMessageContent(text="Half an answer")],
        )

        await core.server.handle_stream_cancelled(thread, [pending], core.context)

        page = await core.store.load_thread_items(
            thread.id, None, 20, "asc", core.context
        )
        return list(page.data)

    assert asyncio.run(exercise()) == []


def test_browser_disconnect_does_not_cancel_a_dispatched_chat_turn() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    revisions = _FakeRevisions(
        chat_entered=entered,
        chat_release=release,
        chat_finished=finished,
        chat_preview=("A partial ",),
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, selector, _events_before = _start_deck_selector(sidecar)
        selected = next(
            choice for choice in sidecar.server.deck_choices if choice.revision_supported
        )
        _post(
            sidecar,
            _action_request(
                thread_id,
                selector,
                _deck_selector_action(selector, selected.deck_id),
            ),
        )
        payload = json.dumps(_followup_message_request(thread_id, "which deck?")).encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request(
            "POST",
            sidecar.server.api_path,
            body=payload,
            headers={"Content-Type": "application/json", "Origin": sidecar.origin},
        )
        response = connection.getresponse()
        assert response.status == 200
        connection.close()

        assert entered.wait(3)
        release.set()
        assert finished.wait(3)
        assert revisions.chatted == [("potential-practice", "which deck?")]
    finally:
        release.set()
        sidecar.close()


def test_browser_disconnect_does_not_cancel_a_confirmed_durable_action() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    revisions = _FakeRevisions(entered=entered, release=release, finished=finished)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_plan(sidecar)
        payload = json.dumps(_confirmation_request(thread_id, widget)).encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request(
            "POST",
            sidecar.server.api_path,
            body=payload,
            headers={"Content-Type": "application/json", "Origin": sidecar.origin},
        )
        response = connection.getresponse()
        assert response.status == 200
        connection.close()

        assert entered.wait(3)
        release.set()
        assert finished.wait(3)
        assert len(revisions.executed) == 1
    finally:
        release.set()
        sidecar.close()


def test_browser_disconnect_does_not_cancel_apply_and_finish() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    revisions = _FakeRevisions(
        finish_entered=entered,
        finish_release=release,
        finish_finished=finished,
    )
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, widget, _events_before = _create_finish_review(sidecar)
        payload = json.dumps(_confirmation_request(thread_id, widget)).encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request(
            "POST",
            sidecar.server.api_path,
            body=payload,
            headers={"Content-Type": "application/json", "Origin": sidecar.origin},
        )
        response = connection.getresponse()
        assert response.status == 200
        connection.close()

        assert entered.wait(3)
        release.set()
        assert finished.wait(3)
        assert len(revisions.finish_executed) == 1
    finally:
        release.set()
        sidecar.close()


def test_memory_store_refuses_a_different_deck_context() -> None:
    async def exercise() -> None:
        store = ScopedMemoryStore("deck-a")
        with pytest.raises(PermissionError, match="another deck scope"):
            store.generate_thread_id(AssistantRequestContext(deck_scope="deck-b"))

    asyncio.run(exercise())
