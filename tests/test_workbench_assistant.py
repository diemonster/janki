"""The isolated conversational ChatKit workbench controller.

Every callback is fake. These tests exercise the ChatKit wire protocol and the
loopback HTTP boundary without contacting OpenAI or another provider.
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from chatkit.icons import IconName
from pydantic import TypeAdapter

from japanese_anki.workbench.assistant import (
    AssistantDeckChoice,
    AssistantRequestContext,
    ChatReply,
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
    create_assistant_core,
)
from japanese_anki.workbench.assistant_http import (
    create_assistant_sidecar as _create_assistant_sidecar,
)

SESSION_TOKEN = "assistant-session-token-000000000000"
REQUEST_FINGERPRINT = "0123456789abcdef" * 4
FINISH_FINGERPRINT = "fedcba9876543210" * 4
PACKAGE_FINGERPRINT = "abcdef0123456789" * 4


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
    prepared: list[tuple[str, str]] = field(default_factory=list)
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
    finish_entered: threading.Event | None = None
    finish_release: threading.Event | None = None
    finish_finished: threading.Event | None = None
    resolve_block_for: str | None = None
    resolve_entered: threading.Event | None = None
    resolve_release: threading.Event | None = None
    resolve_refusal: str | None = None
    deck_choices: tuple[AssistantDeckChoice, ...] = ()

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
    ) -> ChatReply:
        self.chatted.append((deck_scope, message))
        self.chat_histories.append(history)
        if self.chat_entered is not None:
            self.chat_entered.set()
        if self.chat_release is not None and not self.chat_release.wait(3):
            raise AssertionError("the test never released the fake chat")
        progress("Preparing answer")
        progress("Writing answer")
        progress("Saving answer")
        if self.chat_finished is not None:
            self.chat_finished.set()
        return ChatReply(text=f"Answer about {deck_scope}: {message}")

    def prepare_revision(self, *, deck_scope: str, instruction: str) -> RevisionPlan:
        self.prepared.append((deck_scope, instruction))
        return RevisionPlan(
            request_fingerprint=REQUEST_FINGERPRINT,
            target=deck_scope,
            effects=(
                "propose one polite and one casual example per selected card",
                "stage the proposal without changing the deck",
            ),
            disclosures=("The confirmed revision is a paid OpenAI API call.",),
        )

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
        progress("Reading the source")
        progress("Checking the answer's shape")
        progress("Saving proposals")
        if self.finished is not None:
            self.finished.set()
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
        progress("Preparing finish")
        progress("Applying reviewed revision")
        progress("Creating example audio")
        progress("Building Anki package")
        progress("Saving finish receipt")
        if self.finish_finished is not None:
            self.finish_finished.set()
        return RevisionFinishExecution(
            message=("The reviewed revision, example audio, and Anki package are complete."),
            receipt_id=FINISH_FINGERPRINT,
            state="complete",
            target="data/decks/potential.yaml",
            output_path="dist/potential.apkg",
            package_sha256=PACKAGE_FINGERPRINT,
            card_count=16,
        )

    def prepare_source_extraction(self, *, source_path: Path) -> SourceExtractionPlan:
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
    return _create_assistant_sidecar(
        callbacks,
        deck_choices=deck_choices,
        **kwargs,
    )


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


def _prepare_action(widget: dict[str, Any]) -> dict[str, Any]:
    button = next(child for child in widget["widget"]["children"] if child["type"] == "Button")
    return button["onClickAction"]


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
    widget = _widget_items(events)[0]
    return thread_id, assistant, widget, events


def _create_plan(
    sidecar: Any,
    message: str = "Add polite and casual examples",
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    thread_id, _assistant, prepare_widget, _chat_events = _create_chat(sidecar, message)
    prepare_events = _events(
        _post(
            sidecar,
            _action_request(thread_id, prepare_widget, _prepare_action(prepare_widget)),
        )[2]
    )
    return thread_id, _widget_items(prepare_events)[0], prepare_events


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


def _start_deck_selector(
    sidecar: Any,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    status, _headers, body = _post(
        sidecar,
        _message_request("Choose an active deck"),
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


def test_revision_supported_deck_must_also_support_chat() -> None:
    with pytest.raises(
        ValueError,
        match="Every revision-supported deck must also support chat",
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
        assert revisions.prepared == []
        assert revisions.executed == []
        assert [event for event in events if event["type"] == "progress_update"] == []
        assert [event for event in events if event["type"] == "notice"] == []

        root = selector["widget"]
        assert root["type"] == "ListView"
        assert root["status"]["text"] == "Choose the active deck"
        assert [child["type"] for child in root["children"]] == [
            "ListViewItem",
            "ListViewItem",
            "ListViewItem",
        ]
        wire = json.dumps(root, ensure_ascii=False)
        for choice in _assistant_deck_choices():
            assert choice.label in wire
            assert choice.scope not in wire
        assert _assistant_deck_choices()[-1].unavailable_reason in wire

        actions = []
        for choice, row in zip(_assistant_deck_choices(), root["children"], strict=True):
            if choice.chat_supported:
                actions.append(_deck_selector_action(selector, choice.deck_id))
            else:
                assert "onClickAction" not in row
        badges = [
            next(
                child["label"]
                for child in row["children"][0]["children"]
                if child["type"] == "Badge"
            )
            for row in root["children"]
        ]
        assert badges == ["Chat + changes", "Chat + changes", "Chat only"]
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


def test_unselected_new_thread_does_not_guess_a_deck_or_call_chat() -> None:
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
        assert revisions.chatted == []
        assert revisions.chat_histories == []
        assert revisions.prepared == []
        events = _events(body)
        assert len(_widget_items(events)) == 1
        assert "Choose the active deck" in json.dumps(events, ensure_ascii=False)
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
        assert len(_widget_items(_events(body))) == 1
    finally:
        sidecar.close()


def test_chat_only_deck_routes_exact_scope_without_offering_a_change_action() -> None:
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
        assert revisions.prepared == []
        assert revisions.executed == []
    finally:
        sidecar.close()


def test_switching_decks_clears_history_and_invalidates_old_change_bindings() -> None:
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
                _followup_message_request(thread_id, "Improve the first deck"),
            )[2]
        )
        old_prepare_widget = _widget_items(first_chat_events)[0]

        second_selector_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Choose an active deck"),
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

        stale_prepare = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    old_prepare_widget,
                    _prepare_action(old_prepare_widget),
                ),
            )[2]
        )
        assert revisions.prepared == []
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "belongs elsewhere" in event["message"])
            for event in stale_prepare
        )

        second_chat_events = _events(
            _post(
                sidecar,
                _followup_message_request(thread_id, "Improve the second deck"),
            )[2]
        )
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Improve the first deck"),
            ("data/decks/teform-drill.yaml", "Improve the second deck"),
        ]
        assert revisions.chat_histories == [(), ()]
        second_prepare_widget = _widget_items(second_chat_events)[0]
        plan_events = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    second_prepare_widget,
                    _prepare_action(second_prepare_widget),
                ),
            )[2]
        )
        plan_widget = _widget_items(plan_events)[0]
        assert revisions.prepared == [("data/decks/teform-drill.yaml", "Improve the second deck")]

        third_selector = _widget_items(
            _events(
                _post(
                    sidecar,
                    _followup_message_request(thread_id, "Choose an active deck"),
                )[2]
            )
        )[0]
        _post(
            sidecar,
            _action_request(
                thread_id,
                third_selector,
                _deck_selector_action(third_selector, "potential"),
            ),
        )
        stale_confirmation = _events(
            _post(sidecar, _confirmation_request(thread_id, plan_widget))[2]
        )

        assert revisions.executed == []
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "belongs elsewhere" in event["message"])
            for event in stale_confirmation
        )
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
        assert revisions.prepared == []
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
                    _followup_message_request(thread_id, "Choose an active deck"),
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


def test_selector_recovers_a_thread_with_stale_active_deck_metadata() -> None:
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
        old_prepare_widget = _widget_items(old_chat_events)[0]
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
            event["type"] == "notice" and "no longer available" in event["message"]
            for event in recovery_events
        )
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Old request for deck A")
        ]
        recovery_selector = _widget_items(recovery_events)[0]
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

        stale = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    old_prepare_widget,
                    _prepare_action(old_prepare_widget),
                ),
            )[2]
        )
        assert revisions.prepared == []
        assert any(
            event["type"] == "error"
            and ("stale" in event["message"] or "already used" in event["message"])
            for event in stale
        )

        _post(sidecar, _followup_message_request(thread_id, "Question for recovered deck"))
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "Old request for deck A"),
            ("data/decks/potential-practice.yaml", "Question for recovered deck"),
        ]
        assert revisions.chat_histories == [(), ()]
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
                    _followup_message_request(thread_id, "Choose an active deck"),
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
                    _followup_message_request(thread_id, "Choose an active deck"),
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


def test_deck_switch_rechecks_for_a_chat_started_during_resolution() -> None:
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
                    _followup_message_request(thread_id, "Choose an active deck"),
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
        assert chat_entered.wait(3)

        resolve_release.set()
        switcher.join(3)
        assert switch_response and switch_response[0][0] == 200
        refused = _events(switch_response[0][2])
        assert any(
            event["type"] == "error" and "still running" in event["message"]
            for event in refused
        )
        assert revisions.resolved == ["potential", "te-form"]
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "A question for deck A")
        ]

        chat_release.set()
        chatter.join(3)
        assert chat_response and chat_response[0][0] == 200
        _post(sidecar, _followup_message_request(thread_id, "Follow up on deck A"))
        assert revisions.chatted == [
            ("data/decks/potential-practice.yaml", "A question for deck A"),
            ("data/decks/potential-practice.yaml", "Follow up on deck A"),
        ]
        assert revisions.chat_histories[-1] == (
            ("user", "A question for deck A"),
            (
                "assistant",
                "Answer about data/decks/potential-practice.yaml: A question for deck A",
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


def test_thread_stays_busy_until_chat_plan_and_finish_bindings_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=_assistant_deck_choices(),
        session_token=SESSION_TOKEN,
    )
    server = sidecar.server.assistant_core.server
    sidecar.start()
    try:
        thread_id, selector, _ = _start_deck_selector(sidecar)
        _post(
            sidecar,
            _action_request(
                thread_id,
                selector,
                _deck_selector_action(selector, "potential"),
            ),
        )

        original_prepare_widget = type(server)._prepare_widget
        chat_binding_checks: list[bool] = []

        def guarded_prepare_widget(*, capability: str) -> Any:
            chat_binding_checks.append(thread_id in server._busy_threads)
            return original_prepare_widget(capability=capability)

        monkeypatch.setattr(
            type(server),
            "_prepare_widget",
            staticmethod(guarded_prepare_widget),
        )
        chat_events = _events(
            _post(sidecar, _followup_message_request(thread_id, "Prepare a change"))[2]
        )
        prepare_widget = _widget_items(chat_events)[0]
        assert chat_binding_checks == [True]
        assert thread_id not in server._busy_threads

        original_plan_widget = type(server)._plan_widget
        plan_binding_checks: list[bool] = []

        def guarded_plan_widget(
            plan: RevisionPlan,
            *,
            instruction: str,
            capability: str,
        ) -> Any:
            plan_binding_checks.append(thread_id in server._busy_threads)
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
        plan_events = _events(
            _post(
                sidecar,
                _action_request(
                    thread_id,
                    prepare_widget,
                    _prepare_action(prepare_widget),
                ),
            )[2]
        )
        plan_widget = _widget_items(plan_events)[0]
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
        assert b"Ask janki" not in body
        assert b"Questions are read-only" in body
        assert b"explicit deck-change action" in body
        assert b"Choose the active deck inside this chat" in body
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
        assert b'label: "Choose active deck"' in script
        assert b'prompt: "Choose an active deck"' in script
        assert b'label: "What Janki can do"' in script
        assert b'prompt: "What can Janki help me do here?"' in script
        assert b'label: "How changes work"' in script
        assert (
            b'prompt: "Explain how I can prepare and confirm a deck change."'
            in script
        )
        assert b'label: "Create from a source"' in script
        assert b"Summarize what this deck is designed to teach." not in script
        assert b'icon: "book-open"' in script
        assert b'icon: "write"' in script
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


def test_project_only_shell_names_only_source_intake_and_extraction(
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
        assert b"Attach one PDF or photo" in body
        assert b"until a readable deck is configured" in body
        assert b"deck changes additionally require a supported rich drill deck" in body
        assert b"Ask a question in ordinary language" not in body
        assert b"Questions are read-only" not in body
        assert b'placeholder: "Attach a PDF or photo"' in script
        assert b'greeting: "Add a source to Janki"' in script
        assert b"What would you like to do?" not in script
        assert b'label: "What Janki can do"' not in script
        assert b'label: "How changes work"' not in script
        assert b'label: "Create from a source"' not in script
    finally:
        sidecar.close()


def test_project_without_a_readable_deck_does_not_offer_a_dead_end() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_choices=(),
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        question_events = _events(
            _post(sidecar, _message_request("Can you help with a deck?"))[2]
        )
        assert any(
            event["type"] == "notice"
            and "No readable deck is available" in event["message"]
            for event in question_events
        )
        assert not any(
            event["type"] == "notice" and "Choose a deck below" in event["message"]
            for event in question_events
        )
        assert _widget_items(question_events) == []

        for prompt in (
            "What can Janki help me do here?",
            "Explain how I can prepare and confirm a deck change.",
        ):
            events = _events(_post(sidecar, _message_request(prompt))[2])
            answer = next(
                event["item"]["content"][0]["text"]
                for event in events
                if event["type"] == "thread.item.done"
                and event["item"]["type"] == "assistant_message"
            )
            assert "No readable deck is available" in answer
            assert "Choose a readable deck" not in answer
            assert "ask read-only questions" not in answer
            assert _widget_items(events) == []

        choose_events = _events(
            _post(sidecar, _message_request("Choose an active deck"))[2]
        )
        assert any(
            event["type"] == "notice"
            and "No configured deck is available" in event["message"]
            for event in choose_events
        )
        assert _widget_items(choose_events) == []

        assert revisions.chatted == []
        assert revisions.prepared == []
    finally:
        sidecar.close()


def test_chat_only_shell_does_not_offer_a_deck_change_starter() -> None:
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
        assert b"Questions are read-only" in body
        assert b"explicit deck-change action" not in body
        assert b"chat only" in body
        assert b'label: "Choose active deck"' in script
        assert b'label: "What Janki can do"' in script
        assert b'label: "How changes work"' not in script
        assert b'label: "Create from a source"' in script
    finally:
        sidecar.close()


def test_chat_only_project_refuses_typed_deck_change_help_without_a_model_call() -> None:
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
        assert "no Chat + changes deck" in answer
        assert "explicit deck-change action" not in answer
        assert revisions.chatted == []
        assert revisions.prepared == []
    finally:
        sidecar.close()


def test_chat_only_project_describes_capabilities_without_claiming_deck_changes() -> None:
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
        thread_id, _selector, _events_before = _start_deck_selector(sidecar)

        status, _headers, body = _post(
            sidecar,
            _followup_message_request(
                thread_id,
                "What can Janki help me do here?",
            ),
        )

        assert status == 200
        answer = next(
            event["item"]["content"][0]["text"]
            for event in _events(body)
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert "read-only questions" in answer
        assert "no Chat + changes deck" in answer
        assert "reviewed change proposal" not in answer
        assert revisions.chatted == []
        assert revisions.prepared == []
    finally:
        sidecar.close()


def test_revision_project_explains_typed_deck_change_help_without_a_model_call() -> None:
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
        assert "Choose a Chat + changes deck" in answer
        assert "explicit deck-change action" in answer

        capabilities_status, _headers, capabilities_body = _post(
            sidecar,
            _followup_message_request(
                thread_id,
                "What can Janki help me do here?",
            ),
        )
        assert capabilities_status == 200
        capabilities_answer = next(
            event["item"]["content"][0]["text"]
            for event in _events(capabilities_body)
            if event["type"] == "thread.item.done"
            and event["item"]["type"] == "assistant_message"
        )
        assert "A Chat + changes deck" in capabilities_answer
        assert "reviewed change proposal" in capabilities_answer
        assert revisions.chatted == []
        assert revisions.prepared == []
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
        assert revisions.prepared == []
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
                    _followup_message_request(thread_id, "Choose an active deck"),
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
def test_every_composer_message_is_chat_only_even_when_it_sounds_mutating(
    message: str,
) -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, assistant, widget, events = _create_chat(sidecar, message)

        assert revisions.chatted == [("potential-practice", message)]
        assert revisions.chat_histories == [()]
        assert revisions.prepared == []
        assert revisions.executed == []
        assert f"Answer about potential-practice: {message}" in json.dumps(
            assistant, ensure_ascii=False
        )
        assert any(event["type"] == "stream_options" for event in events)
        assert next(event for event in events if event["type"] == "stream_options")[
            "stream_options"
        ] == {"allow_cancel": False}
        assert [event["text"] for event in events if event["type"] == "progress_update"] == [
            "Preparing answer",
            "Writing answer",
            "Saving answer",
        ]

        root = widget["widget"]
        assert root["size"] == "full"
        action = _prepare_action(widget)
        button = next(child for child in root["children"] if child["type"] == "Button")
        assert button["label"] == "Prepare this message as a deck change"
        assert action["type"] == "janki.revision.prepare"
        assert action["handler"] == "server"
        assert len(action["payload"]["capability"]) >= 32
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
        assert len(_widget_items(_events(body))) == 1
        assert revisions.prepared == []
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
        ) -> ChatReply:
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


def test_only_the_explicit_one_use_prepare_action_renders_the_exact_plan() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, _assistant, prepare_widget, _events_before = _create_chat(sidecar)
        request = _action_request(
            thread_id,
            prepare_widget,
            _prepare_action(prepare_widget),
        )

        events = _events(_post(sidecar, request)[2])
        assert revisions.prepared == [("potential-practice", "Add polite and casual examples")]
        assert revisions.executed == []
        widget = _widget_items(events)[0]
        wire = json.dumps(widget, ensure_ascii=False)
        assert "potential-practice" in wire
        assert "Add polite and casual examples" in wire
        assert REQUEST_FINGERPRINT in wire
        root = widget["widget"]
        assert root["size"] == "full"
        assert len(root["children"]) == 1
        body = root["children"][0]
        assert body["type"] == "Col"
        assert body["width"] == "100%"
        assert body["minWidth"] == 0

        effects = _component_with_id(root, "confirmation-effects")
        assert effects["gap"] == 3
        assert [row["children"][1]["children"][0]["value"] for row in effects["children"]] == [
            "propose one polite and one casual example per selected card",
            "stage the proposal without changing the deck",
        ]
        disclosures = _component_with_id(root, "confirmation-disclosures")
        assert [child["value"] for child in disclosures["children"]] == [
            "The confirmed revision is a paid OpenAI API call."
        ]
        fingerprint = _component_with_id(root, "confirmation-fingerprint")
        lines = [child["value"] for child in fingerprint["children"]]
        assert "".join(lines) == REQUEST_FINGERPRINT
        assert all(0 < len(line) <= 32 for line in lines)

        action = widget["widget"]["confirm"]["action"]
        assert action["type"] == "janki.revision.confirm"
        assert action["handler"] == "server"

        repeated = _events(_post(sidecar, request)[2])
        assert len(revisions.prepared) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in repeated
        )
    finally:
        sidecar.close()


def test_prepare_capability_is_consumed_on_thread_or_sender_mismatch() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_assistant, first_widget, _ = _create_chat(
            sidecar, "first exact message"
        )
        second_thread, _second_assistant, second_widget, _ = _create_chat(
            sidecar, "second exact message"
        )

        cross_thread = _action_request(
            second_thread,
            second_widget,
            _prepare_action(first_widget),
        )
        refused = _events(_post(sidecar, cross_thread)[2])
        assert revisions.prepared == []
        assert any(event["type"] == "error" and "stale" in event["message"] for event in refused)

        original = _action_request(
            first_thread,
            first_widget,
            _prepare_action(first_widget),
        )
        replay = _events(_post(sidecar, original)[2])
        assert revisions.prepared == []
        assert any(
            event["type"] == "error" and "already used" in event["message"] for event in replay
        )

        third_thread, _third_assistant, third_widget, _ = _create_chat(
            sidecar, "third exact message"
        )
        second_turn = _events(
            _post(
                sidecar,
                _followup_message_request(third_thread, "another message"),
            )[2]
        )
        other_widget = next(
            widget
            for widget in _widget_items(second_turn)
            if "janki.revision.prepare" in json.dumps(widget)
        )
        wrong_sender = _action_request(
            third_thread,
            other_widget,
            _prepare_action(third_widget),
        )
        sender_refusal = _events(_post(sidecar, wrong_sender)[2])
        assert revisions.prepared == []
        assert any(
            event["type"] == "error" and "belongs elsewhere" in event["message"]
            for event in sender_refusal
        )
        assert first_assistant["id"] != first_widget["id"]
    finally:
        sidecar.close()


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
                _followup_message_request(thread_id, "unrelated follow-up"),
            )[2]
        )
        other_widget = _widget_items(followup_events)[0]
        finish_action = finish_widget["widget"]["confirm"]["action"]

        refused = _events(
            _post(
                sidecar,
                _action_request(thread_id, other_widget, finish_action),
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
        thread_id, _assistant, widget, _ = _create_chat(sidecar)
        invented = _action_request(
            thread_id,
            widget,
            {"type": old_action, "payload": {"capability": "invented"}},
        )

        events = _events(_post(sidecar, invented)[2])

        assert revisions.prepared == []
        assert revisions.executed == []
        assert any(
            event["type"] == "error" and event["message"] == "That assistant action was refused."
            for event in events
        )
    finally:
        sidecar.close()


def test_browser_disconnect_does_not_cancel_a_dispatched_chat_turn() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    revisions = _FakeRevisions(
        chat_entered=entered,
        chat_release=release,
        chat_finished=finished,
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
