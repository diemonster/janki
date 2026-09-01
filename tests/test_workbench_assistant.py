"""The isolated conversational ChatKit workbench controller.

Every callback is fake. These tests exercise the ChatKit wire protocol and the
loopback HTTP boundary without contacting OpenAI or another provider.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from japanese_anki.workbench.assistant import (
    AssistantRequestContext,
    ChatReply,
    RevisionConfirmation,
    RevisionExecution,
    RevisionPlan,
    ScopedMemoryStore,
)
from japanese_anki.workbench.assistant_http import create_assistant_sidecar

SESSION_TOKEN = "assistant-session-token-000000000000"
REQUEST_FINGERPRINT = "0123456789abcdef" * 4


@dataclass
class _FakeRevisions:
    chatted: list[tuple[str, str]] = field(default_factory=list)
    chat_histories: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    prepared: list[tuple[str, str]] = field(default_factory=list)
    executed: list[RevisionConfirmation] = field(default_factory=list)
    entered: threading.Event | None = None
    release: threading.Event | None = None
    finished: threading.Event | None = None
    chat_entered: threading.Event | None = None
    chat_release: threading.Event | None = None
    chat_finished: threading.Event | None = None

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
            target="Potential Practice",
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
        return RevisionExecution()


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
    button = next(
        child for child in widget["widget"]["children"] if child["type"] == "Button"
    )
    return button["onClickAction"]


def _confirmation_request(thread_id: str, widget: dict[str, Any]) -> dict[str, Any]:
    action = widget["widget"]["confirm"]["action"]
    return _action_request(thread_id, widget, action)


def _create_chat(
    sidecar: Any,
    message: str = "Add polite and casual examples",
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    status, _headers, body = _post(sidecar, _message_request(message))
    assert status == 200
    events = _events(body)
    created = next(event for event in events if event["type"] == "thread.created")
    assistant = next(
        event["item"]
        for event in events
        if event["type"] == "thread.item.done"
        and event["item"]["type"] == "assistant_message"
    )
    widget = _widget_items(events)[0]
    return created["thread"]["id"], assistant, widget, events


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


def test_shell_is_a_separate_tokenized_origin_with_only_the_chatkit_cdn() -> None:
    sidecar = create_assistant_sidecar(
        _FakeRevisions(),
        deck_scope="potential-practice",
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
        assert (
            "style-src 'self' "
            "'sha256-G2shiuZXM1qoGNHm7OQ6u7Ye45SO8f8LKO17q0kfGvw='"
        ) in csp
        assert "'unsafe-inline'" not in csp
        assert b"https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" in body
        assert b"Conversation is read-only" in body
        assert b"main-workbench-secret" not in body

        script_status, _script_headers, script = _request(
            sidecar,
            "GET",
            sidecar.server.script_path,
        )
        assert script_status == 200
        assert sidecar.server.api_path.encode() in script
        assert b'domainKey: "domain_pk_localhost_dev"' in script
        assert b'attachments: { enabled: false }' in script
        assert b'placeholder: "Ask about this deck"' in script
        assert b'greeting: "What would you like to know about this deck?"' in script
        assert (
            b"threadItemActions: {\n      feedback: false,\n      retry: false,\n    },"
            in script
        )
        assert b"onClientTool" not in script
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
        assert [
            event["text"] for event in events if event["type"] == "progress_update"
        ] == ["Preparing answer", "Writing answer", "Saving answer"]

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
        thread_id, _assistant, _widget, _events_before = _create_chat(
            sidecar, "which deck?"
        )

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
            assert _post(
                sidecar,
                _followup_message_request(thread_id, f"message {number}"),
            )[0] == 200

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

        assert _post(
            sidecar,
            _followup_message_request(thread_id, "second"),
        )[0] == 200

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
        assert revisions.prepared == [
            ("potential-practice", "Add polite and casual examples")
        ]
        assert revisions.executed == []
        widget = _widget_items(events)[0]
        wire = json.dumps(widget, ensure_ascii=False)
        assert "Potential Practice" in wire
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
        assert [
            row["children"][1]["children"][0]["value"]
            for row in effects["children"]
        ] == [
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
            event["type"] == "error" and "already used" in event["message"]
            for event in repeated
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
        assert any(
            event["type"] == "error" and "stale" in event["message"]
            for event in refused
        )

        original = _action_request(
            first_thread,
            first_widget,
            _prepare_action(first_widget),
        )
        replay = _events(_post(sidecar, original)[2])
        assert revisions.prepared == []
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in replay
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


def test_confirm_stages_one_proposal_and_offers_no_continuation_actions() -> None:
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
        progress = [
            event["text"] for event in events if event["type"] == "progress_update"
        ]
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
        assert b"Refresh the workbench dashboard to review it" in body
        assert b"http://127.0.0.1:" not in body
        assert _widget_items(events) == []

        repeated = _events(_post(sidecar, request)[2])
        assert len(revisions.executed) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in repeated
        )
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
        assert any(
            event["type"] == "error" and "stale" in event["message"]
            for event in first
        )

        request["params"]["action"]["payload"][
            "request_fingerprint"
        ] = REQUEST_FINGERPRINT
        second = _events(_post(sidecar, request)[2])
        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in second
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
            event["type"] == "error"
            and event["message"] == "That assistant action was refused."
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
        payload = json.dumps(_message_request("which deck?")).encode()
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


def test_memory_store_refuses_a_different_deck_context() -> None:
    async def exercise() -> None:
        store = ScopedMemoryStore("deck-a")
        with pytest.raises(PermissionError, match="another deck scope"):
            store.generate_thread_id(AssistantRequestContext(deck_scope="deck-b"))

    asyncio.run(exercise())
