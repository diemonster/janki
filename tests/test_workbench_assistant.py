"""The isolated, deterministic ChatKit workbench controller.

Every callback is fake.  These tests exercise the ChatKit wire protocol and
the loopback HTTP boundary without contacting OpenAI or another provider.
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
    OwnerActionConfirmation,
    OwnerActionExecution,
    OwnerActionPlan,
    RevisionConfirmation,
    RevisionExecution,
    RevisionPlan,
    RevisionRefusal,
    ScopedMemoryStore,
)
from japanese_anki.workbench.assistant_http import create_assistant_sidecar

SESSION_TOKEN = "assistant-session-token-000000000000"


@dataclass
class _FakeRevisions:
    prepared: list[tuple[str, str]] = field(default_factory=list)
    executed: list[RevisionConfirmation] = field(default_factory=list)
    entered: threading.Event | None = None
    release: threading.Event | None = None
    finished: threading.Event | None = None
    chain: bool = False
    followups_prepared: list[str] = field(default_factory=list)
    followup_contexts: list[str] = field(default_factory=list)
    followups_executed: list[OwnerActionConfirmation] = field(default_factory=list)
    fail_followup: str | None = None

    def prepare_revision(self, *, deck_scope: str, instruction: str) -> RevisionPlan:
        self.prepared.append((deck_scope, instruction))
        return RevisionPlan(
            request_fingerprint="request-fingerprint-123",
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
        return RevisionExecution(
            next_action="apply" if self.chain else None,
            continuation_context="staging/proposal.json" if self.chain else None,
        )

    def prepare_followup(
        self,
        kind: str,
        *,
        deck_scope: str,
        continuation_context: str,
    ) -> OwnerActionPlan:
        assert deck_scope == "potential-practice"
        self.followups_prepared.append(kind)
        self.followup_contexts.append(continuation_context)
        details = {
            "apply": (
                "Confirm exact proposal",
                ("Replace 16 cards with the displayed examples",),
                ("No provider call.",),
            ),
            "audio": (
                "Confirm exact example-audio plan",
                (
                    "32 total; 4 current; 2 recoverable; 26 provider required",
                    "Provider openai-realtime; model gpt-realtime-1.5",
                ),
                ("Confirming may make 26 paid provider calls.",),
            ),
            "build": (
                "Confirm exact deck build",
                ("Build 16 cards to dist/potential.apkg",),
                ("No provider call.",),
            ),
        }[kind]
        return OwnerActionPlan(
            kind=kind,
            fingerprint=f"{kind}-fingerprint-123",
            target=f"{kind}-target",
            title=details[0],
            effects=details[1],
            disclosures=details[2],
        )

    def consume_followup(
        self,
        confirmation: OwnerActionConfirmation,
        *,
        progress: Any,
    ) -> OwnerActionExecution:
        self.followups_executed.append(confirmation)
        if confirmation.kind == self.fail_followup:
            raise RevisionRefusal(f"{confirmation.kind} did not complete durably")
        labels = {
            "apply": (
                "Preparing proposal",
                "Re-reading proposal",
                "Applying revision",
                "Archiving proposal",
            ),
            "audio": (
                "Preparing audio",
                "Creating audio",
                "Saving audio",
                "Cleaning up audio",
            ),
            "build": ("Preparing deck", "Building package", "Saving package"),
        }[confirmation.kind]
        for label in labels:
            progress(label)
        next_action = {"apply": "audio", "audio": "build", "build": None}[
            confirmation.kind
        ]
        return OwnerActionExecution(
            message=f"{confirmation.kind} completed durably",
            next_action=next_action,
            continuation_context=(
                f"durable-{confirmation.kind}-result" if next_action is not None else None
            ),
        )


def _message_request(instruction: str = "Add polite and casual examples") -> dict[str, Any]:
    return {
        "type": "threads.create",
        "params": {
            "input": {
                "content": [{"type": "input_text", "text": instruction}],
                "attachments": [],
                "quoted_text": None,
                "inference_options": {},
            }
        },
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


def _create_plan(sidecar: Any) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    status, _headers, body = _post(sidecar, _message_request())
    assert status == 200
    events = _events(body)
    created = next(event for event in events if event["type"] == "thread.created")
    widget_event = next(
        event
        for event in events
        if event["type"] == "thread.item.done" and event["item"]["type"] == "widget"
    )
    return created["thread"]["id"], widget_event["item"], events


def _confirmation_request(thread_id: str, widget: dict[str, Any]) -> dict[str, Any]:
    action = widget["widget"]["confirm"]["action"]
    return {
        "type": "threads.custom_action",
        "params": {
            "thread_id": thread_id,
            "item_id": widget["id"],
            "action": {"type": action["type"], "payload": action["payload"]},
        },
    }


def _action_request(
    thread_id: str,
    widget: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "threads.custom_action",
        "params": {
            "thread_id": thread_id,
            "item_id": widget["id"],
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
        assert b"OpenAI's hosted ChatKit UI" in body
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


def test_one_message_renders_the_exact_plan_without_executing_it() -> None:
    revisions = _FakeRevisions()
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        _thread_id, widget, events = _create_plan(sidecar)

        assert revisions.prepared == [
            ("potential-practice", "Add polite and casual examples")
        ]
        assert revisions.executed == []
        assert any(event["type"] == "stream_options" for event in events)
        assert next(event for event in events if event["type"] == "stream_options")[
            "stream_options"
        ] == {"allow_cancel": False}

        wire = json.dumps(widget, ensure_ascii=False)
        assert "Potential Practice" in wire
        assert "Add polite and casual examples" in wire
        assert "request-fingerprint-123" in wire
        action = widget["widget"]["confirm"]["action"]
        assert action["type"] == "janki.revision.confirm"
        assert action["handler"] == "server"
        assert len(action["payload"]["capability"]) >= 32
    finally:
        sidecar.close()


def test_confirm_consumes_one_exact_plan_and_streams_named_progress() -> None:
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
        assert confirmation.expected_fingerprint == "request-fingerprint-123"
        assert b"Refresh the workbench dashboard to review it" in body
        assert b"http://127.0.0.1:" not in body

        second_status, _second_headers, second_body = _post(sidecar, request)
        assert second_status == 200
        assert len(revisions.executed) == 1
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in _events(second_body)
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
        assert any(event["type"] == "error" and "stale" in event["message"] for event in first)

        request["params"]["action"]["payload"][
            "request_fingerprint"
        ] = "request-fingerprint-123"
        second = _events(_post(sidecar, request)[2])
        assert revisions.executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in second
        )
    finally:
        sidecar.close()


def test_apply_audio_and_build_each_need_a_fresh_owner_confirmation() -> None:
    revisions = _FakeRevisions(chain=True)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, revision_widget, _events_before = _create_plan(sidecar)

        invented = {
            "type": "threads.custom_action",
            "params": {
                "thread_id": thread_id,
                "item_id": revision_widget["id"],
                "action": {
                    "type": "janki.revision.apply.prepare",
                    "payload": {"continuation": "invented"},
                },
            },
        }
        invented_events = _events(_post(sidecar, invented)[2])
        assert revisions.followups_prepared == []
        assert any(event["type"] == "error" for event in invented_events)

        revision_events = _events(
            _post(sidecar, _confirmation_request(thread_id, revision_widget))[2]
        )
        next_widget = _widget_items(revision_events)[0]
        assert "has not been planned or authorized" in json.dumps(next_widget)

        expected_progress = {
            "apply": [
                "Preparing proposal",
                "Re-reading proposal",
                "Applying revision",
                "Archiving proposal",
            ],
            "audio": [
                "Preparing audio",
                "Creating audio",
                "Saving audio",
                "Cleaning up audio",
            ],
            "build": ["Preparing deck", "Building package", "Saving package"],
        }
        completed_kinds: list[str] = []
        for kind in ("apply", "audio", "build"):
            preparation_wire = json.dumps(next_widget, ensure_ascii=False)
            if kind == "apply":
                assert "selected Japanese/English examples" in preparation_wire
                assert "OpenAI-hosted ChatKit UI" in preparation_wire
            elif kind == "audio":
                assert "exact Japanese audio inputs" in preparation_wire
                assert "does not contact the audio provider" in preparation_wire
            else:
                assert "package targets and bound hashes" in preparation_wire
                assert "does not build the package" in preparation_wire
            prepare_request = _action_request(
                thread_id,
                next_widget,
                _prepare_action(next_widget),
            )
            prepare_events = _events(_post(sidecar, prepare_request)[2])
            assert revisions.followups_prepared == [*completed_kinds, kind]
            expected_context = (
                "staging/proposal.json"
                if kind == "apply"
                else f"durable-{completed_kinds[-1]}-result"
            )
            assert revisions.followup_contexts[-1] == expected_context
            plan_widget = _widget_items(prepare_events)[0]
            plan_wire = json.dumps(plan_widget, ensure_ascii=False)
            assert f"{kind}-fingerprint-123" in plan_wire
            if kind == "audio":
                assert "26 provider required" in plan_wire
                assert "gpt-realtime-1.5" in plan_wire
                assert "paid provider calls" in plan_wire
            assert [item.kind for item in revisions.followups_executed] == completed_kinds

            confirmation_request = _confirmation_request(thread_id, plan_widget)
            execution_events = _events(_post(sidecar, confirmation_request)[2])
            assert revisions.followups_executed[-1].kind == kind
            completed_kinds.append(kind)
            progress = [
                event["text"]
                for event in execution_events
                if event["type"] == "progress_update"
            ]
            assert progress == expected_progress[kind]
            assert all("%" not in label and "percent" not in label for label in progress)

            repeated = _events(_post(sidecar, confirmation_request)[2])
            assert len([item for item in revisions.followups_executed if item.kind == kind]) == 1
            assert any(
                event["type"] == "error" and "already used" in event["message"]
                for event in repeated
            )
            following = _widget_items(execution_events)
            if kind == "build":
                assert following == []
            else:
                assert len(following) == 1
                next_widget = following[0]
    finally:
        sidecar.close()


def test_stale_followup_fingerprint_is_consumed_and_never_executed() -> None:
    revisions = _FakeRevisions(chain=True)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, revision_widget, _events_before = _create_plan(sidecar)
        revision_events = _events(
            _post(sidecar, _confirmation_request(thread_id, revision_widget))[2]
        )
        next_widget = _widget_items(revision_events)[0]
        prepare_events = _events(
            _post(
                sidecar,
                _action_request(thread_id, next_widget, _prepare_action(next_widget)),
            )[2]
        )
        plan_widget = _widget_items(prepare_events)[0]
        confirmation = _confirmation_request(thread_id, plan_widget)
        confirmation["params"]["action"]["payload"]["plan_fingerprint"] = "stale"

        refused = _events(_post(sidecar, confirmation)[2])

        assert revisions.followups_executed == []
        assert any(
            event["type"] == "error" and "stale" in event["message"]
            for event in refused
        )
        confirmation["params"]["action"]["payload"][
            "plan_fingerprint"
        ] = "apply-fingerprint-123"
        repeated = _events(_post(sidecar, confirmation)[2])
        assert revisions.followups_executed == []
        assert any(
            event["type"] == "error" and "already used" in event["message"]
            for event in repeated
        )
    finally:
        sidecar.close()


def test_a_refused_durable_action_never_offers_the_next_action() -> None:
    revisions = _FakeRevisions(chain=True, fail_followup="apply")
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        thread_id, revision_widget, _events_before = _create_plan(sidecar)
        revision_events = _events(
            _post(sidecar, _confirmation_request(thread_id, revision_widget))[2]
        )
        next_widget = _widget_items(revision_events)[0]
        prepare_events = _events(
            _post(
                sidecar,
                _action_request(thread_id, next_widget, _prepare_action(next_widget)),
            )[2]
        )
        apply_widget = _widget_items(prepare_events)[0]

        refused = _events(
            _post(sidecar, _confirmation_request(thread_id, apply_widget))[2]
        )

        assert [item.kind for item in revisions.followups_executed] == ["apply"]
        assert _widget_items(refused) == []
        assert any(
            event["type"] == "notice"
            and event["level"] == "danger"
            and "did not complete durably" in event["message"]
            for event in refused
        )
    finally:
        sidecar.close()


def test_an_older_next_button_stays_bound_to_its_own_durable_proposal() -> None:
    @dataclass
    class ContextRevisions(_FakeRevisions):
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
                message="proposal completed",
                next_action="apply",
                continuation_context=f"staging/{confirmation.instruction}.json",
            )

    revisions = ContextRevisions(chain=True)
    sidecar = create_assistant_sidecar(
        revisions,
        deck_scope="potential-practice",
        session_token=SESSION_TOKEN,
    )
    sidecar.start()
    try:
        first_thread, first_plan, _ = _create_plan(sidecar)
        second_status, _headers, second_body = _post(
            sidecar,
            _message_request("second-proposal"),
        )
        assert second_status == 200
        second_events = _events(second_body)
        second_thread = next(
            event["thread"]["id"]
            for event in second_events
            if event["type"] == "thread.created"
        )
        second_plan = _widget_items(second_events)[0]
        first_done = _events(
            _post(sidecar, _confirmation_request(first_thread, first_plan))[2]
        )
        second_done = _events(
            _post(sidecar, _confirmation_request(second_thread, second_plan))[2]
        )
        first_next = _widget_items(first_done)[0]
        assert _widget_items(second_done)

        _post(
            sidecar,
            _action_request(first_thread, first_next, _prepare_action(first_next)),
        )

        assert revisions.followup_contexts == [
            "staging/Add polite and casual examples.json"
        ]
    finally:
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
