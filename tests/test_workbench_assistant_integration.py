"""The optional ChatKit sidecar is wired to one exact revision application plan."""

from __future__ import annotations

import builtins
import hashlib
import http.client
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import operations
from japanese_anki.application import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    OUTCOME_UNKNOWN,
    DispatchFailure,
    ExtractionCompletionError,
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    ExtractionRevision,
    revision,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence
from japanese_anki.workbench import assistant_adapter, assistant_http
from japanese_anki.workbench import server as workbench_server
from japanese_anki.workbench.assistant import (
    RevisionConfirmation,
    RevisionFinishConfirmation,
    RevisionRefusal,
    SourceExtractionConfirmation,
)
from japanese_anki.workbench.assistant_http import create_assistant_sidecar


def _config(tmp_path: Path, *, enabled: bool = True) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        f"[assistant]\nenabled = {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _revision_plan(
    *,
    plan_fingerprint: str = "rendered-plan",
    request_fingerprint: str = "narrow-provider-request-fingerprint",
    selected_record_ids: tuple[str, ...] = ("word:one",),
    provider: str = "claude-code",
    billing_display: str = "Claude Max subscription via Claude Code",
    auth_metadata: dict[str, Any] | None = None,
    transport: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        plan_fingerprint=plan_fingerprint,
        request_fingerprint=request_fingerprint,
        selected_record_ids=selected_record_ids,
        provider=provider,
        billing_display=billing_display,
        auth_metadata=(
            auth_metadata
            if auth_metadata is not None
            else {"auth_method": "claude.ai", "subscription_type": "max"}
        ),
        model="claude-opus-5",
        transport=(
            transport
            if transport is not None
            else {"kind": "claude-code-cli", "cli_version": "2.1.246"}
        ),
        provider_plan=SimpleNamespace(request_bytes=b"exact-provider-request"),
        deck_relative_path="data/decks/potential.yaml",
        owner_instruction="Add examples.",
    )


def _finish_plan(
    tmp_path: Path,
    *,
    fingerprint: str = "f" * 64,
    staging_name: str = "proposal.json",
) -> Any:
    current = (
        ExampleSentence(
            japanese="今は遊べません。",
            furigana="今[いま]は 遊[あそ]べません。",
            english="I cannot play now.",
            register="polite",
        ),
        ExampleSentence(
            japanese="今日は遊べない。",
            furigana="今日[きょう]は 遊[あそ]べない。",
            english="I cannot play today.",
            register="casual",
        ),
    )
    proposed = (
        ExampleSentence(
            japanese="明日は遊べます。",
            furigana="明日[あした]は 遊[あそ]べます。",
            english="I can play tomorrow.",
            register="polite",
        ),
        ExampleSentence(
            japanese="今日は遊べる。",
            furigana="今日[きょう]は 遊[あそ]べる。",
            english="I can play today.",
            register="casual",
        ),
    )
    revision_plan = SimpleNamespace(
        staging_path=tmp_path / "data" / "staging" / staging_name,
        deck_relative_path="data/decks/potential.yaml",
        current_form_note="Old potential note.",
        form_note="Potential expresses ability or possibility.",
        selected_record_ids=("word:one",),
        current_drill_examples={"word:one": current},
        drill_examples={"word:one": proposed},
    )
    counts = SimpleNamespace(
        total=2,
        current=0,
        recoverable=1,
        provider_required=1,
    )
    return SimpleNamespace(
        revision=revision_plan,
        audio=SimpleNamespace(
            example_provider=SimpleNamespace(
                name="openai-realtime",
                access="paid-network",
                settings={"model": "gpt-realtime-1.5"},
            ),
            example_counts=counts,
        ),
        build=SimpleNamespace(
            output_path=tmp_path / "dist" / "potential.apkg",
            card_count=16,
        ),
        authority={"exact": fingerprint},
        fingerprint=fingerprint,
    )


def _seeded_extraction_confirmation(
    tmp_path: Path,
) -> tuple[
    assistant_adapter.RevisionAssistantAdapter,
    SourceExtractionConfirmation,
    ProjectConfig,
]:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    preparation_id = "prepared-source"
    expected = ExtractionDispatchExpectation(
        source=source,
        model="claude-opus-5",
        mode=None,
        source_sha256="a" * 64,
        request_fingerprint="b" * 64,
        replacement_revision=None,
        replacement_confirmed=False,
        staging_path=(config.staging_dir / "lesson.pdf.yaml").resolve(),
        patterns_path=config.patterns_file.resolve(),
        operations_path=config.operations_file.resolve(),
    )
    with adapter._plan_lock:
        adapter._extraction_expectations[preparation_id] = expected
    return (
        adapter,
        SourceExtractionConfirmation(
            preparation_id=preparation_id,
            source_name="lesson.pdf",
            expected_fingerprint="b" * 64,
        ),
        config,
    )


def test_disabled_workbench_never_imports_the_assistant_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {
            "japanese_anki.workbench.assistant_adapter",
            "japanese_anki.workbench.assistant_http",
        } or name.startswith("chatkit"):
            raise AssertionError(f"disabled workbench imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert workbench_server._start_assistant_for(_config(tmp_path, enabled=False)) is None


def test_enabled_workbench_gives_assistant_the_configured_local_inbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = SimpleNamespace(
        deck_scope="data/decks/potential.yaml",
        conversation_available=True,
    )
    expected_sidecar = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        assistant_adapter,
        "discover_revision_adapter",
        lambda _config: (adapter, []),
    )

    def start(callbacks: Any, **kwargs: Any) -> object:
        captured.update(callbacks=callbacks, **kwargs)
        return expected_sidecar

    monkeypatch.setattr(assistant_http, "start_assistant_sidecar", start)

    result = workbench_server._start_assistant_for(config)

    assert result is expected_sidecar
    assert captured == {
        "callbacks": adapter,
        "deck_scope": adapter.deck_scope,
        "inbox_root": config.scan_inbox,
        "conversation_available": True,
    }


def test_enabled_workbench_starts_project_intake_without_a_revision_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    expected_sidecar = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [])

    def start(callbacks: Any, **kwargs: Any) -> object:
        captured.update(callbacks=callbacks, **kwargs)
        return expected_sidecar

    monkeypatch.setattr(assistant_http, "start_assistant_sidecar", start)

    result = workbench_server._start_assistant_for(config)

    assert result is expected_sidecar
    assert captured["callbacks"].deck_path is None
    assert captured["deck_scope"] == "janki-project"
    assert captured["inbox_root"] == config.scan_inbox
    assert captured["conversation_available"] is False


def test_discovery_keeps_project_intake_when_two_revision_decks_are_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "data" / "decks" / "potential.yaml"
    second = tmp_path / "data" / "decks" / "te-form.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [first, second])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {"kind": "conjugation", "drill_examples": {"record": []}},
            [],
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: pytest.fail("ambiguous discovery must not select a deck"),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert adapter.deck_path is None
    assert adapter.deck_scope == "janki-project"
    assert len(warnings) == 1
    assert "exactly one" in warnings[0]
    assert "potential.yaml" in warnings[0]
    assert "te-form.yaml" in warnings[0]
    assert "No deck was guessed" in warnings[0]
    assert "Source intake and extraction remain available" in warnings[0]

    with pytest.raises(RevisionRefusal, match="no unique deck selected"):
        adapter.chat(
            deck_scope=adapter.deck_scope,
            history=(),
            message="Which deck?",
            progress=lambda _label: None,
        )


def test_adapter_plans_the_complete_stored_order_and_displays_application_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    selected = ("word:second", "word:first")
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=selected,
    )
    calls: list[tuple[Path, tuple[str, ...], str]] = []
    application_plan = _revision_plan(
        plan_fingerprint="full-application-plan-fingerprint",
        request_fingerprint="narrow-provider-request-fingerprint",
        selected_record_ids=selected,
    )

    def fake_plan(
        _config: Any,
        target: Path,
        record_ids: tuple[str, ...],
        instruction: str,
    ) -> Any:
        calls.append((target, tuple(record_ids), instruction))
        return application_plan

    monkeypatch.setattr(assistant_adapter.revision, "plan_revision", fake_plan)

    rendered = adapter.prepare_revision(
        deck_scope="data/decks/potential.yaml",
        instruction="Add examples.",
    )

    assert calls == [(deck, selected, "Add examples.")]
    assert rendered.request_fingerprint == "full-application-plan-fingerprint"
    assert "word:second, word:first" in rendered.effects[0]
    wire = "\n".join((*rendered.effects, *rendered.disclosures))
    assert "claude-code" in wire
    assert "Billing: Claude Max subscription via Claude Code" in wire
    assert "Authentication: auth method=claude.ai, subscription type=max" in wire
    assert "Model: claude-opus-5" in wire
    assert "Claude Code version: 2.1.246" in wire
    assert (
        "Exact provider request bytes: 22 bytes; request bytes SHA-256 "
        f"{hashlib.sha256(b'exact-provider-request').hexdigest()}"
    ) in wire
    assert ("Provider request identity: narrow-provider-request-fingerprint") in wire
    assert "Anthropic" not in wire
    assert "API" not in wire


def test_adapter_routes_an_ordinary_question_only_through_the_chat_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    planned = SimpleNamespace(request_fingerprint="chat-request")
    observed: list[tuple[str, str]] = []

    def plan_chat(
        _config: ProjectConfig,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
    ) -> Any:
        assert history == (("user", "earlier"), ("assistant", "Earlier answer."))
        observed.append((deck_scope, message))
        return planned

    monkeypatch.setattr(assistant_adapter.assistant_chat, "plan_chat", plan_chat)
    monkeypatch.setattr(
        assistant_adapter.assistant_chat,
        "run_chat",
        lambda _config, expected, *, progress: (
            SimpleNamespace(
                answer="You are viewing the Potential Practice deck.",
                operation_id="assistant-operation",
            )
            if expected is planned
            else pytest.fail("chat dispatched another plan")
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: pytest.fail("an ordinary question must not become a revision"),
    )

    progress: list[str] = []
    reply = adapter.chat(
        deck_scope=adapter.deck_scope,
        history=(("user", "earlier"), ("assistant", "Earlier answer.")),
        message="which deck?",
        progress=progress.append,
    )

    assert observed == [("data/decks/potential.yaml", "which deck?")]
    assert reply.text == "You are viewing the Potential Practice deck."


def test_adapter_prepares_then_dispatches_one_exact_saved_source_only_after_click(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    target = SimpleNamespace(
        source_sha256="a" * 64,
        staging_path=config.staging_dir / "lesson.yaml",
        patterns_path=config.patterns_file,
        provenance={"request_fingerprint": "b" * 64},
    )
    consent = SimpleNamespace(
        sendable=True,
        target=target,
        refusal="",
        busy="",
        name="lesson.pdf",
        model="claude-opus-5",
        mode=None,
        replacement_revision=None,
        replaces=None,
        replaces_cards=0,
        replaces_state="",
        replaces_grammar="",
        sends_known_words=False,
    )
    monkeypatch.setattr(
        assistant_adapter,
        "describe_extraction",
        lambda fresh, path, *, mode: (
            consent
            if fresh.root == config.root and path == source and mode is None
            else pytest.fail("adapter described a different source")
        ),
    )
    dispatched: list[Any] = []

    def dispatch(fresh: Any, expected: Any, *, progress: Any) -> Any:
        assert fresh.root == config.root
        dispatched.append(expected)
        for label in (
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ):
            progress(label)
        return SimpleNamespace(
            target=config.staging_dir / "lesson.yaml",
            records=12,
        )

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", dispatch)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=None,
        record_ids=(),
    )

    plan = adapter.prepare_source_extraction(source_path=source)

    assert dispatched == []
    assert plan.source_name == "lesson.pdf"
    assert "whole lesson.pdf" in " ".join(plan.effects)
    assert "page ranges" in " ".join(plan.effects)
    assert "paid Anthropic API call" in " ".join(plan.disclosures)
    assert "Claude Pro or Max does not pay" in " ".join(plan.disclosures)
    assert plan.replaces is False
    assert plan.confirm_label == "Send lesson.pdf using claude-opus-5 — paid API call"

    seen_progress: list[str] = []
    result = adapter.consume_replan_and_extract(
        SourceExtractionConfirmation(
            preparation_id=plan.preparation_id,
            source_name=plan.source_name,
            expected_fingerprint=plan.request_fingerprint,
        ),
        progress=seen_progress.append,
    )

    assert len(dispatched) == 1
    expected = dispatched[0]
    assert expected.source == source
    assert expected.request_fingerprint == "b" * 64
    assert expected.source_sha256 == "a" * 64
    assert expected.replacement_confirmed is False
    assert expected.staging_path == (config.staging_dir / "lesson.yaml").resolve()
    assert expected.patterns_path == config.patterns_file.resolve()
    assert expected.operations_path == config.operations_file.resolve()
    assert seen_progress == [
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    assert "12 card proposal(s)" in result.message

    with pytest.raises(RevisionRefusal, match="already used"):
        adapter.consume_replan_and_extract(
            SourceExtractionConfirmation(
                preparation_id=plan.preparation_id,
                source_name=plan.source_name,
                expected_fingerprint=plan.request_fingerprint,
            ),
            progress=lambda _label: None,
        )
    assert len(dispatched) == 1


def test_adapter_replacement_button_is_the_only_event_that_grants_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    revision_snapshot = ExtractionRevision(
        staging_sha256="c" * 64,
        pattern_entry_sha256="d" * 64,
        pattern_reviewed=True,
        pattern_has_patterns=True,
    )
    consent = SimpleNamespace(
        sendable=True,
        target=SimpleNamespace(
            source_sha256="a" * 64,
            staging_path=config.staging_dir / "lesson.yaml",
            patterns_path=config.patterns_file,
            provenance={"request_fingerprint": "b" * 64},
        ),
        refusal="",
        busy="",
        name="lesson.pdf",
        model="claude-opus-5",
        mode=None,
        replacement_revision=revision_snapshot,
        replaces=config.staging_dir / "lesson.yaml",
        replaces_cards=18,
        replaces_state="Cards need edits",
        replaces_grammar="Grammar reviewed",
        sends_known_words=False,
    )
    monkeypatch.setattr(
        assistant_adapter,
        "describe_extraction",
        lambda _config, _path, *, mode: consent,
    )
    dispatched: list[Any] = []
    monkeypatch.setattr(
        assistant_adapter,
        "dispatch_extraction",
        lambda _config, expected, *, progress: (
            dispatched.append(expected)
            or SimpleNamespace(target=config.staging_dir / "lesson.yaml", records=18)
        ),
    )
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=tmp_path / "data" / "decks" / "potential.yaml",
        record_ids=("word:one",),
    )

    plan = adapter.prepare_source_extraction(source_path=source)

    assert plan.replaces is True
    assert plan.confirm_label.startswith("Replace the named review")
    plan_text = " ".join(plan.effects)
    assert "18 cards, Cards need edits" in plan_text
    assert "Grammar reviewed" in plan_text
    assert "not recoverable" in plan_text
    with adapter._plan_lock:
        rendered = adapter._extraction_expectations[plan.preparation_id]
    assert rendered.replacement_revision == revision_snapshot
    assert rendered.replacement_confirmed is False

    adapter.consume_replan_and_extract(
        SourceExtractionConfirmation(
            preparation_id=plan.preparation_id,
            source_name=plan.source_name,
            expected_fingerprint=plan.request_fingerprint,
        ),
        progress=lambda _label: None,
    )

    assert len(dispatched) == 1
    assert dispatched[0].replacement_revision == revision_snapshot
    assert dispatched[0].replacement_confirmed is True


@pytest.mark.parametrize(
    ("phase", "expected_text"),
    [
        ("binding", "no paid request was made"),
        ("preparation", "no paid request was made"),
        ("authorization", "may contain unused authority"),
    ],
)
def test_adapter_preserves_each_pre_dispatch_failure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_text: str,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(JankiError("pre-dispatch refusal"), phase=phase)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert expected_text in text
    assert "nothing was sent" in text
    if phase == "authorization":
        assert "janki operations" in text


@pytest.mark.parametrize(
    ("failure", "expected_text"),
    [
        (
            DispatchFailure("operation-saved", ANSWER_SAVED),
            "operations --show-reply operation-saved",
        ),
        (
            DispatchFailure("operation-empty", ANSWER_EMPTY),
            "captured provider reply contains no answer",
        ),
        (
            DispatchFailure("operation-unavailable", ANSWER_UNAVAILABLE),
            "exact recovery bytes are unavailable",
        ),
        (
            DispatchFailure("operation-forgotten", FORGOTTEN),
            "already forgotten",
        ),
        (
            DispatchFailure("operation-cleanup", FORGOTTEN, cleanup_pending=True),
            "operations --forget operation-cleanup",
        ),
        (
            DispatchFailure(
                "operation-unknown",
                OUTCOME_UNKNOWN,
                money_may_have_been_spent=True,
            ),
            "retry may pay twice",
        ),
    ],
)
def test_adapter_preserves_each_dispatched_recovery_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: DispatchFailure,
    expected_text: str,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        JankiError("provider refused"),
        phase="dispatch",
        operation_id=failure.operation_id,
        failure=failure,
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert expected_text in text
    assert failure.operation_id in text
    assert "may have been billed" in text


def test_adapter_preserves_a_dispatch_journal_failure_without_inviting_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        JankiError("provider connection ended"),
        phase="dispatch",
        operation_id="operation-journal",
        journal_error=operations.OperationError("journal unreadable"),
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert "operation-journal" in text
    assert "journal unreadable" in text
    assert "could not settle" in text
    assert "do not retry" in text


def test_adapter_reports_saved_proposals_after_pattern_store_completion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, config = _seeded_extraction_confirmation(tmp_path)
    staging_path = config.staging_dir / "lesson.yaml"
    cause = ExtractionCompletionError(JankiError("pattern store refused"), staging_path)
    error = ExtractionDispatchError(
        cause,
        phase="completion",
        operation_id="operation-completion",
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value)
    assert str(staging_path) in text
    assert "proposals are saved" in text
    assert "do not repeat extraction" in text


def test_adapter_refuses_blind_retry_after_final_operation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        operations.OperationError("final journal save refused"),
        phase="completion",
        operation_id="operation-final-save",
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert "operation-final-save" in text
    assert "could not prove" in text
    assert "do not retry" in text
    assert "janki operations" in text


def test_assistant_page_does_not_render_provider_disclosure(tmp_path: Path) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[assistant]\nenabled = true\nprovider = "claude-code"\nmodel = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    sidecar = create_assistant_sidecar(
        adapter,
        deck_scope=adapter.deck_scope,
        session_token="assistant-session-token-000000000000",
    )
    sidecar.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request("GET", sidecar.server.shell_path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()

        assert response.status == 200
        assert "Ask uses claude-code" not in body
        assert "Claude Pro/Max subscription" not in body
        assert "claude-opus-5" not in body
        assert "data/assistant" not in body
    finally:
        sidecar.close()


def test_adapter_renders_the_same_confirmation_shape_for_anthropic_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan(
        provider="anthropic-api",
        billing_display="Anthropic API billing",
        auth_metadata={
            "auth_method": "environment-api-key",
            "api_key_source": "ANTHROPIC_API_KEY",
        },
        transport={"kind": "anthropic-messages-api"},
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )

    rendered = adapter.prepare_revision(
        deck_scope="data/decks/potential.yaml",
        instruction="Add examples.",
    )
    wire = "\n".join((*rendered.effects, *rendered.disclosures))

    assert "anthropic-api" in wire
    assert "Billing: Anthropic API billing" in wire
    assert "Authentication: auth method=environment-api-key" in wire
    assert "api key source=ANTHROPIC_API_KEY" in wire
    assert "Claude Code version" not in wire
    assert "Provider request identity: narrow-provider-request-fingerprint" in wire


def test_adapter_reloads_the_on_disk_provider_before_rendering_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[ai]\nrevise_provider = "claude-code"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    config_path.write_text(
        '[ai]\nrevise_provider = "anthropic-api"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    application_plan = _revision_plan(
        provider="anthropic-api",
        billing_display="Anthropic API billing",
        auth_metadata={
            "auth_method": "environment-api-key",
            "api_key_source": "ANTHROPIC_API_KEY",
        },
        transport={"kind": "anthropic-messages-api"},
    )
    observed: list[str] = []

    def fresh_plan(fresh_config: ProjectConfig, *_args: Any, **_kwargs: Any) -> Any:
        observed.append(fresh_config.revise_provider)
        return application_plan

    monkeypatch.setattr(assistant_adapter.revision, "plan_revision", fresh_plan)

    rendered = adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )

    assert observed == ["anthropic-api"]
    wire = "\n".join((*rendered.effects, *rendered.disclosures))
    assert "Billing: Anthropic API billing" in wire
    assert "Authentication: auth method=environment-api-key" in wire


def test_adapter_refuses_a_stale_confirmation_before_run_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: pytest.fail("a stale binding must not execute"),
    )
    adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )

    with pytest.raises(RevisionRefusal, match="stale"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=adapter.deck_scope,
                instruction="Add examples.",
                expected_fingerprint="different-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )


def test_adapter_delegates_replan_and_dispatch_to_run_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    runs: list[Any] = []

    def fake_run(_config: Any, expected: Any, *, progress: Any) -> Any:
        runs.append(expected)
        progress("Reading the source")
        progress("Checking the answer's shape")
        progress("Saving proposals")
        return SimpleNamespace(staging_path=tmp_path / "data" / "staging" / "proposal.json")

    monkeypatch.setattr(assistant_adapter.revision, "run_revision", fake_run)
    finish_plan = _finish_plan(tmp_path)
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda _config, staging_path: (
            finish_plan
            if staging_path == finish_plan.revision.staging_path
            else pytest.fail("finish planned another staged proposal")
        ),
    )
    adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )
    progress: list[str] = []

    result = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use",
            deck_scope=adapter.deck_scope,
            instruction="Add examples.",
            expected_fingerprint="rendered-plan",
            target="data/decks/potential.yaml",
        ),
        progress=progress.append,
    )

    assert runs == [application_plan]
    assert progress == [
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    assert "data/staging/proposal.json" in result.message
    assert result.finish is not None
    assert result.finish.request_fingerprint == "f" * 64
    assert result.finish.target == "data/decks/potential.yaml"
    assert result.finish.current_form_note == "Old potential note."
    assert result.finish.proposed_form_note == ("Potential expresses ability or possibility.")
    assert result.finish.records[0].record_id == "word:one"
    assert result.finish.records[0].proposed_examples[0].japanese == "明日は遊べます。"
    assert result.finish.audio_provider_required == 1
    assert result.finish.output_path == "dist/potential.apkg"
    assert result.finish.card_count == 16
    assert result.finish_unavailable is None


def test_staged_revision_survives_finish_planning_failure_without_inviting_rebill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan()
    staging_path = tmp_path / "data" / "staging" / "paid-proposal.json"
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: SimpleNamespace(staging_path=staging_path),
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(JankiError("build template is missing")),
    )
    adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )

    result = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="revision-capability",
            deck_scope=adapter.deck_scope,
            instruction="Add examples.",
            expected_fingerprint="rendered-plan",
            target="data/decks/potential.yaml",
        ),
        progress=lambda _label: None,
    )

    assert result.finish is None
    assert "data/staging/paid-proposal.json" in result.message
    assert "Do not repeat the paid revision call" in result.message
    assert result.finish_unavailable is not None
    assert "build template is missing" in result.finish_unavailable
    assert "staged proposal remains the deliverable" in result.finish_unavailable
    assert adapter._finish_plans == {}


def test_adapter_replans_exact_finish_then_executes_shared_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    expected = _finish_plan(tmp_path)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    planned: list[Path] = []

    def fresh_plan(_config: ProjectConfig, staging_path: Path) -> Any:
        planned.append(staging_path)
        return expected

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        fresh_plan,
    )
    executed: list[Any] = []

    def execute(_config: ProjectConfig, plan: Any, *, progress: Any) -> Any:
        executed.append(plan)
        for phase in (
            "Preparing finish",
            "Applying reviewed revision",
            "Creating example audio",
            "Building Anki package",
            "Saving finish receipt",
        ):
            progress(phase)
        return SimpleNamespace(
            receipt_id="f" * 64,
            state="complete",
            deck_path=config.deck_dir / "potential.yaml",
            output_path=tmp_path / "dist" / "potential.apkg",
            package_sha256="a" * 64,
            card_count=16,
        )

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        execute,
    )
    phases: list[str] = []

    result = adapter.consume_replan_and_finish(
        RevisionFinishConfirmation(
            preparation_id="finish-preparation",
            expected_fingerprint="f" * 64,
            target="data/decks/potential.yaml",
        ),
        progress=phases.append,
    )

    assert planned == [expected.revision.staging_path]
    assert executed == [expected]
    assert phases == [
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    ]
    assert result.state == "complete"
    assert result.receipt_id == "f" * 64
    assert result.output_path == "dist/potential.apkg"
    assert result.package_sha256 == "a" * 64


def test_adapter_consumes_finish_plan_and_refuses_staged_or_build_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    expected = _finish_plan(tmp_path)
    changed = _finish_plan(tmp_path, fingerprint="e" * 64)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: changed,
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        lambda *_args, **_kwargs: pytest.fail("drift must not execute"),
    )
    confirmation = RevisionFinishConfirmation(
        preparation_id="finish-preparation",
        expected_fingerprint="f" * 64,
        target="data/decks/potential.yaml",
    )

    with pytest.raises(RevisionRefusal, match="changed after you reviewed") as drift:
        adapter.consume_replan_and_finish(
            confirmation,
            progress=lambda _label: None,
        )
    assert "Return to the Workbench" in str(drift.value)
    assert "reopen the existing staged proposal" in str(drift.value)
    assert "do not repeat the paid revise call" in str(drift.value)
    assert "prepare and review a fresh revision" not in str(drift.value)
    with pytest.raises(RevisionRefusal, match="already used"):
        adapter.consume_replan_and_finish(
            confirmation,
            progress=lambda _label: None,
        )


def test_adapter_reports_inspected_recovery_state_after_finish_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=config.deck_dir / "potential.yaml",
        record_ids=("word:one",),
    )
    expected = _finish_plan(tmp_path)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(JankiError("audio provider disconnected")),
    )
    inspected: list[str] = []

    def inspect(_config: ProjectConfig, receipt_id: str) -> Any:
        inspected.append(receipt_id)
        return SimpleNamespace(
            receipt_id=receipt_id,
            state="revision_applied",
            deck_path=config.deck_dir / "potential.yaml",
            output_path=tmp_path / "dist" / "potential.apkg",
            package_sha256=None,
            card_count=None,
        )

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "inspect_revision_finish",
        inspect,
    )

    result = adapter.consume_replan_and_finish(
        RevisionFinishConfirmation(
            preparation_id="finish-preparation",
            expected_fingerprint="f" * 64,
            target="data/decks/potential.yaml",
        ),
        progress=lambda _label: None,
    )

    assert inspected == ["f" * 64]
    assert result.state == "revision_applied"
    assert "audio provider disconnected" in result.message
    assert "reviewed revision is applied and archived" in result.message
    assert "Resume only receipt" in result.message


def test_adapter_reloads_the_on_disk_provider_before_confirmation_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[ai]\nrevise_provider = "claude-code"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    deck = config.deck_dir / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan(provider="claude-code")
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )
    config_path.write_text(
        '[ai]\nrevise_provider = "anthropic-api"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )

    def refuse_switched_provider(
        fresh_config: ProjectConfig,
        _expected: Any,
        *,
        progress: Any,
    ) -> Any:
        del progress
        assert fresh_config.revise_provider == "anthropic-api"
        raise revision.RevisionApplicationError(
            "[revision-request-stale] the configured provider changed; nothing was sent"
        )

    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        refuse_switched_provider,
    )

    with pytest.raises(RevisionRefusal, match="request-stale"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=adapter.deck_scope,
                instruction="Add examples.",
                expected_fingerprint="rendered-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )

    assert not config.operations_file.exists()
    assert not config.staging_dir.exists()


def test_serve_starts_assistant_first_and_closes_it_after_main_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeAssistant:
        url = "http://127.0.0.1:41001/assistant-token/"

        def close(self) -> None:
            events.append("assistant-close")

    assistant = FakeAssistant()

    def start(_config: Any) -> FakeAssistant:
        events.append("assistant-start")
        return assistant

    monkeypatch.setattr(workbench_server, "_start_assistant_for", start)
    session = SimpleNamespace(token="main-workbench-token")

    def open_session(_config: Any, *, assistant_url: str) -> Any:
        events.append("session-open")
        assert assistant_url == assistant.url
        return session

    monkeypatch.setattr(workbench_server.WorkbenchSession, "open", staticmethod(open_session))

    class FakeMainServer:
        expected_host = "127.0.0.1:41002"

        def serve_forever(self) -> None:
            events.append("main-serve")
            raise KeyboardInterrupt

        def server_close(self) -> None:
            events.append("main-close")

    monkeypatch.setattr(workbench_server, "make_server", lambda _session: FakeMainServer())

    url = workbench_server.serve(_config(tmp_path), open_browser=False)

    assert url == "http://127.0.0.1:41002/main-workbench-token/"
    assert events == [
        "assistant-start",
        "session-open",
        "main-serve",
        "main-close",
        "assistant-close",
    ]
    output = capsys.readouterr().out
    assert f"Workbench: {url}" in output
    assert f"Assistant: {assistant.url}" in output
    assert "main-workbench-token" not in assistant.url


def test_expected_application_refusal_becomes_chatkit_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=tmp_path / "data" / "decks" / "potential.yaml",
        record_ids=("word:one",),
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            revision.RevisionApplicationError("nothing was sent")
        ),
    )

    with pytest.raises(RevisionRefusal, match="nothing was sent"):
        adapter.prepare_revision(
            deck_scope=adapter.deck_scope,
            instruction="Add examples.",
        )


def test_prepare_revision_surfaces_any_local_janki_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=tmp_path / "data" / "decks" / "potential.yaml",
        record_ids=("word:one",),
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("prompt unavailable; no provider was contacted")
        ),
    )

    with pytest.raises(RevisionRefusal, match="no provider was contacted"):
        adapter.prepare_revision(
            deck_scope=adapter.deck_scope,
            instruction="Add examples.",
        )


def test_execute_revision_surfaces_any_local_janki_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("missing provider key; no provider was contacted")
        ),
    )
    adapter.prepare_revision(
        deck_scope=adapter.deck_scope,
        instruction="Add examples.",
    )

    with pytest.raises(RevisionRefusal, match="no provider was contacted"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=adapter.deck_scope,
                instruction="Add examples.",
                expected_fingerprint="rendered-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )
