"""The optional ChatKit sidecar is wired to one exact revision application plan."""

from __future__ import annotations

import builtins
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki.application import revision
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench import server as workbench_server
from japanese_anki.workbench.assistant import (
    OwnerActionConfirmation,
    RevisionConfirmation,
    RevisionRefusal,
)


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


def test_discovery_refuses_two_rich_conjugation_decks_without_guessing(
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

    assert adapter is None
    assert len(warnings) == 1
    assert "exactly one" in warnings[0]
    assert "potential.yaml" in warnings[0]
    assert "te-form.yaml" in warnings[0]
    assert "No deck was guessed" in warnings[0]


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
    assert (
        "Provider request identity: narrow-provider-request-fingerprint"
    ) in wire
    assert "Anthropic" not in wire
    assert "API" not in wire


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
    assert result.review_url is None
    assert result.next_action == "apply"


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


def test_adapter_binds_apply_audio_and_build_to_the_existing_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    staging = tmp_path / "data" / "staging" / "proposal.json"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    deck_text = "applied deck bytes"
    deck_sha = hashlib.sha256(deck_text.encode()).hexdigest()
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: SimpleNamespace(revision=SimpleNamespace(text=deck_text)),
    )
    example = ExampleSentence(
        japanese="話せます。",
        furigana="話[はな]せます。",
        romaji="",
        english="I can speak.",
        register="polite",
    )

    class FakeApplyPlan:
        plan_fingerprint = "apply-plan"
        staging_path = staging
        deck_relative_path = "data/decks/potential.yaml"
        selected_record_ids = ("word:one",)
        current_form_note = "Old note."
        current_drill_examples = {"word:one": (example,)}
        operation_id = "operation-one"
        model = "claude-opus-5"
        request_fingerprint = "request-fingerprint"
        intended_deck_sha256 = deck_sha
        form_note = "Use the potential form."
        drill_examples = {"word:one": (example,)}
        archive_path = tmp_path / "data" / "staging" / "done" / "proposal.json"

    apply_plan = FakeApplyPlan()
    monkeypatch.setattr(
        assistant_adapter.revision_apply,
        "RevisionApplyPlan",
        FakeApplyPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision_apply,
        "plan_revision_apply",
        lambda config, path: apply_plan,
    )
    apply_runs: list[Any] = []
    apply_progress: list[str] = []

    def execute_apply(config: Any, plan: Any, *, progress: Any) -> Any:
        apply_runs.append(plan)
        for label in (
            "Preparing proposal",
            "Re-reading proposal",
            "Applying revision",
            "Archiving proposal",
        ):
            progress(label)
        return SimpleNamespace(
            archive_path=apply_plan.archive_path,
            deck_sha256=deck_sha,
        )

    monkeypatch.setattr(
        assistant_adapter.revision_apply,
        "execute_revision_apply",
        execute_apply,
    )

    rendered_apply = adapter.prepare_followup(
        "apply",
        deck_scope=adapter.deck_scope,
        continuation_context="data/staging/proposal.json",
    )
    assert rendered_apply.fingerprint == "apply-plan"
    assert "話せます" in "\n".join(rendered_apply.effects)
    applied = adapter.consume_followup(
        OwnerActionConfirmation(
            capability="one-use",
            deck_scope=adapter.deck_scope,
            kind="apply",
            expected_fingerprint="apply-plan",
            target=rendered_apply.target,
        ),
        progress=apply_progress.append,
    )
    assert apply_runs == [apply_plan]
    assert apply_progress == [
        "Preparing proposal",
        "Re-reading proposal",
        "Applying revision",
        "Archiving proposal",
    ]
    assert applied.next_action == "audio"

    class FakeAudioPlan:
        fingerprint = "audio-plan"
        example_counts = SimpleNamespace(
            total=32,
            current=4,
            recoverable=2,
            provider_required=26,
        )
        example_provider = SimpleNamespace(
            name="openai-realtime",
            access="paid-network",
            voice="cedar",
            speed=0.75,
            settings={"model": "gpt-realtime-1.5"},
        )
        clips = (
            SimpleNamespace(
                state="provider-required",
                request_input="話せます。",
                target="audio/example.wav",
                provider=example_provider,
            ),
        )

    audio_plan = FakeAudioPlan()
    monkeypatch.setattr(assistant_adapter.audio_application, "AudioPlan", FakeAudioPlan)
    monkeypatch.setattr(
        assistant_adapter.audio_application,
        "plan_deck_audio",
        lambda *args, **kwargs: audio_plan,
    )
    audio_runs: list[dict[str, Any]] = []

    def execute_audio(*args: Any, **kwargs: Any) -> Any:
        audio_runs.append(kwargs)
        return SimpleNamespace(
            succeeded=True,
            pending_recovery=False,
            ledger_error=None,
            stopped_by=None,
            state="complete",
            file_count=26,
            up_to_date=4,
        )

    monkeypatch.setattr(
        assistant_adapter.audio_application,
        "execute_deck_audio",
        execute_audio,
    )
    rendered_audio = adapter.prepare_followup(
        "audio",
        deck_scope=adapter.deck_scope,
        continuation_context=deck_sha,
    )
    audio_wire = "\n".join((*rendered_audio.effects, *rendered_audio.disclosures))
    assert "26 clips from the provider" in audio_wire
    assert "model gpt-realtime-1.5" in audio_wire
    assert "26 paid provider calls" in audio_wire
    voiced = adapter.consume_followup(
        OwnerActionConfirmation(
            capability="one-use",
            deck_scope=adapter.deck_scope,
            kind="audio",
            expected_fingerprint="audio-plan",
            target=rendered_audio.target,
        ),
        progress=lambda _label: None,
    )
    assert audio_runs[0]["expected_fingerprint"] == "audio-plan"
    assert audio_runs[0]["force"] is False
    assert audio_runs[0]["prune"] is False
    assert voiced.next_action == "build"

    class FakeBuildPlan:
        fingerprint = "build-plan"
        output_path = tmp_path / "dist" / "potential.apkg"
        card_count = 16
        deck_name = "Potential Practice"
        form = "potential"
        deck_sha256 = deck_sha
        source_sha256 = "b" * 64

    build_plan = FakeBuildPlan()
    monkeypatch.setattr(
        assistant_adapter.deck_build,
        "ConjugationDeckBuildPlan",
        FakeBuildPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.deck_build,
        "plan_conjugation_deck_build",
        lambda *args: build_plan,
    )
    build_runs: list[Any] = []
    monkeypatch.setattr(
        assistant_adapter.deck_build,
        "execute_conjugation_deck_build",
        lambda config, plan: build_runs.append(plan)
        or SimpleNamespace(
            card_count=16,
            output_path=build_plan.output_path,
            package_sha256="a" * 64,
        ),
    )
    rendered_build = adapter.prepare_followup(
        "build",
        deck_scope=adapter.deck_scope,
        continuation_context=deck_sha,
    )
    assert "16 cards" in "\n".join(rendered_build.effects)
    built = adapter.consume_followup(
        OwnerActionConfirmation(
            capability="one-use",
            deck_scope=adapter.deck_scope,
            kind="build",
            expected_fingerprint="build-plan",
            target=rendered_build.target,
        ),
        progress=lambda _label: None,
    )
    assert build_runs == [build_plan]
    assert built.next_action is None


def test_adapter_never_offers_build_after_audio_did_not_finish_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=_config(tmp_path),
        deck_path=deck,
        record_ids=("word:one",),
    )
    deck_text = "applied deck bytes"
    deck_sha = hashlib.sha256(deck_text.encode()).hexdigest()
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: SimpleNamespace(revision=SimpleNamespace(text=deck_text)),
    )

    class FakeAudioPlan:
        fingerprint = "audio-plan"
        example_counts = SimpleNamespace(
            total=2,
            current=0,
            recoverable=0,
            provider_required=2,
        )
        example_provider = SimpleNamespace(
            name="openai-realtime",
            access="paid-network",
            voice="cedar",
            speed=0.75,
            settings={"model": "gpt-realtime-1.5"},
        )
        clips = ()

    monkeypatch.setattr(assistant_adapter.audio_application, "AudioPlan", FakeAudioPlan)
    monkeypatch.setattr(
        assistant_adapter.audio_application,
        "plan_deck_audio",
        lambda *args, **kwargs: FakeAudioPlan(),
    )
    monkeypatch.setattr(
        assistant_adapter.audio_application,
        "execute_deck_audio",
        lambda *args, **kwargs: SimpleNamespace(
            succeeded=False,
            pending_recovery=True,
            ledger_error=None,
            stopped_by="paid response needs recovery",
            state="generation-stopped",
            file_count=0,
            up_to_date=0,
        ),
    )
    plan = adapter.prepare_followup(
        "audio",
        deck_scope=adapter.deck_scope,
        continuation_context=deck_sha,
    )

    with pytest.raises(RevisionRefusal, match="did not complete durably"):
        adapter.consume_followup(
            OwnerActionConfirmation(
                capability="one-use",
                deck_scope=adapter.deck_scope,
                kind="audio",
                expected_fingerprint=plan.fingerprint,
                target=plan.target,
            ),
            progress=lambda _label: None,
        )



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
