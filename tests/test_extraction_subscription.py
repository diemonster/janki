"""Extraction reaching Claude through the owner's subscription by default.

The content pass is unchanged: one source, one prompt, one answer. What
changes is *who is billed for it* and *how the request leaves this machine*.
Every transport probe and dispatch here is faked (IMPLEMENTATION_PLAN rule 6):
these tests are about the selection, the refusals that precede authority, and
the exact bytes recorded, not about the CLI's own behaviour.

The one rule the whole file exists to hold: janki never reaches the paid
Anthropic API because the subscription transport was inconvenient. A missing
CLI, a logged-out login, or any exception is a refusal — the API is used only
when a person wrote ``extract_provider = 'anthropic-api'`` down.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES, _prepared

from conftest import seed_prompts
from japanese_anki import claude_client, cli, extract, operations, prompts, staging
from japanese_anki.application.extraction import (
    ANSWER_SAVED,
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    describe_extraction,
    dispatch_extraction,
    extraction_capture_envelope,
    extraction_capture_parts,
    plan_corpus_extraction,
    prepare_extraction_transport,
    recover_extraction,
    recover_extraction_from_capture,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput
from japanese_anki.workbench import render
from japanese_anki.workbench.assistant_adapter import _extraction_billing_disclosure
from japanese_anki.workbench.dispatch import ExtractionActions

SCENARIO = "table_exhaustive"


def _answer() -> dict[str, Any]:
    """One fixture answer in the shape the extraction schema validates."""
    return json.loads((RESPONSES / f"{SCENARIO}.json").read_text(encoding="utf-8"))


def _stream(answer: Any) -> bytes:
    """A Claude Code stream whose final result carries that structured answer."""
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _project(tmp_path: Path, *, provider: str = "claude-code") -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        f'extract_provider = "{provider}"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _source(config: ProjectConfig, name: str = "lesson.pdf") -> Path:
    """One inbox source, already in the corpus, as every surface requires."""
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 fake")
    return path


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any Anthropic API use a loud failure rather than a silent bill."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _expectation(
    config: ProjectConfig, source: Path, plan: Any
) -> ExtractionDispatchExpectation:
    target = plan.targets[0]
    return ExtractionDispatchExpectation(
        source=source,
        provider=plan.provider,
        model=plan.model,
        mode=plan.mode,
        source_sha256=target.source_sha256,
        request_fingerprint=str(target.provenance["request_fingerprint"]),
        replacement_revision=None,
        replacement_confirmed=False,
        staging_path=target.staging_path,
        patterns_path=target.patterns_path,
        operations_path=config.operations_file,
    )


def _plan(config: ProjectConfig, source: Path, runner: FakeClaudeRunner) -> Any:
    return plan_corpus_extraction(
        config,
        source,
        mode=None,
        model=config.extract_model,
        provider_runner=runner,
        provider_which=_which,
        provider_env={},
    )


def test_extraction_plans_the_subscription_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured default is the owner's login, and it says so exactly."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    assert config.extract_provider == "claude-code"
    source = _source(config)
    runner = FakeClaudeRunner()

    plan = _plan(config, source, runner)

    target = plan.targets[0]
    assert plan.provider == "claude-code"
    assert target.provider_plan is not None
    assert target.provider_plan.provider == "claude-code"
    assert target.provider_plan.model == "claude-opus-5"
    assert target.billing_display == "Claude Max subscription via Claude Code"
    # The provenance the journal records is the provider's own request
    # identity, not a second one computed beside it.
    assert target.provenance["provider"] == "claude-code"
    assert (
        target.provenance["request_fingerprint"]
        == target.provider_plan.request_fingerprint
    )
    assert target.provenance["source_sha256"] == target.source_sha256


def test_a_missing_cli_refuses_before_the_journal_and_never_uses_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No CLI is a refusal. It is never a reason to bill an API key."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)

    with pytest.raises(JankiError):
        plan_corpus_extraction(
            config,
            source,
            mode=None,
            model=config.extract_model,
            provider_runner=FakeClaudeRunner(),
            provider_which=lambda name, **kwargs: None,
            provider_env={},
        )

    assert not config.operations_file.exists()


def test_a_logged_out_login_refuses_before_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login that cannot pay is a refusal, not a fallback to the API."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(auth={"loggedIn": False})

    with pytest.raises(JankiError):
        _plan(config, source, runner)

    assert not config.operations_file.exists()


def test_the_source_block_is_sent_once_and_captured_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paid bytes reach disk, and the streamed frames arrive, before parse."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    reply = _stream(_answer())
    runner = FakeClaudeRunner(reply=reply)
    plan = _plan(config, source, runner)
    expectation = _expectation(config, source, plan)

    outcome = dispatch_extraction(
        config,
        expectation,
        provider_runner=runner,
        provider_which=_which,
        provider_env={},
        provider_spawn=runner.spawn,
    )

    assert outcome is not None
    [(_command, _kwargs)] = runner.spawned
    sent = runner.processes[0].stdin.written.decode("utf-8")
    document = plan.targets[0].item.data_b64
    assert sent.count(document) == 1
    assert source.name in sent

    journal = operations.OperationJournal.load(config.operations_file)
    [entry] = journal.tracked()
    # The captured artifact holds the exact paid bytes, wrapped with the
    # request that produced them.
    saved, saved_request = extraction_capture_parts(
        journal.read_reply(entry.operation_id)
    )
    assert saved == reply
    assert saved_request is not None
    assert saved_request["provider_manifest"]["provider"] == "claude-code"
    assert journal.read_response_frames(entry.operation_id)


def test_a_changed_provider_refuses_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmation naming one transport cannot be spent on another."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    plan = _plan(config, source, runner)
    expectation = dataclasses.replace(
        _expectation(config, source, plan), provider="anthropic-api"
    )

    with pytest.raises(ExtractionDispatchError) as caught:
        dispatch_extraction(
            config,
            expectation,
            provider_runner=runner,
            provider_which=_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )

    assert caught.value.phase == "binding"
    assert not runner.spawned
    assert not config.operations_file.exists()


def test_a_changed_request_refuses_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    plan = _plan(config, source, runner)
    expectation = dataclasses.replace(
        _expectation(config, source, plan), request_fingerprint="0" * 64
    )

    with pytest.raises(ExtractionDispatchError):
        dispatch_extraction(
            config,
            expectation,
            provider_runner=runner,
            provider_which=_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )

    assert not runner.spawned


def test_a_captured_answer_the_schema_rejects_stays_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paid bytes survive a refusal, and re-reading them makes no second call."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    broken = _answer()
    broken["candidates"] = [{"expression": "話す"}]
    runner = FakeClaudeRunner(reply=_stream(broken))
    plan = _plan(config, source, runner)

    with pytest.raises(JankiError):
        dispatch_extraction(
            config,
            _expectation(config, source, plan),
            provider_runner=runner,
            provider_which=_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )

    journal = operations.OperationJournal.load(config.operations_file)
    [entry] = journal.tracked()
    saved, saved_request = extraction_capture_parts(
        journal.read_reply(entry.operation_id)
    )
    assert saved == _stream(broken)
    assert saved_request is not None

    # The same captured plan, re-read. No probe, no spawn, no client.
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("recovery contacted a provider")

    result = recover_extraction(
        plan.targets[0],
        model=plan.model,
        mode=plan.mode,
        raw_reply=_stream(_answer()),
    )
    assert result.candidates
    assert len(runner.spawned) == 1
    assert result.pattern_set.prompt_provenance["provider"] == "claude-code"


def test_recovery_never_probes_or_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    plan = _plan(config, source, runner)
    probes = len(runner.calls)

    result = recover_extraction(
        plan.targets[0],
        model=plan.model,
        mode=plan.mode,
        raw_reply=_stream(_answer()),
    )

    assert result.candidates
    assert len(runner.calls) == probes
    assert not runner.spawned


def test_preparation_precedes_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that disappears after planning leaves no authority behind it."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    plan = _plan(config, source, runner)
    seen = 0

    def vanishing_which(name: str, **kwargs: Any) -> str | None:
        """Present while the dispatch re-plans, gone by the time it prepares."""
        nonlocal seen
        seen += 1
        return _which(name, **kwargs) if seen == 1 else None

    with pytest.raises(ExtractionDispatchError) as caught:
        dispatch_extraction(
            config,
            _expectation(config, source, plan),
            provider_runner=runner,
            provider_which=vanishing_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )

    assert caught.value.phase == "preparation"
    assert not runner.spawned
    assert not config.operations_file.exists()


def test_the_consent_page_names_the_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()

    consent = describe_extraction(
        config,
        source,
        provider_runner=runner,
        provider_which=_which,
        provider_env={},
    )

    assert consent.sendable
    assert consent.provider == "claude-code"
    assert consent.billing_display == "Claude Max subscription via Claude Code"


def test_explicit_anthropic_api_selection_still_calls_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy path is still there — for whoever writes it down."""
    config = _project(tmp_path, provider="anthropic-api")
    source = _source(config)
    calls: list[tuple[Any, ...]] = []

    def fake_parse_call(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        capture = kwargs.get("capture")
        if capture is not None:
            capture(b'{"fixture": true}')
        return claude_client.CallResult(
            parsed=extract.candidate_schema()(**_answer()),
            stop_reason="end_turn",
            refusal=None,
        )

    monkeypatch.setattr(claude_client, "parse_call", fake_parse_call)
    monkeypatch.setattr(claude_client, "prepare_paid_client", lambda *a, **k: object())

    plan = plan_corpus_extraction(
        config, source, mode=None, model=config.extract_model
    )
    assert plan.provider == "anthropic-api"
    assert plan.targets[0].provenance["provider"] == "anthropic"
    assert plan.targets[0].billing_display == "Anthropic API billing"

    outcome = dispatch_extraction(config, _expectation(config, source, plan))

    assert outcome is not None
    assert calls


def test_the_transport_helper_refuses_an_unknown_provider(tmp_path: Path) -> None:
    config = _project(tmp_path)
    with pytest.raises(JankiError):
        prepare_extraction_transport(
            dataclasses.replace(
                plan_corpus_extraction(
                    config,
                    _source(config),
                    mode=None,
                    model=config.extract_model,
                    provider="anthropic-api",
                ),
                provider="mystery",
            )
        )


def test_a_prepared_input_carries_one_content_block(tmp_path: Path) -> None:
    """The block the provider plan is given is the source's own, unaltered."""
    item: PreparedInput = _prepared(tmp_path, "block.pdf")
    block = item.content_block()
    assert block == item.content_block()
    assert isinstance(block, dict)


def _dispatched(
    config: ProjectConfig, source: Path, runner: FakeClaudeRunner, plan: Any
) -> str:
    """Run one call whose answer the schema rejects, and return its operation."""
    with pytest.raises(ExtractionDispatchError) as caught:
        dispatch_extraction(
            config,
            _expectation(config, source, plan),
            provider_runner=runner,
            provider_which=_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )
    assert caught.value.operation_id
    return str(caught.value.operation_id)


def test_the_consent_page_charges_the_subscription_not_an_api_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered page names the account the planned request actually spends."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    consent = describe_extraction(
        config,
        source,
        provider_runner=runner,
        provider_which=_which,
        provider_env={},
    )

    page = render.render_consent(consent, token="t0ken", csrf="csrf", dispatch="cap")

    assert "Claude Max subscription via Claude Code" in page
    assert "Anthropic API credits" not in page
    assert "does not pay for this" not in page


def test_a_bound_capability_carries_the_transport_that_was_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the server dispatches with is the transport the page described."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    consent = describe_extraction(
        config,
        source,
        provider_runner=runner,
        provider_which=_which,
        provider_env={},
    )

    actions = ExtractionActions()
    token = actions.issue(consent, operations_path=config.operations_file)

    assert token
    assert consent.provider == "claude-code"
    # The bound capability the server dispatches from, not a default filled in
    # later: this is the value that becomes the expectation's provider.
    assert actions._pending[token].provider == "claude-code"


def test_the_wizard_disclosure_names_the_account_it_spends() -> None:
    """The Assistant's confirmation says which allowance a click spends."""
    subscription = _extraction_billing_disclosure(
        "claude-code", "Claude Max subscription via Claude Code"
    )
    api = _extraction_billing_disclosure("anthropic-api", "Anthropic API billing")

    assert "subscription" in subscription
    assert "no Anthropic API credit" in subscription.replace(
        "charges no Anthropic API credit", "no Anthropic API credit"
    )
    assert "paid Anthropic API call" in api
    assert "Claude Pro or Max does not pay for it" in api


def test_a_rejected_answer_is_classified_as_saved_not_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real answer the schema refuses is still an answer that was paid for."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    broken = _answer()
    broken["candidates"] = [{"expression": "話す"}]
    runner = FakeClaudeRunner(reply=_stream(broken))
    plan = _plan(config, source, runner)

    with pytest.raises(ExtractionDispatchError) as caught:
        dispatch_extraction(
            config,
            _expectation(config, source, plan),
            provider_runner=runner,
            provider_which=_which,
            provider_env={},
            provider_spawn=runner.spawn,
        )

    assert caught.value.failure is not None
    assert caught.value.failure.outcome == ANSWER_SAVED


def test_show_reply_exports_the_complete_captured_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: Any
) -> None:
    """The sole export surface hands over the whole artifact, byte for byte.

    Not the provider's bytes alone: the manifest saved beside them is what a
    later recovery needs, and an export that quietly dropped it would hand
    somebody a reply they can read and cannot replay.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    broken = _answer()
    broken["candidates"] = [{"expression": "話す"}]
    reply = _stream(broken)
    runner = FakeClaudeRunner(reply=reply)
    plan = _plan(config, source, runner)
    operation_id = _dispatched(config, source, runner, plan)
    before = operations.OperationJournal.load(config.operations_file).read_reply(
        operation_id
    )
    capfdbinary.readouterr()

    code = cli.main(
        ["--root", str(tmp_path), "operations", "--show-reply", operation_id]
    )

    assert code == 0
    printed = capfdbinary.readouterr().out
    assert printed == before
    # The exported artifact carries both halves: the exact provider reply and
    # the request manifest that produced it.
    exported_reply, exported_request = extraction_capture_parts(printed)
    assert exported_reply == reply
    assert exported_request is not None
    assert exported_request["provider_manifest"]["provider"] == "claude-code"
    # Reading it changed nothing.
    after = operations.OperationJournal.load(config.operations_file).read_reply(
        operation_id
    )
    assert after == before


def test_recovery_survives_an_edited_prompt_but_refuses_a_changed_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved request is read back; the current decoder must still fit it."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    plan = _plan(config, source, runner)
    target = plan.targets[0]
    payload = extraction_capture_envelope(target.provenance, _stream(_answer()))

    # The prompts have been edited since. Recovery must not read them at all.
    def refuse_prompt(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("recovery read a prompt file")

    monkeypatch.setattr(prompts, "load", refuse_prompt)
    monkeypatch.setattr(claude_client, "read_style_guide", refuse_prompt)

    recovered = recover_extraction_from_capture(payload, source_name=source.name)
    assert recovered.candidates

    # The response contract has moved. The bytes and the request are kept, and
    # janki says so rather than parsing an old answer under a new shape.
    class Different(BaseModel):
        answer: str

    monkeypatch.setattr(extract, "candidate_schema", lambda: Different)
    with pytest.raises(JankiError) as caught:
        recover_extraction_from_capture(payload, source_name=source.name)

    assert "preserved" in str(caught.value)
    raw, saved = extraction_capture_parts(payload)
    assert raw == _stream(_answer())
    assert saved is not None and saved["provider_manifest"]["provider"] == "claude-code"


def test_recovery_refuses_to_restate_the_answer_in_another_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    target = _plan(config, source, runner).targets[0]
    payload = extraction_capture_envelope(target.provenance, _stream(_answer()))

    with pytest.raises(JankiError):
        recover_extraction_from_capture(payload, source_name=source.name, mode="prose")


def test_a_tampered_capture_wrapper_is_refused_rather_than_shown_as_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named mutation: the wrapper version moves, and nothing pretends to read it."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    target = _plan(config, source, runner).targets[0]
    payload = extraction_capture_envelope(target.provenance, _stream(_answer()))

    decoded = json.loads(payload)
    decoded["janki_extraction_capture"] = 99
    tampered = json.dumps(decoded).encode("utf-8")

    with pytest.raises(JankiError):
        extraction_capture_parts(tampered)
    # And a classifier reading the same artifact reports no answer rather than
    # inventing one out of the wrapper's own fields.
    assert operations.response_answer_text(tampered) == ""
    # The untampered wrapper still yields the exact structured answer.
    assert operations.response_answer_text(payload)


def test_a_moved_request_identity_refuses_the_saved_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named mutation: the saved manifest names a different request from the
    provenance around it, and the staging validator says so by name."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner()
    target = _plan(config, source, runner).targets[0]
    run_id = staging.new_review_run_id()
    block = {"source_fingerprint": target.provenance["source_sha256"]}

    moved = json.loads(json.dumps(dict(target.provenance)))
    moved["provider_manifest"]["request_fingerprint"] = "f" * 64

    with pytest.raises(JankiError) as mutated:
        staging._validate_prompt_provenance(
            {"prompt_provenance": moved, "review_run_id": run_id}, block
        )
    assert "different requests" in str(mutated.value)

    # The unmutated pair gets past the provider-request check; whatever the
    # rest of a staged file still needs is a different complaint.
    with pytest.raises(JankiError) as clean:
        staging._validate_prompt_provenance(
            {
                "prompt_provenance": json.loads(
                    json.dumps(dict(target.provenance))
                ),
                "review_run_id": run_id,
            },
            block,
        )
    assert "different requests" not in str(clean.value)
