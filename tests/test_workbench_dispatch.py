"""W3: one browser consent becomes exactly one journaled extraction.

These are HTTP tests because the boundary is the feature: the form action is
bound to what was rendered, a fresh plan is compared at click time, and only
then may the shared extraction service spend and write.  Every provider here
is fake; the real normalization, journal, capture, and staging paths run.
"""

from __future__ import annotations

import html
import http.client
import json
import threading
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pytest
from test_application_journey import _project, _stage

import japanese_anki.io as janki_io
from conftest import seed_prompts
from japanese_anki import inputs, operations, patterns, staging
from japanese_anki.application.extraction import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    OUTCOME_UNKNOWN,
    DispatchFailure,
    ExtractionCompletionError,
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    ExtractionOutcome,
    dispatch_extraction,
)
from japanese_anki.claude_client import CallResult
from japanese_anki.config import ProjectConfig
from japanese_anki.io import DataError, atomic_write_text_bound
from japanese_anki.staging import read_staging
from japanese_anki.workbench import WorkbenchSession, make_server
from japanese_anki.workbench import server as workbench_server

PDF = b"%PDF-1.7 fake"
RESPONSES = Path(__file__).parent / "fixtures" / "workbench" / "responses"


@pytest.mark.parametrize(
    ("outcome", "cleanup_pending", "changed", "next_step"),
    [
        (ANSWER_SAVED, False, "exact provider answer is saved", "--show-reply op-1"),
        (ANSWER_EMPTY, False, "reply contains no answer", "--show-reply op-1"),
        (ANSWER_UNAVAILABLE, False, "bytes are unavailable", "janki operations"),
        (FORGOTTEN, True, "forget decision is recorded", "--forget op-1"),
        (FORGOTTEN, False, "no recovery answer remains", "Back button"),
        (OUTCOME_UNKNOWN, False, "unsettled provider outcome", "Do not retry"),
    ],
)
def test_paid_failure_pages_preserve_each_exact_journal_outcome(
    outcome: str,
    cleanup_pending: bool,
    changed: str,
    next_step: str,
) -> None:
    view = workbench_server._paid_dispatch_failure_view(
        RuntimeError("provider stopped"),
        DispatchFailure(
            operation_id="op-1",
            outcome=outcome,
            cleanup_pending=cleanup_pending,
        ),
        unchanged="No card was written.",
    )

    assert changed in view.changed
    assert next_step in view.next_step
    assert view.money == (
        "The provider request was dispatched and may already have been billed."
    )


class _DispatchForm(HTMLParser):
    """The successful controls in the paid POST form, as a browser sends them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_form = False
        self.action = ""
        self.fields: dict[str, str] = {}
        self.replacement_offered = False
        self.submitters: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "form":
            self.in_form = (
                values.get("method", "").casefold() == "post"
                and "/extract/" in values.get("action", "")
            )
            if self.in_form:
                self.action = values["action"]
            return
        if not self.in_form or tag not in {"input", "button"}:
            return
        if tag == "button" and values.get("type", "submit").casefold() == "submit":
            self.submitters.append(values)
        name = values.get("name", "")
        if not name:
            return
        if tag == "input" and values.get("type", "").casefold() == "checkbox":
            if name == "replace":
                self.replacement_offered = True
            return
        self.fields[name] = values.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self.in_form:
            self.in_form = False


class _FakeCall:
    """A provider-shaped answer that preserves the capture-before-parse seam."""

    def __init__(
        self,
        scenario: str = "lesson_with_grammar",
        *,
        stop_reason: str = "end_turn",
        entered: threading.Event | None = None,
        allow_answer: threading.Event | None = None,
        captured: threading.Event | None = None,
        allow_validation: threading.Event | None = None,
    ) -> None:
        self.raw = json.loads(
            (RESPONSES / f"{scenario}.json").read_text(encoding="utf-8")
        )
        self.stop_reason = stop_reason
        self.entered = entered
        self.allow_answer = allow_answer
        self.captured = captured
        self.allow_validation = allow_validation
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        model: str,
        system_blocks: Any,
        user_content: Any,
        schema: Any,
        client: Any = None,
        **kwargs: Any,
    ) -> CallResult:
        del client
        self.calls.append(
            {"model": model, "system": system_blocks, "content": user_content}
        )
        if self.entered is not None:
            self.entered.set()
        if self.allow_answer is not None and not self.allow_answer.wait(3):
            raise AssertionError("test never allowed the fake answer to arrive")
        capture = kwargs.get("capture")
        if capture is not None:
            capture(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(self.raw, ensure_ascii=False),
                        }
                    ]
                }
            )
        if self.captured is not None:
            self.captured.set()
        if self.allow_validation is not None and not self.allow_validation.wait(3):
            raise AssertionError("test never allowed the fake answer to validate")
        parsed = schema(**self.raw) if self.stop_reason == "end_turn" else None
        return CallResult(parsed, self.stop_reason, None)


def _corpus(tmp_path: Path, *names: str) -> None:
    _project(tmp_path)
    seed_prompts(tmp_path)
    for name in names or ("lesson.pdf",):
        (tmp_path / "inbox" / name).write_bytes(PDF)


def _session(tmp_path: Path) -> WorkbenchSession:
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def _running(session: WorkbenchSession) -> tuple[Any, threading.Thread]:
    server = make_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(
    server: Any,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=3
    )
    headers: dict[str, str] = {}
    if body is not None:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
        }
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, payload


def _form(server: Any, session: WorkbenchSession, name: str = "lesson.pdf") -> _DispatchForm:
    status, _headers, payload = _request(
        server,
        "GET",
        f"/{session.token}/extract/{quote(name, safe='')}",
    )
    assert status == 200, payload.decode("utf-8")
    form = _DispatchForm()
    form.feed(payload.decode("utf-8"))
    assert form.action
    assert form.fields.get("dispatch")
    return form


def _post(server: Any, form: _DispatchForm, fields: dict[str, str] | None = None):
    submitted = dict(form.fields if fields is None else fields)
    body = urlencode(submitted).encode("utf-8")
    return _request(server, "POST", form.action, body=body)


def _post_in_thread(
    server: Any,
    form: _DispatchForm,
    fields: dict[str, str],
) -> tuple[threading.Thread, list[tuple[int, dict[str, str], bytes]]]:
    responses: list[tuple[int, dict[str, str], bytes]] = []
    thread = threading.Thread(
        target=lambda: responses.append(_post(server, form, fields)),
        daemon=True,
    )
    thread.start()
    return thread, responses


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeCall) -> None:
    monkeypatch.setattr(
        "japanese_anki.extract.claude_client.parse_call",
        fake,
    )


def test_browser_paid_post_delegates_one_exact_request_to_shared_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP owns the one-use click; the application service owns the run."""
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    calls: list[tuple[ProjectConfig, ExtractionDispatchExpectation]] = []
    progress_labels: list[str] = []

    def shared_dispatch(
        config: ProjectConfig,
        expected: ExtractionDispatchExpectation,
        *,
        progress: Any = None,
    ) -> ExtractionOutcome:
        calls.append((config, expected))
        for label in (
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ):
            progress_labels.append(label)
            assert progress is not None
            progress(label)
        return ExtractionOutcome(
            target=config.staging_dir / "lesson.pdf.yaml",
            records=2,
            already_known=0,
            unusable=0,
            duplicates=0,
            coverage_status="selection",
            kept_reviewed_patterns=False,
            source="lesson.pdf",
        )

    monkeypatch.setattr(workbench_server, "dispatch_extraction", shared_dispatch)
    try:
        form = _form(server, session)
        status, _headers, payload = _post(server, form)

        assert status == 200, payload.decode("utf-8")
        assert len(calls) == 1
        config, expected = calls[0]
        assert config == session.config
        assert expected.source == session.source_path("lesson.pdf")
        assert expected.model == form.fields["model"]
        assert (expected.mode or "") == form.fields["mode"]
        assert expected.request_fingerprint == form.fields["request_fingerprint"]
        assert expected.source_sha256
        assert expected.replacement_revision is None
        assert not expected.replacement_confirmed
        assert expected.staging_path == (
            session.config.staging_dir / "lesson.pdf.yaml"
        ).resolve()
        assert expected.patterns_path == session.config.patterns_file.resolve()
        assert expected.operations_path == session.config.operations_file.resolve()
        assert progress_labels == [
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ]
        assert "Review proposed cards" in payload.decode("utf-8")
        assert not session.config.operations_file.exists()
    finally:
        server.shutdown()
        server.server_close()


def _dispatch_expectation(
    session: WorkbenchSession,
    *,
    replacement_confirmed: bool = False,
) -> ExtractionDispatchExpectation:
    consent = session.consent("lesson.pdf")
    assert consent is not None and consent.target is not None
    source = session.source_path("lesson.pdf")
    assert source is not None
    return ExtractionDispatchExpectation(
        provider="anthropic-api",
        source=source,
        model=consent.model,
        mode=consent.mode,
        source_sha256=consent.target.source_sha256,
        request_fingerprint=str(
            consent.target.provenance["request_fingerprint"]
        ),
        replacement_revision=consent.replacement_revision,
        replacement_confirmed=replacement_confirmed,
        staging_path=consent.target.staging_path.resolve(),
        patterns_path=consent.target.patterns_path.resolve(),
        operations_path=session.config.operations_file.resolve(),
    )


def test_shared_dispatch_runs_the_existing_completion_route_and_four_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    monkeypatch.setattr(
        "japanese_anki.application.extraction.claude_client.prepare_paid_client",
        lambda: object(),
    )
    import japanese_anki.application.extraction as extraction_application

    real_complete = extraction_application.complete_extraction
    completed: list[str] = []

    def observed_complete(*args: Any, **kwargs: Any) -> ExtractionOutcome:
        completed.append(str(kwargs["operation_id"]))
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(extraction_application, "complete_extraction", observed_complete)
    labels: list[str] = []

    outcome = dispatch_extraction(session.config, expected, progress=labels.append)

    assert outcome.target == session.config.staging_dir / "lesson.pdf.yaml"
    assert len(fake.calls) == 1
    assert len(completed) == 1
    assert labels == [
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    journal = operations.OperationJournal.load(session.config.operations_file)
    assert journal.operations[completed[0]].state == "committed"


def test_shared_dispatch_replans_and_refuses_rendered_request_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    prompt = tmp_path / "prompts" / "extract-auto.md"
    prompt.write_text(
        prompt.read_text(encoding="utf-8") + "\nChanged after consent.\n",
        encoding="utf-8",
    )

    with pytest.raises(ExtractionDispatchError, match="request changed") as caught:
        dispatch_extraction(session.config, expected, client=object())

    assert caught.value.phase == "binding"
    assert not caught.value.provider_dispatched
    assert fake.calls == []
    assert operations.OperationJournal.load(
        session.config.operations_file
    ).operations == {}


def test_shared_dispatch_binds_source_digest_independently_of_request_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    import japanese_anki.application.extraction as extraction_application

    real_plan = extraction_application.plan_corpus_extraction

    def plan_with_divergent_source(*args: Any, **kwargs: Any):
        planned = real_plan(*args, **kwargs)
        target = replace(planned.targets[0], source_sha256="f" * 64)
        return replace(planned, targets=(target,))

    monkeypatch.setattr(
        extraction_application,
        "plan_corpus_extraction",
        plan_with_divergent_source,
    )

    with pytest.raises(ExtractionDispatchError, match="request changed"):
        dispatch_extraction(session.config, expected, client=object())

    assert operations.OperationJournal.load(
        session.config.operations_file
    ).operations == {}


def _assert_dispatch_path_drift_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_field: str,
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    changed_path = (tmp_path / "changed" / config_field).resolve()
    changed_config = replace(session.config, **{config_field: changed_path})
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)

    with pytest.raises(ExtractionDispatchError, match="destination changed") as caught:
        dispatch_extraction(changed_config, expected, client=object())

    assert caught.value.phase == "binding"
    assert not caught.value.provider_dispatched
    assert fake.calls == []
    assert operations.OperationJournal.load(
        changed_config.operations_file
    ).operations == {}


def test_shared_dispatch_binds_the_rendered_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_dispatch_path_drift_is_refused(
        tmp_path,
        monkeypatch,
        config_field="staging_dir",
    )


def test_shared_dispatch_binds_the_rendered_pattern_store_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_dispatch_path_drift_is_refused(
        tmp_path,
        monkeypatch,
        config_field="patterns_file",
    )


def test_shared_dispatch_binds_the_rendered_operation_journal_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_dispatch_path_drift_is_refused(
        tmp_path,
        monkeypatch,
        config_field="operations_file",
    )


@pytest.mark.parametrize(
    "raised_label",
    (
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ),
)
def test_progress_callback_failure_cannot_interrupt_a_confirmed_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised_label: str,
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    labels: list[str] = []

    def broken_progress(label: str) -> None:
        labels.append(label)
        if label == raised_label:
            raise RuntimeError("the presentation disappeared")

    outcome = dispatch_extraction(
        session.config,
        expected,
        client=object(),
        progress=broken_progress,
    )

    assert outcome.records == 2
    assert len(fake.calls) == 1
    assert labels == [
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    journal = operations.OperationJournal.load(session.config.operations_file)
    assert len(journal.operations) == 1
    assert next(iter(journal.operations.values())).state == "committed"


def test_shared_dispatch_will_not_derive_force_from_a_rendered_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "table_exhaustive", filename="lesson.pdf")
    seed_prompts(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session, replacement_confirmed=False)
    assert expected.replacement_revision is not None
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    before = (tmp_path / "staging" / "lesson.pdf.yaml").read_bytes()
    baseline = operations.OperationJournal.load(
        session.config.operations_file
    ).operations

    with pytest.raises(ExtractionDispatchError, match="Confirm that this re-read"):
        dispatch_extraction(session.config, expected, client=object())

    assert fake.calls == []
    assert (tmp_path / "staging" / "lesson.pdf.yaml").read_bytes() == before
    assert (
        operations.OperationJournal.load(session.config.operations_file).operations
        == baseline
    )


def test_shared_dispatch_refuses_replacement_confirmation_the_plan_never_offered(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = replace(
        _dispatch_expectation(session),
        replacement_confirmed=True,
    )
    assert expected.replacement_revision is None

    with pytest.raises(ExtractionDispatchError, match="did not offer"):
        dispatch_extraction(session.config, expected, client=object())

    assert operations.OperationJournal.load(
        session.config.operations_file
    ).operations == {}


def test_shared_dispatch_reports_shape_when_provider_returns_without_capture_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    expected = _dispatch_expectation(session)
    fake = _FakeCall()

    def without_capture(*args: Any, **kwargs: Any) -> CallResult:
        kwargs.pop("capture", None)
        return fake(*args, **kwargs)

    monkeypatch.setattr(
        "japanese_anki.extract.claude_client.parse_call",
        without_capture,
    )
    labels: list[str] = []

    outcome = dispatch_extraction(
        session.config,
        expected,
        client=object(),
        progress=labels.append,
    )

    assert outcome.records == 2
    assert labels == [
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]


def test_missing_key_refuses_before_journal_authority_or_provider_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)

    def missing_key() -> object:
        raise DataError("ANTHROPIC_API_KEY is not set; no provider was contacted")

    monkeypatch.setattr(
        "japanese_anki.workbench.server.claude_client.prepare_paid_client",
        missing_key,
    )
    try:
        form = _form(server, session)
        status, _headers, payload = _post(server, form)

        text = html.unescape(payload.decode("utf-8"))
        assert status == 409
        assert "ANTHROPIC_API_KEY is not set" in text
        assert "No extraction operation was authorized" in text
        assert "No provider was contacted and no paid call was made" in text
        assert fake.calls == []
        assert not session.config.operations_file.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_post_replans_and_refuses_a_request_changed_since_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        prompt = tmp_path / "prompts" / "extract-auto.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "\nChanged.\n",
                          encoding="utf-8")

        status, _headers, payload = _post(server, form)

        assert status == 409
        text = html.unescape(payload.decode("utf-8"))
        assert "changed after this page was rendered" in text
        assert "Nothing was sent" in text
        assert fake.calls == []
        assert not (tmp_path / "staging" / "lesson.pdf.yaml").exists()
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_post_refuses_source_bytes_changed_since_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        source = session.source_path("lesson.pdf")
        assert source is not None
        source.write_bytes(source.read_bytes() + b"\nchanged source bytes\n")

        status, _headers, payload = _post(server, form)

        assert status == 409
        text = html.unescape(payload.decode("utf-8"))
        assert "changed after this page was rendered" in text
        assert "Nothing was sent" in text
        assert fake.calls == []
        config = ProjectConfig.load(tmp_path)
        assert not (config.staging_dir / "lesson.pdf.yaml").exists()
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_a_staging_file_that_appears_after_consent_is_not_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        target = tmp_path / "staging" / "lesson.pdf.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        sentinel = b"review that appeared after consent\n"
        target.write_bytes(sentinel)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "Staging file already exists" in payload.decode("utf-8")
        assert target.read_bytes() == sentinel
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_a_staging_parent_that_becomes_a_file_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        config.staging_dir.write_text("not a directory\n", encoding="utf-8")

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "non-directory staging parent" in payload.decode("utf-8")
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("swap", ["source", "root"])
def test_a_corpus_path_swap_during_post_replan_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap: str
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    source = config.scan_inbox / "lesson.pdf"
    held = tmp_path / ("held-source.pdf" if swap == "source" else "held-inbox")
    outside = tmp_path / ("outside.pdf" if swap == "source" else "outside-inbox")
    if swap == "source":
        outside.write_bytes(b"private outside bytes")
    else:
        outside.mkdir()
        (outside / "lesson.pdf").write_bytes(b"private outside bytes")
    real_read = inputs._read_fd_bytes
    armed = False
    swapped = False

    def replace_path(descriptor: int) -> bytes:
        nonlocal swapped
        captured = real_read(descriptor)
        if not armed or swapped:
            return captured
        if swap == "source":
            source.rename(held)
            source.symlink_to(outside)
        else:
            config.scan_inbox.rename(held)
            config.scan_inbox.symlink_to(outside, target_is_directory=True)
        swapped = True
        return captured

    monkeypatch.setattr(inputs, "_read_fd_bytes", replace_path)
    try:
        form = _form(server, session)
        armed = True

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "changed while" in payload.decode("utf-8")
        assert "Nothing was sent" in payload.decode("utf-8")
        assert swapped
        assert fake.calls == []
        assert not (config.staging_dir / "lesson.pdf.yaml").exists()
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        if swap == "source":
            if source.is_symlink():
                source.unlink()
            if held.exists():
                held.rename(source)
        else:
            if config.scan_inbox.is_symlink():
                config.scan_inbox.unlink()
            if held.exists():
                held.rename(config.scan_inbox)
        server.shutdown()
        server.server_close()


def test_a_corpus_root_symlink_loop_during_post_replan_is_a_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    held = tmp_path / "held-inbox"
    real_source_path = WorkbenchSession.source_path
    real_resolve = Path.resolve
    swapped = False

    def source_then_replace_root(
        current_session: WorkbenchSession, name: str
    ) -> Path | None:
        nonlocal swapped
        source = real_source_path(current_session, name)
        if source is not None and not swapped:
            config.scan_inbox.rename(held)
            config.scan_inbox.symlink_to(
                config.scan_inbox, target_is_directory=True
            )
            swapped = True
        return source

    def resolve_like_python_311(path: Path, *args: Any, **kwargs: Any) -> Path:
        # Python 3.11 reports a symlink loop from resolve() as RuntimeError;
        # newer pathlib leaves a non-strict loop unresolved. The workbench
        # must not make this resolving preflight on either runtime: its
        # descriptor-bound reader owns containment and formats the refusal.
        if swapped and path == config.scan_inbox:
            raise RuntimeError("Symlink loop from corpus-root replacement")
        return real_resolve(path, *args, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(
            WorkbenchSession, "source_path", source_then_replace_root
        )
        monkeypatch.setattr(Path, "resolve", resolve_like_python_311)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "Nothing was sent" in payload.decode("utf-8")
        assert swapped
        assert fake.calls == []
        assert not (config.staging_dir / "lesson.pdf.yaml").exists()
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        if config.scan_inbox.is_symlink():
            config.scan_inbox.unlink()
        if held.exists():
            held.rename(config.scan_inbox)
        server.shutdown()
        server.server_close()


def test_new_staging_write_is_bound_to_the_absence_shown_at_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-cooperating writer can create the target after every preflight;
    the final atomic replace must still refuse rather than erase it."""
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    target = config.staging_dir / "lesson.pdf.yaml"
    sentinel = b"review created at the final write seam\n"
    real_write = staging.atomic_write_text_bound
    injected = False

    def interfere(path: Path, text: str, **kwargs: Any) -> None:
        nonlocal injected
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(sentinel)
        injected = True
        real_write(path, text, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(staging, "atomic_write_text_bound", interfere)

        status, _headers, payload = _post(server, form)

        assert status == 200
        assert "changed before replace" in html.unescape(payload.decode("utf-8"))
        assert injected
        assert target.read_bytes() == sentinel
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(config.operations_file)
        assert [operation.state for operation in journal.operations.values()] == [
            "result_captured"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_replacement_requires_explicit_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "table_exhaustive", filename="lesson.pdf")
    seed_prompts(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    target = tmp_path / "staging" / "lesson.pdf.yaml"
    before = target.read_bytes()
    try:
        form = _form(server, session)
        assert form.replacement_offered

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "confirm" in payload.decode("utf-8").casefold()
        assert target.read_bytes() == before
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_confirmed_replacement_forces_plan_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    store = patterns.load_store(ProjectConfig.load(tmp_path).patterns_file)
    store["lesson.pdf"] = replace(store["lesson.pdf"], reviewed=True)
    patterns.save_store(ProjectConfig.load(tmp_path).patterns_file, store)
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 200, payload.decode("utf-8")
        records, _meta = read_staging(tmp_path / "staging" / "lesson.pdf.yaml")
        assert {record.expression for record in records} == {"あげる", "もらう"}
        assert not patterns.load_store(
            ProjectConfig.load(tmp_path).patterns_file
        )["lesson.pdf"].reviewed
        assert len(fake.calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_replacement_changed_after_consent_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checked replacement names one exact review, not whichever bytes happen
    to occupy that filename when the old page is submitted."""
    _stage(tmp_path, "table_exhaustive", filename="lesson.pdf")
    seed_prompts(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    target = tmp_path / "staging" / "lesson.pdf.yaml"
    try:
        form = _form(server, session)
        baseline = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations
        changed = target.read_bytes() + b"\n# newer human review\n"
        target.write_bytes(changed)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 409
        assert "review changed after this page was rendered" in html.unescape(
            payload.decode("utf-8")
        )
        assert target.read_bytes() == changed
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == baseline
    finally:
        server.shutdown()
        server.server_close()


def test_replacement_changed_during_the_paid_call_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The completion write is the compare-and-swap; a preflight-only check
    still loses an edit made while the provider is reading."""
    _stage(tmp_path, "table_exhaustive", filename="lesson.pdf")
    seed_prompts(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    target = tmp_path / "staging" / "lesson.pdf.yaml"
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        baseline_ids = set(
            operations.OperationJournal.load(
                ProjectConfig.load(tmp_path).operations_file
            ).operations
        )
        fields = dict(form.fields)
        fields["replace"] = "confirmed"
        post, responses = _post_in_thread(server, form, fields)
        assert entered.wait(1)
        changed = target.read_bytes() + b"\n# review saved while Claude read\n"
        target.write_bytes(changed)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        assert "review changed while the source was being read" in html.unescape(
            responses[0][2].decode("utf-8")
        )
        assert target.read_bytes() == changed
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        )
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        assert journal.operations[added.pop()].state == "result_captured"
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_pattern_review_changed_after_consent_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        baseline = operations.OperationJournal.load(config.operations_file).operations
        store = patterns.load_store(config.patterns_file)
        store["lesson.pdf"] = replace(store["lesson.pdf"], reviewed=True)
        patterns.save_store(config.patterns_file, store)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 409
        assert "grammar review changed after this page was rendered" in html.unescape(
            payload.decode("utf-8")
        )
        assert patterns.load_store(config.patterns_file)["lesson.pdf"].reviewed
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == baseline
    finally:
        server.shutdown()
        server.server_close()


def test_raw_pattern_entry_change_after_consent_is_stale_even_if_it_parses_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        wire = config.patterns_file.read_text(encoding="utf-8")
        changed = wire.replace('"reviewed": false', '"reviewed" : false', 1)
        assert changed != wire
        config.patterns_file.write_text(changed, encoding="utf-8")
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 409
        assert "grammar review changed after this page was rendered" in html.unescape(
            payload.decode("utf-8")
        )
        assert config.patterns_file.read_text(encoding="utf-8") == changed
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_replacement_targets_are_locked_in_deterministic_realpath_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    consent = session.consent("lesson.pdf")
    assert consent is not None and consent.target is not None
    import japanese_anki.application.extraction as extraction_application

    entered: list[Path] = []

    class ObservedLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> None:
            entered.append(self.path)

        def __exit__(self, *args: Any) -> None:
            del args

    monkeypatch.setattr(
        extraction_application,
        "exclusive_path_lock",
        lambda path: ObservedLock(Path(path)),
    )

    revision = extraction_application.extraction_replacement_revision(
        config, consent.target
    )

    expected = sorted(
        (consent.target.staging_path.resolve(), config.patterns_file.resolve()),
        key=str,
    )
    assert revision is not None
    assert entered == expected


def test_pattern_review_changed_during_the_paid_call_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    target = tmp_path / "staging" / "lesson.pdf.yaml"
    before = target.read_bytes()
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        baseline_ids = set(
            operations.OperationJournal.load(config.operations_file).operations
        )
        fields = dict(form.fields)
        fields["replace"] = "confirmed"
        post, responses = _post_in_thread(server, form, fields)
        assert entered.wait(1)
        store = patterns.load_store(config.patterns_file)
        store["lesson.pdf"] = replace(store["lesson.pdf"], reviewed=True)
        patterns.save_store(config.patterns_file, store)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        assert "grammar review changed while the source was being read" in html.unescape(
            responses[0][2].decode("utf-8")
        )
        assert target.read_bytes() == before
        assert patterns.load_store(config.patterns_file)["lesson.pdf"].reviewed
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(config.operations_file)
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        assert journal.operations[added.pop()].state == "result_captured"
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_final_staging_compare_and_swap_preserves_an_unlocked_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The under-lock equality check is not the final seam: a hand editor does
    not take janki's advisory lock, so the atomic replace must compare too."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    import japanese_anki.application.extraction as extraction_application

    real_write = extraction_application.write_staging_under_lock
    injected: list[bytes] = []

    def interfere(path: Path, *args: Any, **kwargs: Any):
        changed = Path(path).read_bytes() + b"\n# edit at the final write seam\n"
        Path(path).write_bytes(changed)
        injected.append(changed)
        return real_write(path, *args, **kwargs)

    try:
        form = _form(server, session)
        baseline_ids = set(
            operations.OperationJournal.load(config.operations_file).operations
        )
        monkeypatch.setattr(
            extraction_application, "write_staging_under_lock", interfere
        )
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 200
        assert "changed content" in html.unescape(payload.decode("utf-8"))
        assert injected
        assert (
            tmp_path / "staging" / "lesson.pdf.yaml"
        ).read_bytes() == injected[0]
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(config.operations_file)
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        assert journal.operations[added.pop()].state == "result_captured"
    finally:
        server.shutdown()
        server.server_close()


def test_final_pattern_compare_and_swap_preserves_an_unlocked_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    real_save = patterns.save_store_under_lock
    injected = False

    def interfere(path: Path, store: Any, **kwargs: Any) -> None:
        nonlocal injected
        live = patterns.load_store(path)
        live["other.pdf"] = patterns.PatternSet(
            source="other.pdf", title="Saved outside janki", reviewed=True
        )
        Path(path).write_text(patterns.render_store(live), encoding="utf-8")
        injected = True
        real_save(path, store, **kwargs)

    try:
        form = _form(server, session)
        baseline_ids = set(
            operations.OperationJournal.load(config.operations_file).operations
        )
        monkeypatch.setattr(patterns, "save_store_under_lock", interfere)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 200
        text = html.unescape(payload.decode("utf-8"))
        assert "changed content" in text
        assert str(config.staging_dir / "lesson.pdf.yaml") in text
        assert "embedded pattern set" in text
        assert "inspect janki operations" not in text
        assert injected
        assert patterns.load_store(config.patterns_file)["other.pdf"].reviewed
        assert len(fake.calls) == 1
        assert (tmp_path / "staging" / "lesson.pdf.yaml").is_file()
        journal = operations.OperationJournal.load(config.operations_file)
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        assert journal.operations[added.pop()].state == "committed"
    finally:
        server.shutdown()
        server.server_close()


def test_pattern_failure_still_names_staging_after_committed_entry_is_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pattern persistence happens after staging is committed. Forget may
    retire that committed journal entry before the pattern CAS reports its
    failure; recovery truth comes from the typed completion failure, not a
    later journal read that can no longer remember the commit."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    real_save = patterns.save_store_under_lock
    forgotten: list[str] = []
    baseline_ids = set(
        operations.OperationJournal.load(config.operations_file).operations
    )

    def interfere(path: Path, store: Any, **kwargs: Any) -> None:
        live = patterns.load_store(path)
        live["other.pdf"] = patterns.PatternSet(
            source="other.pdf", title="Saved outside janki", reviewed=True
        )
        Path(path).write_text(patterns.render_store(live), encoding="utf-8")
        journal = operations.OperationJournal.load(config.operations_file)
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        operation_id = added.pop()
        assert journal.operations[operation_id].state == "committed"
        assert journal.forget([operation_id]) == 1
        forgotten.append(operation_id)
        real_save(path, store, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(patterns, "save_store_under_lock", interfere)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 200
        text = html.unescape(payload.decode("utf-8"))
        assert "changed content" in text
        assert forgotten
        assert str(config.staging_dir / "lesson.pdf.yaml") in text
        assert "embedded pattern set" in text
        assert "no recovery answer remains" not in text
        assert (config.staging_dir / "lesson.pdf.yaml").is_file()
        assert set(
            operations.OperationJournal.load(config.operations_file).operations
        ) == baseline_ids
    finally:
        server.shutdown()
        server.server_close()


def test_first_read_pattern_write_is_bound_to_store_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    real_save = patterns.save_store_under_lock
    injected = False

    def interfere(path: Path, store: Any, **kwargs: Any) -> None:
        nonlocal injected
        other = {
            "other.pdf": patterns.PatternSet(
                source="other.pdf", title="Saved outside janki", reviewed=True
            )
        }
        Path(path).write_text(patterns.render_store(other), encoding="utf-8")
        injected = True
        real_save(path, store, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(patterns, "save_store_under_lock", interfere)

        status, _headers, payload = _post(server, form)

        assert status == 200
        assert "changed before replace" in html.unescape(payload.decode("utf-8"))
        assert injected
        store = patterns.load_store(config.patterns_file)
        assert set(store) == {"other.pdf"}
        assert store["other.pdf"].reviewed
        assert len(fake.calls) == 1
        assert (config.staging_dir / "lesson.pdf.yaml").is_file()
        journal = operations.OperationJournal.load(config.operations_file)
        assert [operation.state for operation in journal.operations.values()] == [
            "committed"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_first_read_pattern_write_is_bound_to_existing_store_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    original = {
        "other.pdf": patterns.PatternSet(
            source="other.pdf", title="Before the call", reviewed=False
        )
    }
    patterns.save_store(config.patterns_file, original)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    real_save = patterns.save_store_under_lock
    injected = False

    def interfere(path: Path, store: Any, **kwargs: Any) -> None:
        nonlocal injected
        live = patterns.load_store(path)
        live["other.pdf"] = replace(live["other.pdf"], reviewed=True)
        Path(path).write_text(patterns.render_store(live), encoding="utf-8")
        injected = True
        real_save(path, store, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(patterns, "save_store_under_lock", interfere)

        status, _headers, payload = _post(server, form)

        assert status == 200
        assert "changed content" in html.unescape(payload.decode("utf-8"))
        assert injected
        store = patterns.load_store(config.patterns_file)
        assert set(store) == {"other.pdf"}
        assert store["other.pdf"].reviewed
        assert len(fake.calls) == 1
        assert (config.staging_dir / "lesson.pdf.yaml").is_file()
        journal = operations.OperationJournal.load(config.operations_file)
        assert [operation.state for operation in journal.operations.values()] == [
            "committed"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_a_late_pattern_store_symlink_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    outside = tmp_path / "outside-patterns.json"
    try:
        form = _form(server, session)
        config.patterns_file.symlink_to(outside)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "symlinked pattern store" in payload.decode("utf-8")
        assert fake.calls == []
        assert config.patterns_file.is_symlink()
        assert not outside.exists()
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_a_broken_pattern_store_parent_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    base = ProjectConfig.load(tmp_path)
    parent = tmp_path / "pattern-parent"
    config = replace(base, patterns_file=parent / "patterns.json")
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        parent.symlink_to(tmp_path / "missing-pattern-parent", target_is_directory=True)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "pattern-store parent" in payload.decode("utf-8")
        assert "Nothing was sent" in payload.decode("utf-8")
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("output", ["staging", "patterns"])
def test_an_output_grandparent_symlink_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    _corpus(tmp_path)
    base = ProjectConfig.load(tmp_path)
    output_root = tmp_path / f"{output}-root"
    if output == "staging":
        config = replace(base, staging_dir=output_root / "nested" / "staging")
    else:
        config = replace(
            base, patterns_file=output_root / "nested" / "patterns.json"
        )
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    outside = tmp_path / f"outside-{output}"
    try:
        form = _form(server, session)
        (outside / "nested").mkdir(parents=True)
        if output == "staging":
            (outside / "nested" / "staging").mkdir()
        output_root.symlink_to(outside, target_is_directory=True)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "safely open target directory" in payload.decode("utf-8")
        assert "Nothing was sent" in payload.decode("utf-8")
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_a_pattern_parent_swap_during_replacement_preserves_the_card_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    base = ProjectConfig.load(tmp_path)
    parent = tmp_path / "pattern-parent"
    config = replace(base, patterns_file=parent / "patterns.json")
    target = config.staging_dir / "lesson.pdf.yaml"
    original_review = target.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    baseline_ids = set(
        operations.OperationJournal.load(config.operations_file).operations
    )
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"
        post, responses = _post_in_thread(server, form, fields)
        assert entered.wait(1)
        parent.rmdir()
        parent.symlink_to(tmp_path / "missing-pattern-parent", target_is_directory=True)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        assert "pattern-store parent" in responses[0][2].decode("utf-8")
        assert target.read_bytes() == original_review
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(config.operations_file)
        added = set(journal.operations) - baseline_ids
        assert len(added) == 1
        assert journal.operations[added.pop()].state == "result_captured"
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_paid_action_token_is_bound_to_its_rendered_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        assert form.fields["dispatch"] != form.fields["csrf"]
        fields = dict(form.fields)
        fields["model"] = "claude-some-other-model"

        status, _headers, _payload = _post(server, form, fields)

        assert status == 409
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_paid_action_token_is_bound_to_its_rendered_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path, "lesson.pdf", "other.pdf")
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        lesson = session.source_path("lesson.pdf")
        assert lesson is not None
        # Isolate the action-name binding from the later fresh-plan checks:
        # even a route alias resolving to the identical source is not the
        # named paid action the page issued.
        monkeypatch.setattr(
            WorkbenchSession,
            "source_path",
            lambda _session, _name: lesson,
        )
        form.action = form.action.rsplit("/", 1)[0] + "/other.pdf"

        status, _headers, _payload = _post(server, form)

        assert status == 409
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_paid_action_token_is_bound_to_its_rendered_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["mode"] = "prose"

        status, _headers, _payload = _post(server, form, fields)

        assert status == 409
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_paid_action_token_is_bound_to_its_rendered_request_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["request_fingerprint"] = "f" * 64
        assert fields["request_fingerprint"] != form.fields["request_fingerprint"]

        status, _headers, _payload = _post(server, form, fields)

        assert status == 409
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_paid_action_token_is_bound_to_whether_replacement_was_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["replacement"] = "1"

        status, _headers, _payload = _post(server, form, fields)

        assert status == 409
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_paid_post_requires_the_clicked_submit_button(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capability rides on the button, not a hidden field an implicit or
    unattended form submission would carry."""
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        assert len(form.submitters) == 2
        guard, paid = form.submitters
        assert "disabled" in guard
        assert guard.get("class") == "implicit-submit-guard"
        assert "name" not in guard and "value" not in guard
        assert paid.get("name") == "dispatch" and paid.get("value")
        assert "disabled" not in paid
        fields = dict(form.fields)
        del fields["dispatch"]

        status, _headers, _payload = _post(server, form, fields)

        assert status == 400
        assert fake.calls == []
        assert operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        ).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_unrelated_grammar_change_after_consent_does_not_stale_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    store = patterns.load_store(config.patterns_file)
    store["other.pdf"] = patterns.PatternSet(source="other.pdf", title="Other")
    patterns.save_store(config.patterns_file, store)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        store = patterns.load_store(config.patterns_file)
        store["other.pdf"] = replace(store["other.pdf"], reviewed=True)
        patterns.save_store(config.patterns_file, store)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"

        status, _headers, payload = _post(server, form, fields)

        assert status == 200, payload.decode("utf-8")
        assert patterns.load_store(config.patterns_file)["other.pdf"].reviewed
        assert len(fake.calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_unrelated_grammar_change_during_call_survives_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    store = patterns.load_store(config.patterns_file)
    store["other.pdf"] = patterns.PatternSet(source="other.pdf", title="Other")
    patterns.save_store(config.patterns_file, store)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        fields = dict(form.fields)
        fields["replace"] = "confirmed"
        post, responses = _post_in_thread(server, form, fields)
        assert entered.wait(1)
        store = patterns.load_store(config.patterns_file)
        store["other.pdf"] = replace(store["other.pdf"], reviewed=True)
        patterns.save_store(config.patterns_file, store)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        assert "Review proposed cards" in responses[0][2].decode("utf-8")
        assert patterns.load_store(config.patterns_file)["other.pdf"].reviewed
        assert len(fake.calls) == 1
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_a_forged_csrf_cannot_spend_or_consume_the_paid_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        forged = dict(form.fields)
        forged["csrf"] = "not-this-session"

        refused = _post(server, form, forged)
        accepted = _post(server, form)

        assert refused[0] == 403
        assert accepted[0] == 200, accepted[2].decode("utf-8")
        assert len(fake.calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_a_non_ascii_csrf_is_a_403_without_consuming_the_paid_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        forged = dict(form.fields)
        forged["csrf"] = "あ"

        refused = _post(server, form, forged)
        accepted = _post(server, form)

        assert refused[0] == 403
        assert accepted[0] == 200, accepted[2].decode("utf-8")
        assert len(fake.calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_paid_action_token_cannot_be_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        first = _post(server, form)
        assert first[0] == 200, first[2].decode("utf-8")
        # Neutralize the ordinary staging collision. Without this, the second
        # POST could reach the provider and the wrong guard would pass the test.
        (tmp_path / "staging" / "lesson.pdf.yaml").unlink()

        second = _post(server, form)

        assert second[0] == 409
        assert len(fake.calls) == 1
        assert len(
            operations.OperationJournal.load(
                ProjectConfig.load(tmp_path).operations_file
            ).operations
        ) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_authorize_race_is_a_409_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        config = ProjectConfig.load(tmp_path)
        journal = operations.OperationJournal.load(config.operations_file)
        journal.authorize(
            "other", kind="extract", source_file="other.pdf",
            source_sha256="a" * 64, request_fp="b" * 64,
            model="claude-opus-5",
        )
        journal.advance("other", "dispatching")

        real_authorize = operations.OperationJournal.authorize
        calls: list[str] = []

        def observed_authorize(self: Any, operation_id: str, **kwargs: Any):
            calls.append(operation_id)
            return real_authorize(self, operation_id, **kwargs)

        monkeypatch.setattr(operations.OperationJournal, "authorize", observed_authorize)
        status, _headers, payload = _post(server, form)

        assert status == 409
        assert calls, "the POST trusted a display instead of reaching authorize"
        assert "will not start another paid call" in payload.decode("utf-8")
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_journal_write_failure_at_authorize_is_a_409_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)

    writes = 0

    def unwritable(*args: Any, **kwargs: Any) -> None:
        nonlocal writes
        del args, kwargs
        writes += 1
        raise DataError("Could not write the operation journal")

    try:
        form = _form(server, session)
        monkeypatch.setattr(operations, "atomic_write_text_bound", unwritable)

        status, _headers, payload = _post(server, form)

        assert status == 409
        text = payload.decode("utf-8")
        assert "Could not write the operation journal" in text
        assert "Nothing was sent" in text
        assert writes == 1
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_an_unsafe_pending_store_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        pending = config.operations_file.parent / operations.PENDING_DIR
        pending.parent.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            pending.write_text("not a directory\n", encoding="utf-8")
        else:
            outside = tmp_path / "outside-pending"
            outside.mkdir()
            pending.symlink_to(outside, target_is_directory=True)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "pending answer store" in payload.decode("utf-8")
        assert "Nothing was sent" in payload.decode("utf-8")
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_an_unwritable_pending_store_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    real_open = janki_io.os.open
    probes = 0

    def refuse_probe(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal probes
        if isinstance(path, str) and path.startswith(".janki-write-probe."):
            probes += 1
            raise PermissionError("the pending store is not writable")
        return real_open(path, *args, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(janki_io.os, "open", refuse_probe)

        status, _headers, payload = _post(server, form)

        assert status == 409
        text = payload.decode("utf-8")
        assert "pending answer store" in text
        assert "not writable" in text
        assert "Nothing was sent" in text
        assert probes == 1
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


def test_a_pending_store_without_atomic_publication_is_refused_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    links = 0

    def refuse_link(*_args: Any, **_kwargs: Any) -> None:
        nonlocal links
        links += 1
        raise PermissionError("hard-link publication is unavailable")

    try:
        form = _form(server, session)
        monkeypatch.setattr(janki_io.os, "link", refuse_link)

        status, _headers, payload = _post(server, form)

        assert status == 409
        text = payload.decode("utf-8")
        assert "pending answer store" in text
        assert "hard-link publication is unavailable" in text
        assert "Nothing was sent" in text
        assert links == 1
        assert fake.calls == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("swap", ["parent", "target"])
def test_a_pending_store_swap_during_the_call_never_writes_outside_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap: str
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    pending = config.operations_file.parent / operations.PENDING_DIR
    outside = tmp_path / ("outside-pending" if swap == "parent" else "outside.json")
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        post, responses = _post_in_thread(server, form, dict(form.fields))
        assert entered.wait(1)
        [operation_id] = operations.OperationJournal.load(
            config.operations_file
        ).operations
        if swap == "parent":
            held = tmp_path / "held-pending"
            pending.rename(held)
            outside.mkdir()
            pending.symlink_to(outside, target_is_directory=True)
        else:
            outside.write_bytes(b"keep this")
            (pending / f"{operation_id}.json").symlink_to(outside)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        text = html.unescape(responses[0][2].decode("utf-8"))
        assert "may already have been billed" in text
        assert len(fake.calls) == 1
        if swap == "parent":
            assert not (outside / f"{operation_id}.json").exists()
        else:
            assert outside.read_bytes() == b"keep this"
        assert not (config.staging_dir / "lesson.pdf.yaml").exists()
        journal = operations.OperationJournal.load(config.operations_file)
        assert journal.operations[operation_id].state == "outcome_unknown"
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_a_late_journal_symlink_is_a_409_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    outside = tmp_path / "outside-operations.json"
    try:
        form = _form(server, session)
        config.operations_file.parent.mkdir(parents=True, exist_ok=True)
        config.operations_file.symlink_to(outside)

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "symlinked operation journal" in payload.decode("utf-8")
        assert fake.calls == []
        assert config.operations_file.is_symlink()
        assert not outside.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_journal_creation_is_bound_to_final_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    sentinel = b'{"version": 1, "operations": {}}\n'
    injected = False

    def interfere(path: Path, text: str, **kwargs: Any) -> None:
        nonlocal injected
        if not injected:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(sentinel)
            injected = True
        atomic_write_text_bound(path, text, **kwargs)

    try:
        form = _form(server, session)
        monkeypatch.setattr(
            operations,
            "atomic_write_text_bound",
            interfere,
            raising=False,
        )

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "changed before replace" in payload.decode("utf-8")
        assert injected
        assert config.operations_file.read_bytes() == sentinel
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_a_late_non_directory_journal_parent_is_a_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    base = ProjectConfig.load(tmp_path)
    parent = tmp_path / "journal-parent"
    config = replace(base, operations_file=parent / "operations.json")
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)
        parent.write_text("not a directory\n", encoding="utf-8")

        status, _headers, payload = _post(server, form)

        assert status == 409
        assert "operation journal parent" in payload.decode("utf-8")
        assert "Nothing was sent" in payload.decode("utf-8")
        assert fake.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_journal_failure_while_settling_a_provider_error_stays_on_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)

    def provider_failure(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("provider connection failed")

    monkeypatch.setattr(
        "japanese_anki.extract.claude_client.parse_call", provider_failure
    )
    real_write = operations.atomic_write_text_bound
    writes = 0

    def fail_third_write(*args: Any, **kwargs: Any) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise DataError("journal became read-only after dispatch")
        real_write(*args, **kwargs)

    monkeypatch.setattr(operations, "atomic_write_text_bound", fail_third_write)
    try:
        form = _form(server, session)

        status, _headers, payload = _post(server, form)

        assert status == 200
        text = html.unescape(payload.decode("utf-8"))
        assert "provider connection failed" in text
        assert "could not settle the operation journal" in text
        assert "journal became read-only after dispatch" in text
        assert text.endswith("</main></body></html>")
        assert writes == 3
        assert not (tmp_path / "staging" / "lesson.pdf.yaml").exists()
        journal = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        )
        assert [operation.state for operation in journal.operations.values()] == [
            "dispatching"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_provider_errors_redact_environment_credentials_from_page_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    secret = "sk-ant-upstream-echo-must-disappear"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    def provider_failure(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise DataError(f"authorization header contained {secret}")

    monkeypatch.setattr(
        "japanese_anki.extract.claude_client.parse_call", provider_failure
    )
    try:
        form = _form(server, session)
        status, _headers, payload = _post(server, form)

        assert status == 200
        assert secret.encode() not in payload
        assert b"[redacted ANTHROPIC_API_KEY]" in payload
        journal_bytes = session.config.operations_file.read_bytes()
        assert secret.encode() not in journal_bytes
        assert b"[redacted ANTHROPIC_API_KEY]" in journal_bytes
    finally:
        server.shutdown()
        server.server_close()


def test_a_blank_provider_exception_still_finishes_the_failure_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)

    def provider_failure(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError()

    monkeypatch.setattr(
        "japanese_anki.extract.claude_client.parse_call", provider_failure
    )
    try:
        form = _form(server, session)
        status, _headers, payload = _post(server, form)

        text = payload.decode("utf-8")
        assert status == 200
        assert "RuntimeError failed without details." in text
        for heading in ("What happened", "What changed", "Money", "What to do next"):
            assert heading in text
        assert text.endswith("</main></body></html>")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("failure_kind", "expected_happened"),
    [
        ("staging-saved", "Proposals remain saved at"),
        ("operation", "OperationError failed without details."),
        ("janki", "DataError failed without details."),
    ],
)
def test_blank_completion_exceptions_still_finish_the_paid_answer_failure_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_happened: str,
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    if failure_kind == "staging-saved":
        failure: Exception = ExtractionCompletionError(
            DataError(),
            tmp_path / "staging" / "lesson.pdf.yaml",
        )
    elif failure_kind == "operation":
        failure = operations.OperationError()
    else:
        failure = DataError()

    def fail_completion(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    import japanese_anki.application.extraction as extraction_application

    monkeypatch.setattr(
        extraction_application, "complete_extraction", fail_completion
    )
    try:
        form = _form(server, session)
        status, _headers, payload = _post(server, form)
    finally:
        server.shutdown()
        server.server_close()

    text = payload.decode("utf-8")
    assert status == 200
    assert expected_happened in text
    for heading in ("What happened", "What changed", "Money", "What to do next"):
        assert heading in text
    assert "provider call completed and may have been billed" in text
    assert text.endswith("</main></body></html>")
    assert len(fake.calls) == 1
    if failure_kind == "operation":
        assert "Saving was refused" in text


def test_staging_directory_failure_after_dispatch_has_a_complete_recovery_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        post, responses = _post_in_thread(server, form, dict(form.fields))
        assert entered.wait(1)
        if config.staging_dir.exists():
            config.staging_dir.rmdir()
        config.staging_dir.write_text("not a directory\n", encoding="utf-8")
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        text = html.unescape(responses[0][2].decode("utf-8"))
        assert "Could not safely open target directory" in text
        assert "answer arrived" in text
        assert text.endswith("</main></body></html>")
        assert len(fake.calls) == 1
        journal = operations.OperationJournal.load(config.operations_file)
        assert [operation.state for operation in journal.operations.values()] == [
            "result_captured"
        ]
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_force_forget_winning_before_completion_is_reported_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A person may discard a captured reply while its request thread is
    still validating. The eventual completion refusal must not claim an
    operation or recovery answer that the winning decision removed."""
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    captured = threading.Event()
    allow_validation = threading.Event()
    fake = _FakeCall(captured=captured, allow_validation=allow_validation)
    _install_fake(monkeypatch, fake)
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        post, responses = _post_in_thread(server, form, dict(form.fields))
        assert captured.wait(1)
        journal = operations.OperationJournal.load(config.operations_file)
        assert len(journal.operations) == 1
        operation_id = next(iter(journal.operations))
        assert journal.operations[operation_id].state == "result_captured"
        assert journal.forget([operation_id], force=True) == 1

        allow_validation.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        text = html.unescape(responses[0][2].decode("utf-8"))
        assert "operation was already forgotten" in text
        assert "no recovery answer remains" in text
        assert "Saving was refused. The answer remains in janki operations" not in text
        assert not (config.staging_dir / "lesson.pdf.yaml").exists()
        assert not operations.OperationJournal.load(
            config.operations_file
        ).operations
    finally:
        allow_validation.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def test_staging_ancestor_swap_after_dispatch_never_creates_outside_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    base = ProjectConfig.load(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    config = replace(
        base,
        staging_dir=data / "staging",
        operations_file=tmp_path / "operations.json",
    )
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    fake = _FakeCall(entered=entered, allow_answer=allow_answer)
    _install_fake(monkeypatch, fake)
    held = tmp_path / "held-data"
    outside = tmp_path / "outside"
    post: threading.Thread | None = None
    try:
        form = _form(server, session)
        post, responses = _post_in_thread(server, form, dict(form.fields))
        assert entered.wait(1)
        data.rename(held)
        outside.mkdir()
        data.symlink_to(outside, target_is_directory=True)
        allow_answer.set()
        post.join(3)

        assert not post.is_alive()
        assert responses and responses[0][0] == 200
        text = html.unescape(responses[0][2].decode("utf-8"))
        assert "answer arrived" in text
        assert "safely open target directory" in text
        assert len(fake.calls) == 1
        assert not (outside / "staging").exists()
        journal = operations.OperationJournal.load(config.operations_file)
        assert [operation.state for operation in journal.operations.values()] == [
            "result_captured"
        ]
    finally:
        allow_answer.set()
        if post is not None:
            post.join(3)
        server.shutdown()
        server.server_close()


def _read_through(response: http.client.HTTPResponse, marker: bytes) -> bytes:
    found = bytearray()
    while marker not in found:
        byte = response.read(1)
        if not byte:
            break
        found.extend(byte)
    assert marker in found, found.decode("utf-8", errors="replace")
    return bytes(found)


def test_progress_names_only_the_four_truthful_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    entered = threading.Event()
    allow_answer = threading.Event()
    captured = threading.Event()
    allow_validation = threading.Event()
    fake = _FakeCall(
        entered=entered,
        allow_answer=allow_answer,
        captured=captured,
        allow_validation=allow_validation,
    )
    _install_fake(monkeypatch, fake)
    connection: http.client.HTTPConnection | None = None
    try:
        form = _form(server, session)
        body = urlencode(form.fields).encode("utf-8")
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=3
        )
        connection.request(
            "POST",
            form.action,
            body=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert entered.wait(1)

        before_answer = _read_through(response, b"Reading the source")
        assert b"Checking the answer" not in before_answer
        allow_answer.set()
        assert captured.wait(1)
        through_capture = before_answer + _read_through(
            response, b"Checking the answer&#x27;s shape"
        )
        assert b"Saving proposals" not in through_capture
        allow_validation.set()
        rendered = html.unescape((through_capture + response.read()).decode("utf-8"))

        labels = (
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        )
        assert all(rendered.count(label) == 1 for label in labels)
        assert [rendered.index(label) for label in labels] == sorted(
            rendered.index(label) for label in labels
        )
        assert "%" not in rendered
        assert "aria-valuenow" not in rendered
        assert "<progress" not in rendered
    finally:
        allow_answer.set()
        allow_validation.set()
        if connection is not None:
            connection.close()
        server.shutdown()
        server.server_close()


def test_browser_success_uses_complete_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    import japanese_anki.application.extraction as extraction_application
    from japanese_anki.application import complete_extraction as real_complete

    calls: list[str] = []

    def observed_complete(*args: Any, **kwargs: Any):
        calls.append(str(kwargs["operation_id"]))
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(
        extraction_application, "complete_extraction", observed_complete
    )
    try:
        form = _form(server, session)

        status, headers, payload = _post(server, form)

        assert status == 200, payload.decode("utf-8")
        assert headers["cache-control"] == "no-store"
        assert calls and len(calls) == 1
        records, _meta = read_staging(tmp_path / "staging" / "lesson.pdf.yaml")
        assert len(records) == 2
        journal = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        )
        assert [operation.state for operation in journal.operations.values()] == [
            "committed"
        ]
        assert "Review proposed cards" in payload.decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()


def test_a_disconnected_progress_writer_does_not_abandon_paid_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once authority is journaled, losing the tab only disables rendering;
    the handler still captures and commits the paid answer."""
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall()
    _install_fake(monkeypatch, fake)
    from japanese_anki.workbench import server as server_module

    class DisconnectedWriter:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.attempts = 0

        def write(self, payload: bytes) -> Any:
            self.attempts += 1
            if self.attempts >= 3:
                raise BrokenPipeError("the browser tab closed")
            return self.wrapped.write(payload)

        def flush(self) -> Any:
            return self.wrapped.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

    real_start = server_module._ExtractionProgress.start
    broken: list[DisconnectedWriter] = []

    def disconnect_after_headers(progress: Any) -> None:
        real_start(progress)
        writer = DisconnectedWriter(progress.handler.wfile)
        broken.append(writer)
        progress.handler.wfile = writer

    monkeypatch.setattr(
        server_module._ExtractionProgress, "start", disconnect_after_headers
    )
    try:
        form = _form(server, session)

        status, _headers, _payload = _post(server, form)

        assert status == 200
        assert broken and broken[0].attempts == 3
        assert len(fake.calls) == 1
        records, _meta = read_staging(tmp_path / "staging" / "lesson.pdf.yaml")
        assert len(records) == 2
        journal = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        )
        assert [operation.state for operation in journal.operations.values()] == [
            "committed"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_a_truncated_answer_plainly_says_nothing_was_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    server, _thread = _running(session)
    fake = _FakeCall(stop_reason="max_tokens")
    _install_fake(monkeypatch, fake)
    try:
        form = _form(server, session)

        status, _headers, payload = _post(server, form)

        assert status == 200
        text = html.unescape(payload.decode("utf-8"))
        assert "ran out of room" in text
        assert "Nothing was staged" in text
        assert "janki operations --show-reply" in text
        assert ".pending" not in text
        assert not (tmp_path / "staging" / "lesson.pdf.yaml").exists()
        journal = operations.OperationJournal.load(
            ProjectConfig.load(tmp_path).operations_file
        )
        assert [operation.state for operation in journal.operations.values()] == [
            "result_captured"
        ]
    finally:
        server.shutdown()
        server.server_close()
