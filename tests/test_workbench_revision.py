"""Main Workbench review and one-confirmation finish for deck revisions."""

from __future__ import annotations

import hashlib
import html
import http.client
import json
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
import yaml
from test_application_revision_finish import (
    _fixture,
    _providers,
    _RealtimeTransport,
)

from japanese_anki.application import audio as audio_application
from japanese_anki.application import revision_finish
from japanese_anki.workbench.render import (
    render_revision,
    render_revision_finish_status,
)
from japanese_anki.workbench.server import WorkbenchSession, make_server


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
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    headers = {"Host": server.expected_host}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, payload


def _hidden_fields(page: bytes) -> dict[str, str]:
    return {
        name: html.unescape(value)
        for name, value in re.findall(
            r'<input type=hidden name="?([^" ]+)"? value="([^"]*)">',
            page.decode("utf-8"),
        )
    }


def _review_url(session: WorkbenchSession, staging: Path) -> str:
    return f"/{session.token}/revisions/{staging.name}"


def _resolved_test_providers(
    config: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str = "test-key",
) -> tuple[_RealtimeTransport, Any, Any]:
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key=api_key,
        transport=transport,
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda _config, _chosen: words,
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda _config, _chosen, _words: sentences,
    )
    return transport, words, sentences


def test_dashboard_lists_only_safe_service_validated_revision_plans(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    linked = config.staging_dir / "revise-linked.json"
    linked.symlink_to(staging)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(server, "GET", f"/{session.token}/")
    finally:
        server.shutdown()
        server.server_close()

    page = payload.decode("utf-8")
    assert status == 200
    assert "Deck revision proposals" in page
    assert "decks/potential.yaml" in page
    assert f"/{session.token}/revisions/{staging.name}" in page
    assert "1 selected card(s)" in page
    assert "revise-linked.json is not a direct regular file and was refused" in page
    assert f"/{session.token}/revisions/{linked.name}" not in page


def test_revision_review_renders_one_exact_apply_and_finish_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    _transport, words, sentences = _resolved_test_providers(config, monkeypatch)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(server, "GET", _review_url(session, staging))
    finally:
        server.shutdown()
        server.server_close()

    page = payload.decode("utf-8")
    assert status == 200
    assert "Old note." in page
    assert "New note." in page
    assert "I can speak." in page
    assert "I am able to speak." in page
    assert "Can you speak?" in page
    assert "Can you speak Japanese?" in page
    assert "I can read." not in page
    assert "openai-realtime" in page
    assert "gpt-realtime-1.5" in page
    assert "paid-network" in page
    assert f"Total clips</dt><dd>{plan.audio.example_counts.total}" in page
    assert (
        f"Provider calls required</dt><dd>{plan.audio.example_counts.provider_required}"
    ) in page
    assert str(plan.build.output_path) in page
    assert f"cards:</b> {plan.build.card_count}" in page
    assert page.count("<form ") == 1
    assert page.count("Apply and finish") == 1
    assert "apply-revision" not in page
    assert "deck-audio" not in page
    assert "deck-build" not in page
    assert "%" not in page
    for phase in (
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    ):
        assert phase in page
    assert _hidden_fields(payload) == {
        "action": "apply-and-finish",
        "csrf": session.csrf_token,
        "plan_fingerprint": plan.fingerprint,
    }


def test_revision_and_receipt_views_escape_every_new_displayed_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    _transport, words, sentences = _resolved_test_providers(config, monkeypatch)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    provider = plan.audio.example_provider
    assert provider is not None
    malicious = '<img src=x onerror="alert(1)">'
    rendered_plan = replace(
        plan,
        audio=replace(
            plan.audio,
            example_provider=replace(
                provider,
                name=malicious,
                settings={"model": malicious},
            ),
        ),
    )
    review = render_revision(rendered_plan, token="token", csrf="csrf")
    result = replace(
        _result_for(plan, "revision_applied"),
        deck_path=Path(malicious),
        output_path=Path(malicious),
    )
    receipt = render_revision_finish_status(result, token="token", csrf="csrf")

    assert malicious not in review
    assert malicious not in receipt
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in review
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in receipt


def test_apply_and_finish_requires_csrf_and_rejects_duplicate_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _resolved_test_providers(config, monkeypatch)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _review_url(session, staging))
        fields = _hidden_fields(page)
        forged = dict(fields, csrf="forged")
        csrf_status, _headers, _payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/finish",
            body=urlencode(forged).encode(),
        )
        duplicated = [
            ("action", "apply-and-finish"),
            ("csrf", session.csrf_token),
            ("csrf", "forged"),
            ("plan_fingerprint", fields["plan_fingerprint"]),
        ]
        duplicate_status, _headers, duplicate_payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/finish",
            body=urlencode(duplicated).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert csrf_status == 403
    assert duplicate_status == 400
    assert b"repeats a field" in duplicate_payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_apply_and_finish_replans_at_click_and_refuses_any_plan_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport, _words, _sentences = _resolved_test_providers(config, monkeypatch)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _review_url(session, staging))
        fields = _hidden_fields(page)
        staging.write_text(
            staging.read_text(encoding="utf-8").replace(
                "New note.", "Proposal changed after rendering."
            ),
            encoding="utf-8",
        )
        status, _headers, payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/finish",
            body=urlencode(fields).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"apply-and-finish plan changed after the page was rendered" in payload
    assert b"This action stopped before any repository write" in payload
    assert deck.read_bytes() == before
    assert staging.exists()
    assert transport.calls == []


def test_one_confirmation_applies_voices_builds_and_streams_named_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport, _words, _sentences = _resolved_test_providers(config, monkeypatch)
    unselected_before = yaml.safe_load(deck.read_text(encoding="utf-8"))["deck"]["drill_examples"][
        "word:読む:よむ"
    ]
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _review_url(session, staging))
        status, _headers, payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/finish",
            body=urlencode(_hidden_fields(page)).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    rendered = payload.decode("utf-8")
    assert status == 200
    phases = (
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    )
    assert [rendered.index(phase) for phase in phases] == sorted(
        rendered.index(phase) for phase in phases
    )
    assert "revision, example audio, and Anki package are complete" in rendered
    assert "Resume this exact finish" not in rendered
    finish_records = list((config.staging_dir / "done" / "revisions").glob("finish-*.json"))
    assert len(transport.calls) == 2
    assert "New note." in deck.read_text(encoding="utf-8")
    assert "audio/janki-" in deck.read_text(encoding="utf-8")
    assert (
        yaml.safe_load(deck.read_text(encoding="utf-8"))["deck"]["drill_examples"]["word:読む:よむ"]
        == unselected_before
    )
    assert "読め" not in json.dumps(transport.calls, ensure_ascii=False)
    assert not staging.exists()
    assert (config.staging_dir / "done" / "revisions" / staging.name).exists()
    assert len(list((config.media_dir / "audio").glob("*.wav"))) == 2
    assert (config.dist_dir / "potential.apkg").read_bytes().startswith(b"PK")
    assert len(finish_records) == 1


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_completed_finish_only_returns_to_dashboard_when_package_needs_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport, _words, _sentences = _resolved_test_providers(config, monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        review_url = _review_url(session, staging)
        _status, _headers, review = _request(server, "GET", review_url)
        assert _status == 200, review.decode("utf-8")
        finish_status, _headers, _payload = _request(
            server,
            "POST",
            review_url + "/finish",
            body=urlencode(_hidden_fields(review)).encode(),
        )
        assert finish_status == 200, _payload.decode("utf-8")
        finish_records = list((config.staging_dir / "done" / "revisions").glob("finish-*.json"))
        assert len(finish_records) == 1
        receipt_id = finish_records[0].stem.removeprefix("finish-")
        output = config.dist_dir / "potential.apkg"

        current_status, _headers, current_dashboard = _request(server, "GET", f"/{session.token}/")
        if damage == "missing":
            output.unlink()
        else:
            output.write_bytes(b"changed package")
        damaged_status, _headers, damaged_dashboard = _request(server, "GET", f"/{session.token}/")
        receipt_url = f"/{session.token}/revision-finishes/{receipt_id}"
        page_status, _headers, receipt_page = _request(server, "GET", receipt_url)
        resume_fields = _hidden_fields(receipt_page)
        resume_status, _headers, resume_page = _request(
            server,
            "POST",
            receipt_url + "/resume",
            body=urlencode(resume_fields).encode(),
        )
        repaired_status, _headers, repaired_dashboard = _request(
            server, "GET", f"/{session.token}/"
        )
    finally:
        server.shutdown()
        server.server_close()

    assert current_status == 200
    assert b"Revision finishes" not in current_dashboard
    assert damaged_status == 200
    assert b"Revision finishes" in damaged_dashboard
    assert b"Package needs exact verification or rebuild" in damaged_dashboard
    assert receipt_url.encode() in damaged_dashboard
    assert page_status == 200
    assert b"Verify or rebuild this exact package" in receipt_page
    assert resume_fields == {
        "action": "resume-apply-and-finish",
        "csrf": session.csrf_token,
        "receipt_id": receipt_id,
    }
    assert resume_status == 200
    assert b"Building Anki package" in resume_page
    assert b"Saving finish receipt" in resume_page
    assert b"Creating example audio" not in resume_page
    assert output.read_bytes().startswith(b"PK")
    assert len(transport.calls) == 2
    assert repaired_status == 200
    assert b"Revision finishes" not in repaired_dashboard


def _result_for(
    plan: revision_finish.RevisionFinishPlan,
    state: str,
) -> revision_finish.RevisionFinishResult:
    provider = plan.audio.example_provider
    assert provider is not None
    return revision_finish.RevisionFinishResult(
        receipt_id=plan.fingerprint,
        state=state,  # type: ignore[arg-type]
        record_path=plan.record_path,
        deck_path=plan.revision.deck_path,
        output_path=plan.build.output_path,
        example_provider_access=provider.access,
        max_provider_calls=plan.provider_required_count,
        package_sha256="a" * 64 if state == "complete" else None,
        card_count=plan.build.card_count if state == "complete" else None,
    )


def test_completed_receipt_view_offers_exact_package_rebuild_only(
    tmp_path: Path,
) -> None:
    receipt_id = "a" * 64
    result = revision_finish.RevisionFinishResult(
        receipt_id=receipt_id,
        state="complete",
        record_path=tmp_path / f"finish-{receipt_id}.json",
        deck_path=tmp_path / "deck.yaml",
        output_path=tmp_path / "deck.apkg",
        example_provider_access="paid-network",
        max_provider_calls=2,
        package_sha256="b" * 64,
        card_count=3,
    )

    page = render_revision_finish_status(result, token="token", csrf="csrf")

    assert page.count("<form ") == 1
    assert "Verify or rebuild this exact package" in page
    assert "It will not repeat revision apply or example audio" in page
    assert _hidden_fields(page.encode()) == {
        "action": "resume-apply-and-finish",
        "csrf": "csrf",
        "receipt_id": receipt_id,
    }
    recovery_page = render_revision_finish_status(
        result,
        token="token",
        csrf="csrf",
        package_needs_rebuild=True,
    )
    assert "disposable Anki package is missing, changed" in recovery_page


def test_completed_finish_discovery_hides_current_and_surfaces_package_damage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, _staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    receipt_id = "a" * 64
    record_path = config.staging_dir / "done" / "revisions" / f"finish-{receipt_id}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{}\n", encoding="utf-8")
    output = config.dist_dir / "potential.apkg"
    output.parent.mkdir(parents=True, exist_ok=True)
    package = b"current package"
    output.write_bytes(package)
    result = revision_finish.RevisionFinishResult(
        receipt_id=receipt_id,
        state="complete",
        record_path=record_path,
        deck_path=deck,
        output_path=output,
        example_provider_access="paid-network",
        max_provider_calls=2,
        package_sha256=hashlib.sha256(package).hexdigest(),
        card_count=1,
    )
    monkeypatch.setattr(
        revision_finish,
        "inspect_revision_finish",
        lambda _config, _receipt_id: result,
    )

    current, current_warnings = session.revision_finishes()
    output.unlink()
    missing, missing_warnings = session.revision_finishes()
    output.write_bytes(b"changed package")
    changed, changed_warnings = session.revision_finishes()

    assert current == []
    assert current_warnings == []
    assert missing == [result]
    assert missing_warnings == []
    assert changed == [result]
    assert changed_warnings == []


def test_partial_finish_surfaces_refreshable_receipt_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    _resolved_test_providers(config, monkeypatch)
    session = WorkbenchSession.open(config)
    planned = session.revision_finish_plan(staging.name)
    assert planned is not None
    partial = _result_for(planned, "revision_applied")
    complete = _result_for(planned, "complete")
    resumed: list[str] = []
    planned.record_path.parent.mkdir(parents=True, exist_ok=True)
    planned.record_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        revision_finish,
        "execute_revision_finish",
        lambda _config, _plan, *, progress: partial,
    )
    monkeypatch.setattr(
        revision_finish,
        "inspect_revision_finish",
        lambda _config, receipt_id: partial if receipt_id == planned.fingerprint else None,
    )

    def resume(
        _config: Any,
        receipt_id: str,
        *,
        progress: Any,
    ) -> revision_finish.RevisionFinishResult:
        resumed.append(receipt_id)
        progress("Building Anki package")
        progress("Saving finish receipt")
        return complete

    monkeypatch.setattr(revision_finish, "resume_revision_finish", resume)
    server, _thread = _running(session)
    try:
        review_url = _review_url(session, staging)
        _status, _headers, page = _request(server, "GET", review_url)
        first_status, _headers, first_payload = _request(
            server,
            "POST",
            review_url + "/finish",
            body=urlencode(_hidden_fields(page)).encode(),
        )
        dashboard_status, _headers, dashboard = _request(server, "GET", f"/{session.token}/")
        receipt_url = f"/{session.token}/revision-finishes/{planned.fingerprint}"
        get_status, _headers, receipt_page = _request(server, "GET", receipt_url)
        resume_fields = _hidden_fields(receipt_page)
        resume_status, _headers, resume_payload = _request(
            server,
            "POST",
            receipt_url + "/resume",
            body=urlencode(resume_fields).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert first_status == 200
    assert planned.fingerprint.encode() in first_payload
    assert b"Resume this exact finish" in first_payload
    assert dashboard_status == 200
    assert b"Revision finishes" in dashboard
    assert receipt_url.encode() in dashboard
    assert b"Waiting to resume from revision applied" in dashboard
    assert get_status == 200
    assert b"revision is applied and archived" in receipt_page
    assert resume_fields == {
        "action": "resume-apply-and-finish",
        "csrf": session.csrf_token,
        "receipt_id": planned.fingerprint,
    }
    assert resume_status == 200
    assert b"Building Anki package" in resume_payload
    assert b"Anki package are complete" in resume_payload
    assert resumed == [planned.fingerprint]


def test_resume_requires_csrf_and_matching_receipt_without_running_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, _staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    receipt_id = "a" * 64
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("resume must not run")

    monkeypatch.setattr(revision_finish, "resume_revision_finish", forbidden)
    server, _thread = _running(session)
    try:
        path = f"/{session.token}/revision-finishes/{receipt_id}/resume"
        forged = {
            "action": "resume-apply-and-finish",
            "csrf": "forged",
            "receipt_id": receipt_id,
        }
        csrf_status, _headers, _payload = _request(
            server, "POST", path, body=urlencode(forged).encode()
        )
        mismatch = dict(forged, csrf=session.csrf_token, receipt_id="b" * 64)
        mismatch_status, _headers, mismatch_payload = _request(
            server, "POST", path, body=urlencode(mismatch).encode()
        )
    finally:
        server.shutdown()
        server.server_close()

    assert csrf_status == 403
    assert mismatch_status == 409
    assert b"different finish receipt" in mismatch_payload
    assert called is False


@pytest.mark.parametrize("state", ["audio_complete", "complete"])
def test_post_audio_resume_failure_never_claims_possible_new_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    config, deck, _staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    receipt_id = "a" * 64
    result = revision_finish.RevisionFinishResult(
        receipt_id=receipt_id,
        state=state,  # type: ignore[arg-type]
        record_path=(config.staging_dir / "done" / "revisions" / f"finish-{receipt_id}.json"),
        deck_path=deck,
        output_path=config.dist_dir / "potential.apkg",
        example_provider_access="paid-network",
        max_provider_calls=2,
        package_sha256="b" * 64 if state == "complete" else None,
        card_count=1 if state == "complete" else None,
    )
    inspections = 0

    def inspect(_config: Any, inspected_receipt_id: str) -> revision_finish.RevisionFinishResult:
        nonlocal inspections
        assert inspected_receipt_id == receipt_id
        inspections += 1
        if inspections == 1:
            return result
        raise OSError("receipt became unreadable")

    def interrupted(*_args: object, **_kwargs: object) -> object:
        raise OSError("local package resume failed")

    monkeypatch.setattr(revision_finish, "inspect_revision_finish", inspect)
    monkeypatch.setattr(revision_finish, "resume_revision_finish", interrupted)
    server, _thread = _running(session)
    try:
        path = f"/{session.token}/revision-finishes/{receipt_id}/resume"
        status, _headers, payload = _request(
            server,
            "POST",
            path,
            body=urlencode(
                {
                    "action": "resume-apply-and-finish",
                    "csrf": session.csrf_token,
                    "receipt_id": receipt_id,
                }
            ).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert b"local package resume failed" in payload
    assert b"Example audio was already durable before this resume" in payload
    assert b"this resume could not contact a paid provider" in payload
    assert b"may have been billed" not in payload
    assert inspections == 2


def _pre_audio_resume_failure_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: str,
    access: audio_application.AudioAccess,
    max_provider_calls: int,
) -> bytes:
    config, deck, _staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    receipt_id = "a" * 64
    result = revision_finish.RevisionFinishResult(
        receipt_id=receipt_id,
        state=state,  # type: ignore[arg-type]
        record_path=(config.staging_dir / "done" / "revisions" / f"finish-{receipt_id}.json"),
        deck_path=deck,
        output_path=config.dist_dir / "potential.apkg",
        example_provider_access=access,
        max_provider_calls=max_provider_calls,
    )
    inspections = 0

    def inspect(
        _config: Any,
        inspected_receipt_id: str,
    ) -> revision_finish.RevisionFinishResult:
        nonlocal inspections
        assert inspected_receipt_id == receipt_id
        inspections += 1
        if inspections == 1:
            return result
        raise OSError("receipt became unreadable")

    def interrupted(*_args: object, **_kwargs: object) -> object:
        raise OSError("pre-audio resume failed")

    monkeypatch.setattr(revision_finish, "inspect_revision_finish", inspect)
    monkeypatch.setattr(revision_finish, "resume_revision_finish", interrupted)
    server, _thread = _running(session)
    try:
        path = f"/{session.token}/revision-finishes/{receipt_id}/resume"
        status, _headers, payload = _request(
            server,
            "POST",
            path,
            body=urlencode(
                {
                    "action": "resume-apply-and-finish",
                    "csrf": session.csrf_token,
                    "receipt_id": receipt_id,
                }
            ).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert b"pre-audio resume failed" in payload
    assert inspections == 2
    return payload


@pytest.mark.parametrize("state", ["authorized", "revision_applied"])
def test_local_network_pre_audio_resume_failure_never_claims_possible_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    payload = _pre_audio_resume_failure_payload(
        tmp_path,
        monkeypatch,
        state=state,
        access="local-network",
        max_provider_calls=2,
    )

    assert b"local-network example provider" in payload
    assert b"could not incur a paid provider charge" in payload
    assert b"may have been billed" not in payload


@pytest.mark.parametrize("state", ["authorized", "revision_applied"])
def test_paid_network_pre_audio_resume_failure_preserves_possible_billing_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    payload = _pre_audio_resume_failure_payload(
        tmp_path,
        monkeypatch,
        state=state,
        access="paid-network",
        max_provider_calls=2,
    )

    assert b"A paid provider call may have been billed" in payload


def test_zero_provider_call_authority_never_claims_possible_resume_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _pre_audio_resume_failure_payload(
        tmp_path,
        monkeypatch,
        state="authorized",
        access="paid-network",
        max_provider_calls=0,
    )

    assert b"authorized no provider-required example clips" in payload
    assert b"could not incur a paid provider charge" in payload
    assert b"may have been billed" not in payload


def test_aggregate_failure_after_stream_start_is_never_a_500_or_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _resolved_test_providers(config, monkeypatch)
    before = deck.read_bytes()

    def interrupted(*_args: object, **_kwargs: object) -> object:
        raise OSError("simulated aggregate interruption")

    monkeypatch.setattr(revision_finish, "execute_revision_finish", interrupted)
    monkeypatch.setattr(revision_finish, "inspect_revision_finish", interrupted)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = _review_url(session, staging)
        _status, _headers, page = _request(server, "GET", url)
        status, _headers, payload = _request(
            server,
            "POST",
            url + "/finish",
            body=urlencode(_hidden_fields(page)).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert b"simulated aggregate interruption" in payload
    assert b"cannot safely prove which finish phases became durable" in payload
    assert b"may have been billed" in payload
    assert b"are complete" not in payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_local_network_initial_finish_failure_never_claims_possible_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _transport, words, _sentences = _resolved_test_providers(config, monkeypatch)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda _config, _chosen, _words: words,
    )
    before = deck.read_bytes()

    def interrupted(*_args: object, **_kwargs: object) -> object:
        raise OSError("simulated local aggregate interruption")

    monkeypatch.setattr(revision_finish, "execute_revision_finish", interrupted)
    monkeypatch.setattr(revision_finish, "inspect_revision_finish", interrupted)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = _review_url(session, staging)
        _status, _headers, page = _request(server, "GET", url)
        status, _headers, payload = _request(
            server,
            "POST",
            url + "/finish",
            body=urlencode(_hidden_fields(page)).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert b"local-network" in page
    assert status == 200
    assert b"simulated local aggregate interruption" in payload
    assert b"This exact plan required no paid provider call" in payload
    assert b"may have been billed" not in payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_superseded_revision_confirmation_ladder_routes_are_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _resolved_test_providers(config, monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        old_apply_status, _headers, _payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/apply",
            body=urlencode(
                {
                    "action": "apply-revision",
                    "csrf": session.csrf_token,
                    "plan_fingerprint": "a" * 64,
                }
            ).encode(),
        )
        old_finish_status, _headers, _payload = _request(
            server,
            "GET",
            f"/{session.token}/decks/{deck.name}/finish",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert old_apply_status == 404
    assert old_finish_status == 404
    assert staging.exists()


def test_revision_finish_never_joins_an_untrusted_receipt_path(tmp_path: Path) -> None:
    config, _deck, _staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, _payload = _request(
            server,
            "GET",
            f"/{session.token}/revision-finishes/%2E%2E%2Ffinish.json",
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 404


def test_broken_revision_review_is_a_409_refusal_page(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    staging.write_text("not json\n", encoding="utf-8")
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(server, "GET", _review_url(session, staging))
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"Could not parse revision" in payload
    assert b"This action stopped before any repository write" in payload
    assert b"This action made no paid provider call" in payload


def test_broken_revision_is_visible_on_dashboard_without_breaking_it(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    staging.write_text("not json\n", encoding="utf-8")
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(server, "GET", f"/{session.token}/")
    finally:
        server.shutdown()
        server.server_close()

    page = payload.decode("utf-8")
    assert status == 200
    assert "Deck revision proposals" in page
    assert "Needs attention" in page
    assert f"Could not open revision {staging.name}" in page
    assert f"/{session.token}/revisions/{staging.name}" not in page
