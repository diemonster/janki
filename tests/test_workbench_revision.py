"""Browser review and exact owner apply for staged deck revisions."""

from __future__ import annotations

import html
import http.client
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from test_application_audio import Provider, RecordingProvider
from test_application_deck_build import _fixture as _deck_fixture
from test_application_revision_apply import _fixture

from japanese_anki.application import audio as audio_application
from japanese_anki.application import revision_apply
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
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=3
    )
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


def _form_fields(page: bytes, action: str) -> dict[str, str]:
    forms = re.findall(r"<form\b.*?</form>", page.decode("utf-8"), re.DOTALL)
    for form in forms:
        fields = _hidden_fields(form.encode("utf-8"))
        if fields.get("action") == action:
            return fields
    raise AssertionError(f"no {action!r} form")


def _review_url(session: WorkbenchSession, staging: Path) -> str:
    return f"/{session.token}/revisions/{staging.name}"


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
    assert f'/{session.token}/revisions/{staging.name}' in page
    assert "1 selected card(s)" in page
    assert "revise-linked.json is not a direct regular file and was refused" in page
    assert f'/{session.token}/revisions/{linked.name}' not in page


def test_revision_review_renders_exact_selected_current_and_proposed_content(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(
            server, "GET", _review_url(session, staging)
        )
    finally:
        server.shutdown()
        server.server_close()

    page = payload.decode("utf-8")
    fields = _hidden_fields(payload)
    assert status == 200
    assert "Old note." in page
    assert "New note." in page
    assert "I can speak." in page
    assert "I am able to speak." in page
    assert "Can you speak?" in page
    assert "Can you speak Japanese?" in page
    assert "polite" in page and "casual" in page
    assert "I can read." not in page
    assert fields == {
        "action": "apply-revision",
        "csrf": session.csrf_token,
        "plan_fingerprint": plan.plan_fingerprint,
    }


def test_revision_apply_requires_csrf_and_leaves_the_deck_unchanged(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        body = urlencode(
            {
                "action": "apply-revision",
                "csrf": "forged",
                "plan_fingerprint": plan.plan_fingerprint,
            }
        ).encode()
        status, _headers, _payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/apply",
            body=body,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 403
    assert deck.read_bytes() == before
    assert staging.exists()


def test_revision_apply_rejects_duplicated_authority_fields(tmp_path: Path) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        pairs = [
            ("action", "apply-revision"),
            ("csrf", session.csrf_token),
            ("csrf", "forged"),
            ("plan_fingerprint", plan.plan_fingerprint),
        ]
        status, _headers, payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/apply",
            body=urlencode(pairs).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 400
    assert b"repeats a field" in payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_revision_apply_replans_at_click_and_refuses_rendered_plan_drift(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _review_url(session, staging)
        )
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
            _review_url(session, staging) + "/apply",
            body=urlencode(fields).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"changed after the page was rendered" in payload
    assert b"This action stopped before any repository write" in payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_revision_apply_errors_are_recoverable_conflicts_never_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck, staging = _fixture(tmp_path)
    before = deck.read_bytes()
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _review_url(session, staging)
        )
        fields = _hidden_fields(page)

        def fail_after_start(*_args: object, **_kwargs: object) -> object:
            raise OSError("simulated interrupted apply")

        monkeypatch.setattr(
            revision_apply, "execute_revision_apply", fail_after_start
        )
        status, _headers, payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/apply",
            body=urlencode(fields).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"simulated interrupted apply" in payload
    assert b"cannot safely prove that repository files are unchanged" in payload
    assert b"This action made no paid provider call" in payload
    assert deck.read_bytes() == before
    assert staging.exists()


def test_successful_revision_apply_points_to_audio_then_build(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _review_url(session, staging)
        )
        status, headers, _payload = _request(
            server,
            "POST",
            _review_url(session, staging) + "/apply",
            body=urlencode(_hidden_fields(page)).encode(),
        )
        assert status == 303
        assert headers["location"] == (
            f"/{session.token}/decks/{deck.name}/finish"
        )
        follow_status, _follow_headers, follow_payload = _request(
            server, "GET", headers["location"]
        )
    finally:
        server.shutdown()
        server.server_close()

    follow = follow_payload.decode("utf-8")
    assert follow_status == 200
    assert "Finish revised deck" in follow
    assert "Example audio" in follow
    assert "Build Anki package" in follow
    assert "separate owner-authorized actions" in follow
    assert "New note." in deck.read_text(encoding="utf-8")
    assert not staging.exists()
    assert (config.staging_dir / "done" / "revisions" / staging.name).exists()


def _fake_providers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    paid: bool = True,
) -> tuple[Provider, RecordingProvider]:
    words = Provider("voicevox", 7)
    sentences = RecordingProvider(
        "openai-realtime" if paid else "voicevox",
        "cedar" if paid else 8,
    )
    sentences.settings["model"] = "fake-realtime-v1" if paid else "fake-local-v1"
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
    return words, sentences


def test_revision_finish_renders_exact_audio_and_build_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _words, _sentences = _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(
            server,
            "GET",
            f"/{session.token}/decks/{deck.name}/finish",
        )
    finally:
        server.shutdown()
        server.server_close()

    page = payload.decode("utf-8")
    assert status == 200
    assert "openai-realtime" in page
    assert "fake-realtime-v1" in page
    assert "paid-network" in page
    assert "Total clips</dt><dd>2" in page
    assert "Already current</dt><dd>0" in page
    assert "Recoverable exact clips</dt><dd>0" in page
    assert "Provider calls required</dt><dd>2" in page
    assert str((config.dist_dir / "potential.apkg").resolve()) in page
    assert "cards:</b> 1" in page
    assert "Audio is not fully current" in page
    assert "Building stays available" in page
    assert _form_fields(payload, "deck-audio")["csrf"] == session.csrf_token
    assert _form_fields(payload, "deck-build")["csrf"] == session.csrf_token


def test_revision_finish_executes_example_audio_with_fake_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _words, sentences = _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)
        status, headers, _payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, "deck-audio")).encode(),
        )
        assert status == 303
        follow_status, _follow_headers, follow_payload = _request(
            server, "GET", headers["location"]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert follow_status == 200
    assert b"Example audio finished" in follow_payload
    assert len(sentences.said) == 2
    assert "audio/janki-" in deck.read_text(encoding="utf-8")
    assert len(list((config.media_dir / "audio").glob("*.wav"))) == 2


def test_revision_finish_build_remains_separately_available_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)
        assert b"Audio is not fully current" in page
        status, headers, _payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, "deck-build")).encode(),
        )
        assert status == 303
        follow_status, _follow_headers, follow_payload = _request(
            server, "GET", headers["location"]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert follow_status == 200
    assert b"Built the Anki package" in follow_payload
    assert (config.dist_dir / "potential.apkg").read_bytes().startswith(b"PK")


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ("deck-audio", b"audio plan changed after the page was rendered"),
        ("deck-build", b"build plan changed after the page was rendered"),
    ],
)
def test_revision_finish_replans_each_action_at_click(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    message: bytes,
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _words, sentences = _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)
        deck.write_text(
            deck.read_text(encoding="utf-8").replace(
                "You can do the action.", "Changed after rendering."
            ),
            encoding="utf-8",
        )
        status, _headers, payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, action)).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert message in payload
    assert b"This action stopped before any repository write" in payload
    assert sentences.said == []
    assert not (config.dist_dir / "potential.apkg").exists()


def test_revision_finish_actions_require_session_csrf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)
        fields = _form_fields(page, "deck-build")
        fields["csrf"] = "forged"
        status, _headers, _payload = _request(
            server, "POST", url, body=urlencode(fields).encode()
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 403
    assert not (config.dist_dir / "potential.apkg").exists()


def test_revision_finish_audio_failure_reports_paid_and_write_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)
        plan = audio_application.plan_deck_audio(config, deck)

        def stopped(*_args: object, **_kwargs: object) -> object:
            return audio_application.AudioExecutionOutcome(
                state="records-stale",
                plan=plan,
                output_dir=config.media_dir / "audio",
                pending_recovery=True,
                record_references_written=False,
                media_published=False,
                ledger_committed=False,
            )

        monkeypatch.setattr(audio_application, "execute_deck_audio", stopped)
        status, _headers, payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, "deck-audio")).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"may have been billed" in payload
    assert b"did not write deck audio references" in payload
    assert b"did not publish canonical media" in payload
    assert b"did not commit canonical audio ledger" in payload
    assert b"Exact recovery is durable" in payload


def test_revision_finish_local_audio_exception_does_not_claim_a_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _fake_providers(monkeypatch, paid=False)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)

        received: dict[str, object] = {}

        def interrupted(*_args: object, **kwargs: object) -> object:
            received.update(kwargs)
            raise OSError("local audio interrupted")

        monkeypatch.setattr(
            audio_application, "execute_deck_audio", interrupted
        )
        status, _headers, payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, "deck-audio")).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"cannot prove whether deck audio references" in payload
    assert b"required no paid provider call" in payload
    assert received["words"] is False
    assert received["examples"] is True
    assert received["force"] is False
    assert received["prune"] is False
    assert received["expected_fingerprint"] == _form_fields(
        page, "deck-audio"
    )["plan_fingerprint"]


def test_revision_finish_build_errors_are_conflicts_never_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _deck_fixture(tmp_path)
    _fake_providers(monkeypatch)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        url = f"/{session.token}/decks/{deck.name}/finish"
        _status, _headers, page = _request(server, "GET", url)

        def interrupted(*_args: object, **_kwargs: object) -> object:
            raise OSError("local build interrupted")

        monkeypatch.setattr(
            "japanese_anki.application.deck_build.execute_conjugation_deck_build",
            interrupted,
        )
        status, _headers, payload = _request(
            server,
            "POST",
            url,
            body=urlencode(_form_fields(page, "deck-build")).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"local build interrupted" in payload
    assert b"cannot safely prove that repository files are unchanged" in payload
    assert b"This action made no paid provider call" in payload


def test_revision_finish_never_joins_an_untrusted_deck_path(tmp_path: Path) -> None:
    config, _deck = _deck_fixture(tmp_path)
    session = WorkbenchSession.open(config)
    server, _thread = _running(session)
    try:
        status, _headers, _payload = _request(
            server,
            "GET",
            f"/{session.token}/decks/%2E%2E%2Fpotential.yaml/finish",
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
        status, _headers, payload = _request(
            server, "GET", _review_url(session, staging)
        )
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
    assert f'/{session.token}/revisions/{staging.name}' not in page
