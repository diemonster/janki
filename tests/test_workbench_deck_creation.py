"""W4.1's preview-first study-deck creator in the localhost workbench."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml
from test_workbench import _request, _running

from japanese_anki.application.deck_creation import (
    StudyDeckCreationError,
    create_study_deck,
    plan_study_deck,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import WorkbenchSession
from japanese_anki.workbench import server as workbench_server

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
"""


def _session(tmp_path: Path) -> WorkbenchSession:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]\n", encoding="utf-8")
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def _url(session: WorkbenchSession) -> str:
    return f"/{session.token}/decks/new"


def _post(
    server: object,
    session: WorkbenchSession,
    pairs: list[tuple[str, str]],
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        _url(session),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(pairs).encode("utf-8"),
    )


def _fingerprint(page: bytes) -> str:
    match = re.search(
        rb'name=plan_fingerprint value="([0-9a-f]{64})"', page
    )
    assert match is not None
    return match.group(1).decode("ascii")


def test_dashboard_opens_creator_with_recognition_on_by_default(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        dashboard = _request(server, "GET", f"/{session.token}/")
        status, _headers, page = _request(server, "GET", _url(session))
    finally:
        server.shutdown()
        server.server_close()

    rendered = page.decode("utf-8")
    assert dashboard[0] == 200
    assert f'href="/{session.token}/decks/new"' in dashboard[2].decode("utf-8")
    assert status == 200
    assert '<input type=checkbox name="recognition" checked>' in rendered
    assert '<input type=checkbox name="production">' in rendered
    assert '<input type=checkbox name="reading">' in rendered
    assert "Preview study deck" in rendered
    assert not session.config.deck_dir.exists()


def test_preview_shows_exact_plan_and_create_publishes_that_deck(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    choices = [
        ("action", "preview"),
        ("csrf", session.csrf_token),
        ("name", "Genki <One>"),
        ("recognition", "on"),
        ("production", "on"),
    ]
    server, _thread = _running(session)
    try:
        status, _headers, preview = _post(server, session, choices)
        fingerprint = _fingerprint(preview)
        planned = plan_study_deck(
            session.config,
            name="Genki <One>",
            recognition=True,
            production=True,
            reading=False,
        )
        assert not planned.path.exists()
        created_status, created_headers, _body = _post(
            server,
            session,
            [
                ("action", "create"),
                ("csrf", session.csrf_token),
                ("name", "Genki <One>"),
                ("recognition", "on"),
                ("production", "on"),
                ("plan_fingerprint", fingerprint),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    rendered = preview.decode("utf-8")
    assert status == 200
    assert "Genki &lt;One&gt;" in rendered
    assert str(planned.path) in rendered
    assert str(planned.output_path) in rendered
    assert "Create study deck" in rendered
    assert created_status == 303
    assert created_headers["location"] == f"/{session.token}/"
    assert planned.path.read_bytes() == planned.yaml_bytes
    assert yaml.safe_load(planned.path.read_bytes())["deck"]["cards"] == {
        "recognition": True,
        "production": True,
        "reading": False,
    }


def test_create_error_after_publication_reports_the_deck_state_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    fields = [
        ("action", "preview"),
        ("csrf", session.csrf_token),
        ("name", "Genki recovery"),
        ("recognition", "on"),
    ]
    planned = plan_study_deck(
        session.config,
        name="Genki recovery",
        recognition=True,
        production=False,
        reading=False,
    )

    def publish_then_fail(
        config: ProjectConfig,
        plan: workbench_server.StudyDeckCreationPlan,
    ) -> None:
        create_study_deck(config, plan)
        raise StudyDeckCreationError(
            "The create response was lost after deck publication."
        )

    server, _thread = _running(session)
    try:
        preview_status, _headers, preview = _post(server, session, fields)
        assert preview_status == 200
        monkeypatch.setattr(
            workbench_server,
            "create_study_deck",
            publish_then_fail,
        )
        status, _headers, body = _post(
            server,
            session,
            [
                ("action", "create"),
                ("csrf", session.csrf_token),
                ("name", "Genki recovery"),
                ("recognition", "on"),
                ("plan_fingerprint", _fingerprint(preview)),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    rendered = body.decode("utf-8")
    assert status == 409
    assert "The create response was lost after deck publication." in rendered
    assert (
        "This action cannot safely prove that repository files are unchanged."
        in rendered
    )
    assert "This action made no paid provider call." in rendered
    assert "Nothing was created" not in rendered
    assert (
        "reload the dashboard and check whether the study deck was published "
        "before trying to create it again" in rendered
    )
    assert planned.path.read_bytes() == planned.yaml_bytes


def test_create_refuses_stale_preview_and_strictly_parses_authority(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    preview_fields = [
        ("action", "preview"),
        ("csrf", session.csrf_token),
        ("name", "Lesson 14"),
        ("recognition", "on"),
    ]
    server, _thread = _running(session)
    try:
        forged = [
            (key, "forged" if key == "csrf" else value)
            for key, value in preview_fields
        ]
        assert _post(server, session, forged)[0] == 403
        non_ascii = [
            (key, "偽" if key == "csrf" else value)
            for key, value in preview_fields
        ]
        assert _post(server, session, non_ascii)[0] == 403
        assert _post(
            server, session, [*preview_fields, ("intake_tag", "forged")]
        )[0] == 400
        assert _post(
            server, session, [*preview_fields, ("recognition", "on")]
        )[0] == 400
        wrong_toggle = [
            (key, "off" if key == "recognition" else value)
            for key, value in preview_fields
        ]
        assert _post(server, session, wrong_toggle)[0] == 400

        status, _headers, preview = _post(server, session, preview_fields)
        assert status == 200
        fingerprint = _fingerprint(preview)
        planned = plan_study_deck(session.config, name="Lesson 14")
        session.config.deck_dir.mkdir(parents=True, exist_ok=True)
        (session.config.deck_dir / "arrived-later.yaml").write_text(
            yaml.safe_dump(
                {
                    "deck": {
                        "name": "Arrived later",
                        "deck_id": 1_500_000_111,
                        "source": "../vocabulary.json",
                        "include_ids": ["word:later:later"],
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        stale_status, _headers, stale_body = _post(
            server,
            session,
            [
                ("action", "create"),
                ("csrf", session.csrf_token),
                ("name", "Lesson 14"),
                ("recognition", "on"),
                ("plan_fingerprint", fingerprint),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert stale_status == 409
    assert b"plan changed after the preview" in stale_body
    assert not planned.path.exists()
