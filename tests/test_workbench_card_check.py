"""W4.1's read-only structural card check in the workbench."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest
from test_workbench import _request, _running
from test_workbench_assignment import _assignment_project

from japanese_anki.workbench import WorkbenchSession
from japanese_anki.workbench import server as server_module


def _source_url(token: str) -> str:
    return f"/{token}/source/{quote('table.pdf', safe='')}"


def test_check_cards_projects_actions_from_the_bound_review_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    bound_panel = session.panel("table.pdf")
    assert bound_panel is not None
    captured: list[tuple[bool, bool, bool]] = []
    real_check = server_module.check_cards

    def check(config: object, records: object, **kwargs: object):
        captured.append(
            (
                records is bound_panel.records,
                kwargs.get("reidentifiable") is bound_panel.reidentifiable,
                kwargs.get("approvable") is bound_panel.has_extraction_lineage,
            )
        )
        return real_check(config, records, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WorkbenchSession, "panel", lambda *_args: bound_panel)
    monkeypatch.setattr(server_module, "check_cards", check)
    before = staging_path.read_bytes()
    source_url = _source_url(session.token)
    server, _thread = _running(session)
    try:
        source = _request(server, "GET", source_url)
        status, _headers, page = _request(server, "GET", source_url + "/check")
    finally:
        server.shutdown()
        server.server_close()

    rendered_source = source[2].decode("utf-8")
    rendered = page.decode("utf-8")
    assert source[0] == 200
    assert f'href="{source_url}/check"' in rendered_source
    assert 'id="card-1"' in rendered_source
    assert status == 200
    assert "Check these cards" in rendered
    assert 'id="card-1"' in rendered
    assert "Review both Japanese examples" in rendered
    assert "Choose a study deck" in rendered
    assert "Source and technical details" in rendered
    assert "example-review-available" in rendered
    assert "deck-unassigned" in rendered
    assert f'href="{source_url}#card-1"' in rendered
    assert captured == [(True, True, True)]
    assert staging_path.read_bytes() == before
