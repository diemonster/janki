"""W4.1's browser controller for exact thematic deck assignment."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pytest
import yaml
from test_application_journey import _stage
from test_workbench import _request, _running

import japanese_anki.application.assignment as assignment_module
from japanese_anki import staging as staging_module
from japanese_anki.config import ProjectConfig
from japanese_anki.io import save_records_json
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import read_staging, write_staging
from japanese_anki.workbench import WorkbenchSession
from japanese_anki.workbench import server as workbench_server


def _write_deck(root: Path, stem: str, name: str, intake_tag: str) -> None:
    document = {
        "deck": {
            "name": name,
            "source": "../vocabulary.json",
            "include_tags": [intake_tag],
            "intake_tag": intake_tag,
        }
    }
    (root / "decks" / f"{stem}.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _assignment_project(tmp_path: Path) -> tuple[WorkbenchSession, Path]:
    _stage(tmp_path, "table_exhaustive", filename="table.pdf")
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    records, meta = read_staging(staging_path)
    records[0] = replace(records[0], tags=["old-intake", "personal"])
    write_staging(staging_path, records, meta, force=True)
    save_records_json(
        tmp_path / "vocabulary.json",
        [replace(records[0], tags=["canonical-only"])],
    )
    _write_deck(tmp_path, "old", "Old deck", "old-intake")
    _write_deck(tmp_path, "lesson", "Lesson deck", "lesson-intake")
    return WorkbenchSession.open(ProjectConfig.load(tmp_path)), staging_path


def _source_url(session: WorkbenchSession) -> str:
    return f"/{session.token}/source/{quote('table.pdf', safe='')}"


def _write_sibling(staging_path: Path) -> Path:
    records, _meta = read_staging(staging_path)
    sibling = replace(
        records[0],
        source=replace(
            records[0].source,
            imported_from="other.pdf",
            row=1,
        ),
    )
    path = staging_path.with_name("other.pdf.yaml")
    write_staging(path, [sibling], {"source_file": "other.pdf"})
    return path


def _fingerprint(page: bytes, card: int = 0) -> str:
    match = re.search(
        rf'<form id="assign-{card}".*?name=plan_fingerprint value="([0-9a-f]+)"',
        page.decode("utf-8"),
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _post(
    server: object,
    session: WorkbenchSession,
    fingerprint: str,
    *,
    pairs: list[tuple[str, str]] | None = None,
    snapshot: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    panel = session.panel("table.pdf")
    assert panel is not None
    submitted_snapshot = snapshot or panel.staging_fingerprint
    fields = pairs or [
        ("action", "assign"),
        ("csrf", session.csrf_token),
        ("staging_snapshot", submitted_snapshot),
        ("card", "0"),
        ("plan_fingerprint", fingerprint),
        ("destination", "lesson"),
    ]
    return _request(
        server,
        "POST",
        _source_url(session) + "/assign",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(fields).encode("utf-8"),
    )


def test_source_page_offers_configured_deck_stems_and_exact_tag_diffs(
    tmp_path: Path,
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    _write_sibling(staging_path)
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", _source_url(session))
        records, meta = read_staging(staging_path)
        meta["source_file"] = "table.pdf/assign"
        write_staging(staging_path, records, meta, force=True)
        subroute = _request(server, "GET", _source_url(session) + "/assign")
        encoded_source = _request(
            server,
            "GET",
            f"/{session.token}/source/{quote('table.pdf/assign', safe='')}",
        )
    finally:
        server.shutdown()
        server.server_close()

    rendered = page.decode("utf-8")
    assert status == 200
    assert subroute[0] == 404
    assert encoded_source[0] == 200
    assert "Old deck" in rendered and "Lesson deck" in rendered
    assert 'name=destination value="lesson"' in rendered
    assert "name=intake_tag" not in rendered
    assert "Tags added</dt><dd>lesson-intake" in rendered
    assert "Tags removed</dt><dd>old-intake" in rendered
    assert "Tags after assignment</dt><dd>personal, lesson-intake" in rendered
    assert (
        "This word is proposed by two sources. Choose which word deck should "
        "teach it; both sources can still remain in its history."
        in rendered
    )


def test_source_page_plans_one_matrix_without_reloading_per_card_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    records, _meta = read_staging(staging_path)
    real_census = assignment_module._word_deck_census
    real_project = assignment_module.project_deck_records
    census_calls = 0
    projected: list[str] = []

    def observe_census(
        config: ProjectConfig,
    ) -> tuple[tuple[object, ...], tuple[str, ...]]:
        nonlocal census_calls
        census_calls += 1
        return real_census(config)

    def observe_project(
        deck_path: Path,
        source_path: Path,
        source_records: Sequence[VocabularyRecord],
    ) -> tuple[dict[str, Any], list[VocabularyRecord]]:
        projected.append(deck_path.name)
        return real_project(deck_path, source_path, source_records)

    monkeypatch.setattr(assignment_module, "_word_deck_census", observe_census)
    monkeypatch.setattr(assignment_module, "project_deck_records", observe_project)

    offers = workbench_server._assignment_offers(
        session.config,
        records,
        staging_path,
    )

    assert len(offers) == 3
    assert all(len(offer.choices) == 2 for offer in offers)
    assert census_calls == 1
    # Two decks are projected once for current ownership, then once for each
    # of the two rendered destinations. The record count is not a multiplier.
    assert projected == ["lesson.yaml", "old.yaml"] * 3


def test_assignment_replans_then_writes_only_the_selected_staging_row(
    tmp_path: Path,
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    before, before_meta = read_staging(staging_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _source_url(session))
        status, headers, _body = _post(server, session, _fingerprint(page))
    finally:
        server.shutdown()
        server.server_close()

    after, after_meta = read_staging(staging_path)
    assert status == 303
    assert "assigned=1" in headers["location"]
    assert after[0] == replace(before[0], tags=["personal", "lesson-intake"])
    assert after[1:] == before[1:]
    assert after_meta == before_meta


def test_assignment_post_rebinds_the_full_rendered_destination_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _source_url(session))
        fingerprint = _fingerprint(page)
        before = staging_path.read_bytes()
        real_plans = workbench_server.plan_deck_assignments

        def change_only_the_unselected_choice(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[assignment_module.DeckAssignmentAttempt, ...], ...]:
            matrix = real_plans(*args, **kwargs)
            row = list(matrix[0])
            old_index = next(
                index
                for index, attempt in enumerate(row)
                if attempt.destination.stem == "old"
            )
            row[old_index] = replace(
                row[old_index],
                plan=None,
                refusal="the unselected destination changed",
            )
            return (tuple(row),)

        monkeypatch.setattr(
            workbench_server,
            "plan_deck_assignments",
            change_only_the_unselected_choice,
        )

        status, _headers, body = _post(server, session, fingerprint)
    finally:
        server.shutdown()
        server.server_close()

    assert status == 409
    assert b"plan changed" in body
    assert staging_path.read_bytes() == before


def test_assignment_refuses_a_changed_plan_or_staging_snapshot(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    session, staging_path = _assignment_project(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _source_url(session))
        old_fingerprint = _fingerprint(page)
        before = staging_path.read_bytes()

        _write_deck(tmp_path, "lesson", "Lesson deck", "replacement-intake")
        status, _headers, body = _post(server, session, old_fingerprint)
        assert status == 409
        assert b"plan changed" in body
        assert staging_path.read_bytes() == before

        _write_deck(tmp_path, "lesson", "Lesson deck", "lesson-intake")
        _status, _headers, fresh_page = _request(
            server, "GET", _source_url(session)
        )
        fresh_fingerprint = _fingerprint(fresh_page)
        sibling = _write_sibling(staging_path)
        status, _headers, body = _post(server, session, fresh_fingerprint)
        assert status == 409
        assert b"plan changed" in body
        assert staging_path.read_bytes() == before

        _status, _headers, fresh_page = _request(
            server, "GET", _source_url(session)
        )
        fresh_fingerprint = _fingerprint(fresh_page)
        rendered_panel = session.panel("table.pdf")
        assert rendered_panel is not None
        records, meta = read_staging(staging_path)
        records[1] = replace(records[1], tags=["concurrent"])
        write_staging(staging_path, records, meta, force=True)
        changed = staging_path.read_bytes()

        status, _headers, body = _post(
            server,
            session,
            fresh_fingerprint,
            snapshot=rendered_panel.staging_fingerprint,
        )
        assert status == 409
        assert b"source changed" in body
        assert staging_path.read_bytes() == changed
        assert sibling.exists()

        _status, _headers, race_page = _request(
            server, "GET", _source_url(session)
        )
        race_fingerprint = _fingerprint(race_page)
        real_render = staging_module.render_staging_update

        def race_after_render(
            captured: bytes, updated: object, *, source: str
        ) -> str:
            text = real_render(captured, updated, source=source)
            live, live_meta = read_staging(staging_path)
            live[2] = replace(live[2], tags=["won-the-race"])
            write_staging(staging_path, live, live_meta, force=True)
            return text

        monkeypatch.setattr(
            staging_module,
            "render_staging_update",
            race_after_render,
        )
        status, _headers, body = _post(server, session, race_fingerprint)
        raced, _meta = read_staging(staging_path)
        assert status == 409
        assert b"final review write" in body
        assert raced[0].tags == ["old-intake", "personal"]
        assert raced[2].tags == ["won-the-race"]
    finally:
        server.shutdown()
        server.server_close()


def test_assignment_requires_exact_fields_and_session_csrf(tmp_path: Path) -> None:
    session, staging_path = _assignment_project(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _source_url(session))
        fingerprint = _fingerprint(page)
        panel = session.panel("table.pdf")
        assert panel is not None
        base = [
            ("action", "assign"),
            ("csrf", session.csrf_token),
            ("staging_snapshot", panel.staging_fingerprint),
            ("card", "0"),
            ("plan_fingerprint", fingerprint),
            ("destination", "lesson"),
        ]
        attempts = [
            (403, [(key, "wrong" if key == "csrf" else value) for key, value in base]),
            (400, [*base, ("intake_tag", "forged")]),
            (400, [*base, ("destination", "lesson")]),
            (
                400,
                [
                    (key, "not-a-deck" if key == "destination" else value)
                    for key, value in base
                ],
            ),
        ]
        before = staging_path.read_bytes()
        for expected, pairs in attempts:
            status, _headers, _body = _post(
                server,
                session,
                fingerprint,
                pairs=pairs,
            )
            assert status == expected
            assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()
