"""W4.1's browser coverage decisions and shared promotion transaction."""

from __future__ import annotations

from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
import yaml
from test_application_journey import _approve_examples, _stage
from test_promote import FakeJpdb, client_for
from test_workbench import _request, _running

from conftest import seed_prompts
from japanese_anki import coverage, jpdb, operations, promote
from japanese_anki.application import coverage as coverage_application
from japanese_anki.application import promotion_action
from japanese_anki.application.coverage import (
    approve_coverage_as_owner,
    plan_coverage,
)
from japanese_anki.application.promotion import decide_promotion, execute_promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.io import load_records, save_records_json
from japanese_anki.staging import read_staging, write_staging
from japanese_anki.workbench import WorkbenchSession
from japanese_anki.workbench import server as workbench_server


class _ActionForms(HTMLParser):
    """Collect successful controls from forms carrying one hidden action."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current: list[tuple[str, str]] | None = None
        self.forms: dict[str, list[tuple[str, str]]] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key: value for key, value in attrs}
        if tag == "form":
            assert self.current is None, "promotion forms must not be nested"
            self.current = []
            return
        if self.current is None or tag not in {"input", "textarea"}:
            return
        name = values.get("name")
        if not name or "disabled" in values:
            return
        kind = (values.get("type") or "text").lower()
        if kind in {"button", "image", "reset", "submit"}:
            return
        if kind in {"checkbox", "radio"} and "checked" not in values:
            return
        self.current.append((name, values.get("value") or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag != "form" or self.current is None:
            return
        actions = [value for name, value in self.current if name == "action"]
        if len(actions) == 1:
            assert actions[0] not in self.forms
            self.forms[actions[0]] = self.current
        self.current = None


def _form(page: bytes, action: str) -> list[tuple[str, str]]:
    parser = _ActionForms()
    parser.feed(page.decode("utf-8"))
    assert action in parser.forms, f"page did not offer {action!r}"
    return list(parser.forms[action])


def _with(
    fields: list[tuple[str, str]], **replacements: str
) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in fields
        if name not in replacements
    ] + list(replacements.items())


def _write_deck(root: Path, *, name: str = "Lesson deck") -> Path:
    path = root / "decks" / "lesson.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": name,
                    "source": "../vocabulary.json",
                    "intake_tag": "lesson-intake",
                    "include_tags": ["lesson-intake"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _promotion_project(
    tmp_path: Path,
) -> tuple[WorkbenchSession, Path, Path, tuple[str, ...]]:
    _stage(tmp_path, "table_exhaustive", filename="table.pdf")
    seed_prompts(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    _approve_examples(tmp_path, staging_path.name)
    records, meta = read_staging(staging_path)
    assigned = [replace(record, tags=["lesson-intake"]) for record in records]
    write_staging(staging_path, assigned, meta, force=True)
    deck_path = _write_deck(tmp_path)
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    return session, staging_path, deck_path, tuple(record.id for record in assigned)


def _add_url(session: WorkbenchSession, source: str = "table.pdf") -> str:
    return f"/{session.token}/source/{quote(source, safe='')}/add"


def _post(
    server: object,
    session: WorkbenchSession,
    fields: list[tuple[str, str]],
    source: str = "table.pdf",
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        _add_url(session, source),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(fields).encode("utf-8"),
    )


def _file_state(path: Path) -> tuple[bool, bytes]:
    return path.is_file(), path.read_bytes() if path.is_file() else b""


def test_owner_coverage_then_promotion_lands_the_previewed_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path, deck_path, expected_ids = _promotion_project(tmp_path)
    reason = "I compared all three source rows with the staged account."
    api = FakeJpdb()
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(api),
    )

    server, _thread = _running(session)
    try:
        status, _headers, coverage_page = _request(
            server, "GET", _add_url(session)
        )
        assert status == 200
        rendered = coverage_page.decode("utf-8")
        assert (
            "Did the extraction account for the source units it promised to cover?"
            in rendered
        )
        assert "I compared the source rows myself" in rendered
        assert "Ask Claude to check completeness" in rendered

        incomplete_owner = _with(
            _form(coverage_page, "owner-coverage"),
            reason=reason,
            compared="not-confirmed",
        )
        assert _post(server, session, incomplete_owner)[0] == 400
        owner = _with(
            _form(coverage_page, "owner-coverage"),
            reason=reason,
            compared="confirmed",
        )
        owner_status, _headers, _body = _post(server, session, owner)
        assert owner_status == 303

        preview_status, _headers, preview = _request(
            server, "GET", _add_url(session)
        )
        assert preview_status == 200
        previewed = preview.decode("utf-8")
        assert "Add 3 cards to your collection" in previewed
        assert "Lesson deck" in previewed

        promote_status, promote_headers, _body = _post(
            server,
            session,
            _form(preview, "promote"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert promote_status == 303
    assert promote_headers["location"].startswith(f"/{session.token}/?")
    assert "source=table.pdf" in promote_headers["location"]
    assert api.bodies, "the browser promotion must run the jpdb reading check"

    canonical = load_records(session.config.normalized_file)
    assert tuple(record.id for record in canonical) == expected_ids
    assert all(record.tags == ["lesson-intake"] for record in canonical)
    assert not staging_path.exists()

    archive_path = session.config.staging_dir / "done" / staging_path.name
    archived, archive_meta = read_staging(archive_path)
    assert tuple(record.id for record in archived) == expected_ids
    archived_approval = archive_meta["coverage"]["approval"]
    assert archived_approval["authority"] == "repository-owner"
    assert archived_approval["reason"] == reason
    _deck, deck_records = resolve_deck_records(deck_path)
    assert {record.id for record in deck_records} == set(expected_ids)


def test_paid_model_coverage_is_one_use_journaled_and_provenanced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path, _deck_path, _expected_ids = _promotion_project(tmp_path)
    sent: list[object] = []

    def review(
        *_args: object,
        capture: object = None,
        **kwargs: object,
    ) -> coverage.CoverageVerdict:
        sent.append(kwargs)
        assert callable(capture)
        capture({"content": [{"type": "text", "text": "exact paid reply"}]})
        return coverage.CoverageVerdict(
            True,
            "Every source row is accounted for.",
            str(kwargs["model"]),
            "f" * 64,
        )

    monkeypatch.setattr(coverage, "review_coverage", review)
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", _add_url(session))
        assert status == 200
        spent_by_mismatch = _form(page, "model-coverage")
        assert dict(spent_by_mismatch).get("coverage_action")
        assert _post(
            server,
            session,
            _with(spent_by_mismatch, model="not-the-rendered-model"),
        )[0] == 409
        assert _post(server, session, spent_by_mismatch)[0] == 409
        assert sent == []

        fresh_status, _headers, fresh_page = _request(
            server, "GET", _add_url(session)
        )
        assert fresh_status == 200
        submission = _form(fresh_page, "model-coverage")
        assert dict(submission).get("coverage_action")

        approved_status, _headers, _body = _post(server, session, submission)
        assert approved_status == 303
    finally:
        server.shutdown()
        server.server_close()

    assert len(sent) == 1
    coverage_entries = [
        entry
        for entry in operations.OperationJournal.load(
            session.config.operations_file
        ).operations.values()
        if entry.kind == "coverage"
    ]
    assert len(coverage_entries) == 1
    [entry] = coverage_entries
    assert entry.state == "committed"
    submitted = dict(submission)
    assert entry.request_fp == submitted["request_fingerprint"]
    assert entry.source_sha256 == submitted["source_fingerprint"]

    _records, meta = read_staging(staging_path)
    approval = meta["coverage"]["approval"]
    assert approval["authority"] == "model"
    assert approval["model"] == entry.model
    assert approval["prompt_fingerprint"] == submitted["prompt_fingerprint"]
    assert approval["reason"] == "Every source row is accounted for."

    # Once dispatch began, even a second failure while settling the journal
    # must keep conservative billing language and the operation handle. The
    # generic pre-dispatch handler's “Nothing was sent” suffix would be false.
    failure_root = tmp_path / "classifier-failure"
    failure_root.mkdir()
    failure_session, _staging, _deck, _ids = _promotion_project(failure_root)

    def provider_failure(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("provider failed after dispatch")

    def classifier_failure(*_args: object, **_kwargs: object) -> object:
        raise operations.OperationError("journal could not be settled")

    captured_failures: list[coverage_application.CoverageRunError] = []
    real_run_model_coverage = workbench_server.run_model_coverage

    def observe_run_failure(*args: object, **kwargs: object) -> object:
        try:
            return real_run_model_coverage(*args, **kwargs)  # type: ignore[arg-type]
        except coverage_application.CoverageRunError as exc:
            captured_failures.append(exc)
            raise

    failure_server, _thread = _running(failure_session)
    try:
        failure_status, _headers, failure_page = _request(
            failure_server, "GET", _add_url(failure_session)
        )
        assert failure_status == 200
        failure_submission = _form(failure_page, "model-coverage")
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(coverage, "review_coverage", provider_failure)
            failure_patch.setattr(
                coverage_application,
                "classify_dispatch_failure",
                classifier_failure,
            )
            failure_patch.setattr(
                workbench_server,
                "run_model_coverage",
                observe_run_failure,
            )
            refusal_status, _headers, refusal_body = _post(
                failure_server,
                failure_session,
                failure_submission,
            )
    finally:
        failure_server.shutdown()
        failure_server.server_close()

    refusal_text = refusal_body.decode("utf-8")
    assert refusal_status == 409
    assert "provider failed after dispatch" in refusal_text
    assert "journal could not be settled" in refusal_text
    assert "may have been billed" in refusal_text
    assert "Nothing was sent" not in refusal_text
    [unsettled] = [
        item
        for item in operations.OperationJournal.load(
            failure_session.config.operations_file
        ).operations.values()
        if item.kind == "coverage"
    ]
    assert unsettled.state == "dispatching"
    assert unsettled.operation_id in refusal_text
    [captured_failure] = captured_failures
    assert captured_failure.failure.outcome == "outcome_unknown"
    assert captured_failure.failure.money_may_have_been_spent is True


def test_a_changed_deck_makes_the_promotion_preview_stale_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path, _deck_path, _expected_ids = _promotion_project(tmp_path)
    approve_coverage_as_owner(
        session.config,
        plan_coverage(session.config, staging_path),
        reason="I compared all three source rows.",
    )
    archive_path = session.config.staging_dir / "done" / staging_path.name
    initial_api = FakeJpdb()
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(initial_api),
    )

    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", _add_url(session))
        assert status == 200
        stale_submission = _form(page, "promote")
        assert _post(
            server,
            session,
            _with(stale_submission, csrf="not-this-session"),
        )[0] == 403
        assert _post(
            server,
            session,
            [*stale_submission, ("unexpected", "field")],
        )[0] == 400

        # The owner name is part of what the preview says. The selector still
        # chooses the same exact rows, so only a fresh-plan comparison catches
        # that the checked page is no longer the page being acted on.
        _write_deck(tmp_path, name="Renamed lesson deck")
        paths = (
            session.config.normalized_file,
            staging_path,
            session.config.ledger_file,
            session.config.operations_file,
            archive_path,
        )
        before = tuple(_file_state(path) for path in paths)

        refused_status, _headers, refusal = _post(
            server, session, stale_submission
        )
        after = tuple(_file_state(path) for path in paths)

        _write_deck(tmp_path)
        race_status, _headers, race_page = _request(
            server, "GET", _add_url(session)
        )
        assert race_status == 200
        race_submission = _form(race_page, "promote")
        before_race = tuple(_file_state(path) for path in paths)

        class _DeckChangingJpdb(FakeJpdb):
            def __call__(self, *args: object, **kwargs: object) -> object:
                if not self.bodies:
                    _write_deck(tmp_path, name="Changed during reading checks")
                return super().__call__(*args, **kwargs)  # type: ignore[arg-type]

        api = _DeckChangingJpdb()
        monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
        monkeypatch.setattr(
            jpdb,
            "JpdbClient",
            lambda *_args, **_kwargs: client_for(api),
        )
        race_refused, _headers, race_body = _post(
            server, session, race_submission
        )
        after_race = tuple(_file_state(path) for path in paths)
    finally:
        server.shutdown()
        server.server_close()

    assert refused_status == 409
    assert b"changed after" in refusal or b"stale" in refusal
    assert after == before
    assert initial_api.bodies == []
    assert race_refused == 409
    assert b"changed after" in race_body or b"stale" in race_body
    assert api.bodies
    assert after_race == before_race

    # The archive's human note is durable metadata even though it does not
    # change row accounting. Bind its exact bytes to the rendered preview so
    # POST cannot rebuild the archive and erase a concurrent archive-only note.
    archive_root = tmp_path / "archive-note"
    archive_root.mkdir()
    _stage(archive_root, "table_exhaustive", filename="archive.pdf")
    seed_prompts(archive_root)
    archive_staging = archive_root / "staging" / "archive.pdf.yaml"
    _approve_examples(archive_root, archive_staging.name)
    archive_config = ProjectConfig.load(archive_root)
    (archive_root / "decks" / "all.yaml").write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": "Every reviewed card",
                    "source": "../vocabulary.json",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    archive_session = WorkbenchSession.open(archive_config)
    approve_coverage_as_owner(
        archive_config,
        plan_coverage(archive_config, archive_staging),
        reason="I compared all three source rows.",
    )
    archive_records, archive_meta = read_staging(archive_staging)
    archive_baseline = decide_promotion(
        archive_config,
        archive_staging,
        skip_reading_check=True,
    )
    assert archive_baseline.readings is not None
    archived_record = archive_baseline.readings.promoted[0]
    done_path = archive_config.staging_dir / "done" / archive_staging.name
    done_path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        done_path,
        [archived_record],
        promote.archive_meta(archive_meta, 1),
    )
    stable_api = FakeJpdb()
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(stable_api),
    )
    archive_server, _thread = _running(archive_session)
    try:
        archive_status, _headers, archive_page = _request(
            archive_server,
            "GET",
            _add_url(archive_session, "archive.pdf"),
        )
        assert archive_status == 200
        archive_submission = _form(archive_page, "promote")
        archived_records, archived_meta = read_staging(done_path)
        archived_meta["review_notes"] = (
            "A concurrent archive-only note.\n\n"
            + str(archived_meta["review_notes"])
        )
        write_staging(done_path, archived_records, archived_meta, force=True)
        archive_paths = (
            archive_config.normalized_file,
            archive_staging,
            archive_config.ledger_file,
            done_path,
        )
        before_archive_note = tuple(_file_state(path) for path in archive_paths)
        archive_refused, _headers, archive_body = _post(
            archive_server,
            archive_session,
            archive_submission,
            "archive.pdf",
        )
        after_archive_note = tuple(_file_state(path) for path in archive_paths)
    finally:
        archive_server.shutdown()
        archive_server.server_close()
    assert archive_refused == 409
    assert b"changed after" in archive_body or b"stale" in archive_body
    assert stable_api.bodies == []
    assert after_archive_note == before_archive_note

    # A second edit can land after POST's fresh replan, immediately before the
    # shared transaction. The execution CAS binds exact archive wire too; a
    # parsed-equivalent comment must survive and no canonical write may land.
    cas_api = FakeJpdb()
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(cas_api),
    )
    cas_server, _thread = _running(archive_session)
    try:
        cas_status, _headers, cas_page = _request(
            cas_server,
            "GET",
            _add_url(archive_session, "archive.pdf"),
        )
        assert cas_status == 200
        cas_submission = _form(cas_page, "promote")
        before_cas = tuple(
            _file_state(path)
            for path in (
                archive_config.normalized_file,
                archive_staging,
                archive_config.ledger_file,
            )
        )
        original_execute = workbench_server.execute_promotion
        cas_suffix = b"\n# changed at the transaction seam\n"

        def change_archive_before_execute(*args: object, **kwargs: object) -> object:
            done_path.write_bytes(done_path.read_bytes() + cas_suffix)
            return original_execute(*args, **kwargs)  # type: ignore[arg-type]

        with monkeypatch.context() as cas_patch:
            cas_patch.setattr(
                workbench_server,
                "execute_promotion",
                change_archive_before_execute,
            )
            cas_refused, _headers, cas_body = _post(
                cas_server,
                archive_session,
                cas_submission,
                "archive.pdf",
            )
        after_cas = tuple(
            _file_state(path)
            for path in (
                archive_config.normalized_file,
                archive_staging,
                archive_config.ledger_file,
            )
        )
    finally:
        cas_server.shutdown()
        cas_server.server_close()
    assert cas_refused == 409
    assert b"record-archive-stale" in cas_body
    assert cas_api.bodies
    assert after_cas == before_cas
    assert done_path.read_bytes().endswith(cas_suffix)

    # The consulted decision must use the same prelookup preservation census
    # as the final offline A plan. A transient record in an excluded static
    # source changes only stored_ids: ownership intentionally ignores it, but
    # reading/remint decisions must not execute from that B snapshot.
    stored_root = tmp_path / "stored-id-a-b-a"
    stored_root.mkdir()
    stored_session, stored_staging, _deck, stored_ids = _promotion_project(
        stored_root
    )
    approve_coverage_as_owner(
        stored_session.config,
        plan_coverage(stored_session.config, stored_staging),
        reason="I compared all three source rows.",
    )
    stored_records, _stored_meta = read_staging(stored_staging)
    external_source = stored_root / "static.json"
    save_records_json(external_source, [])
    source_a = external_source.read_bytes()
    save_records_json(external_source, [stored_records[0]])
    source_b = external_source.read_bytes()
    external_source.write_bytes(source_a)
    (stored_root / "decks" / "preservation.yaml").write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": "Excluded preservation source",
                    "source": "../static.json",
                    "exclude_ids": [stored_ids[0]],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stored_offline = decide_promotion(
        stored_session.config,
        stored_staging,
        skip_reading_check=None,
    )
    assert stored_ids[0] not in stored_offline.stored_ids
    stored_fingerprint = promotion_action.promotion_preview_fingerprint(
        stored_offline
    )
    original_decide = promotion_action.decide_promotion
    stored_calls = 0

    def transient_stored_id(*args: object, **kwargs: object) -> object:
        nonlocal stored_calls
        stored_calls += 1
        if stored_calls == 1:
            external_source.write_bytes(source_b)
            try:
                return original_decide(*args, **kwargs)  # type: ignore[arg-type]
            finally:
                external_source.write_bytes(source_a)
        return original_decide(*args, **kwargs)  # type: ignore[arg-type]

    stored_api = FakeJpdb()
    stored_paths = (
        stored_session.config.normalized_file,
        stored_staging,
        stored_session.config.ledger_file,
    )
    before_stored = tuple(_file_state(path) for path in stored_paths)
    with monkeypatch.context() as stored_patch:
        stored_patch.setattr(
            promotion_action, "decide_promotion", transient_stored_id
        )
        with pytest.raises(
            promotion_action.PromotionActionError,
            match="promotion-preview-stale",
        ):
            promotion_action.resolve_promotion_for_execution(
                stored_session.config,
                stored_offline,
                client_factory=lambda: client_for(stored_api),
                expected_preview_fingerprint=stored_fingerprint,
            )
    assert stored_calls == 2
    assert stored_api.bodies
    assert external_source.read_bytes() == source_a
    assert tuple(_file_state(path) for path in stored_paths) == before_stored

    # The same preservation source can change after the resolver's last read.
    # Execution rechecks declared IDs at the canonical commit seam, because an
    # excluded record is intentionally invisible to ownership but still owns
    # its stable Anki identity.
    commit_api = FakeJpdb()
    commit_decision = promotion_action.resolve_promotion_for_execution(
        stored_session.config,
        stored_offline,
        client_factory=lambda: client_for(commit_api),
        expected_preview_fingerprint=stored_fingerprint,
    )
    before_commit = tuple(_file_state(path) for path in stored_paths)
    external_source.write_bytes(source_b)
    with pytest.raises(JankiError, match="promotion-input-stale"):
        execute_promotion(stored_session.config, commit_decision)
    assert commit_api.bodies
    assert tuple(_file_state(path) for path in stored_paths) == before_commit
    assert external_source.read_bytes() == source_b

    # An offline deck refusal is provisional when jpdb may hold the row that
    # caused it. The first check reveals the still-pending coverage gate; only
    # after that decision does another check render the exact one-card landing
    # plan. Promotion needs a separate confirmation of that checked preview.
    collision_root = tmp_path / "post-reading"
    collision_root.mkdir()
    _stage(collision_root, "table_exhaustive", filename="lesson.pdf")
    seed_prompts(collision_root)
    collision_config = ProjectConfig.load(collision_root)
    collision_staging = collision_root / "staging" / "lesson.pdf.yaml"
    _approve_examples(collision_root, collision_staging.name)
    collision_records, collision_meta = read_staging(collision_staging)
    assigned_records = [
        replace(record, tags=["land"] if index == 0 else [])
        for index, record in enumerate(collision_records)
    ]
    write_staging(
        collision_staging,
        assigned_records,
        collision_meta,
        force=True,
    )
    def write_collision_deck(name: str = "Landing deck") -> None:
        (collision_root / "decks" / "all.yaml").write_text(
            yaml.safe_dump(
                {
                    "deck": {
                        "name": name,
                        "source": "../vocabulary.json",
                        "intake_tag": "land",
                        "include_tags": ["land"],
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    write_collision_deck()
    landing_record = assigned_records[0]
    parses: dict[str, list[object]] = {}
    senses: dict[tuple[int, int], dict[str, object]] = {}
    for index, record in enumerate(assigned_records):
        vid = 999_001 + index * 2
        sid = vid + 1
        dictionary_reading = record.reading if index == 0 else "ちがう"
        parses[record.expression] = [
            vid,
            sid,
            record.expression,
            dictionary_reading,
            [],
            100,
            [],
        ]
        senses[(vid, sid)] = {
            "reading": dictionary_reading,
            "alt_sids": [],
        }
    collision_api = FakeJpdb(
        parses,
        senses,
    )
    collision_session = WorkbenchSession.open(collision_config)
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(collision_api),
    )

    collision_server, _thread = _running(collision_session)
    try:
        preview_status, _headers, preview = _request(
            collision_server,
            "GET",
            _add_url(collision_session, "lesson.pdf"),
        )
        assert preview_status == 200
        assert b"A reading check can change this provisional result" in preview
        reading_check = _form(preview, "check-readings")
        before_precheck = (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        )
        assert _post(
            collision_server,
            collision_session,
            _with(reading_check, action="promote"),
            "lesson.pdf",
        )[0] == 409
        assert collision_api.bodies == []
        assert (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        ) == before_precheck
        coverage_status, _headers, coverage_page = _post(
            collision_server,
            collision_session,
            reading_check,
            "lesson.pdf",
        )
        assert coverage_status == 200, coverage_page.decode("utf-8")
        assert b"Coverage needs a decision" in coverage_page
        assert load_records(collision_config.normalized_file) == []
        owner = _with(
            _form(coverage_page, "owner-coverage"),
            compared="confirmed",
            reason="I compared the complete source with this account.",
        )
        assert _post(
            collision_server, collision_session, owner, "lesson.pdf"
        )[0] == 303

        next_status, _headers, next_page = _request(
            collision_server,
            "GET",
            _add_url(collision_session, "lesson.pdf"),
        )
        assert next_status == 200
        before_checked = (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        )
        checked_status, _headers, checked_page = _post(
            collision_server,
            collision_session,
            _form(next_page, "check-readings"),
            "lesson.pdf",
        )
        after_checked = (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        )
        assert checked_status == 200
        assert b"jpdb has now been checked" in checked_page
        assert b"Add 1 card to your collection" in checked_page
        assert after_checked == before_checked

        checked_submission = _form(checked_page, "promote-checked")
        write_collision_deck("Changed after the checked preview")
        before_checked_stale = (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        )
        checked_refusal, _headers, checked_refusal_body = _post(
            collision_server,
            collision_session,
            checked_submission,
            "lesson.pdf",
        )
        after_checked_stale = (
            collision_config.normalized_file.read_bytes(),
            collision_staging.read_bytes(),
        )
        assert checked_refusal == 409
        assert b"changed" in checked_refusal_body
        assert b"Nothing was promoted" in checked_refusal_body
        assert after_checked_stale == before_checked_stale

        write_collision_deck()
        fresh_status, _headers, fresh_page = _request(
            collision_server,
            "GET",
            _add_url(collision_session, "lesson.pdf"),
        )
        assert fresh_status == 200
        rechecked_status, _headers, rechecked_page = _post(
            collision_server,
            collision_session,
            _form(fresh_page, "check-readings"),
            "lesson.pdf",
        )
        assert rechecked_status == 200
        promoted_status, _headers, _body = _post(
            collision_server,
            collision_session,
            _form(rechecked_page, "promote-checked"),
            "lesson.pdf",
        )
    finally:
        collision_server.shutdown()
        collision_server.server_close()

    assert promoted_status == 303
    assert collision_api.bodies
    assert [record.id for record in load_records(collision_config.normalized_file)] == [
        landing_record.id
    ]
    remaining, _meta = read_staging(collision_staging)
    assert [record.id for record in remaining] == [
        record.id for record in assigned_records[1:]
    ]
    assert all(record.source.raw_fields.get("hold_reason") for record in remaining)
