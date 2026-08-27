"""Strict W5 finish forms and the one-use dictionary decision capability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from test_promote import FakeJpdb, client_for
from test_workbench import _request, _running
from test_workbench_promotion import _form, _promotion_project

from japanese_anki import jpdb, kanji
from japanese_anki.application.coverage import (
    approve_coverage_as_owner,
    plan_coverage,
)
from japanese_anki.application.enrichment import DictionaryEnrichmentDecision
from japanese_anki.application.finish import (
    FinishOwnerScope,
    FinishScope,
    records_revision_fingerprint,
)
from japanese_anki.application.kanji_addition import KanjiAdditionPlan
from japanese_anki.application.promotion import decide_promotion, execute_promotion
from japanese_anki.enrich import EnrichResult
from japanese_anki.io import RecordsRevision, load_records, save_records_json
from japanese_anki.kanji import KanjiInfo, KanjiStore, load_store, save_store
from japanese_anki.models import VocabularyRecord
from japanese_anki.workbench import server as workbench_server
from japanese_anki.workbench.finish import (
    AudioExamplesSubmission,
    AudioWordsSubmission,
    BuildSubmission,
    DictionaryAction,
    DictionaryActions,
    DictionaryCommitSubmission,
    DictionaryPlanSubmission,
    FinishFormError,
    KanjiAddSubmission,
    parse_finish_form,
)
from japanese_anki.workbench.render import render_finish

SCOPE = "a" * 64
PLAN = "b" * 64


def _body(**fields: str) -> bytes:
    return urlencode(fields).encode("utf-8")


def _decision() -> DictionaryEnrichmentDecision:
    root = Path("/exact/repository")
    output = root / "vocabulary.json"
    return DictionaryEnrichmentDecision(
        repository_root=root,
        output_path=output,
        ledger_path=root / "ledger.json",
        kanji_path=root / "kanji.json",
        record_ids=("word:\u540d:\u306a",),
        force_fields=(),
        output_revision=RecordsRevision(output, "[]"),
        result=EnrichResult(),
        fingerprint=PLAN,
    )


def _scope() -> FinishScope:
    root = Path("/exact/repository")
    record_id = "word:\u540d:\u306a"
    owner = FinishOwnerScope(
        stem="lesson",
        deck_path=root / "decks" / "lesson.yaml",
        record_ids=(record_id,),
    )
    return FinishScope(
        receipt_id="c" * 64,
        source_file="lesson.pdf",
        archive_path=root / "staging" / "done" / "lesson.yaml",
        archive_run_fingerprint="d" * 64,
        archive_start_index=0,
        review_run_id=None,
        canonical_path=root / "vocabulary.json",
        canonical_revision=records_revision_fingerprint(
            RecordsRevision(root / "vocabulary.json", "[]")
        ),
        deck_configuration_revision="f" * 64,
        record_ids=(record_id,),
        owner_stems=("lesson",),
        owner_groups=(owner,),
        fingerprint=SCOPE,
    )


def _promoted_finish(tmp_path: Path) -> tuple[object, str, tuple[str, ...]]:
    session, staging_path, _deck_path, promoted_ids = _promotion_project(tmp_path)
    approve_coverage_as_owner(
        session.config,
        plan_coverage(session.config, staging_path),
        reason="I compared the three rows for this finish-flow test.",
    )
    execution = execute_promotion(
        session.config,
        decide_promotion(
            session.config,
            staging_path,
            source="table.pdf",
            skip_reading_check=True,
        ),
    )
    assert execution.receipt_id is not None
    return (
        session,
        f"/{session.token}/finish/{execution.receipt_id}",
        promoted_ids,
    )


def _post_finish(
    server: object,
    finish_url: str,
    fields: list[tuple[str, str]],
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        finish_url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=urlencode(fields).encode("utf-8"),
    )


def _replace_field(
    fields: list[tuple[str, str]], name: str, value: str
) -> list[tuple[str, str]]:
    return [(key, value if key == name else old) for key, old in fields]


def test_finish_form_parser_accepts_only_the_six_frozen_submission_shapes() -> None:
    dictionary_plan = parse_finish_form(
        _body(action="dictionary-plan", csrf="csrf", scope_fingerprint=SCOPE)
    )
    dictionary_commit = parse_finish_form(
        _body(
            action="dictionary-commit",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            dictionary_action="capability",
            plan_fingerprint=PLAN,
        )
    )
    kanji_add = parse_finish_form(
        _body(
            action="kanji-add",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            plan_fingerprint=PLAN,
        )
    )
    audio_words = parse_finish_form(
        _body(
            action="audio-words",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            plan_fingerprint=PLAN,
        )
    )
    audio_examples = parse_finish_form(
        _body(
            action="audio-examples",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            plan_fingerprint=PLAN,
        )
    )
    build = parse_finish_form(
        _body(
            action="build",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            plan_fingerprint=PLAN,
        )
    )

    assert dictionary_plan == DictionaryPlanSubmission(
        action="dictionary-plan",
        csrf="csrf",
        scope_fingerprint=SCOPE,
    )
    assert dictionary_commit == DictionaryCommitSubmission(
        action="dictionary-commit",
        csrf="csrf",
        scope_fingerprint=SCOPE,
        dictionary_action="capability",
        plan_fingerprint=PLAN,
    )
    assert kanji_add == KanjiAddSubmission(
        action="kanji-add",
        csrf="csrf",
        scope_fingerprint=SCOPE,
        plan_fingerprint=PLAN,
    )
    assert audio_words == AudioWordsSubmission(
        action="audio-words",
        csrf="csrf",
        scope_fingerprint=SCOPE,
        plan_fingerprint=PLAN,
    )
    assert audio_examples == AudioExamplesSubmission(
        action="audio-examples",
        csrf="csrf",
        scope_fingerprint=SCOPE,
        plan_fingerprint=PLAN,
    )
    assert build == BuildSubmission(
        action="build",
        csrf="csrf",
        scope_fingerprint=SCOPE,
        plan_fingerprint=PLAN,
    )
    with pytest.raises(FrozenInstanceError):
        dictionary_plan.csrf = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        audio_words.csrf = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        audio_examples.csrf = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        build.csrf = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("action", ["audio-words", "audio-examples"])
@pytest.mark.parametrize("forbidden", ["ids", "provider", "force", "prune"])
def test_audio_finish_forms_reject_authority_fields(
    action: str, forbidden: str
) -> None:
    fields = {
        "action": action,
        "csrf": "csrf",
        "scope_fingerprint": SCOPE,
        "plan_fingerprint": PLAN,
        forbidden: "attacker-chosen",
    }

    with pytest.raises(FinishFormError, match="not one this page offered"):
        parse_finish_form(_body(**fields))


@pytest.mark.parametrize("action", ["audio-words", "audio-examples"])
@pytest.mark.parametrize("repeated", ["action", "csrf", "scope_fingerprint", "plan_fingerprint"])
def test_audio_finish_forms_reject_duplicate_fields(
    action: str, repeated: str
) -> None:
    fields = [
        ("action", action),
        ("csrf", "csrf"),
        ("scope_fingerprint", SCOPE),
        ("plan_fingerprint", PLAN),
    ]
    duplicate_value = action if repeated == "action" else dict(fields)[repeated]
    fields.append((repeated, duplicate_value))

    with pytest.raises(FinishFormError, match="repeats a field"):
        parse_finish_form(urlencode(fields).encode("utf-8"))


@pytest.mark.parametrize("action", ["audio-words", "audio-examples"])
@pytest.mark.parametrize(
    "fields",
    [
        {"csrf": "csrf", "scope_fingerprint": SCOPE},
        {
            "csrf": "csrf",
            "scope_fingerprint": SCOPE.upper(),
            "plan_fingerprint": PLAN,
        },
        {
            "csrf": "csrf",
            "scope_fingerprint": SCOPE,
            "plan_fingerprint": PLAN[:-1],
        },
    ],
)
def test_audio_finish_forms_reject_missing_or_malformed_fingerprints(
    action: str, fields: dict[str, str]
) -> None:
    with pytest.raises(FinishFormError):
        parse_finish_form(_body(action=action, **fields))


@pytest.mark.parametrize(
    "body",
    [
        urlencode(
            [
                ("action", "dictionary-plan"),
                ("action", "dictionary-plan"),
                ("csrf", "csrf"),
                ("scope_fingerprint", SCOPE),
            ]
        ).encode("utf-8"),
        _body(action="dictionary-plan", csrf="csrf"),
        _body(
            action="dictionary-plan",
            csrf="csrf",
            scope_fingerprint=SCOPE,
            unexpected="value",
        ),
        _body(action="not-offered", csrf="csrf", scope_fingerprint=SCOPE),
    ],
)
def test_finish_form_parser_rejects_malformed_or_nonexact_forms(body: bytes) -> None:
    with pytest.raises(FinishFormError):
        parse_finish_form(body)


def test_finish_form_parser_rejects_invalid_utf8_at_the_decode_boundary() -> None:
    with pytest.raises(FinishFormError, match="malformed"):
        parse_finish_form(
            b"action=dictionary-plan&csrf=\xff&scope_fingerprint=" + SCOPE.encode("ascii")
        )
    with pytest.raises(FinishFormError, match="malformed"):
        parse_finish_form(
            b"action=dictionary-plan&csrf=%FF&scope_fingerprint="
            + SCOPE.encode("ascii")
        )


@pytest.mark.parametrize(
    ("fields", "invalid_name"),
    [
        (
            {
                "action": "dictionary-plan",
                "csrf": "csrf",
                "scope_fingerprint": SCOPE.upper(),
            },
            "scope fingerprint",
        ),
        (
            {
                "action": "dictionary-commit",
                "csrf": "csrf",
                "scope_fingerprint": SCOPE[:-1],
                "dictionary_action": "capability",
                "plan_fingerprint": PLAN,
            },
            "scope fingerprint",
        ),
        (
            {
                "action": "dictionary-commit",
                "csrf": "csrf",
                "scope_fingerprint": SCOPE,
                "dictionary_action": "capability",
                "plan_fingerprint": PLAN.upper(),
            },
            "plan fingerprint",
        ),
        (
            {
                "action": "kanji-add",
                "csrf": "csrf",
                "scope_fingerprint": "g" * 64,
                "plan_fingerprint": PLAN,
            },
            "scope fingerprint",
        ),
        (
            {
                "action": "kanji-add",
                "csrf": "csrf",
                "scope_fingerprint": SCOPE,
                "plan_fingerprint": "0" * 63,
            },
            "plan fingerprint",
        ),
    ],
)
def test_every_finish_fingerprint_is_lowercase_sha256(
    fields: dict[str, str], invalid_name: str
) -> None:
    with pytest.raises(FinishFormError, match=invalid_name):
        parse_finish_form(_body(**fields))


def test_dictionary_actions_bind_the_exact_decision_and_are_one_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = iter(("same", "same", "fresh"))
    monkeypatch.setattr(
        "japanese_anki.workbench.finish.secrets.token_urlsafe",
        lambda _size: next(generated),
    )
    actions = DictionaryActions()
    decision = _decision()
    scope = _scope()

    class LockCheckedPending(dict[str, object]):
        def _inside_lock(self) -> None:
            assert actions._lock.locked()  # type: ignore[attr-defined]

        def __len__(self) -> int:
            self._inside_lock()
            return super().__len__()

        def __contains__(self, key: object) -> bool:
            self._inside_lock()
            return super().__contains__(key)

        def __setitem__(self, key: str, value: object) -> None:
            self._inside_lock()
            super().__setitem__(key, value)

        def pop(self, key: str, default: object = None) -> object:
            self._inside_lock()
            return super().pop(key, default)

    actions._pending = LockCheckedPending()  # type: ignore[assignment]

    first = actions.issue(scope, decision)
    second = actions.issue(scope, decision)

    assert first == "same"
    assert second == "fresh", "a pending token collision must be retried"
    action = actions.consume(first)
    assert action is not None
    assert action.receipt_id == scope.receipt_id
    assert action.scope_fingerprint == SCOPE
    assert action.decision is decision
    assert actions.consume(first) is None


def test_dictionary_actions_evict_the_oldest_unused_page_when_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = iter(f"token-{number}" for number in range(9))
    monkeypatch.setattr(
        "japanese_anki.workbench.finish.secrets.token_urlsafe",
        lambda _size: next(tokens),
    )
    actions = DictionaryActions()
    decision = _decision()
    scope = _scope()

    issued = [actions.issue(scope, decision) for _number in range(9)]

    assert actions.consume(issued[0]) is None
    assert actions.consume(issued[1]) is not None
    assert actions.consume(issued[-1]) is not None


def test_dictionary_action_issue_refuses_scope_path_or_force_widening() -> None:
    actions = DictionaryActions()
    scope = _scope()
    decision = _decision()
    variants = (
        replace(decision, record_ids=("word:outside:outside",)),
        replace(decision, output_path=Path("/other/vocabulary.json")),
        replace(
            decision,
            output_revision=RecordsRevision(Path("/other/vocabulary.json"), "[]"),
        ),
        replace(
            decision,
            output_revision=RecordsRevision(decision.output_path, "[ ]"),
        ),
        replace(decision, force_fields=("pitch_accent",)),
    )

    for changed in variants:
        with pytest.raises(FinishFormError, match="exact finish scope"):
            actions.issue(scope, changed)


def test_invalid_encoded_finish_receipt_does_not_consume_dictionary_action(
    tmp_path: Path,
) -> None:
    session, _finish_url, _promoted_ids = _promoted_finish(tmp_path)
    action = DictionaryAction(
        receipt_id="c" * 64,
        scope_fingerprint=SCOPE,
        decision=_decision(),
    )
    session.dictionary_actions._pending["still-live"] = action  # type: ignore[attr-defined]
    invalid_url = f"/{session.token}/finish/%E9%A3%9F"
    body = _body(
        action="dictionary-commit",
        csrf=session.csrf_token,
        scope_fingerprint=SCOPE,
        dictionary_action="still-live",
        plan_fingerprint=PLAN,
    )
    server, _thread = _running(session)
    try:
        status, _headers, _response = _request(
            server,
            "POST",
            invalid_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 404
    assert session.consume_dictionary_action("still-live") is action


def test_finish_render_escapes_scope_warning_and_dictionary_diff() -> None:
    scope = replace(
        _scope(),
        source_file='<script id="source">bad()</script>',
        owner_stems=("<owner>",),
        owner_groups=(replace(_scope().owner_groups[0], stem="<owner>"),),
    )
    hostile_record = VocabularyRecord(
        id=scope.record_ids[0],
        expression="名",
        reading="な",
        meanings=["name"],
    )
    decision = replace(
        _decision(),
        result=EnrichResult(
            records=[hostile_record],
            changes={
                hostile_record.id: {
                    "part_of_speech": (None, '<img src=x onerror="bad()">')
                }
            },
            warnings=['<script id="warning">bad()</script>'],
            looked_up=1,
        ),
    )
    plan = KanjiAdditionPlan(
        project_root=Path("/exact/repository"),
        canonical_path=scope.canonical_path,
        canonical_fingerprint="1" * 64,
        kanji_path=Path("/exact/repository/kanji.json"),
        record_ids=scope.record_ids,
        characters=("名",),
        already_known=(),
        to_fetch=("名",),
        refresh=False,
        targeted=True,
        fingerprint="2" * 64,
    )

    page = render_finish(
        scope,
        plan,
        token="token",
        csrf="csrf",
        dictionary_decision=decision,
        dictionary_action="action",
    )

    assert '<script id="source">' not in page
    assert '<script id="warning">' not in page
    assert '<img src=x onerror="bad()">' not in page
    assert "<owner>" not in page
    assert "&lt;script id=&quot;source&quot;&gt;bad()&lt;/script&gt;" in page
    assert "&lt;script id=&quot;warning&quot;&gt;bad()&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=&quot;bad()&quot;&gt;" in page
    assert "&lt;owner&gt;" in page


def test_finish_page_previews_exact_cards_and_routes_the_bound_full_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path, deck_path, promoted_ids = _promotion_project(tmp_path)
    older = VocabularyRecord(
        id="word:older:older",
        expression="OLDER-SHOULD-NOT-APPEAR",
        reading="older",
        meanings=["an older card in the complete deck"],
        tags=["lesson-intake"],
    )
    save_records_json(session.config.normalized_file, [older])
    approve_coverage_as_owner(
        session.config,
        plan_coverage(session.config, staging_path),
        reason="I compared the three rows for this exact-preview test.",
    )
    promotion = execute_promotion(
        session.config,
        decide_promotion(
            session.config,
            staging_path,
            source="table.pdf",
            skip_reading_check=True,
        ),
    )
    assert promotion.receipt_id is not None
    finish_url = f"/{session.token}/finish/{promotion.receipt_id}"
    calls: list[tuple[str, str, str]] = []

    def execute(
        _config: object,
        receipt_id: str,
        *,
        expected_scope_fingerprint: str,
        expected_plan_fingerprint: str,
    ) -> object:
        calls.append(
            (
                receipt_id,
                expected_scope_fingerprint,
                expected_plan_fingerprint,
            )
        )
        if len(calls) == 1:
            return SimpleNamespace(
                state="complete",
                package_paths=(session.config.dist_dir / "lesson.apkg",),
                build_error=None,
                ledger_error=None,
            )
        return SimpleNamespace(
            state="build_and_ledger_incomplete",
            package_paths=(session.config.dist_dir / "lesson.apkg",),
            build_error="fixture later-deck refusal",
            ledger_error="fixture export-history refusal",
        )

    monkeypatch.setattr(
        workbench_server.build_application,
        "execute_finish_build",
        execute,
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        assert b"Local network provider" in page
        assert b"voicevox" in page
        assert b"final package contains each complete current study deck" in page
        assert _form(page, "audio-words")
        assert _form(page, "audio-examples")
        build_form = _form(page, "build")
        assert b"Sync Anki before importing" in page
        assert b"Merge Notetypes" in page
        assert b"this batch's audio: words" in page
        assert b"examples" in page

        preview_status, _headers, preview = _request(
            server,
            "GET",
            f"{finish_url}?preview={deck_path.stem}",
        )
        assert preview_status == 200
        for record_id in promoted_ids:
            record = next(
                item
                for item in load_records(session.config.normalized_file)
                if item.id == record_id
            )
            assert record.expression.encode("utf-8") in preview
        assert older.expression.encode("utf-8") not in preview
        assert b"4 note(s) in the full deck" in preview

        built, build_headers, _body = _post_finish(
            server,
            finish_url,
            build_form,
        )
        assert built == 303
        assert build_headers["location"].endswith("?build=complete")
        final_status, _headers, final_page = _request(
            server,
            "GET",
            build_headers["location"],
        )
        partial_status, _headers, partial_body = _post_finish(
            server,
            finish_url,
            build_form,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert final_status == 200
    assert b"Built every receipted study deck" in final_page
    assert partial_status == 409
    assert b"Package(s) written" in partial_body
    assert b"lesson.apkg" in partial_body
    assert b"fixture later-deck refusal" in partial_body
    assert b"fixture export-history refusal" in partial_body
    assert len(calls) == 2
    submitted = dict(build_form)
    for call in calls:
        assert call == (
            finish_url.rsplit("/", 1)[-1],
            submitted["scope_fingerprint"],
            submitted["plan_fingerprint"],
        )


def test_finish_audio_posts_bind_the_scope_and_report_partial_transaction_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, promoted_ids = _promoted_finish(tmp_path)
    calls: list[tuple[tuple[str, ...], bool, bool, str, bool, bool]] = []

    def execute(
        _config: object,
        record_ids: tuple[str, ...],
        *,
        words: bool,
        examples: bool,
        expected_fingerprint: str,
        force: bool,
        prune: bool,
    ) -> object:
        calls.append(
            (
                record_ids,
                words,
                examples,
                expected_fingerprint,
                force,
                prune,
            )
        )
        plan = workbench_server.audio_application.plan_targeted_audio(
            _config,
            record_ids,
            words=words,
            examples=examples,
            force=force,
        )
        if words:
            plan = replace(
                plan,
                word_counts=replace(
                    plan.word_counts,
                    total=3,
                    current=0,
                    recoverable=2,
                    provider_required=1,
                ),
            )
            return SimpleNamespace(succeeded=True, plan=plan, file_count=1)
        if len(calls) in {3, 4}:
            assert plan.example_provider is not None
            recoverable = plan.example_counts.total if len(calls) == 4 else 0
            plan = replace(
                plan,
                example_counts=replace(
                    plan.example_counts,
                    current=0,
                    recoverable=recoverable,
                    provider_required=plan.example_counts.total - recoverable,
                ),
                example_provider=replace(
                    plan.example_provider,
                    name="openai",
                    access="paid-network",
                ),
            )
        return SimpleNamespace(
            succeeded=False,
            plan=plan,
            stopped_by="fixture provider stopped",
            prune_error=None,
            ledger_error="fixture ledger refusal",
            record_references_written=True,
            media_published=True,
            ledger_committed=False,
            pending_recovery=True,
        )

    monkeypatch.setattr(
        workbench_server.audio_application,
        "execute_targeted_audio",
        execute,
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        word_form = _form(page, "audio-words")
        example_form = _form(page, "audio-examples")

        stale, _headers, stale_body = _post_finish(
            server,
            finish_url,
            _replace_field(word_form, "scope_fingerprint", "e" * 64),
        )
        assert calls == []

        word_status, word_headers, _body = _post_finish(
            server,
            finish_url,
            word_form,
        )
        banner_status, _headers, banner = _request(
            server,
            "GET",
            word_headers["location"],
        )
        example_status, _headers, example_body = _post_finish(
            server,
            finish_url,
            example_form,
        )
        paid_status, _headers, paid_body = _post_finish(
            server,
            finish_url,
            example_form,
        )
        recovery_status, _headers, recovery_body = _post_finish(
            server,
            finish_url,
            example_form,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert stale == 409
    assert b"exact finish scope changed" in stale_body
    assert word_status == 303
    assert word_headers["location"].endswith(
        "?audio=words&created=1&recovered=2&current=0"
    )
    assert banner_status == 200
    assert b"Finished word audio: created 1 clip(s), recovered 2" in banner
    assert b"kept 0 already-current clip(s)" in banner
    assert example_status == 409
    assert b"required no paid provider call" in example_body
    assert b"may have been billed" not in example_body
    assert b"fixture provider stopped" in example_body
    assert b"fixture ledger refusal" in example_body
    assert b"wrote its record references" in example_body
    assert b"published canonical media" in example_body
    assert b"did not commit canonical audio ledger" in example_body
    assert b"pending_audio" in example_body
    assert paid_status == 409
    assert b"paid provider call may have been billed" in paid_body
    assert b"required no paid provider call" not in paid_body
    assert recovery_status == 409
    assert b"required no paid provider call" in recovery_body
    assert b"may have been billed" not in recovery_body
    assert calls == [
        (
            promoted_ids,
            True,
            False,
            dict(word_form)["plan_fingerprint"],
            False,
            False,
        ),
        (
            promoted_ids,
            False,
            True,
            dict(example_form)["plan_fingerprint"],
            False,
            False,
        ),
        (
            promoted_ids,
            False,
            True,
            dict(example_form)["plan_fingerprint"],
            False,
            False,
        ),
        (
            promoted_ids,
            False,
            True,
            dict(example_form)["plan_fingerprint"],
            False,
            False,
        ),
    ]


def test_wrong_csrf_refuses_before_jpdb_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    constructed: list[object] = []
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: constructed.append(object()),
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _replace_field(_form(page, "dictionary-plan"), "csrf", "wrong"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 403
    assert b"did not come from this workbench session" in body
    assert constructed == []


def test_stale_finish_scope_refuses_before_jpdb_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    constructed: list[object] = []
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: constructed.append(object()),
    )

    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        records = load_records(session.config.normalized_file)
        save_records_json(
            session.config.normalized_file,
            [
                *records,
                VocabularyRecord(
                    id="word:scope-changed:scope-changed",
                    expression="scope-changed",
                    reading="scope-changed",
                    meanings=["scope changed"],
                ),
            ],
        )
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(page, "dictionary-plan"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"Nothing was looked up" in body
    assert constructed == []


def test_repository_change_during_jpdb_refuses_without_capability_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, promoted_ids = _promoted_finish(tmp_path)
    dictionary = FakeJpdb(
        {
            "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
            "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
            "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
        }
    )
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(dictionary),
    )
    original_plan = workbench_server.plan_dictionary_enrichment

    def plan_then_change(*args: object, **kwargs: object) -> object:
        decision = original_plan(*args, **kwargs)  # type: ignore[arg-type]
        records = load_records(session.config.normalized_file)
        save_records_json(
            session.config.normalized_file,
            [replace(records[0], usage_notes="concurrent edit"), *records[1:]],
        )
        return decision

    monkeypatch.setattr(
        workbench_server,
        "plan_dictionary_enrichment",
        plan_then_change,
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(page, "dictionary-plan"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"changed during the dictionary lookup" in body
    assert dictionary.bodies
    assert session.dictionary_actions._pending == {}  # type: ignore[attr-defined]
    current = {record.id: record for record in load_records(session.config.normalized_file)}
    assert all(current[record_id].frequency_rank is None for record_id in promoted_ids)


def test_stale_kanji_plan_refuses_before_any_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    fetched: list[str] = []
    monkeypatch.setattr(
        kanji,
        "fetch_kanji",
        lambda character: fetched.append(character) or KanjiInfo(character=character),
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        save_store(
            session.config.kanji_file,
            KanjiStore(entries={"走": KanjiInfo(character="走")}),
        )
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(page, "kanji-add"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"missing kanji changed" in body
    assert fetched == []


def test_dictionary_ledger_partial_reports_records_saved_and_attribution_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    dictionary = FakeJpdb(
        {
            "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
            "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
            "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
        }
    )
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(dictionary),
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        preview_status, _headers, preview = _post_finish(
            server, finish_url, _form(page, "dictionary-plan")
        )
        assert preview_status == 200
        monkeypatch.setattr(
            workbench_server,
            "commit_dictionary_enrichment",
            lambda *_args, **_kwargs: SimpleNamespace(
                state="committed_ledger_incomplete",
                output_path=session.config.normalized_file,
                ledger_error="ledger is read-only",
            ),
        )
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(preview, "dictionary-commit"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"records are saved" in body
    assert b"ledger attribution did not" in body
    assert b"ledger is read-only" in body


def test_stale_deck_ownership_before_dictionary_commit_prevents_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    dictionary = FakeJpdb(
        {
            "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
            "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
            "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
        }
    )
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(dictionary),
    )
    commits: list[object] = []
    original_commit = workbench_server.commit_dictionary_enrichment

    def observe_commit(*args: object, **kwargs: object) -> object:
        commits.append(object())
        return original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        workbench_server,
        "commit_dictionary_enrichment",
        observe_commit,
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        preview_status, _headers, preview = _post_finish(
            server, finish_url, _form(page, "dictionary-plan")
        )
        assert preview_status == 200
        before = session.config.normalized_file.read_bytes()
        [deck_path] = session.config.deck_dir.glob("*.yaml")
        deck_path.write_text(
            deck_path.read_text(encoding="utf-8").replace(
                "lesson-intake", "different-owner"
            ),
            encoding="utf-8",
        )
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(preview, "dictionary-commit"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"owner changed" in body or b"ownership" in body
    assert commits == []
    assert session.config.normalized_file.read_bytes() == before


def test_dictionary_capability_replay_cannot_write_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)
    dictionary = FakeJpdb(
        {
            "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
            "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
            "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
        }
    )
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(dictionary),
    )
    commits = 0
    original_commit = workbench_server.commit_dictionary_enrichment

    def count_commit(*args: object, **kwargs: object) -> object:
        nonlocal commits
        commits += 1
        return original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        workbench_server,
        "commit_dictionary_enrichment",
        count_commit,
    )
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        preview_status, _headers, preview = _post_finish(
            server, finish_url, _form(page, "dictionary-plan")
        )
        assert preview_status == 200
        commit_form = _form(preview, "dictionary-commit")
        first, _headers, _body = _post_finish(server, finish_url, commit_form)
        assert first == 303
        after_first = session.config.normalized_file.read_bytes()
        replay, _headers, body = _post_finish(server, finish_url, commit_form)
    finally:
        server.shutdown()
        server.server_close()

    assert replay == 409
    assert b"already been used or expired" in body
    assert commits == 1
    assert session.config.normalized_file.read_bytes() == after_first


def test_mixed_kanji_result_reports_exact_saved_and_failed_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, finish_url, _promoted_ids = _promoted_finish(tmp_path)

    def fetch(character: str) -> KanjiInfo:
        if character == "食":
            raise kanji.KanjiError("fixture refusal")
        return KanjiInfo(character=character)

    monkeypatch.setattr(kanji, "fetch_kanji", fetch)
    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        refused, _headers, body = _post_finish(
            server,
            finish_url,
            _form(page, "kanji-add"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert refused == 409
    assert b"Saved 2 new kanji lookup(s)" in body
    assert b"1 lookup(s) failed" in body
    assert b"\xe9\xa3\x9f: fixture refusal" in body
    assert set(load_store(session.config.kanji_file).entries) == {"走", "飲"}


def test_finish_page_keeps_dictionary_and_kanji_on_the_receipted_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, staging_path, _deck_path, promoted_ids = _promotion_project(tmp_path)
    outside = VocabularyRecord(
        id="word:外:そと",
        expression="外",
        reading="そと",
        meanings=["outside"],
        tags=["lesson-intake"],
    )
    save_records_json(session.config.normalized_file, [outside])
    approve_coverage_as_owner(
        session.config,
        plan_coverage(session.config, staging_path),
        reason="I compared the three rows for this finish-flow test.",
    )
    execution = execute_promotion(
        session.config,
        decide_promotion(
            session.config,
            staging_path,
            source="table.pdf",
            skip_reading_check=True,
        ),
    )
    assert execution.receipt_id is not None
    finish_url = f"/{session.token}/finish/{execution.receipt_id}"

    dictionary = FakeJpdb(
        {
            "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
            "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
            "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
        }
    )
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "test-key")
    monkeypatch.setattr(
        jpdb,
        "JpdbClient",
        lambda *_args, **_kwargs: client_for(dictionary),
    )
    fetched: list[str] = []

    def fetch(character: str) -> KanjiInfo:
        fetched.append(character)
        return KanjiInfo(character=character)

    monkeypatch.setattr(kanji, "fetch_kanji", fetch)

    def post(
        server: object, fields: list[tuple[str, str]]
    ) -> tuple[int, dict[str, str], bytes]:
        return _request(
            server,
            "POST",
            finish_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urlencode(fields).encode("utf-8"),
        )

    server, _thread = _running(session)
    try:
        status, _headers, page = _request(server, "GET", finish_url)
        assert status == 200
        assert dictionary.bodies == [], "opening the page is local"
        assert b"Networked" in page
        assert b"no model call" in page

        plan_status, _headers, dictionary_preview = post(
            server,
            _form(page, "dictionary-plan"),
        )
        assert plan_status == 200
        assert b"Save these dictionary facts" in dictionary_preview
        before_commit = {
            record.id: record for record in load_records(session.config.normalized_file)
        }
        assert before_commit[outside.id] == outside
        assert not any(
            before_commit[record_id].frequency_rank for record_id in promoted_ids
        ), "the networked preview must not save before confirmation"

        commit_status, commit_headers, _body = post(
            server,
            _form(dictionary_preview, "dictionary-commit"),
        )
        assert commit_status == 303
        assert "dictionary=committed" in commit_headers["location"]
        after_dictionary = {
            record.id: record for record in load_records(session.config.normalized_file)
        }
        assert after_dictionary[outside.id] == outside
        assert all(after_dictionary[record_id].frequency_rank for record_id in promoted_ids)
        parsed = [body["text"][0] for body in dictionary.bodies if "text" in body]
        assert parsed == ["走る", "食べる", "飲む"]

        refreshed_status, _headers, refreshed = _request(
            server,
            "GET",
            commit_headers["location"],
        )
        assert refreshed_status == 200
        kanji_status, kanji_headers, _body = post(
            server,
            _form(refreshed, "kanji-add"),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert kanji_status == 303
    assert "kanji=saved" in kanji_headers["location"]
    assert fetched == ["走", "食", "飲"]
    assert set(load_store(session.config.kanji_file).entries) == {"走", "食", "飲"}
    assert "外" not in load_store(session.config.kanji_file).entries
