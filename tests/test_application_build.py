"""Exact W5 finish previews and owner-deck build transactions."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from japanese_anki import ledger
from japanese_anki.application import build as build_application
from japanese_anki.application.finish import (
    FinishOwnerScope,
    FinishScope,
    records_revision_fingerprint,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import (
    BuildResult,
    resolve_card_types,
    resolve_deck_records,
)
from japanese_anki.io import records_revision
from japanese_anki.models import VocabularyRecord


def _record(expression: str, *, tag: str, meaning: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{expression}",
        expression=expression,
        reading=expression,
        meanings=[meaning],
        tags=[tag],
    )


def _deck(
    path: Path,
    *,
    tag: str,
    output: str,
    cards: dict[str, bool],
    override: tuple[str, str] | None = None,
    name: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "deck": {
            "name": f"{tag.title()} deck" if name is None else name,
            "source": "../vocabulary.json",
            "intake_tag": tag,
            "include_tags": [tag],
            "output": output,
            "cards": cards,
        }
    }
    if override is not None:
        record_id, meaning = override
        payload["notes"] = [{"id": record_id, "meanings": [meaning]}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _BuildFixture:
    config: ProjectConfig
    scope: FinishScope
    beta_receipt_ids: tuple[str, ...]
    alpha_receipt_id: str
    beta_extra_id: str
    alpha_extra_id: str
    third_id: str


def _fixture(tmp_path: Path) -> _BuildFixture:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'dist_dir = "packages"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    beta_second = _record("beta-second", tag="beta", meaning="source second")
    alpha_receipt = _record("alpha-new", tag="alpha", meaning="alpha new")
    beta_first = _record("beta-first", tag="beta", meaning="source first")
    beta_extra = _record("beta-extra", tag="beta", meaning="existing beta")
    alpha_extra = _record("alpha-extra", tag="alpha", meaning="existing alpha")
    third = _record("third-only", tag="third", meaning="not an owner")
    records = [
        beta_second,
        alpha_receipt,
        beta_first,
        beta_extra,
        alpha_extra,
        third,
    ]
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    _deck(
        tmp_path / "decks" / "alpha.yaml",
        tag="alpha",
        output="alpha-finish.apkg",
        cards={"recognition": True, "production": False, "reading": True},
        name="",
    )
    _deck(
        tmp_path / "decks" / "beta.yaml",
        tag="beta",
        output="beta-finish.apkg",
        cards={"recognition": True, "production": True, "reading": False},
        override=(beta_second.id, "deck-resolved second"),
    )
    _deck(
        tmp_path / "decks" / "third.yaml",
        tag="third",
        output="third.apkg",
        cards={"recognition": True, "production": False, "reading": False},
    )
    config = ProjectConfig.load(tmp_path)
    canonical_revision = records_revision_fingerprint(
        records_revision(config.normalized_file)
    )
    _paths, deck_revision = build_application._deck_configuration_snapshot(config)
    beta_receipt_ids = (beta_second.id, beta_first.id)
    scope = FinishScope(
        receipt_id="a" * 64,
        source_file="lesson.pdf",
        archive_path=(config.staging_dir / "done" / "lesson.yaml").resolve(),
        archive_run_fingerprint="b" * 64,
        archive_start_index=0,
        review_run_id=None,
        canonical_path=config.normalized_file.resolve(),
        canonical_revision=canonical_revision,
        deck_configuration_revision=deck_revision,
        record_ids=(beta_second.id, alpha_receipt.id, beta_first.id),
        owner_stems=("beta", "alpha", "beta"),
        owner_groups=(
            FinishOwnerScope(
                stem="beta",
                deck_path=(config.deck_dir / "beta.yaml").resolve(),
                record_ids=beta_receipt_ids,
            ),
            FinishOwnerScope(
                stem="alpha",
                deck_path=(config.deck_dir / "alpha.yaml").resolve(),
                record_ids=(alpha_receipt.id,),
            ),
        ),
        fingerprint="c" * 64,
    )
    return _BuildFixture(
        config=config,
        scope=scope,
        beta_receipt_ids=beta_receipt_ids,
        alpha_receipt_id=alpha_receipt.id,
        beta_extra_id=beta_extra.id,
        alpha_extra_id=alpha_extra.id,
        third_id=third.id,
    )


def _fake_builder(calls: list[tuple[str, tuple[str, ...], object]]):
    def build(
        deck_path: Path,
        config: ProjectConfig,
        output_path: Path | None = None,
        include_ids: object = None,
    ) -> BuildResult:
        assert output_path is not None
        deck_config, records = resolve_deck_records(deck_path)
        record_ids = tuple(record.id for record in records)
        calls.append((deck_path.stem, record_ids, include_ids))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"package:{deck_path.stem}".encode())
        return BuildResult(
            output_path=output_path,
            deck_name=str(deck_config["name"]),
            note_count=len(records),
            card_types=tuple(resolve_card_types(deck_config, config)),
            media_count=0,
            record_ids=record_ids,
        )

    return build


def test_plan_keeps_receipt_preview_order_but_binds_complete_owner_decks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    plan = build_application.plan_finish_build(fixture.config, fixture.scope)

    assert tuple(deck.stem for deck in plan.decks) == ("beta", "alpha")
    beta, alpha = plan.decks
    assert tuple(record.id for record in beta.preview_records) == (
        fixture.beta_receipt_ids
    )
    assert beta.preview_records[0].meanings == ["deck-resolved second"]
    assert fixture.beta_extra_id in {record.id for record in beta.records}
    assert fixture.alpha_extra_id in {record.id for record in alpha.records}
    assert fixture.third_id not in {
        record.id for deck in plan.decks for record in deck.records
    }
    assert beta.card_types == ("recognition", "production")
    assert alpha.card_types == ("recognition", "reading")
    assert alpha.name == ""
    assert beta.output_path == (fixture.config.dist_dir / "beta-finish.apkg")
    assert alpha.output_path == (fixture.config.dist_dir / "alpha-finish.apkg")

    real_resolve = build_application.resolve_deck_records

    def resolve_with_changed_full_member(deck_path: Path):
        deck_config, records = real_resolve(deck_path)
        if deck_path.stem == "beta":
            records = [
                replace(record, meanings=["changed existing beta"])
                if record.id == fixture.beta_extra_id
                else record
                for record in records
            ]
        return deck_config, records

    monkeypatch.setattr(
        build_application, "resolve_deck_records", resolve_with_changed_full_member
    )

    changed = build_application.plan_finish_build(fixture.config, fixture.scope)

    assert changed.fingerprint != plan.fingerprint


def test_execute_builds_only_receipt_owners_as_full_decks_and_records_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    plan = build_application.plan_finish_build(fixture.config, fixture.scope)
    calls: list[tuple[str, tuple[str, ...], object]] = []
    active_locks: set[Path] = set()
    lock_entries: list[Path] = []
    real_lock = build_application.exclusive_path_lock

    @contextmanager
    def tracked_lock(path: Path):
        target = path.resolve()
        lock_entries.append(target)
        with real_lock(path):
            active_locks.add(target)
            try:
                yield
            finally:
                active_locks.remove(target)

    fake_build = _fake_builder(calls)

    def build_while_locked(*args: object, **kwargs: object) -> BuildResult:
        deck_path = Path(args[0]).resolve()
        assert (fixture.config.root / ".janki-audio-operation") in active_locks
        assert fixture.config.deck_dir.resolve() in active_locks
        assert fixture.config.normalized_file.resolve() in active_locks
        assert deck_path in active_locks
        return fake_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        build_application, "resolve_finish_scope", lambda _config, _receipt: fixture.scope
    )
    monkeypatch.setattr(build_application, "exclusive_path_lock", tracked_lock)
    monkeypatch.setattr(build_application, "build_deck", build_while_locked)

    execution = build_application.execute_finish_build(
        fixture.config,
        fixture.scope.receipt_id,
        expected_scope_fingerprint=fixture.scope.fingerprint,
        expected_plan_fingerprint=plan.fingerprint,
    )

    assert execution.state == "complete"
    assert lock_entries[0] == fixture.config.root / ".janki-audio-operation"
    assert tuple(stem for stem, _ids, _include in calls) == ("beta", "alpha")
    assert all(include_ids is None for _stem, _ids, include_ids in calls)
    assert fixture.beta_extra_id in calls[0][1]
    assert fixture.alpha_extra_id in calls[1][1]
    assert all(fixture.third_id not in record_ids for _stem, record_ids, _ in calls)
    assert execution.package_paths == (
        fixture.config.dist_dir / "beta-finish.apkg",
        fixture.config.dist_dir / "alpha-finish.apkg",
    )

    book = ledger.load(fixture.config.ledger_file)
    built_ids = {record_id for _stem, ids, _include in calls for record_id in ids}
    assert set(book.records) == built_ids
    assert fixture.third_id not in book.records
    for stem, record_ids, _include in calls:
        for record_id in record_ids:
            export = book.records[record_id]["exports"][stem]
            assert export["missing"] == ["accent", "audio", "examples"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("scope", "finish-build-scope-stale"),
        ("plan", "finish-build-plan-stale"),
        ("pending", "finish-build-audio-pending"),
    ],
)
def test_stale_or_pending_dispatch_refuses_before_any_package_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    plan = build_application.plan_finish_build(fixture.config, fixture.scope)
    monkeypatch.setattr(
        build_application, "resolve_finish_scope", lambda _config, _receipt: fixture.scope
    )
    if case == "pending":
        monkeypatch.setattr(
            build_application.ledger,
            "load",
            lambda _path: SimpleNamespace(pending_audio={"paid": {}}),
        )

    def unexpected_build(*_args: object, **_kwargs: object) -> BuildResult:
        raise AssertionError("a refusal reached the package writer")

    monkeypatch.setattr(build_application, "build_deck", unexpected_build)

    with pytest.raises(build_application.FinishBuildError, match=message):
        build_application.execute_finish_build(
            fixture.config,
            fixture.scope.receipt_id,
            expected_scope_fingerprint=(
                "0" * 64 if case == "scope" else fixture.scope.fingerprint
            ),
            expected_plan_fingerprint=(
                "0" * 64 if case == "plan" else plan.fingerprint
            ),
        )

    assert not fixture.config.dist_dir.exists()


def test_ledger_save_failure_reports_packages_as_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    plan = build_application.plan_finish_build(fixture.config, fixture.scope)
    calls: list[tuple[str, tuple[str, ...], object]] = []
    saves = 0
    monkeypatch.setattr(
        build_application, "resolve_finish_scope", lambda _config, _receipt: fixture.scope
    )
    monkeypatch.setattr(build_application, "build_deck", _fake_builder(calls))

    def fail_save(_book: ledger.Ledger) -> None:
        nonlocal saves
        saves += 1
        raise ledger.LedgerError("disk full after packages landed")

    monkeypatch.setattr(ledger.Ledger, "save", fail_save)

    execution = build_application.execute_finish_build(
        fixture.config,
        fixture.scope.receipt_id,
        expected_scope_fingerprint=fixture.scope.fingerprint,
        expected_plan_fingerprint=plan.fingerprint,
    )

    assert saves == 1
    assert execution.state == "ledger_incomplete"
    assert execution.ledger_error == "disk full after packages landed"
    assert len(execution.results) == 2
    assert all(path.is_file() for path in execution.package_paths)


@pytest.mark.parametrize(
    ("case", "state", "result_count"),
    [
        ("later_build", "build_incomplete", 1),
        ("result_check", "build_incomplete", 2),
        ("name_check", "build_incomplete", 2),
        ("export_record", "ledger_incomplete", 2),
        ("build_and_ledger", "build_and_ledger_incomplete", 1),
    ],
)
def test_failure_after_a_package_lands_returns_every_known_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    state: str,
    result_count: int,
) -> None:
    fixture = _fixture(tmp_path)
    plan = build_application.plan_finish_build(fixture.config, fixture.scope)
    calls: list[tuple[str, tuple[str, ...], object]] = []
    fake_build = _fake_builder(calls)
    real_export = ledger.Ledger.record_export
    real_save = ledger.Ledger.save
    saves = 0

    monkeypatch.setattr(
        build_application, "resolve_finish_scope", lambda _config, _receipt: fixture.scope
    )

    def sometimes_failing_build(*args: object, **kwargs: object) -> BuildResult:
        deck_path = Path(args[0])
        if deck_path.stem == "alpha" and case in {
            "later_build",
            "build_and_ledger",
        }:
            raise build_application.FinishBuildError("alpha exporter refused")
        result = fake_build(*args, **kwargs)  # type: ignore[arg-type]
        if deck_path.stem == "alpha" and case == "result_check":
            return replace(result, note_count=result.note_count + 1)
        if deck_path.stem == "alpha" and case == "name_check":
            return replace(result, deck_name="different deck")
        return result

    def sometimes_failing_export(
        book: ledger.Ledger,
        record_id: str,
        deck_stem: str,
        *,
        gaps: object = (),
        at: str | None = None,
    ) -> bool:
        if deck_stem == "alpha" and case == "export_record":
            raise ledger.LedgerError("could not record alpha export")
        return real_export(book, record_id, deck_stem, gaps=gaps, at=at)  # type: ignore[arg-type]

    def counted_save(book: ledger.Ledger) -> None:
        nonlocal saves
        saves += 1
        if case == "build_and_ledger":
            raise ledger.LedgerError("ledger also failed")
        real_save(book)

    monkeypatch.setattr(build_application, "build_deck", sometimes_failing_build)
    monkeypatch.setattr(ledger.Ledger, "record_export", sometimes_failing_export)
    monkeypatch.setattr(ledger.Ledger, "save", counted_save)

    execution = build_application.execute_finish_build(
        fixture.config,
        fixture.scope.receipt_id,
        expected_scope_fingerprint=fixture.scope.fingerprint,
        expected_plan_fingerprint=plan.fingerprint,
    )

    assert saves == 1
    assert execution.state == state
    assert len(execution.results) == result_count
    assert all(path.is_file() for path in execution.package_paths)
    if case == "export_record":
        assert execution.build_error is None
        assert "could not record alpha export" in (execution.ledger_error or "")
    else:
        assert execution.build_error
    if case == "build_and_ledger":
        assert execution.ledger_error == "ledger also failed"
    elif case != "export_record":
        assert execution.ledger_error is None
