"""Durable, exact W5 finish scopes recovered from promotion receipts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from japanese_anki import staging
from japanese_anki.application import finish
from japanese_anki.application import promotion as promotion_application
from japanese_anki.application.finish import (
    FinishReceipt,
    FinishScopeError,
    list_finish_receipts,
    resolve_finish_scope,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import RecordsRevision
from japanese_anki.models import SourceReference, VocabularyRecord


@pytest.mark.parametrize(
    ("source_file", "receipt_id", "record_count", "message"),
    [
        (" ", "a" * 64, 1, "nonblank source file"),
        ("lesson.pdf", "A" * 64, 1, "lowercase SHA-256"),
        ("lesson.pdf", "a" * 64, 0, "at least one record"),
    ],
    ids=["blank-source", "invalid-receipt", "empty-batch"],
)
def test_finish_receipt_rejects_invalid_durable_handles(
    source_file: str,
    receipt_id: str,
    record_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FinishReceipt(
            source_file=source_file,
            receipt_id=receipt_id,
            record_count=record_count,
        )


def _record(
    expression: str,
    reading: str,
    *,
    tag: str,
    meaning: str = "something",
) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=[meaning],
        tags=[tag],
        source=SourceReference(type="extract", imported_from="lesson.pdf"),
    )


def _deck(path: Path, *, tag: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": f"{tag.title()} deck",
                    "source": source,
                    "intake_tag": tag,
                    "include_tags": [tag],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "staging" / "done").mkdir(parents=True)
    _deck(
        tmp_path / "decks" / "alpha.yaml",
        tag="alpha",
        source="../vocabulary.json",
    )
    # Nested on purpose: owner stems are resolved through status.deck_files,
    # never by assuming every deck is directly under deck_dir.
    _deck(
        tmp_path / "decks" / "nested" / "beta.yaml",
        tag="beta",
        source="../../vocabulary.json",
    )
    return ProjectConfig.load(tmp_path)


def _archive(
    config: ProjectConfig,
    filename: str,
    records: list[VocabularyRecord],
    owners: dict[str, str],
    *,
    source: str = "lesson.pdf",
    review_run_id: str | None = None,
) -> tuple[Path, str]:
    path = config.staging_dir / "done" / filename
    meta: dict[str, object] = {"source_file": source}
    if review_run_id is not None:
        meta["review_run_id"] = review_run_id
    run_fingerprint = promotion_application._archive_run_fingerprint(meta)
    record_ids = tuple(record.id for record in records)
    receipt_id = promotion_application._promotion_receipt_id(
        filename,
        run_fingerprint,
        0,
        source,
        record_ids,
        owners,
    )
    batch = promotion_application.PromotionBatch(
        receipt_id=receipt_id,
        archive_file=filename,
        archive_run_fingerprint=run_fingerprint,
        archive_start_index=0,
        source_file=source,
        review_run_id=review_run_id,
        promoted_ids=record_ids,
        owner_stems=owners,
    )
    meta[staging.PROMOTION_BATCHES_KEY] = [batch.to_dict()]
    staging.write_staging(path, records, meta)
    return path, receipt_id


def _exact_scope(tmp_path: Path) -> tuple[ProjectConfig, str, tuple[VocabularyRecord, ...]]:
    first = _record("一", "いち", tag="beta")
    second = _record("二", "に", tag="alpha")
    third = _record("三", "さん", tag="beta")
    outside = _record("外", "そと", tag="beta")
    config = _project(tmp_path, [first, second, third, outside])
    _, receipt_id = _archive(
        config,
        "lesson.yaml",
        [first, second, third],
        {first.id: "beta", second.id: "alpha", third.id: "beta"},
        review_run_id="12345678-1234-4123-8123-123456789abc",
    )
    return config, receipt_id, (first, second, third)


def test_receipt_resolves_one_exact_ordered_scope_grouped_by_stored_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, receipt_id, records = _exact_scope(tmp_path)
    calls = 0
    real_load = finish.load_records_snapshot

    def counted_load(path: Path):
        nonlocal calls
        calls += 1
        return real_load(path)

    monkeypatch.setattr(finish, "load_records_snapshot", counted_load)

    scope = resolve_finish_scope(config, receipt_id)

    assert calls == 1, "the exact canonical records and revision come from one read"
    assert scope.receipt_id == receipt_id
    assert scope.source_file == "lesson.pdf"
    assert scope.review_run_id == "12345678-1234-4123-8123-123456789abc"
    assert scope.archive_path == (config.staging_dir / "done" / "lesson.yaml").resolve()
    assert scope.record_ids == tuple(record.id for record in records)
    assert scope.owner_stems == ("beta", "alpha", "beta")
    assert scope.canonical_path == config.normalized_file.resolve()
    assert len(scope.canonical_revision) == 64
    assert len(scope.fingerprint) == 64
    assert tuple(
        (group.stem, group.deck_path, group.record_ids)
        for group in scope.owner_groups
    ) == (
        (
            "beta",
            (config.deck_dir / "nested" / "beta.yaml").resolve(),
            (records[0].id, records[2].id),
        ),
        (
            "alpha",
            (config.deck_dir / "alpha.yaml").resolve(),
            (records[1].id,),
        ),
    )
    assert "word:外:そと" not in scope.record_ids


def test_scan_ignores_nested_non_staging_and_non_regular_entries(tmp_path: Path) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    done = config.staging_dir / "done"
    (done / "ignored.txt").write_text("not: [valid", encoding="utf-8")
    (done / "nested").mkdir()
    (done / "nested" / "invalid.yaml").write_text("not: [valid", encoding="utf-8")
    (done / "directory.yml").mkdir()
    outside = tmp_path / "invalid.yaml"
    outside.write_text("not: [valid", encoding="utf-8")
    (done / "linked.yml").symlink_to(outside)

    assert resolve_finish_scope(config, receipt_id).receipt_id == receipt_id


def test_done_namespace_is_bound_before_scan_and_through_archive_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    done = config.staging_dir / "done"
    replacement = tmp_path / "replacement-done"
    replacement.mkdir()
    (replacement / "malformed.yaml").write_text("not: [valid", encoding="utf-8")
    held = done.with_name("held-done")
    real_scandir = os.scandir
    swapped = False

    def swap_after_binding(path: int | str | os.PathLike[str]):
        nonlocal swapped
        if isinstance(path, int) and not swapped:
            swapped = True
            done.rename(held)
            done.symlink_to(replacement, target_is_directory=True)
        return real_scandir(path)

    monkeypatch.setattr(finish.os, "scandir", swap_after_binding)

    scope = resolve_finish_scope(config, receipt_id)

    assert swapped
    assert scope.archive_path == done / "lesson.yaml"


def test_malformed_unrelated_supported_archive_fails_closed(tmp_path: Path) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    (config.staging_dir / "done" / "unrelated.yml").write_text(
        "not: [valid", encoding="utf-8"
    )

    with pytest.raises(FinishScopeError, match="unrelated.yml"):
        resolve_finish_scope(config, receipt_id)


def test_each_receipt_selects_only_its_batch_from_one_cumulative_archive(
    tmp_path: Path,
) -> None:
    earlier = _record("一", "いち", tag="beta")
    later = _record("二", "に", tag="alpha")
    outside = _record("外", "そと", tag="beta")
    config = _project(tmp_path, [earlier, later, outside])
    filename = "cumulative.yaml"
    path = config.staging_dir / "done" / filename
    meta: dict[str, object] = {"source_file": "lesson.pdf"}
    run_fingerprint = promotion_application._archive_run_fingerprint(meta)

    def batch(record: VocabularyRecord, owner: str) -> promotion_application.PromotionBatch:
        owners = {record.id: owner}
        receipt = promotion_application._promotion_receipt_id(
            filename,
            run_fingerprint,
            0,
            "lesson.pdf",
            (record.id,),
            owners,
        )
        return promotion_application.PromotionBatch(
            receipt_id=receipt,
            archive_file=filename,
            archive_run_fingerprint=run_fingerprint,
            archive_start_index=0,
            source_file="lesson.pdf",
            review_run_id=None,
            promoted_ids=(record.id,),
            owner_stems=owners,
        )

    earlier_batch = batch(earlier, "beta")
    later_batch = batch(later, "alpha")
    meta[staging.PROMOTION_BATCHES_KEY] = [
        earlier_batch.to_dict(),
        later_batch.to_dict(),
    ]
    staging.write_staging(path, [earlier, later], meta)

    assert list_finish_receipts(config) == (
        FinishReceipt(
            source_file="lesson.pdf",
            receipt_id=earlier_batch.receipt_id,
            record_count=1,
        ),
        FinishReceipt(
            source_file="lesson.pdf",
            receipt_id=later_batch.receipt_id,
            record_count=1,
        ),
    )

    earlier_scope = resolve_finish_scope(config, earlier_batch.receipt_id)
    later_scope = resolve_finish_scope(config, later_batch.receipt_id)

    assert earlier_scope.record_ids == (earlier.id,)
    assert earlier_scope.owner_stems == ("beta",)
    assert later_scope.record_ids == (later.id,)
    assert later_scope.owner_stems == ("alpha",)
    assert outside.id not in earlier_scope.record_ids
    assert outside.id not in later_scope.record_ids

    with pytest.raises(FinishScopeError, match="no done archive contains"):
        resolve_finish_scope(config, "f" * 64)


def test_archive_is_validated_against_its_actual_filename(tmp_path: Path) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    original = config.staging_dir / "done" / "lesson.yaml"
    renamed = original.with_name("renamed.yaml")
    original.rename(renamed)

    with pytest.raises(FinishScopeError, match="renamed.yaml.*done archive file"):
        resolve_finish_scope(config, receipt_id)


def test_receipt_discovery_rejects_a_handle_that_no_longer_binds_its_batch(
    tmp_path: Path,
) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    archive = config.staging_dir / "done" / "lesson.yaml"
    archived, meta = staging.read_staging(archive)
    replacement = ("0" if receipt_id[-1] != "0" else "1")
    meta[staging.PROMOTION_BATCHES_KEY][0]["receipt_id"] = (
        receipt_id[:-1] + replacement
    )
    staging.write_staging(archive, archived, meta, force=True)

    with pytest.raises(FinishScopeError, match="does not bind its exact source"):
        list_finish_receipts(config)


def test_duplicate_receipt_matches_are_refused_globally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    archive = config.staging_dir / "done" / "lesson.yaml"
    copied = archive.with_name("copy.yml")
    copied.write_bytes(archive.read_bytes())
    archived, meta = staging.read_staging(archive)
    (batch,) = promotion_application.promotion_batches(
        meta,
        archived=archived,
        archive_file=archive.name,
    )

    def repeated_batch(*_args: object, **_kwargs: object):
        return (batch,)

    monkeypatch.setattr(finish, "promotion_batches", repeated_batch)

    with pytest.raises(FinishScopeError, match="more than one done archive"):
        resolve_finish_scope(config, receipt_id)


@pytest.mark.parametrize(
    ("canonical", "message"),
    [
        ("missing", "is missing from the canonical collection"),
        ("duplicate", "occurs more than once in the canonical collection"),
    ],
)
def test_receipted_ids_must_resolve_once_in_the_canonical_snapshot(
    tmp_path: Path, canonical: str, message: str
) -> None:
    record = _record("一", "いち", tag="alpha")
    canonical_records = [] if canonical == "missing" else [record, record]
    config = _project(tmp_path, canonical_records)
    _, receipt_id = _archive(config, "lesson.yaml", [record], {record.id: "alpha"})

    with pytest.raises(FinishScopeError, match=message):
        resolve_finish_scope(config, receipt_id)


def test_current_exact_owner_must_equal_the_receipts_stored_owner(tmp_path: Path) -> None:
    archived = _record("一", "いち", tag="alpha")
    current = _record("一", "いち", tag="beta")
    config = _project(tmp_path, [current])
    _, receipt_id = _archive(
        config,
        "lesson.yaml",
        [archived],
        {archived.id: "alpha"},
    )

    with pytest.raises(FinishScopeError, match="owner changed.*alpha.*beta"):
        resolve_finish_scope(config, receipt_id)


def test_deck_change_during_owner_proof_refuses_a_mixed_finish_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, receipt_id, _records = _exact_scope(tmp_path)
    real_require = finish.require_exact_deck_ownership
    real_lock = finish.exclusive_path_lock
    changed = False
    deck_lock_held = False

    @contextmanager
    def tracked_deck_lock(path: Path):
        nonlocal deck_lock_held
        assert path == config.deck_dir
        with real_lock(path):
            deck_lock_held = True
            try:
                yield
            finally:
                deck_lock_held = False

    def change_after_owner_proof(*args: object, **kwargs: object):
        nonlocal changed
        assert deck_lock_held, "ownership must remain under the deck-set writer lock"
        ownership = real_require(*args, **kwargs)  # type: ignore[arg-type]
        deck = config.deck_dir / "alpha.yaml"
        deck.write_text(
            deck.read_text(encoding="utf-8") + "# changed during owner proof\n",
            encoding="utf-8",
        )
        changed = True
        return ownership

    monkeypatch.setattr(
        finish,
        "require_exact_deck_ownership",
        change_after_owner_proof,
    )
    monkeypatch.setattr(finish, "exclusive_path_lock", tracked_deck_lock)

    with pytest.raises(FinishScopeError, match="configured deck changed"):
        resolve_finish_scope(config, receipt_id)

    assert changed


def test_finish_fingerprint_binds_every_authoritative_scope_claim(tmp_path: Path) -> None:
    config, receipt_id, records = _exact_scope(tmp_path)
    scope = resolve_finish_scope(config, receipt_id)

    def changed_record_id(value: str) -> finish.FinishScope:
        old = records[0].id
        changed_groups = tuple(
            replace(
                group,
                record_ids=tuple(value if item == old else item for item in group.record_ids),
            )
            for group in scope.owner_groups
        )
        return replace(
            scope,
            record_ids=(value, *scope.record_ids[1:]),
            owner_groups=changed_groups,
        )

    def changed_owner(value: str) -> finish.FinishScope:
        changed_group = replace(scope.owner_groups[0], stem=value)
        return replace(
            scope,
            owner_stems=(value, scope.owner_stems[1], value),
            owner_groups=(changed_group, scope.owner_groups[1]),
        )

    variants: tuple[Callable[[], finish.FinishScope], ...] = (
        lambda: replace(scope, receipt_id="f" * 64),
        lambda: replace(scope, source_file="other.pdf"),
        lambda: replace(scope, archive_path=scope.archive_path.with_name("other.yaml")),
        lambda: replace(scope, archive_run_fingerprint="f" * 64),
        lambda: replace(scope, archive_start_index=scope.archive_start_index + 1),
        lambda: replace(
            scope, review_run_id="abcdefab-cdef-4abc-8def-abcdefabcdef"
        ),
        lambda: changed_record_id("word:changed:changed"),
        lambda: changed_owner("changed-owner"),
        lambda: replace(
            scope,
            owner_groups=(
                replace(
                    scope.owner_groups[0],
                    deck_path=scope.owner_groups[0].deck_path.with_name("moved.yaml"),
                ),
                scope.owner_groups[1],
            ),
        ),
        lambda: replace(
            scope, canonical_path=scope.canonical_path.with_name("other.json")
        ),
        lambda: replace(scope, canonical_revision="f" * 64),
        lambda: replace(scope, deck_configuration_revision="f" * 64),
    )

    assert finish._scope_fingerprint(scope) == scope.fingerprint
    for mutate in variants:
        changed = mutate()
        assert finish._scope_fingerprint(changed) != scope.fingerprint


def test_records_revision_fingerprint_distinguishes_missing_and_exact_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vocabulary.json"

    assert finish.records_revision_fingerprint(RecordsRevision(path, None)) == (
        hashlib.sha256(b"missing\0").hexdigest()
    )
    assert finish.records_revision_fingerprint(RecordsRevision(path, "[]\n")) == (
        hashlib.sha256(b"present\0[]\n").hexdigest()
    )


@pytest.mark.parametrize("receipt_id", ["A" * 64, "f" * 63, "not-a-digest"])
def test_receipt_handle_must_be_exact_lowercase_sha256(
    tmp_path: Path, receipt_id: str
) -> None:
    config = _project(tmp_path, [])

    with pytest.raises(FinishScopeError, match="lowercase SHA-256"):
        resolve_finish_scope(config, receipt_id)
