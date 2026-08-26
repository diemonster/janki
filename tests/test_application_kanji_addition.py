"""Bounded kanji reference work shared by the CLI and workbench."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import kanji
from japanese_anki.application import kanji_addition
from japanese_anki.application.kanji_addition import (
    KanjiAdditionError,
    execute_kanji_addition,
    plan_kanji_addition,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import exclusive_path_lock
from japanese_anki.kanji import KanjiInfo, KanjiStore
from japanese_anki.models import VocabularyRecord


def _record(expression: str, reading: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["something"],
    )


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'kanji_file = "kanji.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def test_targeted_plan_is_exact_nonempty_and_only_names_missing_scoped_kanji(
    tmp_path: Path,
) -> None:
    selected = _record("名前", "なまえ")
    outside = _record("前線", "ぜんせん")
    config = _project(tmp_path, [selected, outside])
    kanji.save_store(
        config.kanji_file,
        KanjiStore(entries={"名": KanjiInfo(character="名")}),
    )

    plan = plan_kanji_addition(config, [selected.id])

    assert plan.record_ids == (selected.id,)
    assert plan.characters == ("名", "前")
    assert plan.already_known == ("名",)
    assert plan.to_fetch == ("前",)
    assert "線" not in plan.characters, "the unselected record stays out of scope"
    with pytest.raises(KanjiAdditionError, match="at least one exact"):
        plan_kanji_addition(config, [])
    with pytest.raises(KanjiAdditionError, match="No canonical record"):
        plan_kanji_addition(config, ["word:missing:missing"])


def test_execution_fetches_outside_lock_and_preserves_a_concurrent_store_addition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])
    locked_paths: set[Path] = set()
    inside_fetch = False
    real_load = kanji.load_store
    real_save = kanji.save_store

    @contextmanager
    def tracked_lock(path: Path):
        resolved = Path(path).resolve()
        assert resolved not in locked_paths
        locked_paths.add(resolved)
        try:
            with exclusive_path_lock(path):
                yield
        finally:
            locked_paths.remove(resolved)

    monkeypatch.setattr(kanji_addition, "exclusive_path_lock", tracked_lock)

    def checked_load(path: Path) -> KanjiStore:
        if not inside_fetch:
            assert config.normalized_file.resolve() in locked_paths, (
                "the canonical lock must span the kanji reload"
            )
            assert config.kanji_file.resolve() in locked_paths, (
                "the execution reload must be inside the kanji writer lock"
            )
        return real_load(path)

    def checked_save(path: Path, store: KanjiStore) -> None:
        if not inside_fetch:
            assert config.normalized_file.resolve() in locked_paths, (
                "the canonical lock must span the kanji save"
            )
            assert config.kanji_file.resolve() in locked_paths, (
                "the execution save must be inside the kanji writer lock"
            )
        real_save(path, store)

    monkeypatch.setattr(kanji, "load_store", checked_load)
    monkeypatch.setattr(kanji, "save_store", checked_save)

    def fetch(character: str) -> KanjiInfo:
        nonlocal inside_fetch
        assert not locked_paths, "network fetch must not hold either file lock"
        if character == "名":
            inside_fetch = True
            try:
                concurrent = kanji.load_store(config.kanji_file)
                concurrent.entries["線"] = KanjiInfo(character="線")
                kanji.save_store(config.kanji_file, concurrent)
            finally:
                inside_fetch = False
            return KanjiInfo(character="名")
        raise KanjiAdditionError("upstream refused 前")

    result = execute_kanji_addition(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
        fetch=fetch,
    )

    assert result.successes == ("名",)
    assert [(item.character, item.message) for item in result.failures] == [
        ("前", "upstream refused 前")
    ]
    assert result.added == ("名",)
    assert result.store_entries == 2
    assert set(real_load(config.kanji_file).entries) == {"名", "線"}


def test_execution_refuses_mutated_and_stale_exact_scopes_before_writing(
    tmp_path: Path,
) -> None:
    selected = _record("名前", "なまえ")
    same_characters = _record("名前", "みょうぜん")
    replacement = _record("本", "ほん")
    config = _project(tmp_path, [selected, same_characters])
    canonical_before = config.normalized_file.read_bytes()
    plan = plan_kanji_addition(config, [selected.id])
    calls: list[str] = []

    changed_selected = replace(selected, meanings=["changed after planning"])
    config.normalized_file.write_text(
        json.dumps([changed_selected.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(KanjiAdditionError, match="kanji-scope-stale"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=lambda character: calls.append(character)
            or KanjiInfo(character=character),
        )
    assert calls == []
    assert not config.kanji_file.exists()

    config.normalized_file.write_bytes(canonical_before)
    mutated = replace(plan, to_fetch=("線",))
    widened = replace(plan, record_ids=(selected.id, same_characters.id))
    for changed_plan in (mutated, widened):
        with pytest.raises(KanjiAdditionError, match="kanji-plan-stale"):
            execute_kanji_addition(
                config,
                changed_plan,
                expected_fingerprint=plan.fingerprint,
                fetch=lambda character: calls.append(character)
                or KanjiInfo(character=character),
            )
    assert calls == []
    assert not config.kanji_file.exists()

    def change_during_fetch(character: str) -> KanjiInfo:
        calls.append(character)
        config.normalized_file.write_text(
            json.dumps([replacement.to_dict()], ensure_ascii=False),
            encoding="utf-8",
        )
        return KanjiInfo(character=character)

    with pytest.raises(KanjiAdditionError, match="changed while kanji facts"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=change_during_fetch,
        )
    assert calls
    assert not config.kanji_file.exists()


def test_execution_holds_canonical_lock_through_the_kanji_save_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = _record("名", "な")
    replacement = _record("本", "ほん")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])
    real_lock = exclusive_path_lock
    lock_paths: list[Path] = []

    @contextmanager
    def race_at_first_execution_lock(path: Path):
        resolved = Path(path).resolve()
        lock_paths.append(resolved)
        if len(lock_paths) == 1:
            config.normalized_file.write_text(
                json.dumps([replacement.to_dict()], ensure_ascii=False),
                encoding="utf-8",
            )
        with real_lock(path):
            yield

    monkeypatch.setattr(
        kanji_addition,
        "exclusive_path_lock",
        race_at_first_execution_lock,
    )

    with pytest.raises(KanjiAdditionError, match="changed while kanji facts"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=lambda character: KanjiInfo(character=character),
        )

    assert lock_paths == [config.normalized_file.resolve()]
    assert not config.kanji_file.exists()


def test_execution_refuses_when_planned_known_kanji_disappears_during_fetch(
    tmp_path: Path,
) -> None:
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    kanji.save_store(
        config.kanji_file,
        KanjiStore(entries={"名": KanjiInfo(character="名")}),
    )
    plan = plan_kanji_addition(config, [selected.id])

    def delete_known(character: str) -> KanjiInfo:
        kanji.save_store(config.kanji_file, KanjiStore())
        return KanjiInfo(character=character)

    with pytest.raises(KanjiAdditionError, match="kanji-store-stale"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=delete_known,
        )

    assert kanji.load_store(config.kanji_file).entries == {}
