"""Bounded kanji reference work shared by the CLI and workbench.

Two sources per character, decided independently: KANJIDIC's reference data
and jpdb's reported reading percentages. Both lookups are supplied here as
seams — nothing reaches the network, and every project points the raw page
cache at its own tmp directory so a stray real call cannot read whatever the
developer's machine had cached.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import jpdb_kanji, kanji
from japanese_anki.application import kanji_addition
from japanese_anki.application.kanji_addition import (
    KanjiAdditionError,
    execute_kanji_addition,
    plan_kanji_addition,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import exclusive_path_lock
from japanese_anki.jpdb_kanji import CharacterReadings
from japanese_anki.kanji import KanjiInfo, KanjiStore
from japanese_anki.models import VocabularyRecord


def _record(expression: str, reading: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["something"],
    )


def _readings(character: str, *, sha: str = "b" * 64) -> CharacterReadings:
    """One provider answer, carried through verbatim by the store."""
    return CharacterReadings(
        character=character,
        source_url=f"https://jpdb.io/kanji/{character}",
        fetched_at_utc="2026-09-07T00:00:00Z",
        sha256=sha,
        groups=(),
    )


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'kanji_file = "kanji.json"\n'
        'jpdb_readings_file = "jpdb_readings.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(root)


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
        fetch_readings=_readings,
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
            fetch_readings=lambda character: calls.append(character)
            or _readings(character),
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
                fetch_readings=lambda character: calls.append(character)
                or _readings(character),
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
            fetch_readings=_readings,
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
            fetch_readings=_readings,
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
            fetch_readings=_readings,
        )

    assert kanji.load_store(config.kanji_file).entries == {}


# --- the provider store, decided on its own ----------------------------------


def test_the_two_stores_get_separate_scope_decisions(tmp_path: Path) -> None:
    """A character KANJIDIC already covers is still missing from the provider
    store, and that is exactly the character whose word card shows no reported
    readings."""
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    kanji.save_store(
        config.kanji_file,
        KanjiStore(entries={"名": KanjiInfo(character="名")}),
    )
    jpdb_kanji.save_readings(config.jpdb_readings_file, {"前": _readings("前")})

    plan = plan_kanji_addition(config, [selected.id])

    assert plan.already_known == ("名",) and plan.to_fetch == ("前",)
    assert plan.readings_already_known == ("前",)
    assert plan.readings_to_fetch == ("名",)
    assert plan.jpdb_readings_path == config.jpdb_readings_file.resolve()
    assert plan.readings_fingerprint, "the bytes the decision was read from"


def test_saved_provider_facts_are_reused_and_refresh_replaces_them(
    tmp_path: Path,
) -> None:
    selected = _record("名", "な")
    config = _project(tmp_path, [selected])
    jpdb_kanji.save_readings(
        config.jpdb_readings_file, {"名": _readings("名", sha="a" * 64)}
    )

    reused = plan_kanji_addition(config, [selected.id])
    assert reused.readings_to_fetch == ()

    refreshing = kanji_addition.plan_corpus_kanji_addition(config, refresh=True)
    assert refreshing.readings_to_fetch == ("名",)
    result = execute_kanji_addition(
        config,
        refreshing,
        expected_fingerprint=refreshing.fingerprint,
        fetch=lambda character: KanjiInfo(character=character),
        fetch_readings=lambda character: _readings(character, sha="d" * 64),
    )

    assert result.reading_replaced == ("名",)
    assert result.readings_saved is True
    saved = jpdb_kanji.load_readings(config.jpdb_readings_file)
    assert saved["名"].sha256 == "d" * 64


def test_a_facts_file_that_moved_after_planning_refuses_before_any_request(
    tmp_path: Path,
) -> None:
    """The to-fetch list was decided from those exact bytes. A file that has
    moved may already hold the pages this run is about to pay to request."""
    selected = _record("名", "な")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])
    jpdb_kanji.save_readings(config.jpdb_readings_file, {"名": _readings("名")})
    asked: list[str] = []

    with pytest.raises(KanjiAdditionError, match="kanji-readings-scope-stale"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=lambda character: asked.append(character)
            or KanjiInfo(character=character),
            fetch_readings=lambda character: asked.append(character)
            or _readings(character),
        )

    assert asked == [], "nothing was requested"
    assert not config.kanji_file.exists()


def test_a_concurrent_provider_entry_survives_this_run(tmp_path: Path) -> None:
    """Additive, like the reference merge: an unrelated lookup that finished
    while this one was on the network is preserved rather than overwritten."""
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])

    def concurrent(character: str) -> CharacterReadings:
        if character == "名":
            saved = jpdb_kanji.load_readings(config.jpdb_readings_file)
            saved["線"] = _readings("線")
            saved["前"] = _readings("前", sha="e" * 64)
            jpdb_kanji.save_readings(config.jpdb_readings_file, saved)
        return _readings(character)

    result = execute_kanji_addition(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
        fetch=lambda character: KanjiInfo(character=character),
        fetch_readings=concurrent,
    )

    assert result.reading_added == ("名",)
    assert result.reading_preserved_concurrent == ("前",)
    assert result.reading_store_entries == 3
    saved = jpdb_kanji.load_readings(config.jpdb_readings_file)
    assert set(saved) == {"名", "前", "線"}
    assert saved["前"].sha256 == "e" * 64, "the concurrent write was not undone"


def test_execution_refuses_when_planned_known_facts_disappear_during_fetch(
    tmp_path: Path,
) -> None:
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    jpdb_kanji.save_readings(config.jpdb_readings_file, {"名": _readings("名")})
    plan = plan_kanji_addition(config, [selected.id])

    def delete_known(character: str) -> CharacterReadings:
        jpdb_kanji.save_readings(config.jpdb_readings_file, {})
        return _readings(character)

    with pytest.raises(KanjiAdditionError, match="kanji-readings-store-stale"):
        execute_kanji_addition(
            config,
            plan,
            expected_fingerprint=plan.fingerprint,
            fetch=lambda character: KanjiInfo(character=character),
            fetch_readings=delete_known,
        )

    assert jpdb_kanji.load_readings(config.jpdb_readings_file) == {}


def test_a_provider_failure_leaves_the_others_saved_and_is_reported(
    tmp_path: Path,
) -> None:
    selected = _record("名前", "なまえ")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])

    def refuse_one(character: str) -> CharacterReadings:
        if character == "前":
            raise jpdb_kanji.KanjiReadingsError("jpdb said 404 for 前")
        return _readings(character)

    result = execute_kanji_addition(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
        fetch=lambda character: KanjiInfo(character=character),
        fetch_readings=refuse_one,
    )

    assert result.reading_successes == ("名",)
    assert [(item.character, item.message) for item in result.reading_failures] == [
        ("前", "jpdb said 404 for 前")
    ]
    assert set(jpdb_kanji.load_readings(config.jpdb_readings_file)) == {"名"}


def test_a_plan_with_nothing_missing_saves_neither_store(tmp_path: Path) -> None:
    selected = _record("名", "な")
    config = _project(tmp_path, [selected])
    kanji.save_store(
        config.kanji_file, KanjiStore(entries={"名": KanjiInfo(character="名")})
    )
    jpdb_kanji.save_readings(config.jpdb_readings_file, {"名": _readings("名")})
    plan = plan_kanji_addition(config, [selected.id])
    before = config.jpdb_readings_file.read_bytes()

    result = execute_kanji_addition(
        config,
        plan,
        expected_fingerprint=plan.fingerprint,
        fetch=lambda character: pytest.fail("nothing should be fetched"),
        fetch_readings=lambda character: pytest.fail("nothing should be fetched"),
    )

    assert (result.saved, result.readings_saved) == (False, False)
    assert (result.store_entries, result.reading_store_entries) == (1, 1)
    assert config.jpdb_readings_file.read_bytes() == before


def test_a_plan_whose_readings_scope_was_edited_refuses(tmp_path: Path) -> None:
    selected = _record("名", "な")
    config = _project(tmp_path, [selected])
    plan = plan_kanji_addition(config, [selected.id])
    widened = replace(plan, readings_to_fetch=("名", "線"))

    with pytest.raises(KanjiAdditionError, match="kanji-plan-stale"):
        execute_kanji_addition(
            config,
            widened,
            expected_fingerprint=plan.fingerprint,
            fetch=lambda character: pytest.fail("nothing should be fetched"),
            fetch_readings=lambda character: pytest.fail("nothing should be fetched"),
        )
