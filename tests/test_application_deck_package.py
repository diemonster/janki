"""Exact application plans for every configured deck package."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_application_audio_completion_proof as proof_fixtures

from japanese_anki import jpdb_kanji, kanji, kanji_notes, ledger
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_build, deck_package
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import FIELD_NAMES, BuildResult
from japanese_anki.identifiers import character_record_id
from japanese_anki.io import (
    RecordsRevision,
    exclusive_path_lock,
    load_records,
    records_json_text,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _project(tmp_path: Path) -> tuple[ProjectConfig, dict[str, Path]]:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "data/ledger.json"\n'
        'media_dir = "data/media"\n'
        'kanji_file = "data/kanji.json"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n'
        'jpdb_readings_file = "data/jpdb_readings.json"\n'
        'patterns_file = "data/patterns.json"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    media = config.media_dir / "audio" / "speak.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"exact audio bytes")
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        part_of_speech="verb",
        verb_group="godan",
        pitch_accent=["LHHH"],
        audio="audio/speak.wav",
        examples=[
            ExampleSentence(
                japanese="日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
                register="polite",
                audio="audio/speak.wav",
            )
        ],
    )
    _write_json(config.normalized_file, [record.to_dict()])
    _write_json(config.kanji_file, {})
    jpdb_kanji.save_readings(config.jpdb_readings_file, {})
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {
            "話": kanji_notes.CharacterNote(
                character="話",
                id=character_record_id("話"),
                meanings=("talk", "speak"),
                stroke_count=13,
            )
        },
    )
    config.deck_dir.mkdir(parents=True)
    vocabulary = config.deck_dir / "lesson.yaml"
    vocabulary.write_text(
        "deck:\n"
        "  name: Lesson deck\n"
        "  source: ../normalized/vocabulary.json\n"
        "  output: lesson.apkg\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: false\n"
        "    reading: false\n",
        encoding="utf-8",
    )
    pattern = config.deck_dir / "rules.yaml"
    pattern.write_text(
        "deck:\n"
        "  kind: pattern\n"
        "  name: Rules\n"
        "  deck_id: 31\n"
        "  model_id: 32\n"
        "  document: rules.pdf\n"
        "  output: rules.apkg\n",
        encoding="utf-8",
    )
    _write_json(
        config.patterns_file,
        {
            "rules.pdf": {
                "kind": "pattern",
                "title": "Rules",
                "reviewed": True,
                "patterns": [
                    {
                        "template": "う → って",
                        "gloss": "te-form ending",
                        "examples": ["かう ⇨ かって"],
                        "where": "row 1",
                    }
                ],
            }
        },
    )
    conjugation = config.deck_dir / "potential.yaml"
    character = config.deck_dir / "kanji.yaml"
    character.write_text(
        "deck:\n"
        "  kind: kanji\n"
        "  name: Character deck\n"
        "  deck_id: 35\n"
        "  model_id: 36\n"
        "  source: ../kanji_notes.json\n"
        "  output: kanji.apkg\n"
        "  include_ids: [kanji:話]\n",
        encoding="utf-8",
    )
    conjugation.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential practice\n"
        "  deck_id: 33\n"
        "  model_id: 34\n"
        "  source: ../normalized/vocabulary.json\n"
        "  output: potential.apkg\n"
        "  include_ids: [word:話す:はなす]\n",
        encoding="utf-8",
    )
    return config, {
        "vocabulary": vocabulary,
        "pattern": pattern,
        "conjugation": conjugation,
        "character": character,
        "source": config.normalized_file,
        "kanji": config.kanji_file,
        "kanji_notes": config.kanji_notes_file,
        "readings": config.jpdb_readings_file,
        "media": media,
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_vocabulary_plan_binds_every_package_input_without_writing(
    tmp_path: Path,
) -> None:
    config, paths = _project(tmp_path)
    before = _snapshot(tmp_path)

    plan = deck_package.plan_deck_package(config, paths["vocabulary"])

    assert _snapshot(tmp_path) == before
    assert plan.kind == "vocabulary"
    assert plan.deck_name == "Lesson deck"
    assert plan.output_path == (config.dist_dir / "lesson.apkg").resolve()
    assert plan.note_count == 1
    assert plan.card_count == 1
    assert plan.card_types == ("recognition",)
    assert plan.record_ids == ("word:話す:はなす",)
    assert plan.deck_input.path == paths["vocabulary"].resolve()
    assert [item.label for item in plan.source_inputs] == [
        "record source",
        "kanji reference store",
        "jpdb reading facts",
    ]
    assert tuple(item.path for item in plan.source_inputs) == (
        paths["source"].resolve(),
        paths["kanji"].resolve(),
        paths["readings"].resolve(),
    )
    assert tuple(item.path.name for item in plan.template_inputs) == (
        "recognition-front.html",
        "recognition-back.html",
        "style.css",
    )
    assert tuple(item.path for item in plan.media_inputs) == (paths["media"].resolve(),)
    for item in plan.all_inputs:
        assert item.sha256 == hashlib.sha256(item.path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "changed", ["deck", "source", "template", "media", "kanji", "readings"]
)
def test_vocabulary_fingerprint_binds_each_exact_input(
    tmp_path: Path,
    changed: str,
) -> None:
    config, paths = _project(tmp_path)
    before = deck_package.plan_deck_package(config, paths["vocabulary"])
    if changed == "deck":
        path = paths["vocabulary"]
        path.write_text(
            path.read_text(encoding="utf-8") + "  description: Changed\n", encoding="utf-8"
        )
    elif changed == "source":
        value = json.loads(paths["source"].read_text(encoding="utf-8"))
        value[0]["meanings"] = ["to converse"]
        _write_json(paths["source"], value)
    elif changed == "template":
        path = config.template_dir / "recognition-front.html"
        path.write_text(path.read_text(encoding="utf-8") + "\n<!-- changed -->\n", encoding="utf-8")
    elif changed == "media":
        paths["media"].write_bytes(b"changed exact audio bytes")
    elif changed == "kanji":
        paths["kanji"].write_text("{ }\n", encoding="utf-8")
    else:
        # A word card's character block draws these figures, so replacing them
        # changes the package the same way a template edit does.
        jpdb_kanji.save_readings(
            paths["readings"],
            {
                "話": jpdb_kanji.CharacterReadings(
                    character="話",
                    source_url="https://jpdb.io/kanji/%E8%A9%B1",
                    fetched_at_utc="2026-09-07T00:00:00Z",
                    sha256="e" * 64,
                    groups=(),
                )
            },
        )

    after = deck_package.plan_deck_package(config, paths["vocabulary"])

    assert after.fingerprint != before.fingerprint


def test_package_fingerprint_binds_exporter_configuration(tmp_path: Path) -> None:
    config, paths = _project(tmp_path)
    before = deck_package.plan_deck_package(config, paths["vocabulary"])

    after = deck_package.plan_deck_package(
        replace(config, max_meanings=config.max_meanings + 1),
        paths["vocabulary"],
    )

    assert after.configuration_fingerprint != before.configuration_fingerprint
    assert after.fingerprint != before.fingerprint


def test_vocabulary_execution_replans_builds_and_records_export_history(
    tmp_path: Path,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])

    result = deck_package.execute_deck_package(config, plan)

    assert result.output_path == plan.output_path
    assert result.output_path.read_bytes().startswith(b"PK")
    assert result.package_sha256 == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    assert result.note_count == 1
    assert result.card_count == 1
    assert result.card_types == ("recognition",)
    assert result.media_count == 1
    assert (
        ledger.load(config.ledger_file).unexported(paths["vocabulary"].stem, plan.record_ids) == []
    )


def test_vocabulary_execution_reports_landed_package_if_export_history_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])

    def fail_save(_book: ledger.Ledger) -> None:
        raise ledger.LedgerError("fixture ledger refusal")

    monkeypatch.setattr(ledger.Ledger, "save", fail_save)

    with pytest.raises(
        deck_package.DeckPackageError,
        match=r"Package .* landed with SHA-256 .* export history could not be saved",
    ):
        deck_package.execute_deck_package(config, plan)

    assert plan.output_path.read_bytes().startswith(b"PK")


def test_vocabulary_execution_refuses_input_drift_before_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    paths["media"].write_bytes(b"changed after confirmation")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> BuildResult:
        nonlocal called
        called = True
        raise AssertionError("stale package plan reached the exporter")

    monkeypatch.setattr(deck_package, "build_deck", forbidden)

    with pytest.raises(deck_package.DeckPackageError, match="plan changed"):
        deck_package.execute_deck_package(config, plan)

    assert called is False
    assert not plan.output_path.exists()


def test_vocabulary_execution_refuses_pending_audio_before_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    monkeypatch.setattr(
        deck_package.ledger,
        "load",
        lambda _path: SimpleNamespace(pending_audio={"unfinished": {}}),
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> BuildResult:
        nonlocal called
        called = True
        raise AssertionError("pending audio reached the package exporter")

    monkeypatch.setattr(deck_package, "build_deck", forbidden)

    with pytest.raises(deck_package.DeckPackageError, match="pending audio recovery"):
        deck_package.execute_deck_package(config, plan)

    assert called is False


def test_vocabulary_execution_detects_input_drift_during_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    real_build = deck_package.build_deck

    def raced(*args: object, **kwargs: object) -> BuildResult:
        result = real_build(*args, **kwargs)
        paths["media"].write_bytes(b"changed at exporter seam")
        return result

    monkeypatch.setattr(deck_package, "build_deck", raced)

    with pytest.raises(deck_package.DeckPackageError, match="plan changed"):
        deck_package.execute_deck_package(config, plan)

    assert plan.output_path.exists(), "dist output is disposable after a detected race"
    assert not config.ledger_file.exists(), "an unaccepted package is not export history"


def test_vocabulary_execution_never_follows_a_late_output_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    config.dist_dir.mkdir()
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    outside = tmp_path.with_name(f"{tmp_path.name}-outside.apkg")
    sentinel = b"outside package must not be replaced"
    outside.write_bytes(sentinel)
    real_build = deck_package.build_deck

    def raced(*args: object, **kwargs: object) -> BuildResult:
        plan.output_path.symlink_to(outside)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(deck_package, "build_deck", raced)

    with pytest.raises(JankiError):
        deck_package.execute_deck_package(config, plan)

    assert plan.output_path.is_symlink()
    assert outside.read_bytes() == sentinel


def test_vocabulary_execution_preserves_a_late_regular_output_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    config.dist_dir.mkdir()
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    sentinel = b"a separately created package must not be replaced"
    real_build = deck_package.build_deck

    def raced(*args: object, **kwargs: object) -> BuildResult:
        plan.output_path.write_bytes(sentinel)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(deck_package, "build_deck", raced)

    with pytest.raises(JankiError):
        deck_package.execute_deck_package(config, plan)

    assert plan.output_path.read_bytes() == sentinel


def test_vocabulary_execution_binds_an_existing_output_inode_even_for_same_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    config.dist_dir.mkdir()
    existing = config.dist_dir / "lesson.apkg"
    sentinel = b"same package bytes under a different inode"
    existing.write_bytes(sentinel)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    real_build = deck_package.build_deck

    def raced(*args: object, **kwargs: object) -> BuildResult:
        replacement = existing.with_suffix(".replacement")
        replacement.write_bytes(sentinel)
        replacement.replace(existing)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(deck_package, "build_deck", raced)

    with pytest.raises(JankiError):
        deck_package.execute_deck_package(config, plan)

    assert existing.read_bytes() == sentinel


def test_pattern_execution_never_follows_a_late_output_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    config.dist_dir.mkdir()
    plan = deck_package.plan_deck_package(config, paths["pattern"])
    outside = tmp_path.with_name(f"{tmp_path.name}-outside-pattern.apkg")
    sentinel = b"outside pattern package must not be replaced"
    outside.write_bytes(sentinel)
    real_build = deck_package.pattern_cards.build_pattern_deck

    def raced(*args: object, **kwargs: object) -> tuple[Path, int]:
        plan.output_path.symlink_to(outside)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(deck_package.pattern_cards, "build_pattern_deck", raced)

    with pytest.raises(JankiError):
        deck_package.execute_deck_package(config, plan)

    assert plan.output_path.is_symlink()
    assert outside.read_bytes() == sentinel


def test_vocabulary_execution_holds_every_input_and_output_lock_through_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["vocabulary"])
    real_lock = deck_package.exclusive_path_lock
    active: dict[Path, int] = {}

    @contextmanager
    def tracking_lock(path: Path):
        resolved = Path(path).resolve()
        with real_lock(path):
            active[resolved] = active.get(resolved, 0) + 1
            try:
                yield
            finally:
                active[resolved] -= 1

    def build(
        _deck: Path,
        _config: ProjectConfig,
        output_path: Path,
        **_output_binding: object,
    ) -> BuildResult:
        required = {
            config.root.resolve() / ".janki-audio-operation",
            config.deck_dir.resolve(),
            plan.output_path,
            *(item.path for item in plan.all_inputs),
        }
        assert required <= {path for path, count in active.items() if count > 0}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"PK lock proof")
        return BuildResult(
            output_path=output_path,
            deck_name=plan.deck_name,
            note_count=plan.note_count,
            card_types=plan.card_types,
            media_count=len(plan.media_inputs),
            record_ids=plan.record_ids,
        )

    monkeypatch.setattr(deck_package, "exclusive_path_lock", tracking_lock)
    monkeypatch.setattr(deck_package, "build_deck", build)

    result = deck_package.execute_deck_package(config, plan)

    assert result.output_path == plan.output_path


def test_pattern_plan_and_execution_use_reviewed_store_and_pattern_exporter(
    tmp_path: Path,
) -> None:
    config, paths = _project(tmp_path)

    plan = deck_package.plan_deck_package(config, paths["pattern"])

    assert plan.kind == "pattern"
    assert plan.variant == "rules.pdf"
    assert plan.card_types == ("rule",)
    assert plan.note_count == plan.card_count == 1
    assert plan.record_ids == ()
    assert plan.source_inputs[0].path == config.patterns_file.resolve()
    assert tuple(item.path.name for item in plan.template_inputs) == (
        "pattern-front.html",
        "pattern-back.html",
        "style.css",
    )
    result = deck_package.execute_deck_package(config, plan)
    assert result.output_path.read_bytes().startswith(b"PK")
    assert result.card_types == ("rule",)
    assert result.media_count == 0


def test_conjugation_plan_wraps_and_executes_existing_plan_bound_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _project(tmp_path)
    plan = deck_package.plan_deck_package(config, paths["conjugation"])
    assert plan.kind == "conjugation"
    assert plan.variant == "potential"
    assert plan.card_types == ("drill",)
    assert plan.record_ids == ("word:話す:はなす",)
    assert isinstance(plan.conjugation_plan, deck_build.ConjugationDeckBuildPlan)
    assert plan.conjugation_plan.record_ids == plan.record_ids
    calls: list[deck_build.ConjugationDeckBuildPlan] = []

    def execute(
        current: ProjectConfig,
        expected: deck_build.ConjugationDeckBuildPlan,
    ) -> deck_build.ConjugationDeckBuildResult:
        assert current is config
        calls.append(expected)
        return deck_build.ConjugationDeckBuildResult(
            output_path=expected.output_path,
            card_count=expected.card_count,
            package_sha256="a" * 64,
        )

    monkeypatch.setattr(deck_build, "execute_conjugation_deck_build_locked", execute)

    result = deck_package.execute_deck_package(config, plan)

    assert calls == [plan.conjugation_plan]
    assert result.card_count == plan.card_count
    assert result.package_sha256 == "a" * 64


def test_package_plan_refuses_record_scope_unconfigured_deck_and_escaping_output(
    tmp_path: Path,
) -> None:
    config, paths = _project(tmp_path)
    with pytest.raises(deck_package.DeckPackageError, match="complete configured deck"):
        deck_package.plan_deck_package(
            config,
            paths["vocabulary"],
            record_ids=("word:話す:はなす",),
        )
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(paths["vocabulary"].read_bytes())
    with pytest.raises(deck_package.DeckPackageError, match="configured deck"):
        deck_package.plan_deck_package(config, outside)
    paths["vocabulary"].write_text(
        paths["vocabulary"]
        .read_text(encoding="utf-8")
        .replace("output: lesson.apkg", "output: ../outside.apkg"),
        encoding="utf-8",
    )
    with pytest.raises(deck_package.DeckPackageError, match="direct .apkg"):
        deck_package.plan_deck_package(config, paths["vocabulary"])


def test_package_planning_refuses_a_deck_number_it_cannot_use(tmp_path: Path) -> None:
    """Display-only planning reports the deck and the key, not a `TypeError`.

    Media planning now reads `deck_id`/`model_id`, so a deck file saying
    `deck_id:` with no value reaches `int(None)` here. This path catches
    `JankiError`, `OSError` and `ValueError` — a `TypeError` is none of those and
    would leave the browser and the Assistant's build action with a traceback
    naming nothing.

    Mutation: let the renderer's conversion raise whatever `int()` raises.
    """

    config, paths = _project(tmp_path)
    deck = paths["vocabulary"]
    deck.write_text(
        deck.read_text(encoding="utf-8") + "  deck_id:\n", encoding="utf-8"
    )

    with pytest.raises(deck_package.DeckPackageError) as refusal:
        deck_package.plan_deck_package(config, deck)

    assert "deck.deck_id must be an integer" in str(refusal.value)
    assert str(deck) in str(refusal.value)


def test_package_refuses_a_configured_deck_replaced_by_an_in_repository_symlink(
    tmp_path: Path,
) -> None:
    config, paths = _project(tmp_path)
    deck = paths["vocabulary"]
    replacement = deck.with_name("replacement.txt")
    replacement.write_bytes(deck.read_bytes())
    deck.unlink()
    deck.symlink_to(replacement.name)

    with pytest.raises(deck_package.DeckPackageError, match="symlink|regular|safely"):
        deck_package.plan_deck_package(config, deck)


def test_character_plan_binds_the_curated_store_and_publishes_once(
    tmp_path: Path,
) -> None:
    """A character package is planned and executed through the same service as
    every other deck. Its inputs are the curated notes and the character
    templates — the facts file is not one, because every figure a card shows
    was copied onto its note when the note was written."""
    config, paths = _project(tmp_path)

    plan = deck_package.plan_deck_package(config, paths["character"])

    assert plan.kind == "kanji"
    assert plan.deck_name == "Character deck"
    assert plan.note_count == 1
    assert plan.card_count == 1
    assert plan.card_types == ("recognition",)
    assert plan.record_ids == ("kanji:話",)
    assert [item.label for item in plan.source_inputs] == ["character note store"]
    assert tuple(item.path for item in plan.source_inputs) == (
        paths["kanji_notes"].resolve(),
    )
    assert tuple(item.path.name for item in plan.template_inputs) == (
        "kanji-recognition-front.html",
        "kanji-recognition-back.html",
        "style.css",
    )
    assert plan.media_inputs == ()

    result = deck_package.execute_deck_package(config, plan)

    assert result.output_path == (config.dist_dir / "kanji.apkg").resolve()
    assert result.note_count == 1 and result.card_count == 1
    assert (
        result.package_sha256
        == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    )


def test_a_character_package_refuses_a_plan_its_notes_no_longer_match(
    tmp_path: Path,
) -> None:
    """The curated store is an input like any other: editing a note after the
    plan was rendered means the confirmed package is not the one that would be
    built."""
    config, paths = _project(tmp_path)
    stale = deck_package.plan_deck_package(config, paths["character"])
    kanji_notes.save_notes(
        paths["kanji_notes"],
        {
            "話": kanji_notes.CharacterNote(
                character="話",
                id=character_record_id("話"),
                meanings=("to speak",),
                stroke_count=13,
            )
        },
    )

    with pytest.raises(deck_package.DeckPackageError, match="changed after"):
        deck_package.execute_deck_package(config, stale)
    assert not (config.dist_dir / "kanji.apkg").exists()


# --- S6-B: projection realization, private preparation, publication ---------
#
# Every proof below is minted by the real audio writer over a real temporary
# repository. `prepare_deck_package` builds with the owning exporter and the
# archive is read back with stdlib zipfile/sqlite3, so the inventory assertions
# are about the artifact rather than about the builder's word for it.


def _finish_project(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    """A repository whose reviewed records are not voiced yet."""

    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "data/ledger.json"\n'
        'media_dir = "data/media"\n'
        'kanji_file = "data/kanji.json"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n'
        'jpdb_readings_file = "data/jpdb_readings.json"\n'
        'patterns_file = "data/patterns.json"\n'
        'operations_file = "data/operations.json"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    _write_json(config.kanji_file, {})
    jpdb_kanji.save_readings(config.jpdb_readings_file, {})
    image = config.media_dir / "images" / "speak.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\n exact image bytes")
    config.normalized_file.parent.mkdir(parents=True, exist_ok=True)
    config.normalized_file.write_text(
        records_json_text(
            [
                replace(
                    proof_fixtures.speaking_record(),
                    image="images/speak.png",
                    tags=["week-1", "verb"],
                )
            ]
        ),
        encoding="utf-8",
    )
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    deck = config.deck_dir / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson deck\n"
        "  source: ../normalized/vocabulary.json\n"
        "  output: lesson.apkg\n"
        "  deck_id: 41\n"
        "  model_id: 42\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n"
        "    reading: false\n",
        encoding="utf-8",
    )
    return config, deck


def _projected_after_audio(
    plan: audio_application.AudioPlan,
    records: list[VocabularyRecord],
) -> list[VocabularyRecord]:
    """The canonical records the audio phase will write, from the plan alone.

    Target names are ``fp(record.id)`` and ``fp(record.id + example.japanese)``,
    which nothing after enrichment changes, so the post-audio collection is
    computable before a single WAV exists — which is what makes the whole-deck
    projection exact.
    """

    projected: list[VocabularyRecord] = []
    for record in records:
        clips = [clip for clip in plan.clips if clip.record_id == record.id]
        word = next((clip for clip in clips if clip.kind == "word"), None)
        examples = []
        for example in record.examples:
            prefix = f"janki-{ledger.example_audio_filename_fingerprint(record, example)}"
            match = next(
                (
                    clip
                    for clip in clips
                    if clip.kind == "example"
                    and clip.target == f"{prefix}{clip.provider.suffix}"
                ),
                None,
            )
            examples.append(
                example if match is None else replace(example, audio=f"audio/{match.target}")
            )
        projected.append(
            replace(
                record,
                audio=record.audio if word is None else f"audio/{word.target}",
                examples=examples,
            )
        )
    return projected


def _prepared_reference_bytes(tmp_path: Path) -> tuple[bytes, bytes]:
    """Exactly what the reference-facts writer will publish, rendered privately."""

    scratch = tmp_path / "reference-scratch"
    scratch.mkdir(exist_ok=True)
    store = scratch / "kanji.json"
    kanji.save_store(
        store,
        kanji.KanjiStore(
            {
                "話": kanji.KanjiInfo(
                    character="話",
                    stroke_count=13,
                    meanings=("talk", "speak"),
                )
            }
        ),
    )
    readings = scratch / "jpdb_readings.json"
    jpdb_kanji.save_readings(
        readings,
        {
            "話": jpdb_kanji.CharacterReadings(
                character="話",
                source_url="https://jpdb.io/kanji/%E8%A9%B1",
                fetched_at_utc="2026-09-12T00:00:00Z",
                sha256="c" * 64,
                groups=(),
            )
        },
    )
    return store.read_bytes(), readings.read_bytes()


@dataclass(frozen=True)
class _Finished:
    """One repository at exactly the moment `packaged` is allowed to start."""

    config: ProjectConfig
    deck: Path
    projection: deck_package.DeckPackagePlan
    proof: audio_application.AudioCompletionProof
    records: list[VocabularyRecord]
    clip_targets: tuple[str, ...]


def _finish_through_audio(
    tmp_path: Path,
    *,
    absent_references: bool = False,
    write_references: bool = True,
    extra_records: Sequence[VocabularyRecord] = (),
    template_edits: Mapping[str, str] | None = None,
) -> _Finished:
    config, deck = _finish_project(tmp_path)
    # Both applied before anything is planned, so the projection binds them:
    # `extra_records` puts a second record in the deck that the job's audio scope
    # does not cover, and `template_edits` is an ordinary card-design change.
    for name, text in (template_edits or {}).items():
        (config.template_dir / name).write_text(text, encoding="utf-8")
    if extra_records:
        config.normalized_file.write_text(
            records_json_text(
                [*load_records(config.normalized_file.resolve()), *extra_records]
            ),
            encoding="utf-8",
        )
    words = proof_fixtures.word_engine()
    sentences = proof_fixtures.sentence_engine()
    ids = [proof_fixtures.SPEAK]
    plan = audio_application.plan_targeted_audio(
        config, ids, words=True, examples=True,
        word_provider=words, sentence_provider=sentences,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    live = load_records(config.normalized_file.resolve())
    projected = _projected_after_audio(plan, live)
    revision = RecordsRevision(
        config.normalized_file.resolve(), records_json_text(projected)
    )
    kanji_bytes, readings_bytes = _prepared_reference_bytes(tmp_path)
    references: dict[Path, str | None] = (
        {config.kanji_file.resolve(): None, config.jpdb_readings_file.resolve(): None}
        if absent_references
        else {
            config.kanji_file.resolve(): hashlib.sha256(kanji_bytes).hexdigest(),
            config.jpdb_readings_file.resolve(): hashlib.sha256(
                readings_bytes
            ).hexdigest(),
        }
    )
    projection = deck_package.plan_vocabulary_deck_package_revision(
        config,
        deck,
        projected,
        revision,
        media_sha256={
            **{
                audio_application.media_target_path(config, clip.target): None
                for clip in plan.clips
            },
            # Not audio: its bytes exist already, so the projection fixes them.
            (config.media_dir.resolve() / "images" / "speak.png"): hashlib.sha256(
                (config.media_dir / "images" / "speak.png").read_bytes()
            ).hexdigest(),
        },
        reference_sha256=references,
    )
    if write_references:
        # The reference-facts writer's own apply, after the projection bound it.
        config.kanji_file.write_bytes(kanji_bytes)
        config.jpdb_readings_file.write_bytes(readings_bytes)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        outcome = audio_application.execute_targeted_audio_locked(
            config, ids, words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=sentences,
        )
        assert outcome.succeeded, outcome.state
        assert config.normalized_file.read_text(encoding="utf-8") == revision.text
        fresh_audio = audio_application.plan_targeted_audio(
            config, ids, words=True, examples=True,
            word_provider=words, sentence_provider=sentences,
        )
        proof = audio_application.prove_audio_completion(
            config,
            fresh_audio,
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                ids,
                words=True,
                examples=True,
            ),
        )
    return _Finished(
        config=config,
        deck=deck,
        projection=projection,
        proof=proof,
        records=load_records(config.normalized_file.resolve()),
        clip_targets=tuple(sorted({clip.target for clip in plan.clips})),
    )


@contextmanager
def _audio_operation(config: ProjectConfig):
    """The finish holds this from the audio phase through publication."""

    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        yield


def test_a_projection_binds_prepared_reference_hashes_and_is_realized_after_them(
    tmp_path: Path,
) -> None:
    """The two reference stores are projected, everything else is read.

    Mutation: consult `reference_sha256` at a third `_input` call, or fall back
    to the live bytes when a prepared hash is supplied.
    """

    finished = _finish_through_audio(tmp_path)
    kanji_bytes, readings_bytes = _prepared_reference_bytes(tmp_path)
    labels = [item.label for item in finished.projection.source_inputs]

    assert labels == ["record source", "kanji reference store", "jpdb reading facts"]
    assert finished.projection.source_inputs[1].sha256 == (
        hashlib.sha256(kanji_bytes).hexdigest()
    )
    assert finished.projection.source_inputs[2].sha256 == (
        hashlib.sha256(readings_bytes).hexdigest()
    )
    projected_media = {
        item.path.name: item.sha256 for item in finished.projection.media_inputs
    }
    assert sorted(value is None for value in projected_media.values()) == [
        False,
        True,
        True,
        True,
    ]
    assert projected_media["speak.png"] is not None

    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    realized = {item.path: item.sha256 for item in preparation.plan.media_inputs}
    assert set(realized) == {item.path for item in finished.projection.media_inputs}
    assert all(value is not None for value in realized.values())
    assert realized[
        finished.config.media_dir.resolve() / "images" / "speak.png"
    ] == projected_media["speak.png"]
    assert preparation.plan.source_inputs == finished.projection.source_inputs
    assert preparation.plan.template_inputs == finished.projection.template_inputs
    assert preparation.plan.record_ids == finished.projection.record_ids


@pytest.mark.parametrize(
    "supplied",
    ["third path", "repeated path", "malformed hash"],
)
def test_reference_hashes_cover_exactly_the_two_stores(
    tmp_path: Path,
    supplied: str,
) -> None:
    """Anything but those two reads, or a bad hash, refuses.

    Mutation: accept any repository path as a reference override.
    """

    config, deck = _finish_project(tmp_path)
    records = load_records(config.normalized_file.resolve())
    revision = records_revision(config.normalized_file.resolve())
    if supplied == "third path":
        references = {config.template_dir.resolve() / "style.css": "a" * 64}
        expected = "is another input"
    elif supplied == "repeated path":
        references = {
            config.kanji_file.resolve(): "a" * 64,
            config.root.resolve() / "data" / ".." / "data" / "kanji.json": "b" * 64,
        }
        expected = "repeats"
    else:
        references = {config.kanji_file.resolve(): "not-a-hash"}
        expected = "malformed"

    with pytest.raises(deck_package.DeckPackageError, match=expected):
        deck_package.plan_vocabulary_deck_package_revision(
            config,
            deck,
            records,
            revision,
            media_sha256={},
            reference_sha256=references,
        )


def test_a_none_reference_means_absent_and_is_never_a_wildcard(
    tmp_path: Path,
) -> None:
    """`None` projects the *missing* file the fresh plan has to reproduce.

    Mutation: treat a `None` reference as "any current bytes are acceptable".
    """

    finished = _finish_through_audio(
        tmp_path, absent_references=True, write_references=False
    )
    assert finished.projection.source_inputs[1].sha256 is None

    # The stores exist on disk, so "absent" is not what the fresh plan finds.
    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="kanji reference store|source input"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    finished.config.kanji_file.unlink()
    finished.config.jpdb_readings_file.unlink()
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    assert preparation.plan.source_inputs[1].sha256 is None


def _doctored(
    plan: deck_package.DeckPackagePlan,
    **values: object,
) -> deck_package.DeckPackagePlan:
    return replace(plan, **values)  # type: ignore[arg-type]


def test_the_comparator_allows_only_the_enumerated_media_slots(
    tmp_path: Path,
) -> None:
    """Pure: two plans and one already-revalidated proof, no disk read.

    Mutation: reuse the permissive build comparator, which accepts any SHA-256
    at a projected `None`.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    fresh = preparation.plan

    deck_package.assert_projection_realized(
        finished.projection, fresh, audio_completion=finished.proof
    )

    invented = _doctored(
        fresh,
        media_inputs=tuple(
            replace(item, sha256=hashlib.sha256(b"never proved").hexdigest())
            if index == 0
            else item
            for index, item in enumerate(fresh.media_inputs)
        ),
    )
    with pytest.raises(
        deck_package.DeckPackageError, match="does not hold the bytes"
    ):
        deck_package.assert_projection_realized(
            finished.projection, invented, audio_completion=finished.proof
        )


@pytest.mark.parametrize(
    "change", ["appeared", "vanished", "fixed hash moved", "note count"]
)
def test_the_comparator_refuses_each_other_difference_by_name(
    tmp_path: Path,
    change: str,
) -> None:
    """Anything outside the enumerated slots refuses, naming the input.

    Mutation: compare the whole plan fingerprint, or skip the media path set.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    fresh = preparation.plan
    stray = finished.config.media_dir.resolve() / "audio" / "janki-stray.wav"
    if change == "appeared":
        doctored = _doctored(
            fresh,
            media_inputs=fresh.media_inputs
            + (
                deck_package.DeckPackageInput(
                    "packaged media", stray, hashlib.sha256(b"stray").hexdigest()
                ),
            ),
        )
        expected = "appeared after the projection"
    elif change == "vanished":
        doctored = _doctored(fresh, media_inputs=fresh.media_inputs[:-1])
        expected = "vanished after the projection"
    elif change == "fixed hash moved":
        fixed = replace(
            finished.projection,
            media_inputs=tuple(
                replace(item, sha256=hashlib.sha256(b"fixed").hexdigest())
                for item in finished.projection.media_inputs
            ),
        )
        with pytest.raises(
            deck_package.DeckPackageError, match="changed after the projection"
        ):
            deck_package.assert_projection_realized(
                fixed, fresh, audio_completion=finished.proof
            )
        return
    else:
        doctored = _doctored(fresh, note_count=fresh.note_count + 1)
        expected = "note count changed"

    with pytest.raises(deck_package.DeckPackageError, match=expected):
        deck_package.assert_projection_realized(
            finished.projection, doctored, audio_completion=finished.proof
        )


def test_a_proof_over_other_canonical_records_refuses_in_the_pure_comparator(
    tmp_path: Path,
) -> None:
    """Artifact scope, not job identity — but a foreign collection still refuses.

    Mutation: drop the canonical digest comparison between plan and proof.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    elsewhere = replace(
        finished.proof, canonical_sha256=hashlib.sha256(b"another job").hexdigest()
    )

    with pytest.raises(
        deck_package.DeckPackageError, match="does not cover the exact canonical"
    ):
        deck_package.assert_projection_realized(
            finished.projection, preparation.plan, audio_completion=elsewhere
        )


def _archive_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _mutate_collection(payload: bytes, statement: str) -> bytes:
    scratch = Path(tempfile.mkdtemp(prefix="janki-tamper-")) / "collection.anki2"
    scratch.write_bytes(payload)
    connection = sqlite3.connect(scratch)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return scratch.read_bytes()


def _tampering(monkeypatch: pytest.MonkeyPatch, mutate) -> None:
    """Let the real exporter build, then damage the staged artifact in place.

    The staged SHA and inode are bound *after* the builder returns, so a tamper
    here is invisible to every hash the preparation carries — which is the
    point: the inventory has to read the archive to find it.
    """

    real_build = deck_package.build_deck

    def build(*args: object, **kwargs: object) -> BuildResult:
        result = real_build(*args, **kwargs)
        members = _archive_members(result.output_path)
        _write_archive(result.output_path, mutate(members))
        return result

    monkeypatch.setattr(deck_package, "build_deck", build)


def test_the_inventory_reads_the_real_archive_note_by_note(tmp_path: Path) -> None:
    """GUIDs and counts are not the check; exact field bytes are.

    Mutation: compare the packaged GUID set and note count only.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    inventory = preparation.inventory

    assert inventory.deck_id == 41
    assert inventory.model_id == 42
    assert inventory.card_types == ("recognition", "production")
    assert inventory.field_names == tuple(FIELD_NAMES)
    assert [name for name, _qfmt, _afmt in inventory.templates] == [
        "Recognition",
        "Production",
    ]
    assert inventory.stylesheet_sha256 == hashlib.sha256(
        (finished.config.template_dir / "style.css").read_bytes()
    ).hexdigest()
    assert [note.record_id for note in inventory.notes] == [proof_fixtures.SPEAK]
    note = inventory.notes[0]
    assert note.fields[FIELD_NAMES.index("Expression")] == "話す"
    assert note.tags == ("week-1", "verb")
    assert note.card_ordinals == (0, 1)
    packaged = dict(inventory.media)
    assert set(packaged) == {*finished.clip_targets, "speak.png"}
    for target in finished.clip_targets:
        source = finished.config.media_dir.resolve() / "audio" / target
        assert packaged[target] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_stored_exported_and_unique_sentence_counts_stay_three_numbers(
    tmp_path: Path,
) -> None:
    """A record may store more sentences than a card has slots for.

    Mutation: report the stored slot count as the exported one.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert preparation.inventory.stored_sentence_slots == 3
    assert preparation.inventory.exported_sentence_slots == 2
    assert preparation.inventory.sentence_clips == 2
    assert any("reaches no field" in warning for warning in preparation.warnings)


def test_exported_sentence_fields_resolve_to_their_proven_clips(
    tmp_path: Path,
) -> None:
    """Each drawn sentence slot, checked against the proof by name.

    Mutation: check only the aggregate packaged media count.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    fields = preparation.inventory.notes[0].fields
    polite = finished.proof.slots_for(proof_fixtures.SPEAK, "example", 0)
    casual = finished.proof.slots_for(proof_fixtures.SPEAK, "example", 2)
    word = finished.proof.slots_for(proof_fixtures.SPEAK, "word", None)

    assert fields[FIELD_NAMES.index("ExampleAudio")] == f"[sound:{polite.target}]"
    assert fields[FIELD_NAMES.index("CasualAudio")] == f"[sound:{casual.target}]"
    assert fields[FIELD_NAMES.index("Audio")] == f"[sound:{word.target}]"
    assert polite.target != word.target
    packaged = dict(preparation.inventory.media)
    assert packaged[polite.target] == polite.media_sha256
    assert packaged[casual.target] == casual.media_sha256


def test_a_note_drawing_another_clip_than_the_proof_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A word clip cannot satisfy a sentence slot.

    Mutation: drop the per-slot sound-field comparison.
    """

    finished = _finish_through_audio(tmp_path)
    polite = finished.proof.slots_for(proof_fixtures.SPEAK, "example", 0)
    word = finished.proof.slots_for(proof_fixtures.SPEAK, "word", None)

    def swap(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"],
            "update notes set flds = replace(flds, "
            f"'[sound:{polite.target}]', '[sound:{word.target}]')",
        )
        return members

    _tampering(monkeypatch, swap)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not draw its proven sentence clip"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_a_packaged_clip_without_its_proven_bytes_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest name is not the clip; the bytes are.

    Mutation: compare packaged media names without their hashes.
    """

    finished = _finish_through_audio(tmp_path)

    def rewrite(members: dict[str, bytes]) -> dict[str, bytes]:
        manifest = json.loads(members["media"].decode("utf-8"))
        index = next(
            key for key, name in manifest.items() if name.startswith("janki-")
        )
        members[index] = b"silence where a clip used to be"
        return members

    _tampering(monkeypatch, rewrite)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not hold its proven bytes"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_a_packaged_image_without_its_bound_bytes_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media the audio proof says nothing about is still bound by the plan.

    Mutation: validate only the media the completion proof covers.
    """

    finished = _finish_through_audio(tmp_path)

    def rewrite(members: dict[str, bytes]) -> dict[str, bytes]:
        manifest = json.loads(members["media"].decode("utf-8"))
        index = next(key for key, name in manifest.items() if name == "speak.png")
        members[index] = b"another picture entirely"
        return members

    _tampering(monkeypatch, rewrite)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="speak.png does not hold the bound bytes"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_a_non_audio_field_edited_in_the_archive_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The archive must carry the exporter's exact rendering, field by field.

    Mutation: compare the note GUID set instead of the field bytes.
    """

    finished = _finish_through_audio(tmp_path)

    def edit(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"],
            "update notes set flds = replace(flds, 'to speak', 'to converse')",
        )
        return members

    _tampering(monkeypatch, edit)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact Meanings"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


@pytest.mark.parametrize("part", ["css", "template"])
def test_the_archived_notetype_must_carry_the_bound_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    part: str,
) -> None:
    """Hashing the template files does not prove the archive carries them.

    Mutation: compare only the external template inputs, not the archived
    stylesheet and card templates.
    """

    finished = _finish_through_audio(tmp_path)
    expected = "stylesheet" if part == "css" else "card templates"

    def edit(members: dict[str, bytes]) -> dict[str, bytes]:
        statement = (
            "update col set models = replace(models, '.card {', '.card-tampered {')"
            if part == "css"
            else "update col set models = replace(models, '{{Expression}}', "
            "'{{Reading}}')"
        )
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"], statement
        )
        return members

    _tampering(monkeypatch, edit)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match=expected
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_an_archive_outside_the_genanki_layout_refuses_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One supported layout, and nothing is guessed at.

    Mutation: fall back to reading whatever collection member is present.
    """

    finished = _finish_through_audio(tmp_path)

    def rename(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki21"] = members.pop("collection.anki2")
        return members

    _tampering(monkeypatch, rename)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError,
        match="Unsupported package archive layout",
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_prepare_stages_privately_and_touches_no_target_or_ledger(
    tmp_path: Path,
) -> None:
    """`packaged` builds an artifact; it publishes nothing.

    Mutation: build straight at the confirmed output path.
    """

    finished = _finish_through_audio(tmp_path)
    target = finished.projection.output_path
    ledger_before = finished.config.ledger_file.read_bytes()

    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert not target.exists()
    assert finished.config.ledger_file.read_bytes() == ledger_before
    assert preparation.staged_path.parent == (
        finished.config.dist_dir.resolve() / deck_package.PREPARED_PACKAGE_DIR_NAME
    )
    assert preparation.staged_path.name == f"{preparation.preparation_id}.apkg"
    assert preparation.staged_sha256 == hashlib.sha256(
        preparation.staged_path.read_bytes()
    ).hexdigest()
    assert preparation.staged_sha256 == preparation.package_sha256
    assert preparation.output_before_revision is None
    assert preparation.output_before_identity is None
    assert preparation.ledger_before_sha256 != preparation.ledger_after_sha256
    assert [entry.record_id for entry in preparation.export_delta] == [
        proof_fixtures.SPEAK
    ]
    assert preparation.export_delta[0].deck_stem == "lesson"
    assert preparation.supersedes == ""


def test_the_packaged_receipt_carries_evidence_and_no_download_offer(
    tmp_path: Path,
) -> None:
    """`packaged` is a build proof; `complete` is minted after the preview.

    Mutation: expose a download token or a finish action from this receipt.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    receipt = preparation.packaged_receipt(finished.config)

    assert receipt["output_path"] == "dist/lesson.apkg"
    assert receipt["package_sha256"] == preparation.staged_sha256
    assert receipt["inventory_fingerprint"] == preparation.inventory.fingerprint
    assert receipt["audio_selection"]["exported_sentence_slots"] == 2
    assert receipt["audio_completion_fingerprint"] == finished.proof.fingerprint
    assert not any(
        key in receipt
        for key in ("download", "token", "complete", "finish", "url", "resolver")
    )


def test_the_receipt_counts_the_cards_the_archive_was_proven_to_carry(
    tmp_path: Path,
) -> None:
    """A completed build proof reports the expansion, not the capacity.

    `plan.card_count` is `len(records) × len(card_types)`, computed before
    anything is built: it is what the deck's configuration *allows*. Prompting a
    production front on `{{Furigana}}` is an ordinary card-design change, and
    this fixture's record has no furigana, so genanki writes no production card
    for it — `Model._req` is `all[Furigana]` and the field is empty. The archive
    then holds one card where the product says two, and the `packaged` receipt a
    preview and a download offer are later minted from must say the number the
    archive was validated against, per note.

    The plan keeps its estimate: the projection/fresh comparison stays strict and
    exact, and nothing here refuses a deck for expanding into fewer cards than
    its directions offer.

    Mutation: publish `plan.card_count` as the completed archive's card total.
    """

    finished = _finish_through_audio(
        tmp_path,
        template_edits={
            "production-front.html": '<div class="prompt">{{Furigana}}</div>\n'
        },
    )
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
        receipt = preparation.packaged_receipt(finished.config)
        result = deck_package.publish_prepared_deck_package(
            finished.config, preparation
        )

    assert preparation.inventory.card_types == ("recognition", "production")
    assert [note.card_ordinals for note in preparation.inventory.notes] == [(0,)]
    assert preparation.plan.card_count == 2, "the configured capacity is unchanged"
    assert receipt["card_count"] == 1
    assert result.card_count == 1
    assert preparation.inventory.card_count == 1


def test_the_receipt_separates_whole_deck_totals_from_the_audio_selection(
    tmp_path: Path,
) -> None:
    """§7.12: every count says which scope it is, side by side in one payload.

    The three sentence numbers come from `audio_completion.slots`, so they
    describe the job's selection — a deck record the audio scope never covered
    contributes nothing to them — while the note, card and media totals describe
    the whole deck. Unlabelled and adjacent, six numbers at two scopes read as
    one scope, and the surface wave would display them that way.

    Mutation: emit the sentence numbers beside the whole-deck totals with no
    scope and no record ids.
    """

    finished = _finish_through_audio(
        tmp_path,
        extra_records=[
            VocabularyRecord(
                id="word:ねこ:ねこ",
                expression="ねこ",
                reading="ねこ",
                meanings=["cat"],
                part_of_speech="noun",
            )
        ],
    )
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    receipt = preparation.packaged_receipt(finished.config)

    # Sorted by record id, which puts the kana-only second record first.
    assert [note.record_id for note in preparation.inventory.notes] == [
        "word:ねこ:ねこ",
        proof_fixtures.SPEAK,
    ]
    assert [note.card_ordinals for note in preparation.inventory.notes] == [
        (0, 1),
        (0, 1),
    ]
    assert receipt["note_count"] == 2, "the whole deck, not the job's selection"
    assert receipt["card_count"] == 4
    assert receipt["card_types"] == ["recognition", "production"]
    assert receipt["media_count"] == 4, "three clips and the image, whole deck"
    assert receipt["audio_selection"] == {
        "record_ids": [proof_fixtures.SPEAK],
        "stored_sentence_slots": 3,
        "exported_sentence_slots": 2,
        "sentence_clips": 2,
    }


def test_publication_republishes_the_exact_artifact_and_the_frozen_delta(
    tmp_path: Path,
) -> None:
    """Publication copies those exact bytes and applies before to after.

    Mutation: rebuild at publication instead of publishing the staged bytes.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
        result = deck_package.publish_prepared_deck_package(
            finished.config, preparation
        )

    published = preparation.plan.output_path
    assert result.output_path == published
    assert published.read_bytes() == preparation.staged_path.read_bytes()
    assert result.package_sha256 == preparation.staged_sha256
    book = ledger.load(finished.config.ledger_file)
    assert book.unexported("lesson", preparation.plan.record_ids) == []
    assert hashlib.sha256(
        book.serialized_text().encode("utf-8")
    ).hexdigest() == preparation.ledger_after_sha256


def test_publication_refuses_a_target_replaced_between_intent_and_publish(
    tmp_path: Path,
) -> None:
    """The bound absence is part of the intent.

    Mutation: publish without the bound output expectation.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
        sentinel = b"a separately created package must not be replaced"
        preparation.plan.output_path.write_bytes(sentinel)
        before = hashlib.sha256(
            ledger.load(finished.config.ledger_file).serialized_text().encode("utf-8")
        ).hexdigest()

        # The strict re-plan sees an output revision this intent never bound and
        # refuses before the bound writer is even asked.
        with pytest.raises(deck_package.DeckPackageError, match="plan changed after"):
            deck_package.publish_prepared_deck_package(finished.config, preparation)

    assert preparation.plan.output_path.read_bytes() == sentinel
    assert before == preparation.ledger_before_sha256
    assert hashlib.sha256(
        ledger.load(finished.config.ledger_file).serialized_text().encode("utf-8")
    ).hexdigest() == preparation.ledger_before_sha256


def test_publication_refuses_a_staged_artifact_replaced_at_a_new_inode(
    tmp_path: Path,
) -> None:
    """Same bytes, another file: the staged artifact is bound by inode too.

    Mutation: re-hash the staged path without binding its identity.
    """

    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
        replacement = preparation.staged_path.with_suffix(".replacement")
        replacement.write_bytes(preparation.staged_path.read_bytes())
        replacement.replace(preparation.staged_path)

        with pytest.raises(
            deck_package.DeckPackageError, match="replaced by another file"
        ):
            deck_package.publish_prepared_deck_package(finished.config, preparation)

    assert not preparation.plan.output_path.exists()


def _prepared(tmp_path: Path) -> tuple[_Finished, deck_package.DeckPackagePreparation]:
    finished = _finish_through_audio(tmp_path)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    return finished, preparation


def _publish_bytes(preparation: deck_package.DeckPackagePreparation) -> None:
    """Land the exact artifact without the ledger write, as a crash would."""

    target = preparation.plan.output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(preparation.staged_path.read_bytes())


def _ledger_state(config: ProjectConfig) -> str:
    return hashlib.sha256(
        ledger.load(config.ledger_file).serialized_text().encode("utf-8")
    ).hexdigest()


def test_recovery_recognizes_its_own_output_before_any_strict_replan(
    tmp_path: Path,
) -> None:
    """After the target lands, a blind re-plan would refuse this job's own work.

    Mutation: re-plan and compare strictly before looking at the target.
    """

    finished, preparation = _prepared(tmp_path)
    _publish_bytes(preparation)
    assert _ledger_state(finished.config) == preparation.ledger_before_sha256

    with _audio_operation(finished.config):
        result = deck_package.recover_prepared_deck_package(
            finished.config, preparation
        )

    assert result is not None
    assert result.package_sha256 == preparation.staged_sha256
    assert _ledger_state(finished.config) == preparation.ledger_after_sha256


def test_recovery_after_the_ledger_landed_records_the_receipt_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Both halves already done: complete, idempotently.

    Mutation: replay the frozen delta onto the after-state.
    """

    finished, preparation = _prepared(tmp_path)
    with _audio_operation(finished.config):
        deck_package.publish_prepared_deck_package(finished.config, preparation)
        after = finished.config.ledger_file.read_bytes()

        result = deck_package.recover_prepared_deck_package(
            finished.config, preparation
        )

    assert result is not None
    assert result.package_sha256 == preparation.staged_sha256
    assert finished.config.ledger_file.read_bytes() == after


def test_recovery_refuses_a_third_ledger_state_and_preserves_it(
    tmp_path: Path,
) -> None:
    """Before to after, exactly — never a merge onto whatever is current.

    Mutation: apply the delta to any observed ledger.
    """

    finished, preparation = _prepared(tmp_path)
    _publish_bytes(preparation)
    book = ledger.load(finished.config.ledger_file)
    book.record_export("word:読む:よむ", "other", at="2026-09-12")
    book.save()
    third = finished.config.ledger_file.read_bytes()

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="neither the prepared before state"
    ):
        deck_package.recover_prepared_deck_package(finished.config, preparation)

    assert finished.config.ledger_file.read_bytes() == third


def test_publication_refuses_a_third_ledger_state_before_writing_the_target(
    tmp_path: Path,
) -> None:
    """Every bound component is checked before the write, not after it.

    §7.5 requires an apply to precheck its entire bound set under its locks
    before any effect. The ledger is bound before→after and the delta is frozen,
    so an owner's own export landing between the intent and the publication makes
    this publication unfinishable — and it is knowable for the price of a read
    that publication was already going to do. Writing the artifact first leaves
    a published package whose history refuses, and a retry of `publish` then
    refuses at `_assert_same` because the target moved, so only recovery can
    speak about it.

    Asserting that it raises is not enough: before the fix it raised too, after
    writing the package. The target must not exist and the owner's ledger bytes
    must be exactly theirs.

    Mutation: precheck the ledger after the artifact lands.
    """

    finished, preparation = _prepared(tmp_path)
    book = ledger.load(finished.config.ledger_file)
    book.record_export("word:読む:よむ", "other", at="2026-09-12")
    book.save()
    third = finished.config.ledger_file.read_bytes()
    target = preparation.plan.output_path
    assert not target.exists()

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="neither the prepared before state"
    ):
        deck_package.publish_prepared_deck_package(finished.config, preparation)

    assert not target.exists(), "the package must not be published into a dead end"
    assert finished.config.ledger_file.read_bytes() == third


def test_publication_binds_the_output_directory_the_preparation_captured(
    tmp_path: Path,
) -> None:
    """§7.11 names `expected_directory_identity` in the publication call.

    The preparation's own `expected_directory_identity` is the **private**
    staging directory's and would be the wrong thing to pass: what publication
    has to bind is the directory the confirmed target lives in. Without it the
    writer creates a missing parent, so a `dist/` replaced between intent and
    publication is written into rather than refused.

    The staged artifact is moved back into the new `dist/` deliberately, keeping
    its own inode and bytes: the refusal has to be about the output parent while
    everything else this intent bound is still exactly available.

    Mutation: publish without binding the output parent, or reuse the staging
    directory's identity for it.
    """

    finished, preparation = _prepared(tmp_path)
    assert (
        preparation.output_directory_identity
        != preparation.expected_directory_identity
    ), "two directories, two bindings"
    staging = preparation.staged_path.parent
    distribution = finished.config.dist_dir
    replaced = distribution.with_name("dist-replaced")
    distribution.rename(replaced)
    distribution.mkdir()
    (replaced / staging.name).rename(staging)

    staged = preparation.staged_path.stat()
    assert (staged.st_dev, staged.st_ino) == preparation.staged_identity
    assert hashlib.sha256(preparation.staged_path.read_bytes()).hexdigest() == (
        preparation.staged_sha256
    )

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="Bound target directory changed"
    ):
        deck_package.publish_prepared_deck_package(finished.config, preparation)

    assert not preparation.plan.output_path.exists()
    assert _ledger_state(finished.config) == preparation.ledger_before_sha256


def test_recovery_publishes_when_the_target_is_still_the_bound_state(
    tmp_path: Path,
) -> None:
    """A crash after the intent and before publication resumes from the intent.

    Mutation: recover from a fresh projection instead of the persisted plan.
    """

    finished, preparation = _prepared(tmp_path)
    assert not preparation.plan.output_path.exists()

    with _audio_operation(finished.config):
        result = deck_package.recover_prepared_deck_package(
            finished.config, preparation
        )

    assert result is not None
    assert preparation.plan.output_path.read_bytes() == (
        preparation.staged_path.read_bytes()
    )
    assert _ledger_state(finished.config) == preparation.ledger_after_sha256


def test_recovery_refuses_an_external_third_target_revision(tmp_path: Path) -> None:
    """A valid zip at the right name is not adoption evidence.

    Mutation: adopt any existing package at the confirmed output path.
    """

    finished, preparation = _prepared(tmp_path)
    external = b"PK someone else's package"
    preparation.plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    preparation.plan.output_path.write_bytes(external)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="is preserved"
    ):
        deck_package.recover_prepared_deck_package(finished.config, preparation)

    assert preparation.plan.output_path.read_bytes() == external


def test_a_vanished_stage_allows_a_new_superseding_preparation(
    tmp_path: Path,
) -> None:
    """`dist/` is disposable, so the rebuild is free — and immutable.

    Mutation: mutate the old intent's staged path instead of minting a new one.
    """

    finished, preparation = _prepared(tmp_path)
    preparation.staged_path.unlink()

    with _audio_operation(finished.config):
        assert (
            deck_package.recover_prepared_deck_package(finished.config, preparation)
            is None
        )
        superseding = deck_package.prepare_deck_package(
            finished.config,
            finished.projection,
            audio_completion=finished.proof,
            supersedes=preparation,
        )
        result = deck_package.publish_prepared_deck_package(
            finished.config, superseding
        )

    assert superseding.preparation_id != preparation.preparation_id
    assert superseding.supersedes == preparation.preparation_id
    assert preparation.supersedes == ""
    # A rebuild mints a new artifact SHA and must still reproduce the content.
    assert superseding.inventory.fingerprint == preparation.inventory.fingerprint
    assert result.output_path.read_bytes() == superseding.staged_path.read_bytes()


def test_a_removed_private_staging_directory_is_a_vanished_stage(
    tmp_path: Path,
) -> None:
    """The whole private directory gone is the same row as the file gone.

    `dist/` is disposable by design, so this node removes `dist/.janki-prepared/`
    outright in the same checkout: the directory does not exist at all, not
    merely empty. §7.11's table answers "staged artifact missing, target matches
    `output_before`" with a free rebuild under a new `preparation_id`, and the
    caller's cue for that is `None`. Binding the directory first turned the
    commonest form of the absent stage into a refusal the caller could only tell
    apart from a genuinely *replaced* directory by matching message text.

    Mutation: bind the directory's identity before asking whether it is there.
    """

    finished, preparation = _prepared(tmp_path)
    shutil.rmtree(preparation.staged_path.parent)

    with _audio_operation(finished.config):
        assert (
            deck_package.recover_prepared_deck_package(finished.config, preparation)
            is None
        )
        superseding = deck_package.prepare_deck_package(
            finished.config,
            finished.projection,
            audio_completion=finished.proof,
            supersedes=preparation,
        )
        result = deck_package.publish_prepared_deck_package(
            finished.config, superseding
        )

    assert superseding.preparation_id != preparation.preparation_id
    assert superseding.supersedes == preparation.preparation_id
    assert preparation.supersedes == ""
    assert superseding.inventory.fingerprint == preparation.inventory.fingerprint
    assert result.output_path.read_bytes() == superseding.staged_path.read_bytes()


@pytest.mark.parametrize("removal", ["staged file", "private directory"])
def test_publication_refuses_a_vanished_stage_in_this_modules_own_error(
    tmp_path: Path,
    removal: str,
) -> None:
    """Publishing a vanished stage is a named refusal, not a bare `OSError`.

    `recover_prepared_deck_package` answers the absent stage with `None` so the
    caller can mint a superseding preparation, and it depends on the raw
    `FileNotFoundError` to tell that row apart from a *replaced* directory. But
    `publish_prepared_deck_package` is a public entry point of this module: a
    caller that reaches it with a disposable `dist/` already cleaned gets an
    exception outside `DeckPackageError` naming neither the preparation nor the
    path, and `JankiError` handlers do not see it. Both ways in — the staged
    file gone and the whole private directory gone — arrive at the one
    `_staged_bytes` call publication makes, so both are parameterized here.

    The refusal has to be actionable, so the preparation id and the staged path
    are both asserted, and the underlying absence is kept as the cause. Nothing
    may have been spent: the target must still be absent and the ledger must
    still be the state the preparation bound.

    Mutation: publish the staged bytes without normalizing their absence.
    """

    finished, preparation = _prepared(tmp_path)
    if removal == "staged file":
        preparation.staged_path.unlink()
    else:
        shutil.rmtree(preparation.staged_path.parent)
    assert _ledger_state(finished.config) == preparation.ledger_before_sha256

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError
    ) as refusal:
        deck_package.publish_prepared_deck_package(finished.config, preparation)

    message = str(refusal.value)
    assert str(preparation.staged_path) in message
    assert preparation.preparation_id in message
    assert isinstance(refusal.value.__cause__, FileNotFoundError)
    assert not preparation.plan.output_path.exists()
    assert _ledger_state(finished.config) == preparation.ledger_before_sha256


def test_a_superseding_attempt_refuses_a_different_bound_inventory(
    tmp_path: Path,
) -> None:
    """"From the same bound input inventory" is checked, not assumed.

    Mutation: accept any rebuild as a superseding attempt.
    """

    finished, preparation = _prepared(tmp_path)
    preparation.staged_path.unlink()
    doctored = replace(
        preparation,
        inventory=replace(
            preparation.inventory, fingerprint=hashlib.sha256(b"other").hexdigest()
        ),
    )

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="same bound input inventory"
    ):
        deck_package.prepare_deck_package(
            finished.config,
            finished.projection,
            audio_completion=finished.proof,
            supersedes=doctored,
        )


def test_a_replaced_private_staging_directory_refuses(tmp_path: Path) -> None:
    """The bound private directory is part of the artifact's identity.

    Mutation: read the staged path without binding its directory.
    """

    finished, preparation = _prepared(tmp_path)
    staging = preparation.staged_path.parent
    moved = staging.with_name(".janki-prepared-old")
    staging.rename(moved)
    staging.mkdir()
    (staging / preparation.staged_path.name).write_bytes(
        (moved / preparation.staged_path.name).read_bytes()
    )

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="staging directory was replaced"
    ):
        deck_package.recover_prepared_deck_package(finished.config, preparation)


def test_the_frozen_export_date_replays_on_a_later_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume the next day replays the prepared bytes, not a fresh date.

    Mutation: call `record_export` without its frozen `at`.
    """

    finished = _finish_through_audio(tmp_path)

    class _Day:
        @staticmethod
        def today() -> date:
            return date(2026, 9, 12)

    monkeypatch.setattr(deck_package, "date", _Day)
    with _audio_operation(finished.config):
        preparation = deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
    assert [entry.at for entry in preparation.export_delta] == ["2026-09-12"]

    class _NextDay:
        @staticmethod
        def today() -> date:
            return date(2026, 9, 13)

    monkeypatch.setattr(deck_package, "date", _NextDay)
    with _audio_operation(finished.config):
        deck_package.publish_prepared_deck_package(finished.config, preparation)

    saved = ledger.load(finished.config.ledger_file)
    entry = saved.records[proof_fixtures.SPEAK]["exports"]["lesson"]
    assert (entry if isinstance(entry, str) else entry["at"]) == "2026-09-12"
    assert _ledger_state(finished.config) == preparation.ledger_after_sha256


def test_a_prepared_package_round_trips_through_its_wire(tmp_path: Path) -> None:
    """The persisted intent is the only thing recovery consumes.

    Mutation: skip the fingerprint recomputation in `from_wire`.
    """

    finished, preparation = _prepared(tmp_path)
    wire = json.loads(
        json.dumps(preparation.to_wire(finished.config), ensure_ascii=False)
    )

    restored = deck_package.DeckPackagePreparation.from_wire(finished.config, wire)

    assert restored == preparation
    assert restored.plan == preparation.plan
    assert restored.audio_completion == finished.proof
    assert restored.inventory == preparation.inventory
    with _audio_operation(finished.config):
        result = deck_package.publish_prepared_deck_package(finished.config, restored)
    assert result.package_sha256 == preparation.staged_sha256

    tampered = json.loads(
        json.dumps(preparation.to_wire(finished.config), ensure_ascii=False)
    )
    tampered["staged_sha256"] = hashlib.sha256(b"other artifact").hexdigest()
    with pytest.raises(deck_package.DeckPackageError, match="own fingerprint"):
        deck_package.DeckPackagePreparation.from_wire(finished.config, tampered)

    unknown = json.loads(
        json.dumps(preparation.to_wire(finished.config), ensure_ascii=False)
    )
    unknown["schema"] = "janki-prepared-deck-package-v99"
    with pytest.raises(deck_package.DeckPackageError, match="Unknown prepared"):
        deck_package.DeckPackagePreparation.from_wire(finished.config, unknown)


def test_the_wire_validates_its_paths_and_digests(tmp_path: Path) -> None:
    """Shape, repository-relative paths and digests, all checked on the way in.

    Mutation: accept an absolute or escaping staged path.
    """

    finished, preparation = _prepared(tmp_path)
    base = json.loads(
        json.dumps(preparation.to_wire(finished.config), ensure_ascii=False)
    )
    escaping = dict(base, staged_path="../outside.apkg")
    with pytest.raises(deck_package.DeckPackageError, match="not repository-relative"):
        deck_package.DeckPackagePreparation.from_wire(finished.config, escaping)

    malformed = dict(base, ledger_before_sha256="not-a-digest")
    with pytest.raises(deck_package.DeckPackageError, match="digest is malformed"):
        deck_package.DeckPackagePreparation.from_wire(finished.config, malformed)

    inventory = json.loads(json.dumps(base, ensure_ascii=False))
    inventory["inventory"]["notes"][0]["fields"][1] = "tampered"
    with pytest.raises(
        deck_package.DeckPackageError, match="inventory does not match its own"
    ):
        deck_package.DeckPackagePreparation.from_wire(finished.config, inventory)


def test_prepare_refuses_a_reference_store_edited_outside_the_prepared_payload(
    tmp_path: Path,
) -> None:
    """A bound input that moved refuses, naming it.

    Mutation: compare the source inputs as a set of paths only.
    """

    finished = _finish_through_audio(tmp_path)
    finished.config.kanji_file.write_text("{ }\n", encoding="utf-8")

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="kanji.json changed after the projection"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_prepare_refuses_a_canonical_change_through_the_proof(tmp_path: Path) -> None:
    """The proof is revalidated before anything is compared.

    Mutation: compare the plans without revalidating the proof.
    """

    finished = _finish_through_audio(tmp_path)
    records = load_records(finished.config.normalized_file.resolve())
    edited = [replace(record, meanings=["to converse"]) for record in records]
    finished.config.normalized_file.write_text(
        records_json_text(edited), encoding="utf-8"
    )

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="audio completion proof is no longer valid"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_prepare_refuses_a_template_edited_after_the_projection(
    tmp_path: Path,
) -> None:
    """Templates are bound inputs like any other.

    Mutation: skip the template input comparison.
    """

    finished = _finish_through_audio(tmp_path)
    front = finished.config.template_dir / "recognition-front.html"
    front.write_text(front.read_text(encoding="utf-8") + "\n<!-- moved -->\n", "utf-8")

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError,
        match="recognition-front.html changed after the projection",
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_prepare_refuses_while_an_unrelated_audio_transaction_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live write-ahead row is still a repository-wide refusal.

    Mutation: drop `_assert_pending_audio_clear` from the prepared path.
    """

    finished = _finish_through_audio(tmp_path)
    real_load = deck_package.ledger.load

    def pending(path: Path, **kwargs: object) -> object:
        book = real_load(path, **kwargs)
        book.pending_audio["unfinished"] = {"record_id": "word:x:x"}
        return book

    monkeypatch.setattr(deck_package.ledger, "load", pending)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="pending audio recovery"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_a_packaged_card_outside_the_built_deck_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """genanki expands one note over the enabled templates; both are checked.

    Mutation: record the card ordinals without comparing their deck.
    """

    finished = _finish_through_audio(tmp_path)

    def elsewhere(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"], "update cards set did = 99"
        )
        return members

    _tampering(monkeypatch, elsewhere)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="belongs to deck 99"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )


def test_a_packaged_note_missing_every_card_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An archive with no cards carries the whole deck and studies nothing.

    Every ordinal in an empty set is in range, so a bounds check accepts this:
    the notes, the exact fields, the tags and every media byte are present, and
    Anki imports a deck with nothing to review. What the archive has to carry
    is the expansion the bound renderer's notes produce, note by note.

    Mutation: record whatever ordinals the archive happens to hold.
    """

    finished = _finish_through_audio(tmp_path)

    def unexpanded(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"], "delete from cards"
        )
        return members

    _tampering(monkeypatch, unexpanded)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact cards"
    ) as refusal:
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert proof_fixtures.SPEAK in str(refusal.value)
    assert "the archive has no card" in str(refusal.value)
    assert "the build renders cards 0, 1" in str(refusal.value)


def test_a_packaged_note_repeating_one_direction_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two recognition cards are not a recognition and a production card.

    Both rows are in range and the count still matches the plan's
    ``card_count``, so neither a bounds check nor an aggregate total sees it;
    the production direction the owner enabled is simply absent from the deck
    they would import.

    Mutation: compare the ordinal set rather than the multiset genanki writes.
    """

    finished = _finish_through_audio(tmp_path)

    def collapsed(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"], "update cards set ord = 0"
        )
        return members

    _tampering(monkeypatch, collapsed)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact cards"
    ) as refusal:
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert proof_fixtures.SPEAK in str(refusal.value)
    assert "the archive has cards 0, 0" in str(refusal.value)
    assert "the build renders cards 0, 1" in str(refusal.value)


def test_a_packaged_note_with_an_extra_copy_of_one_card_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both directions are present and one of them is there twice.

    The ordinal *set* is right, so only the multiset sees this: the learner gets
    a third card that reviews the recognition side again and Anki schedules it
    separately for the life of the deck. Kept as its own case because a set
    comparison passes the collapsed archive above — that one loses a direction,
    this one does not.
    """

    finished = _finish_through_audio(tmp_path)

    def duplicated(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"],
            "insert into cards select id + 1000, nid, did, ord, mod, usn, type, "
            'queue, due, ivl, factor, reps, lapses, "left", odue, odid, flags, '
            "data from cards where ord = 0",
        )
        return members

    _tampering(monkeypatch, duplicated)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact cards"
    ) as refusal:
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert "the archive has cards 0, 0, 1" in str(refusal.value)
    assert "the build renders cards 0, 1" in str(refusal.value)


def test_the_inventory_expects_the_expansion_the_owning_exporter_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the archive is held to is genanki's expansion, not the directions.

    The two agree for every record this project can load: a word note needs at
    least one meaning (`validate_record`) and empty ones are dropped on load,
    so neither enabled template ever loses its question and both cards are
    always expanded. The real fixture's archive is therefore identical either
    way, and only the seam can say where the expectation came from — so the
    owning exporter is made to report a suppressed direction here. The builder
    still packages the notes themselves, so the archive keeps both cards, and
    the extra one has to be refused exactly as a missing one is. Asserting one
    card per enabled direction instead would refuse a package genanki wrote
    correctly the first time a template stops needing a field.

    Mutation: take the expectation from `card_types` instead of the expansion.
    """

    finished = _finish_through_audio(tmp_path)
    expand = deck_package.anki_exporter.expand_deck

    def recognition_only(*args: object, **kwargs: object):
        return tuple(
            replace(item, card_ordinals=item.card_ordinals[:1])
            for item in expand(*args, **kwargs)
        )

    monkeypatch.setattr(
        deck_package.anki_exporter, "expand_deck", recognition_only
    )

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact cards"
    ) as refusal:
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )

    assert "the archive has cards 0, 1" in str(refusal.value)
    assert "the build renders card 0." in str(refusal.value)


def test_a_packaged_note_with_dropped_tags_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tags reach no plan, no result and no hash; only the inventory sees them.

    Mutation: drop the note tags at the genanki boundary.
    """

    finished = _finish_through_audio(tmp_path)

    def strip(members: dict[str, bytes]) -> dict[str, bytes]:
        members["collection.anki2"] = _mutate_collection(
            members["collection.anki2"], "update notes set tags = ''"
        )
        return members

    _tampering(monkeypatch, strip)

    with _audio_operation(finished.config), pytest.raises(
        deck_package.DeckPackageError, match="does not carry the exact tags"
    ):
        deck_package.prepare_deck_package(
            finished.config, finished.projection, audio_completion=finished.proof
        )
