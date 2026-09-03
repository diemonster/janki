"""Exact application plans for every configured deck package."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import ledger
from japanese_anki.application import deck_build, deck_package
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import BuildResult
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
        "source": config.normalized_file,
        "kanji": config.kanji_file,
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
    ]
    assert tuple(item.path for item in plan.source_inputs) == (
        paths["source"].resolve(),
        paths["kanji"].resolve(),
    )
    assert tuple(item.path.name for item in plan.template_inputs) == (
        "recognition-front.html",
        "recognition-back.html",
        "style.css",
    )
    assert tuple(item.path for item in plan.media_inputs) == (paths["media"].resolve(),)
    for item in plan.all_inputs:
        assert item.sha256 == hashlib.sha256(item.path.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", ["deck", "source", "template", "media", "kanji"])
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
    else:
        paths["kanji"].write_text("{ }\n", encoding="utf-8")

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
