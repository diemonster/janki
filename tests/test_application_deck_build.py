"""Exact, deck-scoped builds for rich conjugation practice decks."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

from japanese_anki import cli, ledger
from japanese_anki.application import deck_build
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import RecordsRevision
from japanese_anki.models import VocabularyRecord

_PATTERN_TEMPLATES = (
    "pattern-front.html",
    "pattern-back.html",
    "style.css",
)


def _fixture(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "ledger.json"\n'
        'media_dir = "media"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        part_of_speech="verb",
        verb_group="godan",
    )
    config.normalized_file.write_text(
        json.dumps([record.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    config.deck_dir.mkdir()
    deck = config.deck_dir / "potential.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential practice\n"
        "  deck_id: 19\n"
        "  model_id: 20\n"
        "  source: ../vocabulary.json\n"
        "  output: potential.apkg\n"
        "  include_ids:\n"
        "    - word:話す:はなす\n"
        "  form_note: You can do the action.\n"
        "  drill_examples:\n"
        "    word:話す:はなす:\n"
        "      - japanese: 日本語が話せます。\n"
        "        furigana: '日本語[にほんご]が 話[はな]せます。'\n"
        "        english: I can speak Japanese.\n"
        "        register: polite\n"
        "      - japanese: 日本語が話せる？\n"
        "        furigana: '日本語[にほんご]が 話[はな]せる？'\n"
        "        english: Can you speak Japanese?\n"
        "        register: casual\n",
        encoding="utf-8",
    )
    return config, deck


def _add_audio(config: ProjectConfig, deck: Path) -> tuple[Path, Path]:
    polite = config.media_dir / "audio" / "polite.wav"
    casual = config.media_dir / "audio" / "casual.wav"
    polite.parent.mkdir(parents=True)
    polite.write_bytes(b"polite exact bytes")
    casual.write_bytes(b"casual exact bytes")
    text = deck.read_text(encoding="utf-8")
    text = text.replace(
        "        register: polite\n",
        "        register: polite\n        audio: audio/polite.wav\n",
    ).replace(
        "        register: casual\n",
        "        register: casual\n        audio: audio/casual.wav\n",
    )
    deck.write_text(text, encoding="utf-8")
    return polite.resolve(), casual.resolve()


def test_deck_build_plans_without_writing_then_builds_the_exact_package(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)

    plan = deck_build.plan_conjugation_deck_build(config, deck)

    assert plan.deck_path == deck.resolve()
    assert plan.output_path == (config.dist_dir / "potential.apkg").resolve()
    assert plan.deck_name == "Potential practice"
    assert plan.form == "potential"
    assert plan.card_count == 1
    assert not plan.output_path.exists(), "planning is display-only"

    result = deck_build.execute_conjugation_deck_build(config, plan)

    assert result.output_path == plan.output_path
    assert result.card_count == 1
    assert result.output_path.read_bytes().startswith(b"PK")
    assert len(result.package_sha256) == 64


@pytest.mark.parametrize("filename", _PATTERN_TEMPLATES)
def test_deck_build_plan_binds_every_template_path_hash_and_fingerprint(
    tmp_path: Path,
    filename: str,
) -> None:
    config, deck = _fixture(tmp_path)
    before = deck_build.plan_conjugation_deck_build(config, deck)
    before_by_name = {item.path.name: item for item in before.template_inputs}

    assert tuple(item.path.name for item in before.template_inputs) == _PATTERN_TEMPLATES
    assert before_by_name[filename].path == (config.template_dir / filename).resolve()
    assert before_by_name[filename].sha256 == hashlib.sha256(
        (config.template_dir / filename).read_bytes()
    ).hexdigest()

    path = config.template_dir / filename
    path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    after = deck_build.plan_conjugation_deck_build(config, deck)
    after_by_name = {item.path.name: item for item in after.template_inputs}

    assert after_by_name[filename].sha256 != before_by_name[filename].sha256
    assert after.fingerprint != before.fingerprint
    assert {
        name: item.sha256
        for name, item in after_by_name.items()
        if name != filename
    } == {
        name: item.sha256
        for name, item in before_by_name.items()
        if name != filename
    }


def test_deck_build_plan_binds_every_referenced_media_path_hash_and_fingerprint(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    polite, casual = _add_audio(config, deck)

    before = deck_build.plan_conjugation_deck_build(config, deck)

    assert before.media_inputs == (
        deck_build.ConjugationMediaInput(
            path=casual,
            sha256=hashlib.sha256(casual.read_bytes()).hexdigest(),
        ),
        deck_build.ConjugationMediaInput(
            path=polite,
            sha256=hashlib.sha256(polite.read_bytes()).hexdigest(),
        ),
    )

    polite.write_bytes(b"changed exact bytes")
    after = deck_build.plan_conjugation_deck_build(config, deck)

    assert after.media_inputs[1].sha256 != before.media_inputs[1].sha256
    assert after.media_inputs[0] == before.media_inputs[0]
    assert after.fingerprint != before.fingerprint


def test_deck_build_refuses_media_drift_before_calling_the_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    polite, _casual = _add_audio(config, deck)
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    polite.write_bytes(b"changed after confirmation")
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("media drift must refuse before package generation")

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", forbidden)

    with pytest.raises(deck_build.DeckBuildError, match="plan changed"):
        deck_build.execute_conjugation_deck_build(config, plan)

    assert called is False
    assert not plan.output_path.exists()


def test_deck_build_refuses_media_swapped_during_package_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    polite, _casual = _add_audio(config, deck)
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    real_build = pattern_cards.build_conjugation_deck

    def raced_build(
        deck_path: Path,
        current: ProjectConfig,
        records: tuple[VocabularyRecord, ...],
        output: Path,
    ) -> tuple[Path, int]:
        result = real_build(deck_path, current, records, output)
        polite.write_bytes(b"swapped at the build seam")
        return result

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", raced_build)

    with pytest.raises(deck_build.DeckBuildError, match="during package generation"):
        deck_build.execute_conjugation_deck_build(config, plan)

    assert plan.output_path.exists(), "the dist artifact is disposable after refusal"


def test_projected_build_names_future_media_but_is_not_executable_without_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    polite, casual = _add_audio(config, deck)
    polite.unlink()
    casual.unlink()
    revision = RecordsRevision(deck.resolve(), deck.read_text(encoding="utf-8"))
    projected = deck_build.plan_conjugation_deck_build_revision(
        config,
        deck,
        revision,
        projected_media={polite: None, casual: None},
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("an unbound projected build cannot package media")

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", forbidden)

    assert {item.path for item in projected.media_inputs} == {polite, casual}
    assert all(item.sha256 is None for item in projected.media_inputs)
    with pytest.raises(deck_build.DeckBuildError, match="not fully produced"):
        deck_build.execute_conjugation_deck_build_locked(config, projected)

    assert called is False
    assert not projected.output_path.exists()


def test_projected_build_refuses_a_future_media_target_the_deck_does_not_reference(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    polite, casual = _add_audio(config, deck)
    revision = RecordsRevision(deck.resolve(), deck.read_text(encoding="utf-8"))

    with pytest.raises(deck_build.DeckBuildError, match="target set"):
        deck_build.plan_conjugation_deck_build_revision(
            config,
            deck,
            revision,
            projected_media={
                polite: hashlib.sha256(polite.read_bytes()).hexdigest(),
                casual: hashlib.sha256(casual.read_bytes()).hexdigest(),
                config.media_dir / "audio" / "not-referenced.wav": None,
            },
        )


@pytest.mark.parametrize("filename", _PATTERN_TEMPLATES)
def test_deck_build_refuses_template_drift_before_calling_the_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    config, deck = _fixture(tmp_path)
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    path = config.template_dir / filename
    path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("template drift must refuse before package generation")

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", forbidden)

    with pytest.raises(deck_build.DeckBuildError, match="plan changed"):
        deck_build.execute_conjugation_deck_build(config, plan)

    assert called is False
    assert not plan.output_path.exists()


def test_deck_build_holds_every_template_and_media_lock_through_package_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    media = set(_add_audio(config, deck))
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    real_lock = deck_build.exclusive_path_lock
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
        _records: tuple[VocabularyRecord, ...],
        output: Path,
    ) -> tuple[Path, int]:
        expected = set(pattern_cards.pattern_template_paths(config.template_dir)) | media
        assert expected <= {path for path, count in active.items() if count > 0}
        output.write_bytes(b"PK template-lock proof")
        return output, 1

    monkeypatch.setattr(deck_build, "exclusive_path_lock", tracking_lock)
    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", build)

    result = deck_build.execute_conjugation_deck_build(config, plan)

    assert result.output_path == plan.output_path
    assert result.card_count == 1


def test_deck_build_replans_at_click_and_refuses_stale_deck_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _fixture(tmp_path)
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "You can do the action.", "Changed after confirmation."
        ),
        encoding="utf-8",
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("a stale build must not publish")

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", forbidden)

    with pytest.raises(deck_build.DeckBuildError, match="plan changed"):
        deck_build.execute_conjugation_deck_build(config, plan)

    assert called is False
    assert not plan.output_path.exists()


def test_deck_build_refuses_while_any_paid_audio_transaction_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _fixture(tmp_path)
    plan = deck_build.plan_conjugation_deck_build(config, deck)
    book = ledger.Ledger(path=config.ledger_file)
    owner = pattern_cards.drill_audio_owner_id(
        19, "potential", "word:話す:はなす"
    )
    identity = book._pending_audio_identity(
        owner,
        of="example",
        target="pending.wav",
        request_input="日本語が話せます。",
        forced_accent=False,
        content_fp="a" * 64,
        provider="openai-realtime",
        voice="cedar",
        speed=1.0,
        settings={},
    )
    key = book._pending_audio_key(identity)
    book.record_pending_audio(
        owner,
        of="example",
        target="pending.wav",
        request_input="日本語が話せます。",
        forced_accent=False,
        content_fp="a" * 64,
        provider="openai-realtime",
        voice="cedar",
        speed=1.0,
        settings={},
        staged_file=f".pending/{key}-{'b' * 64}.stage",
        staged_sha256="b" * 64,
    )
    book.save()
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pending audio must refuse before build")

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", forbidden)

    with pytest.raises(
        deck_build.DeckBuildError,
        match="repository has pending audio recovery",
    ):
        deck_build.execute_conjugation_deck_build(config, plan)

    assert called is False


def test_deck_build_refuses_unconfigured_decks(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(deck.read_bytes())

    with pytest.raises(deck_build.DeckBuildError, match="configured deck"):
        deck_build.plan_conjugation_deck_build(config, outside)


def test_deck_build_refuses_outputs_outside_dist(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "output: potential.apkg", "output: ../outside.apkg"
        ),
        encoding="utf-8",
    )
    with pytest.raises(deck_build.DeckBuildError, match="direct .apkg"):
        deck_build.plan_conjugation_deck_build(config, deck)


def test_projected_deck_build_equals_ordinary_plan_after_exact_bytes_land(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    before = deck.read_text(encoding="utf-8")
    projected_source = config.root / "projected-vocabulary.json"
    projected_source.write_bytes(config.normalized_file.read_bytes())
    projected_text = before.replace(
        "name: Potential practice\n", "name: Projected practice\n"
    ).replace(
        "source: ../vocabulary.json\n", "source: ../projected-vocabulary.json\n"
    ).replace("output: potential.apkg\n", "output: projected.apkg\n")
    revision = RecordsRevision(deck.resolve(), projected_text)

    projected = deck_build.plan_conjugation_deck_build_revision(
        config,
        deck,
        revision,
    )

    assert deck.read_text(encoding="utf-8") == before
    assert projected.deck_name == "Projected practice"
    assert projected.source_path == projected_source.resolve()
    assert projected.output_path == (config.dist_dir / "projected.apkg").resolve()
    assert projected.card_count == 1
    deck.write_text(projected_text, encoding="utf-8")
    assert projected == deck_build.plan_conjugation_deck_build(config, deck)


def test_projected_deck_build_refuses_a_missing_snapshot_source(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    projected_text = deck.read_text(encoding="utf-8").replace(
        "source: ../vocabulary.json", "source: ../missing.json"
    )

    with pytest.raises(deck_build.DeckBuildError, match="no collection"):
        deck_build.plan_conjugation_deck_build_revision(
            config,
            deck,
            RecordsRevision(deck.resolve(), projected_text),
        )


def test_projected_deck_build_refuses_a_non_conjugation_snapshot(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    projected_text = deck.read_text(encoding="utf-8").replace(
        "kind: conjugation", "kind: vocabulary"
    )

    with pytest.raises(deck_build.DeckBuildError, match="conjugation deck"):
        deck_build.plan_conjugation_deck_build_revision(
            config,
            deck,
            RecordsRevision(deck.resolve(), projected_text),
        )


def test_projected_deck_build_refuses_a_snapshot_for_another_path(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)
    revision = RecordsRevision(
        (config.deck_dir / "another.yaml").resolve(),
        deck.read_text(encoding="utf-8"),
    )

    with pytest.raises(deck_build.DeckBuildError, match="cannot bind"):
        deck_build.plan_conjugation_deck_build_revision(config, deck, revision)


def test_projected_deck_build_refuses_a_missing_snapshot(
    tmp_path: Path,
) -> None:
    config, deck = _fixture(tmp_path)

    with pytest.raises(deck_build.DeckBuildError, match="snapshot is missing"):
        deck_build.plan_conjugation_deck_build_revision(
            config,
            deck,
            RecordsRevision(deck.resolve(), None),
        )


def test_projected_deck_build_refuses_source_drift_during_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    revision = RecordsRevision(deck.resolve(), deck.read_text(encoding="utf-8"))
    real_revision = deck_build.records_revision

    def drift(path: Path) -> RecordsRevision:
        captured = real_revision(path)
        if path.resolve() == config.normalized_file.resolve():
            return RecordsRevision(captured.path, f"{captured.text}\n")
        return captured

    monkeypatch.setattr(deck_build, "records_revision", drift)

    with pytest.raises(deck_build.DeckBuildError, match="source changed"):
        deck_build.plan_conjugation_deck_build_revision(config, deck, revision)


def test_projected_deck_build_refuses_configured_deck_drift_during_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    revision = RecordsRevision(deck.resolve(), deck.read_text(encoding="utf-8"))
    real_revision = deck_build.records_revision
    deck_reads = 0

    def drift(path: Path) -> RecordsRevision:
        nonlocal deck_reads
        captured = real_revision(path)
        if path.resolve() == deck.resolve():
            deck_reads += 1
            if deck_reads > 1:
                return RecordsRevision(captured.path, f"{captured.text}\n")
        return captured

    monkeypatch.setattr(deck_build, "records_revision", drift)

    with pytest.raises(deck_build.DeckBuildError, match="configured deck changed"):
        deck_build.plan_conjugation_deck_build_revision(config, deck, revision)


def test_cli_conjugation_build_routes_through_the_shared_application_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _fixture(tmp_path)
    real_plan = deck_build.plan_conjugation_deck_build
    real_execute = deck_build.execute_conjugation_deck_build_locked
    called: list[str] = []

    def plan(current: ProjectConfig, target: Path | str):
        called.append("plan")
        return real_plan(current, target)

    def execute(
        current: ProjectConfig,
        expected: deck_build.ConjugationDeckBuildPlan,
    ):
        called.append("execute")
        return real_execute(current, expected)

    monkeypatch.setattr(deck_build, "plan_conjugation_deck_build", plan)
    monkeypatch.setattr(
        deck_build,
        "execute_conjugation_deck_build_locked",
        execute,
    )

    assert cli.main(["--root", str(config.root), "build", str(deck)]) == 0
    assert called == ["plan", "execute"]
