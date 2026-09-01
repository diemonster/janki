"""Exact, deck-scoped builds for rich conjugation practice decks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from japanese_anki import cli, ledger
from japanese_anki.application import deck_build
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import pattern_cards
from japanese_anki.models import VocabularyRecord


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
