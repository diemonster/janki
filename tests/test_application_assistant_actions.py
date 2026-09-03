"""Closed Assistant action plans over existing audio/build services."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from japanese_anki.application import assistant_actions
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_package as deck_package_application
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.config import ProjectConfig
from japanese_anki.models import ExampleSentence, VocabularyRecord


@dataclass(frozen=True)
class Intent:
    kind: str
    resource_ids: tuple[str, ...]
    record_ids: tuple[str, ...] = ()
    instruction: str = "Do exactly this action."
    options: Mapping[str, object] = field(default_factory=dict)


class Provider:
    launch_hint = "not used"
    suffix = ".wav"

    def __init__(self, name: str, voice: int | str) -> None:
        self.name = name
        self.voice = voice
        self.speed = 0.75
        self.settings = {"model": "test-model"}
        self._base_url = name
        self._transport = self.synthesize

    def available(self) -> bool:
        raise AssertionError("action planning must not contact a provider")

    def synthesize(self, _text: str, *, forced_accent: bool) -> bytes:
        raise AssertionError(f"action planning must not dispatch audio ({forced_accent=})")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _project(tmp_path: Path) -> tuple[ProjectConfig, Path, Path, Path, tuple[str, str]]:
    (tmp_path / "janki.toml").write_text(
        '[project]\nname = "Action fixture"\n'
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "data/ledger.json"\n'
        'media_dir = "data/media"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    first = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        part_of_speech="verb",
        verb_group="godan",
        pitch_accent=["LHHH"],
        examples=[
            ExampleSentence(
                japanese="日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
                register="polite",
            )
        ],
    )
    second = VocabularyRecord(
        id="word:遊ぶ:あそぶ",
        expression="遊ぶ",
        reading="あそぶ",
        meanings=["to play"],
        part_of_speech="verb",
        verb_group="godan",
        pitch_accent=["LHHH"],
        examples=[
            ExampleSentence(
                japanese="友達と遊びます。",
                furigana="友達[ともだち]と 遊[あそ]びます。",
                english="I play with a friend.",
                register="polite",
            )
        ],
    )
    _write_json(config.normalized_file, [first.to_dict(), second.to_dict()])
    config.deck_dir.mkdir(parents=True)
    vocabulary = config.deck_dir / "lesson.yaml"
    vocabulary.write_text(
        "deck:\n"
        "  name: Lesson deck\n"
        "  source: ../normalized/vocabulary.json\n"
        "  include_ids: [word:話す:はなす]\n",
        encoding="utf-8",
    )
    conjugation = config.deck_dir / "potential.yaml"
    conjugation.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential practice\n"
        "  deck_id: 19\n"
        "  model_id: 20\n"
        "  source: ../normalized/vocabulary.json\n"
        "  output: potential.apkg\n"
        "  include_ids: [word:話す:はなす]\n"
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
    pattern = config.deck_dir / "rules.yaml"
    pattern.write_text(
        "deck:\n"
        "  kind: pattern\n"
        "  name: Rules\n"
        "  deck_id: 21\n"
        "  model_id: 22\n"
        "  document: rules.pdf\n",
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
    return config, vocabulary, conjugation, pattern, (first.id, second.id)


def _resource_id(config: ProjectConfig, deck: Path) -> str:
    return AssistantContextBroker(config).resource_id_for_deck(deck)


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_audio_plan_resolves_opaque_vocabulary_deck_and_is_side_effect_free(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    before = _snapshot_files(tmp_path)

    plan = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), instruction="Fill missing audio."),
        word_provider=words,
        sentence_provider=sentences,
    )

    assert _snapshot_files(tmp_path) == before
    assert plan.kind == "generate_audio"
    assert plan.resource_id == resource_id
    assert plan.deck_path == vocabulary.resolve()
    assert plan.deck_name == "Lesson deck"
    assert plan.deck_kind == "vocabulary"
    assert plan.requested_record_ids == ()
    assert isinstance(plan.service_plan, audio_application.AudioPlan)
    assert plan.service_plan.record_ids == (ids[0],)
    assert plan.service_plan.words is True
    assert plan.service_plan.examples is True
    assert plan.service_plan.force is False
    assert plan.projection["service_fingerprint"] == plan.service_fingerprint
    assert plan.projection["target"]["resource_id"] == resource_id
    assert plan.projection["target"]["configured_file"] == "data/decks/lesson.yaml"
    assert plan.projection["writes"] == {
        "ledger": "data/ledger.json",
        "media_directory": "data/media/audio",
    }
    assert str(config.root.resolve()) not in plan.projection_wire
    assert plan.fingerprint == hashlib.sha256(plan.projection_wire.encode("utf-8")).hexdigest()


def test_audio_plan_binds_explicit_record_to_the_selected_deck(tmp_path: Path) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    providers = {
        "word_provider": Provider("voicevox", 7),
        "sentence_provider": Provider("openai-realtime", "cedar"),
    }

    selected = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), (ids[0],)),
        **providers,
    )

    assert selected.requested_record_ids == (ids[0],)
    assert selected.service_plan.record_ids == (ids[0],)
    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="not a card in deck 'Lesson deck'",
    ):
        assistant_actions.plan_action(
            config,
            Intent("generate_audio", (resource_id,), (ids[1],)),
            **providers,
        )


def test_audio_plan_uses_safe_deck_defaults_or_two_explicit_clip_choices(
    tmp_path: Path,
) -> None:
    config, vocabulary, conjugation, _pattern, _ids = _project(tmp_path)
    providers = {
        "word_provider": Provider("voicevox", 7),
        "sentence_provider": Provider("openai-realtime", "cedar"),
    }

    default_vocabulary = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (_resource_id(config, vocabulary),)),
        **providers,
    )
    explicit_examples = assistant_actions.plan_action(
        config,
        Intent(
            "generate_audio",
            (_resource_id(config, vocabulary),),
            options={"audio_words": False, "audio_examples": True},
        ),
        **providers,
    )
    default_conjugation = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (_resource_id(config, conjugation),)),
        **providers,
    )

    assert default_vocabulary.projection["options"] == {
        "words": True,
        "examples": True,
        "force": False,
        "prune": False,
        "clip_classes_source": "deck-default",
    }
    assert explicit_examples.projection["options"] == {
        "words": False,
        "examples": True,
        "force": False,
        "prune": False,
        "clip_classes_source": "owner-explicit",
    }
    assert explicit_examples.service_plan.words is False
    assert explicit_examples.service_plan.examples is True
    assert default_conjugation.projection["options"] == {
        "words": False,
        "examples": True,
        "force": False,
        "prune": False,
        "clip_classes_source": "deck-default",
    }


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"audio_words": True}, "both audio_words and audio_examples"),
        (
            {"audio_words": False, "audio_examples": False},
            "at least one clip class",
        ),
        (
            {"audio_words": 1, "audio_examples": True},
            "true or false",
        ),
        ({"audio_force": "yes"}, "true or false"),
        ({"audio_prune": "yes"}, "true or false"),
        ({"force": True}, "unsupported option"),
    ],
)
def test_audio_options_refuse_partial_ambiguous_or_untyped_choices(
    tmp_path: Path,
    options: dict[str, object],
    match: str,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)

    with pytest.raises(assistant_actions.AssistantActionError, match=match):
        assistant_actions.plan_action(
            config,
            Intent(
                "generate_audio",
                (_resource_id(config, vocabulary),),
                options=options,
            ),
            word_provider=Provider("voicevox", 7),
            sentence_provider=Provider("openai-realtime", "cedar"),
        )


def test_conjugation_audio_refuses_word_or_repository_prune_choices(
    tmp_path: Path,
) -> None:
    config, _vocabulary, conjugation, _pattern, _ids = _project(tmp_path)
    resource_id = _resource_id(config, conjugation)
    providers = {
        "word_provider": Provider("voicevox", 7),
        "sentence_provider": Provider("openai-realtime", "cedar"),
    }

    with pytest.raises(assistant_actions.AssistantActionError, match="examples only"):
        assistant_actions.plan_action(
            config,
            Intent(
                "generate_audio",
                (resource_id,),
                options={"audio_words": True, "audio_examples": True},
            ),
            **providers,
        )
    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="repository-wide prune",
    ):
        assistant_actions.plan_action(
            config,
            Intent(
                "generate_audio",
                (resource_id,),
                options={"audio_prune": True},
            ),
            **providers,
        )


def test_force_and_prune_are_bound_to_exact_replacements_and_deletions(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    providers = {
        "word_provider": Provider("voicevox", 7),
        "sentence_provider": Provider("openai-realtime", "cedar"),
    }
    baseline = assistant_actions.plan_action(
        config,
        Intent(
            "generate_audio",
            (resource_id,),
            options={"audio_words": True, "audio_examples": False},
        ),
        **providers,
    )
    [clip] = baseline.service_plan.clips
    audio_dir = config.media_dir / "audio"
    audio_dir.mkdir(parents=True)
    existing = audio_dir / clip.target
    existing.write_bytes(b"existing selected clip")
    stale = audio_dir / "janki-stale.wav"
    stale.write_bytes(b"unreferenced generated clip")

    plan = assistant_actions.plan_action(
        config,
        Intent(
            "generate_audio",
            (resource_id,),
            options={
                "audio_words": True,
                "audio_examples": False,
                "audio_force": True,
                "audio_prune": True,
            },
        ),
        **providers,
    )

    assert plan.service_plan.force is True
    assert plan.projection["options"] == {
        "words": True,
        "examples": False,
        "force": True,
        "prune": True,
        "clip_classes_source": "owner-explicit",
    }
    [replacement] = plan.projection["force_replacements"]
    assert replacement["record_id"] == clip.record_id
    assert replacement["kind"] == "word"
    assert replacement["target"] == clip.target
    current = replacement["existing"]
    assert current["file"] == f"data/media/audio/{clip.target}"
    assert current["entry_type"] == "regular-file"
    assert current["bytes"] == len(b"existing selected clip")
    assert current["sha256"] == hashlib.sha256(b"existing selected clip").hexdigest()
    details = existing.stat()
    assert current["identity"] == [
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    ]
    deletion = plan.projection["cleanup"]["media_files_removed"]
    assert [item["file"] for item in deletion] == ["data/media/audio/janki-stale.wav"]
    assert deletion[0]["sha256"] == hashlib.sha256(
        b"unreferenced generated clip"
    ).hexdigest()
    assert plan.projection["cleanup"]["scope"] == (
        "repository-wide-unreferenced-janki-audio"
    )
    assert plan.fingerprint != baseline.fingerprint


def test_assistant_audio_refuses_a_symlinked_audio_directory_before_prune(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-audio"
    outside.mkdir()
    outside_clip = outside / "janki-outside.wav"
    outside_clip.write_bytes(b"must remain outside the repository")
    config.media_dir.mkdir(parents=True)
    (config.media_dir / "audio").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="audio directory.*symbolic link",
    ):
        assistant_actions.plan_action(
            config,
            Intent(
                "generate_audio",
                (_resource_id(config, vocabulary),),
                options={"audio_prune": True},
            ),
            word_provider=Provider("voicevox", 7),
            sentence_provider=Provider("openai-realtime", "cedar"),
        )

    assert outside_clip.read_bytes() == b"must remain outside the repository"


def test_audio_action_fingerprint_binds_instruction_and_provider_plan(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    first = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), instruction="Fill missing audio."),
        word_provider=words,
        sentence_provider=sentences,
    )
    changed_instruction = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), instruction="Prepare missing audio."),
        word_provider=words,
        sentence_provider=sentences,
    )
    changed_provider = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), instruction="Fill missing audio."),
        word_provider=words,
        sentence_provider=Provider("openai-realtime", "marin"),
    )

    assert changed_instruction.fingerprint != first.fingerprint
    assert changed_provider.service_fingerprint != first.service_fingerprint
    assert changed_provider.fingerprint != first.fingerprint


def test_audio_refuses_a_deck_owned_card_version_the_canonical_writer_cannot_update(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    vocabulary.write_text(
        vocabulary.read_text(encoding="utf-8")
        + "notes:\n"
        + f"  - id: {ids[0]}\n"
        + "    expression: 話す\n"
        + "    reading: はなす\n"
        + "    meanings: [to converse]\n",
        encoding="utf-8",
    )

    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="deck-specific version",
    ):
        assistant_actions.plan_action(
            config,
            Intent("generate_audio", (resource_id,)),
            word_provider=Provider("voicevox", 7),
            sentence_provider=Provider("openai-realtime", "cedar"),
        )


def test_audio_execution_replans_then_calls_existing_targeted_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    plan = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), (ids[0],)),
        word_provider=words,
        sentence_provider=sentences,
    )
    calls: list[tuple[ProjectConfig, tuple[str, ...], dict[str, object]]] = []
    sentinel = audio_application.AudioExecutionOutcome(
        state="no-records",
        plan=None,
        output_dir=config.media_dir / "audio",
        no_records=True,
    )

    def execute(
        current: ProjectConfig,
        record_ids: tuple[str, ...],
        **options: object,
    ) -> audio_application.AudioExecutionOutcome:
        calls.append((current, tuple(record_ids), options))
        return sentinel

    monkeypatch.setattr(audio_application, "execute_targeted_audio_locked", execute)
    monkeypatch.setattr(
        audio_application,
        "execute_targeted_audio",
        lambda *_args, **_kwargs: pytest.fail(
            "Assistant execution must use the already locked audio writer"
        ),
    )

    result = assistant_actions.execute_action(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert result.result is sentinel
    assert result.plan.fingerprint == plan.fingerprint
    assert calls == [
        (
            config,
            (ids[0],),
            {
                "words": True,
                "examples": True,
                "expected_fingerprint": plan.service_fingerprint,
                "force": False,
                "prune": False,
                "progress": None,
                "word_provider": words,
                "sentence_provider": sentences,
            },
        )
    ]


def test_audio_execution_replans_under_audio_lock_and_preserves_explicit_choices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    plan = assistant_actions.plan_action(
        config,
        Intent(
            "generate_audio",
            (_resource_id(config, vocabulary),),
            (ids[0],),
            options={
                "audio_words": False,
                "audio_examples": True,
                "audio_force": True,
                "audio_prune": True,
            },
        ),
        word_provider=words,
        sentence_provider=sentences,
    )
    calls: list[tuple[ProjectConfig, tuple[str, ...], dict[str, object]]] = []
    sentinel = audio_application.AudioExecutionOutcome(
        state="complete",
        plan=plan.service_plan,
        output_dir=config.media_dir / "audio",
        prune_requested=True,
    )

    def execute_locked(
        current: ProjectConfig,
        record_ids: tuple[str, ...],
        **options: object,
    ) -> audio_application.AudioExecutionOutcome:
        calls.append((current, tuple(record_ids), options))
        return sentinel

    monkeypatch.setattr(audio_application, "execute_targeted_audio_locked", execute_locked)
    monkeypatch.setattr(
        audio_application,
        "execute_targeted_audio",
        lambda *_args, **_kwargs: pytest.fail(
            "Assistant execution must retain the lock across re-plan and dispatch"
        ),
    )

    result = assistant_actions.execute_action(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert result.result is sentinel
    assert calls == [
        (
            config,
            (ids[0],),
            {
                "words": False,
                "examples": True,
                "expected_fingerprint": plan.service_fingerprint,
                "force": True,
                "prune": True,
                "progress": None,
                "word_provider": words,
                "sentence_provider": sentences,
            },
        )
    ]


def test_audio_execution_refuses_fresh_plan_drift_before_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, vocabulary, _conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    plan = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,)),
        word_provider=words,
        sentence_provider=sentences,
    )
    vocabulary.write_text(
        vocabulary.read_text(encoding="utf-8").replace(ids[0], ids[1]),
        encoding="utf-8",
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("stale action reached the existing writer")

    monkeypatch.setattr(audio_application, "execute_targeted_audio", forbidden)

    with pytest.raises(assistant_actions.AssistantActionError, match="plan changed"):
        assistant_actions.execute_action(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert called is False


@pytest.mark.parametrize("changed_binding", ["forced-replacement", "prune-candidate"])
def test_audio_execution_refuses_changed_destructive_file_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_binding: str,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    resource_id = _resource_id(config, vocabulary)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")
    baseline = assistant_actions.plan_action(
        config,
        Intent(
            "generate_audio",
            (resource_id,),
            options={"audio_words": True, "audio_examples": False},
        ),
        word_provider=words,
        sentence_provider=sentences,
    )
    [clip] = baseline.service_plan.clips
    audio_dir = config.media_dir / "audio"
    audio_dir.mkdir(parents=True)
    target = (
        audio_dir / clip.target
        if changed_binding == "forced-replacement"
        else audio_dir / "janki-stale.wav"
    )
    target.write_bytes(b"rendered bytes")
    options = {
        "audio_words": True,
        "audio_examples": False,
        "audio_force": changed_binding == "forced-replacement",
        "audio_prune": changed_binding == "prune-candidate",
    }
    plan = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), options=options),
        word_provider=words,
        sentence_provider=sentences,
    )
    target.write_bytes(b"different bytes after confirmation rendered")
    monkeypatch.setattr(
        audio_application,
        "execute_targeted_audio_locked",
        lambda *_args, **_kwargs: pytest.fail(
            "a changed destructive binding must not reach the audio writer"
        ),
    )

    with pytest.raises(assistant_actions.AssistantActionError, match="plan changed"):
        assistant_actions.execute_action(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )


def test_conjugation_audio_maps_card_ids_to_distinct_drill_audio_owners(
    tmp_path: Path,
) -> None:
    config, _vocabulary, conjugation, _pattern, ids = _project(tmp_path)
    resource_id = _resource_id(config, conjugation)

    plan = assistant_actions.plan_action(
        config,
        Intent("generate_audio", (resource_id,), (ids[0],)),
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai-realtime", "cedar"),
    )

    assert isinstance(plan.service_plan, audio_application.AudioPlan)
    assert plan.service_plan.record_ids == (f"drill-audio:19:potential:{ids[0]}",)
    assert plan.service_plan.words is False
    assert plan.service_plan.examples is True
    assert plan.projection["target"]["requested_record_ids"] == [ids[0]]
    assert plan.projection["target"]["audio_owner_ids"] == [f"drill-audio:19:potential:{ids[0]}"]


def test_build_plan_and_execution_use_shared_package_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _vocabulary, conjugation, _pattern, _ids = _project(tmp_path)
    resource_id = _resource_id(config, conjugation)
    before = _snapshot_files(tmp_path)
    plan = assistant_actions.plan_action(
        config,
        Intent("build_deck", (resource_id,), instruction="Build this deck."),
    )

    assert _snapshot_files(tmp_path) == before
    assert plan.kind == "build_deck"
    assert plan.deck_path == conjugation.resolve()
    assert isinstance(plan.service_plan, deck_package_application.DeckPackagePlan)
    assert plan.service_plan.kind == "conjugation"
    assert plan.projection["target"]["card_count"] == 1
    assert plan.projection["writes"] == {
        "package": "dist/potential.apkg",
        "package_precondition": {
            "state": "absent",
            "sha256": None,
            "identity": None,
        },
    }
    assert plan.projection["billing_class"] == "local"
    assert str(config.root.resolve()) not in plan.projection_wire
    calls: list[deck_package_application.DeckPackagePlan] = []
    sentinel = deck_package_application.DeckPackageResult(
        output_path=plan.service_plan.output_path,
        note_count=1,
        card_count=1,
        card_types=("drill",),
        media_count=0,
        package_sha256="1" * 64,
    )

    def execute(
        current: ProjectConfig,
        expected: deck_package_application.DeckPackagePlan,
    ) -> deck_package_application.DeckPackageResult:
        assert current is config
        calls.append(expected)
        return sentinel

    monkeypatch.setattr(deck_package_application, "execute_deck_package", execute)

    result = assistant_actions.execute_action(config, plan)

    assert result.result is sentinel
    assert calls == [plan.service_plan]


def test_unknown_ambiguous_and_out_of_scope_targets_refuse_precisely(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, pattern, ids = _project(tmp_path)
    vocabulary_id = _resource_id(config, vocabulary)
    pattern_id = _resource_id(config, pattern)
    providers = {
        "word_provider": Provider("voicevox", 7),
        "sentence_provider": Provider("openai-realtime", "cedar"),
    }

    with pytest.raises(assistant_actions.AssistantActionError, match="Unknown configured"):
        assistant_actions.plan_action(
            config, Intent("generate_audio", ("resource_not_from_catalog",)), **providers
        )
    with pytest.raises(assistant_actions.AssistantActionError, match="exactly one"):
        assistant_actions.plan_action(
            config,
            Intent("generate_audio", (vocabulary_id, pattern_id)),
            **providers,
        )
    with pytest.raises(assistant_actions.AssistantActionError, match="supplied twice"):
        assistant_actions.plan_action(
            config,
            Intent("generate_audio", (vocabulary_id,), (ids[0], ids[0])),
            **providers,
        )
    with pytest.raises(assistant_actions.AssistantActionError, match="Pattern deck"):
        assistant_actions.plan_action(config, Intent("generate_audio", (pattern_id,)), **providers)


def test_build_supports_vocabulary_and_refuses_record_narrowing(tmp_path: Path) -> None:
    config, vocabulary, conjugation, _pattern, ids = _project(tmp_path)
    vocabulary_id = _resource_id(config, vocabulary)
    conjugation_id = _resource_id(config, conjugation)

    plan = assistant_actions.plan_action(config, Intent("build_deck", (vocabulary_id,)))

    assert isinstance(plan.service_plan, deck_package_application.DeckPackagePlan)
    assert plan.service_plan.kind == "vocabulary"
    assert plan.projection["target"]["card_types"] == ["recognition", "production"]
    assert plan.projection["writes"]["export_history"] == "data/ledger.json"
    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="complete configured deck",
    ):
        assistant_actions.plan_action(config, Intent("build_deck", (conjugation_id,), (ids[0],)))

    with pytest.raises(assistant_actions.AssistantActionError, match="does not accept audio"):
        assistant_actions.plan_action(
            config,
            Intent(
                "build_deck",
                (vocabulary_id,),
                options={"audio_force": True},
            ),
        )


def test_build_projection_names_the_exact_existing_package_it_will_replace(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    config.dist_dir.mkdir(parents=True)
    existing = config.dist_dir / "lesson.apkg"
    existing.write_bytes(b"the currently published package")

    plan = assistant_actions.plan_action(
        config,
        Intent("build_deck", (_resource_id(config, vocabulary),)),
    )

    precondition = plan.projection["writes"]["package_precondition"]
    assert precondition == {
        "state": "replace_exact",
        "sha256": hashlib.sha256(existing.read_bytes()).hexdigest(),
        "identity": [existing.stat().st_dev, existing.stat().st_ino],
    }


def test_build_action_supports_reviewed_pattern_decks(tmp_path: Path) -> None:
    config, _vocabulary, _conjugation, pattern, _ids = _project(tmp_path)

    plan = assistant_actions.plan_action(
        config,
        Intent("build_deck", (_resource_id(config, pattern),)),
    )

    assert isinstance(plan.service_plan, deck_package_application.DeckPackagePlan)
    assert plan.service_plan.kind == "pattern"
    assert plan.projection["target"]["card_types"] == ["rule"]
    assert plan.projection["target"]["card_count"] == 1
    assert plan.projection["inputs"]["source"]["file"] == "data/patterns.json"
    assert plan.projection["writes"] == {
        "package": "dist/rules.apkg",
        "package_precondition": {
            "state": "absent",
            "sha256": None,
            "identity": None,
        },
    }


def test_action_plan_requires_build_service_to_target_the_same_deck(
    tmp_path: Path,
) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)
    plan = assistant_actions.plan_action(
        config,
        Intent("build_deck", (_resource_id(config, vocabulary),)),
    )
    other = config.deck_dir / "other.yaml"
    other.write_text(vocabulary.read_text(encoding="utf-8"), encoding="utf-8")
    other_service = deck_package_application.plan_deck_package(config, other)

    with pytest.raises(ValueError, match="same configured deck"):
        replace(
            plan,
            service_plan=other_service,
            service_fingerprint=other_service.fingerprint,
        )


def test_unhandled_action_kind_never_reaches_a_service(tmp_path: Path) -> None:
    config, vocabulary, _conjugation, _pattern, _ids = _project(tmp_path)

    with pytest.raises(
        assistant_actions.AssistantActionError,
        match="not handled by the audio/build broker",
    ):
        assistant_actions.plan_action(
            config,
            Intent("promote_staging", (_resource_id(config, vocabulary),)),
        )
