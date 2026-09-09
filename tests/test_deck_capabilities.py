"""The fixed media-capability table, and the census and packaging it governs.

Every deck kind states once what it owns, synthesizes and packages. A known
kind that reaches this table with no row is refused before anything censuses,
publishes, packages or prunes media for it — an empty owner set would make a
clip somebody paid for look unreferenced.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import kanji_notes, ledger, status
from japanese_anki.application import audio as audio_application
from japanese_anki.application import deck_capabilities, deck_package
from japanese_anki.application.deck_capabilities import (
    DeckCapabilityError,
    DeckMediaCapability,
    capability,
    durable_media_owners,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import anki
from japanese_anki.io import DataError, records_revision
from japanese_anki.models import ExampleSentence, VocabularyRecord

# The exact table §8 states, transcribed independently of the module under
# test so a silent edit to a column shows up here as a disagreement.
EXPECTED_TABLE = {
    #  kind          vocabulary  drills  word   example  media
    "": (True, False, True, True, True),
    "vocabulary": (True, False, True, True, True),
    "pattern": (True, False, False, False, False),
    "conjugation": (True, True, False, True, True),
    "kanji": (False, False, False, False, False),
}


def _record(expression: str, reading: str, *sentences: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["something"],
        part_of_speech="verb",
        verb_group="godan",
        examples=[ExampleSentence(japanese=item) for item in sentences],
    )


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'dist_dir = "dist"\n'
        'kanji_notes_file = "kanji_notes.json"\n'
        'patterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    return config


def _character_store(config: ProjectConfig) -> Path:
    """The curated character store: a JSON *object*, never a record list."""
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {
            "理": kanji_notes.CharacterNote(
                character="理",
                id="kanji:理",
                meanings=("logic", "reason"),
                stroke_count=11,
            )
        },
    )
    return config.kanji_notes_file


def _character_deck(config: ProjectConfig, name: str = "characters.yaml") -> Path:
    _character_store(config)
    deck = config.deck_dir / name
    deck.write_text(
        "deck:\n"
        "  kind: kanji\n"
        "  name: Characters\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        "  source: ../kanji_notes.json\n"
        "  include_ids: [kanji:理]\n",
        encoding="utf-8",
    )
    return deck


def _conjugation_deck(config: ProjectConfig, name: str = "potential.yaml") -> Path:
    deck = config.deck_dir / name
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential\n"
        "  deck_id: 1047286103\n"
        "  model_id: 1607392351\n"
        "  source: ../vocabulary.json\n"
        '  include_ids: ["word:話す:はなす"]\n'
        "  drill_examples:\n"
        '    "word:話す:はなす":\n'
        "      - japanese: 日本語が話せます。\n"
        "        furigana: 日本語[にほんご]が 話[はな]せます。\n"
        "        english: I can speak Japanese.\n"
        "        register: polite\n"
        "      - japanese: 英語も話せる？\n"
        "        furigana: 英語[えいご]も 話[はな]せる？\n"
        "        english: Can you speak English too?\n"
        "        register: casual\n",
        encoding="utf-8",
    )
    return deck


def _mixed_repository(tmp_path: Path) -> tuple[ProjectConfig, dict[str, Path]]:
    """One deck of every known kind, in one configured deck directory."""
    speak = _record("話す", "はなす", "話します。")
    read = _record("読む", "よむ", "本を読みます。")
    config = _project(tmp_path, [speak, read])

    plain = config.deck_dir / "a-plain.yaml"
    plain.write_text(
        "deck:\n"
        "  name: Plain\n"
        "  deck_id: 11\n"
        "  model_id: 12\n"
        "  source: ../vocabulary.json\n"
        '  include_ids: ["word:読む:よむ"]\n',
        encoding="utf-8",
    )
    lesson = config.deck_dir / "b-lesson.yaml"
    lesson.write_text(
        "deck:\n"
        "  kind: vocabulary\n"
        "  name: Lesson\n"
        "  deck_id: 21\n"
        "  model_id: 22\n"
        "  source: ../vocabulary.json\n"
        '  include_ids: ["word:話す:はなす"]\n'
        "notes:\n"
        '  - id: "word:話す:はなす"\n'
        "    audio: audio/inline-override.mp3\n",
        encoding="utf-8",
    )
    rules = config.deck_dir / "c-rules.yaml"
    rules.write_text(
        "deck:\n"
        "  kind: pattern\n"
        "  name: Rules\n"
        "  deck_id: 31\n"
        "  model_id: 32\n"
        "  document: rules.pdf\n",
        encoding="utf-8",
    )
    (tmp_path / "patterns.json").write_text(
        json.dumps(
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
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    conjugation = _conjugation_deck(config, "d-potential.yaml")
    character = _character_deck(config, "e-characters.yaml")
    return config, {
        "plain": plain,
        "lesson": lesson,
        "pattern": rules,
        "conjugation": conjugation,
        "character": character,
    }


class _RecordsLoaderSpy:
    """Records every path the exporter hands to the vocabulary loader."""

    def __init__(self, real) -> None:  # type: ignore[no-untyped-def]
        self._real = real
        self.paths: list[Path] = []

    def __call__(self, path: Path):  # type: ignore[no-untyped-def]
        self.paths.append(Path(path).resolve())
        return self._real(path)


def _spy_on_vocabulary_loader(monkeypatch: pytest.MonkeyPatch) -> _RecordsLoaderSpy:
    spy = _RecordsLoaderSpy(anki.load_records)
    monkeypatch.setattr(anki, "load_records", spy)
    return spy


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------


def test_the_table_states_one_exact_row_for_every_known_deck_kind() -> None:
    assert set(anki.KNOWN_DECK_KINDS) == set(EXPECTED_TABLE)
    for kind, expected in EXPECTED_TABLE.items():
        entry = capability(kind)
        assert isinstance(entry, DeckMediaCapability)
        assert entry.kind == kind
        assert (
            entry.parses_source_as_vocabulary,
            entry.owns_drill_examples,
            entry.synthesizes_word_audio,
            entry.synthesizes_example_audio,
            entry.packages_media,
        ) == expected


def test_a_kind_string_the_exporter_refuses_is_refused_here_too() -> None:
    with pytest.raises(DeckCapabilityError, match="patern"):
        capability("patern")


def test_a_known_kind_with_no_row_refuses_rather_than_defaulting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(anki, "KNOWN_DECK_KINDS", (*anki.KNOWN_DECK_KINDS, "flashcards"))

    with pytest.raises(DeckCapabilityError) as refusal:
        capability("flashcards")

    message = str(refusal.value)
    assert "flashcards" in message
    assert "deck_capabilities" in message


def test_the_capability_of_a_deck_is_read_from_its_own_kind(tmp_path: Path) -> None:
    config, paths = _mixed_repository(tmp_path)

    assert deck_capabilities.deck_capability(paths["plain"]).kind == ""
    assert deck_capabilities.deck_capability(paths["lesson"]).kind == "vocabulary"
    assert deck_capabilities.deck_capability(paths["pattern"]).kind == "pattern"
    assert deck_capabilities.deck_capability(paths["conjugation"]).kind == "conjugation"
    assert deck_capabilities.deck_capability(paths["character"]).kind == "kanji"
    assert config.deck_dir.is_dir()


def test_a_supplied_revision_is_classified_by_its_own_bytes(tmp_path: Path) -> None:
    """`78da954`'s rule: proposed deck bytes classify themselves."""
    config, paths = _mixed_repository(tmp_path)
    deck = paths["lesson"]
    revision = records_revision(deck)
    assert revision.text is not None
    proposed = replace(
        revision,
        text=revision.text.replace("kind: vocabulary", "kind: kanji"),
    )

    assert deck_capabilities.deck_capability(deck, revision=proposed).kind == "kanji"
    assert deck_capabilities.deck_capability(deck).kind == "vocabulary"
    assert durable_media_owners(config, deck, revision=proposed) == []
    assert durable_media_owners(config, deck) != []


# ---------------------------------------------------------------------------
# C5 — the table and the exporter cannot drift
# ---------------------------------------------------------------------------


def test_the_table_agrees_with_the_exporter_on_what_parses_as_vocabulary(
    tmp_path: Path,
) -> None:
    """For every known kind, a deck holding one record either yields that
    record version or yields nothing — and the column says which."""
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    for index, kind in enumerate(anki.KNOWN_DECK_KINDS):
        deck = config.deck_dir / f"kind-{index}.yaml"
        deck.write_text(
            "deck:\n"
            + (f"  kind: {kind}\n" if kind else "")
            + f"  name: Deck {index}\n"
            f"  deck_id: {100 + index}\n"
            f"  model_id: {200 + index}\n"
            "  source: ../vocabulary.json\n",
            encoding="utf-8",
        )
        declared = anki.deck_declared_record_versions(deck)
        assert bool(declared) is capability(kind).parses_source_as_vocabulary, kind
        assert durable_media_owners(config, deck) == declared, kind
        deck.unlink()


# ---------------------------------------------------------------------------
# C1 — everything a mixed-kind repository keeps alive
# ---------------------------------------------------------------------------


def test_a_targeted_run_protects_declared_versions_overrides_and_drill_owners(
    tmp_path: Path,
) -> None:
    config, paths = _mixed_repository(tmp_path)

    plan = audio_application.plan_targeted_audio(
        config,
        ["word:話す:はなす"],
        examples=True,
    )

    protected = plan.protected_records
    ids = [record.id for record in protected]
    # The canonical collection plus the declared versions of the plain,
    # vocabulary and conjugation decks — and one more 話す for the inline
    # override. Nothing from the character store, whose notes are not words.
    assert ids.count("word:読む:よむ") == 4
    assert ids.count("word:話す:はなす") == 5
    assert not [record_id for record_id in ids if record_id.startswith("kanji:")]
    # The inline `notes:` override is a durable version of its own: it names a
    # media file the source record does not.
    assert "audio/inline-override.mp3" in {record.audio for record in protected}
    # The conjugation deck's synthetic owners.
    drills = sorted(
        record.id for record in protected if record.id.startswith("drill-audio:")
    )
    assert drills == [
        "drill-audio:1047286103:potential:word:話す:はなす",
    ]
    assert paths["character"].is_file()


def test_durable_media_owners_is_the_per_deck_answer_for_every_kind(
    tmp_path: Path,
) -> None:
    config, paths = _mixed_repository(tmp_path)

    plain = durable_media_owners(config, paths["plain"])
    lesson = durable_media_owners(config, paths["lesson"])
    pattern_owners = durable_media_owners(config, paths["pattern"])
    conjugation = durable_media_owners(config, paths["conjugation"])
    character = durable_media_owners(config, paths["character"])

    assert [record.id for record in plain] == ["word:話す:はなす", "word:読む:よむ"]
    assert [record.id for record in lesson] == [
        "word:話す:はなす",
        "word:読む:よむ",
        "word:話す:はなす",
    ]
    assert lesson[-1].audio == "audio/inline-override.mp3"
    # A pattern deck declares no source and no notes, and owns no drills.
    assert pattern_owners == []
    assert [record.id for record in conjugation] == [
        "word:話す:はなす",
        "word:読む:よむ",
        "drill-audio:1047286103:potential:word:話す:はなす",
    ]
    assert character == []


# ---------------------------------------------------------------------------
# C2 — an unmapped known kind refuses before any census
# ---------------------------------------------------------------------------


def _unmapped_kind_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectConfig, Path]:
    """A repository whose last deck states a known kind the table has no row for."""
    monkeypatch.setattr(anki, "KNOWN_DECK_KINDS", (*anki.KNOWN_DECK_KINDS, "flashcards"))
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    # Sorted first, so a census that ran before the refusal would already have
    # parsed a source.
    ordinary = config.deck_dir / "a-lesson.yaml"
    ordinary.write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  deck_id: 21\n"
        "  model_id: 22\n"
        "  source: ../vocabulary.json\n",
        encoding="utf-8",
    )
    unmapped = config.deck_dir / "z-flashcards.yaml"
    unmapped.write_text(
        "deck:\n"
        "  kind: flashcards\n"
        "  name: Flashcards\n"
        "  deck_id: 41\n"
        "  model_id: 42\n"
        "  source: ../vocabulary.json\n",
        encoding="utf-8",
    )
    return config, unmapped


def test_an_unmapped_known_kind_refuses_before_the_census_reads_a_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _unmapped = _unmapped_kind_repository(tmp_path, monkeypatch)
    spy = _spy_on_vocabulary_loader(monkeypatch)

    with pytest.raises(DeckCapabilityError, match="flashcards"):
        audio_application.plan_targeted_audio(
            config,
            ["word:話す:はなす"],
            examples=True,
        )

    assert spy.paths == []


def test_an_unmapped_known_kind_refuses_a_corpus_prune_before_it_removes_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _unmapped = _unmapped_kind_repository(tmp_path, monkeypatch)
    orphan = config.media_dir / "audio" / "janki-orphan.wav"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"paid bytes nothing references yet")
    spy = _spy_on_vocabulary_loader(monkeypatch)

    with pytest.raises(DeckCapabilityError, match="flashcards"):
        audio_application.execute_corpus_audio(config, examples=True, prune=True)

    assert orphan.is_file()
    assert spy.paths == []


# ---------------------------------------------------------------------------
# C3 — and before any package planner
# ---------------------------------------------------------------------------


def test_an_unmapped_known_kind_refuses_the_package_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, unmapped = _unmapped_kind_repository(tmp_path, monkeypatch)

    with pytest.raises(deck_package.DeckPackageError, match="flashcards"):
        deck_package.plan_deck_package(config, unmapped)


def test_the_capability_refusal_precedes_the_package_planner_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kind with a planner but no row must not reach that planner."""
    config = _project(tmp_path, [_record("話す", "はなす")])
    deck = _character_deck(config)
    monkeypatch.setattr(
        deck_capabilities,
        "_TABLE",
        {kind: row for kind, row in deck_capabilities._TABLE.items() if kind != "kanji"},
    )
    planned: list[Path] = []
    monkeypatch.setattr(
        deck_package,
        "_plan_kanji",
        lambda _config, target: planned.append(target),
    )

    with pytest.raises(deck_package.DeckPackageError, match="kanji"):
        deck_package.plan_deck_package(config, deck)

    assert planned == []


# ---------------------------------------------------------------------------
# C4 — the character store is a dependency, not a word list
# ---------------------------------------------------------------------------


def test_a_character_store_is_never_handed_to_the_vocabulary_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path, [_record("話す", "はなす", "話します。")])
    deck = _character_deck(config)
    spy = _spy_on_vocabulary_loader(monkeypatch)
    censused: list[Path] = []
    monkeypatch.setattr(
        anki,
        "deck_declared_record_versions",
        lambda path: censused.append(Path(path)) or [],
    )

    assert durable_media_owners(config, deck) == []

    assert config.kanji_notes_file.resolve() not in spy.paths
    # The column short-circuits: the vocabulary census is not asked a question
    # whose only correct answer is "none of my business".
    assert censused == []


def test_a_character_decks_source_stays_in_the_audio_lock_and_staleness_set(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path, [_record("話す", "はなす", "話します。")])
    deck = _character_deck(config)
    store = config.kanji_notes_file.resolve()
    deck_paths = status.deck_files(config)

    source_paths = audio_application._audio_deck_source_paths(deck_paths)
    assert store in source_paths

    revisions = {store: records_revision(store)}
    audio_application._assert_audio_owners_current(
        revisions,
        config=config,
        deck_paths=deck_paths,
        source_paths=source_paths,
    )

    notes = dict(kanji_notes.load_notes(config.kanji_notes_file))
    notes["説"] = kanji_notes.CharacterNote(
        character="説",
        id="kanji:説",
        meanings=("explanation",),
    )
    kanji_notes.save_notes(config.kanji_notes_file, notes)

    with pytest.raises(DataError, match="changed after its locked snapshot"):
        audio_application._assert_audio_owners_current(
            revisions,
            config=config,
            deck_paths=deck_paths,
            source_paths=source_paths,
        )
    assert deck.is_file()


# ---------------------------------------------------------------------------
# C6 — the repository-wide packaging block is deliberately unchanged
# ---------------------------------------------------------------------------


def _templated_project(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    config = _project(tmp_path, [_record("話す", "はなす", "話します。")])
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    deck = config.deck_dir / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  deck_id: 21\n"
        "  model_id: 22\n"
        "  source: ../vocabulary.json\n"
        "  output: lesson.apkg\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: false\n"
        "    reading: false\n",
        encoding="utf-8",
    )
    return config, deck


def test_an_unrelated_open_audio_wal_row_still_blocks_every_package(
    tmp_path: Path,
) -> None:
    config, deck = _templated_project(tmp_path)
    other = _conjugation_deck(config)
    plan = deck_package.plan_deck_package(config, deck)

    book = ledger.Ledger(path=config.ledger_file)
    owner = "drill-audio:1047286103:potential:word:話す:はなす"
    request = {
        "of": "example",
        "target": "pending.wav",
        "request_input": "日本語が話せます。",
        "forced_accent": False,
        "content_fp": "a" * 64,
        "provider": "openai-realtime",
        "voice": "cedar",
        "speed": 1.0,
        "settings": {},
    }
    key = book._pending_audio_key(book._pending_audio_identity(owner, **request))
    book.record_pending_audio(
        owner,
        **request,
        staged_file=f".pending/{key}-{'b' * 64}.stage",
        staged_sha256="b" * 64,
    )
    book.save()

    with pytest.raises(deck_package.DeckPackageError, match="pending audio recovery"):
        deck_package.execute_deck_package(config, plan)

    assert not (config.dist_dir / "lesson.apkg").exists()
    assert other.is_file()


# ---------------------------------------------------------------------------
# The synthesis and packaging columns govern their own decisions
# ---------------------------------------------------------------------------


def test_deck_scoped_audio_asks_only_for_the_clips_its_kind_synthesizes(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path, [_record("話す", "はなす", "話します。")])
    deck = _conjugation_deck(config)

    with pytest.raises(audio_application.AudioPlanError, match="word audio"):
        audio_application.plan_deck_audio(config, deck, words=True, examples=True)
    with pytest.raises(audio_application.AudioPlanError, match="example audio"):
        audio_application.plan_deck_audio(config, deck, examples=False)

    plan = audio_application.plan_deck_audio(config, deck, examples=True)
    assert plan.example_counts.total == 2


def test_deck_scoped_audio_refuses_a_kind_that_owns_no_drill_examples(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path, [_record("話す", "はなす", "話します。")])
    deck = config.deck_dir / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  deck_id: 21\n"
        "  model_id: 22\n"
        "  source: ../vocabulary.json\n",
        encoding="utf-8",
    )

    with pytest.raises(audio_application.AudioPlanError, match="drill examples"):
        audio_application.plan_deck_audio(config, deck, examples=True)


def test_a_kind_that_packages_no_media_may_not_bind_media_inputs(
    tmp_path: Path,
) -> None:
    config, paths = _mixed_repository(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    plan = deck_package.plan_deck_package(config, paths["pattern"])
    assert plan.media_inputs == ()
    assert capability("pattern").packages_media is False

    with pytest.raises(deck_package.DeckPackageError, match="packages no media"):
        deck_package._finish_plan(
            config,
            repository_root=plan.repository_root,
            deck_path=plan.deck_path,
            output_path=plan.output_path,
            kind=plan.kind,
            deck_name=plan.deck_name,
            variant=plan.variant,
            card_types=plan.card_types,
            note_count=plan.note_count,
            card_count=plan.card_count,
            record_ids=plan.record_ids,
            deck_input=plan.deck_input,
            source_inputs=plan.source_inputs,
            template_inputs=plan.template_inputs,
            # Any repository input stands in: the refusal is about the column,
            # not about which bytes were named.
            media_inputs=(plan.deck_input,),
            conjugation_plan=None,
        )
