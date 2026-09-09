"""Display-only planning for W5's exact audio actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import cli, kanji_notes
from japanese_anki import ledger as ledger_mod
from japanese_anki.application import audio as audio_application
from japanese_anki.application.audio import (
    AudioPlanError,
    plan_audio_records,
    plan_corpus_audio,
    plan_targeted_audio,
)
from japanese_anki.application.deck_capabilities import DeckCapabilityError
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import anki
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    load_records_snapshot,
    load_structured,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import TtsError, openai_realtime


class Provider:
    launch_hint = "start it"
    speed = 1.0

    def __init__(
        self,
        name: str,
        voice: int | str,
        *,
        suffix: str = ".wav",
        speed: float = 1.0,
        destination: str | None = None,
        settings: dict[str, str] | None = None,
        transport: object | None = None,
    ) -> None:
        self.name = name
        self.voice = voice
        self.suffix = suffix
        self.speed = speed
        self._base_url = destination or name
        self.settings = dict(settings or {})
        self._transport = transport or self.synthesize

    def available(self) -> bool:
        raise AssertionError("planning must not probe or contact the provider")

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        raise AssertionError("planning must not dispatch audio")


class RecordingProvider(Provider):
    def __init__(self, name: str, voice: int | str) -> None:
        super().__init__(name, voice)
        self.said: list[tuple[str, bool]] = []

    def available(self) -> bool:
        return True

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        self.said.append((text_or_kana, forced_accent))
        return f"audio:{text_or_kana}".encode()


def _record(
    expression: str,
    reading: str,
    *sentences: str,
) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["something"],
        examples=[ExampleSentence(japanese=item) for item in sentences],
    )


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _drill_project(tmp_path: Path) -> tuple[ProjectConfig, Path, VocabularyRecord]:
    item = replace(
        _record("話す", "はなす"),
        part_of_speech="verb",
        verb_group="godan",
    )
    config = _project(tmp_path, [item])
    config.deck_dir.mkdir(parents=True)
    deck = config.deck_dir / "potential.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  # This learner note must survive generated audio updates.\n"
        '  name: "Potential"\n'
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
        "        spoken_japanese: 英語も、話せる？\n"
        "        register: casual\n",
        encoding="utf-8",
    )
    return config, deck, item


def _character_deck(config: ProjectConfig) -> Path:
    """A configured character deck reading the curated character store.

    Its `source:` is a JSON *object* of character notes rather than a list of
    vocabulary records, which is exactly what the audio owner census used to
    hand to the vocabulary loader.
    """
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    deck = config.deck_dir / "genki-ii-kanji.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Genki II Kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        "  source: ../data/kanji_notes.json\n"
        "  include_ids: [kanji:理]\n",
        encoding="utf-8",
    )
    config.kanji_notes_file.parent.mkdir(parents=True, exist_ok=True)
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {
            "理": kanji_notes.CharacterNote(
                character="理",
                id="kanji:理",
                meanings=("logic", "reason"),
                stroke_count=11,
                kanjidic_readings=(
                    kanji_notes.KanjidicReading(kind="on", reading="リ"),
                ),
                sources=("kanjiapi.dev (KANJIDIC2)",),
            )
        },
    )
    return deck


def _word_identity(item: VocabularyRecord, provider: Provider) -> tuple[str, str, bool, str]:
    utterance, forced, _warning = ledger_mod.word_audio_request(item)
    target = f"janki-{ledger_mod.word_audio_filename_fingerprint(item)}{provider.suffix}"
    return (
        target,
        utterance,
        forced,
        ledger_mod.word_audio_content_fingerprint(item),
    )


def _make_word_current(config: ProjectConfig, item: VocabularyRecord, provider: Provider) -> None:
    target, _utterance, _forced, content_fp = _word_identity(item, provider)
    audio_dir = config.media_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / target).write_bytes(b"current")
    book = ledger_mod.Ledger(path=config.ledger_file)
    book.record_audio(
        item.id,
        file=target,
        of="word",
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
        content_fp=content_fp,
    )
    book.save()


def test_targeted_audio_plan_binds_exact_ids_providers_and_distinct_clip_counts(
    tmp_path: Path,
) -> None:
    selected = _record("話す", "はなす", "話します。", "話します。", "話した。")
    outside = _record("本", "ほん", "本です。")
    config = _project(tmp_path, [selected, outside])

    plan = plan_targeted_audio(
        config,
        [selected.id],
        words=True,
        examples=True,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai-realtime", "cedar"),
    )

    assert plan.record_ids == (selected.id,)
    assert plan.word_counts == audio_application.AudioClipCounts(
        total=1,
        current=0,
        recoverable=0,
        provider_required=1,
    )
    assert plan.example_counts == audio_application.AudioClipCounts(
        total=2,
        current=0,
        recoverable=0,
        provider_required=2,
    )
    assert (plan.clip_count, plan.provider_required_count) == (3, 3)
    assert plan.word_provider is not None
    assert (plan.word_provider.name, plan.word_provider.access) == (
        "voicevox",
        "local-network",
    )
    assert plan.example_provider is not None
    assert (plan.example_provider.name, plan.example_provider.access) == (
        "openai-realtime",
        "paid-network",
    )
    assert len(plan.fingerprint) == 64


def test_targeted_example_planning_reads_a_character_deck_as_the_kind_it_is(
    tmp_path: Path,
) -> None:
    """A character deck owns no word or sentence audio.

    Its `source:` is the curated character store, and handing that store to the
    vocabulary loader refused the whole plan — a sentence request blocked by a
    deck that can never hold one of its clips.
    """
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    deck = _character_deck(config)
    deck_before = deck.read_text(encoding="utf-8")

    plan = plan_targeted_audio(
        config,
        [item.id],
        examples=True,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai-realtime", "cedar"),
    )

    assert plan.record_ids == (item.id,)
    assert plan.example_counts == audio_application.AudioClipCounts(
        total=1,
        current=0,
        recoverable=0,
        provider_required=1,
    )
    assert [record.id for record in plan.protected_records] == [item.id]
    assert deck.read_text(encoding="utf-8") == deck_before


def test_targeted_example_audio_is_generated_with_a_character_deck_configured(
    tmp_path: Path,
) -> None:
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    deck = _character_deck(config)
    deck_before = deck.read_text(encoding="utf-8")
    store_before = config.kanji_notes_file.read_text(encoding="utf-8")
    sentences = RecordingProvider("openai-realtime", "cedar")

    outcome = audio_application.execute_targeted_audio(
        config,
        [item.id],
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=sentences,
    )

    assert outcome.succeeded, outcome.stopped_by
    assert [spoken for spoken, _forced in sentences.said] == ["話します。"]
    voiced = load_records_snapshot(config.normalized_file)[0][0]
    assert voiced.examples[0].audio.startswith("audio/janki-")
    assert (config.media_dir / voiced.examples[0].audio).is_file()
    assert deck.read_text(encoding="utf-8") == deck_before
    assert config.kanji_notes_file.read_text(encoding="utf-8") == store_before


def test_a_character_decks_source_still_stales_a_running_audio_transaction(
    tmp_path: Path,
) -> None:
    """The census stops reading the character store as a word list; it does not
    stop depending on it. Every configured deck's `source:` is still locked and
    revalidated before paid bytes reach canonical media."""
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    _character_deck(config)

    class EditsCharacterStoreDuringSynthesis(RecordingProvider):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.said:
                store = dict(kanji_notes.load_notes(config.kanji_notes_file))
                store["説"] = kanji_notes.CharacterNote(
                    character="説",
                    id="kanji:説",
                    meanings=("explanation",),
                )
                kanji_notes.save_notes(config.kanji_notes_file, store)
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    sentences = EditsCharacterStoreDuringSynthesis("openai-realtime", "cedar")
    outcome = audio_application.execute_targeted_audio(
        config,
        [item.id],
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=sentences,
    )

    assert outcome.state == "finalization-stopped"
    assert "changed after its locked snapshot" in (outcome.stopped_by or "")
    assert outcome.pending_recovery is True
    assert outcome.media_published is False


def test_a_deck_kind_with_no_media_capability_stops_the_run_before_any_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known kind the capability table has no row for is not "no media".

    Answering it with an empty owner set would leave whatever it keeps alive
    looking unreferenced, so the whole transaction refuses before a clip is
    synthesized and before anything is published or pruned.
    """
    monkeypatch.setattr(anki, "KNOWN_DECK_KINDS", (*anki.KNOWN_DECK_KINDS, "flashcards"))
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    config.deck_dir.mkdir(parents=True)
    (config.deck_dir / "flashcards.yaml").write_text(
        "deck:\n"
        "  kind: flashcards\n"
        "  name: Flashcards\n"
        "  deck_id: 41\n"
        "  model_id: 42\n"
        "  source: ../vocabulary.json\n",
        encoding="utf-8",
    )
    normalized_before = config.normalized_file.read_text(encoding="utf-8")
    sentences = RecordingProvider("openai-realtime", "cedar")

    with pytest.raises(DeckCapabilityError, match="flashcards"):
        audio_application.execute_targeted_audio(
            config,
            [item.id],
            examples=True,
            word_provider=RecordingProvider("voicevox", 7),
            sentence_provider=sentences,
        )

    assert sentences.said == []
    assert not config.ledger_file.exists()
    assert not (config.media_dir / "audio").exists()
    assert config.normalized_file.read_text(encoding="utf-8") == normalized_before


def test_an_unknown_deck_kind_still_refuses_the_durable_audio_owner_census(
    tmp_path: Path,
) -> None:
    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    (config.deck_dir / "typo.yaml").write_text(
        "deck:\n"
        "  name: Typo\n"
        "  kind: kanjy\n"
        "  deck_id: 1500000003\n"
        "  model_id: 1500000103\n",
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="unknown deck kind 'kanjy'"):
        plan_targeted_audio(
            config,
            [item.id],
            examples=True,
            word_provider=Provider("voicevox", 7),
            sentence_provider=Provider("openai-realtime", "cedar"),
        )


def test_empty_targeted_audio_never_widens_to_the_corpus(tmp_path: Path) -> None:
    config = _project(tmp_path, [_record("話す", "はなす")])
    provider = Provider("voicevox", 7)

    with pytest.raises(AudioPlanError, match="empty scope never means"):
        plan_targeted_audio(
            config,
            [],
            words=True,
            word_provider=provider,
        )
    with pytest.raises(AudioPlanError, match="empty scope never means"):
        audio_application.execute_targeted_audio(
            config,
            [],
            words=True,
            word_provider=provider,
        )

    corpus = plan_corpus_audio(
        config,
        words=True,
        word_provider=provider,
    )
    assert corpus.targeted is False
    assert corpus.record_ids == ("word:話す:はなす",)


def test_cli_planner_uses_its_already_locked_record_universe_without_reloading(
    tmp_path: Path,
) -> None:
    normalized = _record("本", "ほん")
    owner_resolved = _record("話す", "はなす")
    config = _project(tmp_path, [normalized])
    _ignored, revision = load_records_snapshot(config.normalized_file)

    plan = plan_audio_records(
        config,
        [owner_resolved],
        revision,
        [owner_resolved.id],
        words=True,
        word_provider=Provider("voicevox", 7),
    )

    assert plan.record_ids == (owner_resolved.id,)
    assert plan.word_counts.total == 1


def test_audio_plan_binds_and_refuses_colliding_deck_owned_references(
    tmp_path: Path,
) -> None:
    selected = _record("話す", "はなす")
    protected = _record("本", "ほん")
    config = _project(tmp_path, [selected])
    records, revision = load_records_snapshot(config.normalized_file)
    provider = Provider("voicevox", 7)

    before = plan_audio_records(
        config,
        records,
        revision,
        [selected.id],
        words=True,
        word_provider=provider,
        protected_records=[protected],
    )
    protected.meanings = ["a changed deck-owned version"]
    changed = plan_audio_records(
        config,
        records,
        revision,
        [selected.id],
        words=True,
        word_provider=provider,
        protected_records=[protected],
    )
    assert changed.fingerprint != before.fingerprint

    target, _utterance, _forced, _content_fp = _word_identity(selected, provider)
    protected.audio = f"audio/{target}"
    with pytest.raises(AudioPlanError, match="Different audio identities"):
        plan_audio_records(
            config,
            records,
            revision,
            [selected.id],
            words=True,
            word_provider=provider,
            protected_records=[protected],
        )


def test_current_word_media_is_counted_current_until_force_requires_a_provider(
    tmp_path: Path,
) -> None:
    provider = Provider("voicevox", 7)
    item = _record("話す", "はなす")
    target, _utterance, _forced, _content_fp = _word_identity(item, provider)
    item.audio = f"audio/{target}"
    config = _project(tmp_path, [item])
    _make_word_current(config, item, provider)

    ordinary = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        word_provider=provider,
    )
    forced = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        force=True,
        word_provider=provider,
    )

    assert ordinary.word_counts == audio_application.AudioClipCounts(
        total=1,
        current=1,
        recoverable=0,
        provider_required=0,
    )
    assert forced.word_counts == audio_application.AudioClipCounts(
        total=1,
        current=0,
        recoverable=0,
        provider_required=1,
    )


def test_exact_orphan_stage_is_counted_recoverable_without_adopting_its_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Provider("voicevox", 7)
    item = _record("話す", "はなす")
    config = _project(tmp_path, [item])
    target, utterance, forced, content_fp = _word_identity(item, provider)
    book = ledger_mod.Ledger(path=config.ledger_file)
    monkeypatch.setattr(audio_application.ledger, "load", lambda path: book)
    key = book.pending_audio_key_for(
        item.id,
        of="word",
        target=target,
        request_input=utterance,
        forced_accent=forced,
        content_fp=content_fp,
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
    )
    paid_bytes = b"already paid"
    digest = hashlib.sha256(paid_bytes).hexdigest()
    pending = config.media_dir / "audio" / ".pending"
    pending.mkdir(parents=True)
    stage = pending / f"{key}-{digest}.stage"
    stage.write_bytes(paid_bytes)

    plan = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        word_provider=provider,
    )

    assert plan.word_counts == audio_application.AudioClipCounts(
        total=1,
        current=0,
        recoverable=1,
        provider_required=0,
    )
    assert not config.ledger_file.exists(), "planning never adopts the orphan row"
    assert book.pending_audio == {}, "the caller's in-memory ledger stays read-only"
    [clip] = plan.clips
    assert (clip.recovery_key, clip.recovery_sha256, clip.recovery_source) == (
        key,
        digest,
        "orphan-stage",
    )

    alternate_key = "f" * 64
    alternate_stage = pending / f"{alternate_key}-{digest}.stage"
    stage.replace(alternate_stage)
    with monkeypatch.context() as changed_key:
        changed_key.setattr(
            ledger_mod.Ledger,
            "pending_audio_key_for",
            lambda self, *args, **kwargs: alternate_key,
        )
        rekeyed = plan_targeted_audio(
            config,
            [item.id],
            words=True,
            word_provider=provider,
        )
    assert rekeyed.clips[0].recovery_key == alternate_key
    assert rekeyed.fingerprint != plan.fingerprint
    alternate_stage.replace(stage)

    replacement_bytes = b"a different paid render"
    replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
    stage.unlink()
    replacement_stage = pending / f"{key}-{replacement_digest}.stage"
    replacement_stage.write_bytes(replacement_bytes)
    replacement = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        word_provider=provider,
    )
    assert replacement.clips[0].recovery_sha256 == replacement_digest
    assert replacement.fingerprint != plan.fingerprint

    assert book.record_pending_audio(
        item.id,
        of="word",
        target=target,
        request_input=utterance,
        forced_accent=forced,
        content_fp=content_fp,
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
        staged_file=f".pending/{replacement_stage.name}",
        staged_sha256=replacement_digest,
    ) == key
    registered = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        word_provider=provider,
    )
    assert registered.clips[0].recovery_source == "pending-stage"
    assert registered.fingerprint != replacement.fingerprint

    replacement_stage.replace(config.media_dir / "audio" / target)
    published = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        word_provider=provider,
    )
    assert published.clips[0].recovery_source == "canonical-target"
    assert published.fingerprint != registered.fingerprint
    assert not config.ledger_file.exists(), "all test WAL changes stayed in memory"


def test_corrupt_exact_recovery_refuses_unless_force_explicitly_replaces_it(
    tmp_path: Path,
) -> None:
    provider = Provider("voicevox", 7)
    item = _record("話す", "はなす")
    config = _project(tmp_path, [item])
    target, utterance, forced, content_fp = _word_identity(item, provider)
    book = ledger_mod.Ledger(path=config.ledger_file)
    key = book.pending_audio_key_for(
        item.id,
        of="word",
        target=target,
        request_input=utterance,
        forced_accent=forced,
        content_fp=content_fp,
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
    )
    pending = config.media_dir / "audio" / ".pending"
    pending.mkdir(parents=True)
    (pending / f"{key}-{'0' * 64}.stage").write_bytes(b"not that digest")

    with pytest.raises(AudioPlanError, match="corrupt; use --force"):
        plan_targeted_audio(
            config,
            [item.id],
            words=True,
            word_provider=provider,
        )

    forced = plan_targeted_audio(
        config,
        [item.id],
        words=True,
        force=True,
        word_provider=provider,
    )
    assert forced.word_counts.provider_required == 1
    assert forced.word_counts.recoverable == 0
    assert not config.ledger_file.exists()


def test_example_plan_displays_the_exact_spoken_japanese_request(tmp_path: Path) -> None:
    item = _record("止む", "やむ", "雨、まだ止まないの？")
    item.examples[0].spoken_japanese = "雨、まだやまないの？"
    config = _project(tmp_path, [item])

    plan = plan_targeted_audio(
        config,
        [item.id],
        examples=True,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai", "onyx", suffix=".mp3"),
    )

    [clip] = plan.clips
    assert clip.request_input == "雨、まだやまないの？"
    assert clip.target == (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(item, item.examples[0])}"
        ".mp3"
    )
    assert clip.content_fingerprint == ledger_mod.example_audio_content_fingerprint(
        item.examples[0]
    )


def test_audio_fingerprint_binds_canonical_revision_even_for_the_same_records(
    tmp_path: Path,
) -> None:
    item = _record("話す", "はなす")
    config = _project(tmp_path, [item])
    records, revision = load_records_snapshot(config.normalized_file)
    provider = Provider("voicevox", 7)

    original = plan_audio_records(
        config,
        records,
        revision,
        [item.id],
        words=True,
        word_provider=provider,
    )
    whitespace_only_revision = replace(
        revision,
        text=f"{revision.text or ''}\n",
    )
    changed = plan_audio_records(
        config,
        records,
        whitespace_only_revision,
        [item.id],
        words=True,
        word_provider=provider,
    )

    assert changed.fingerprint != original.fingerprint


def test_audio_fingerprint_distinguishes_targeted_from_same_one_record_corpus(
    tmp_path: Path,
) -> None:
    item = _record("話す", "はなす")
    config = _project(tmp_path, [item])
    records, revision = load_records_snapshot(config.normalized_file)
    provider = Provider("voicevox", 7)

    targeted = plan_audio_records(
        config,
        records,
        revision,
        [item.id],
        words=True,
        word_provider=provider,
    )
    corpus = plan_audio_records(
        config,
        records,
        revision,
        None,
        words=True,
        word_provider=provider,
    )

    assert targeted.record_ids == corpus.record_ids
    assert targeted.fingerprint != corpus.fingerprint


def test_cli_routes_exact_ids_and_provider_resolution_through_audio_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunnableProvider(Provider):
        def available(self) -> bool:
            return True

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            return b"audio"

    item = _record("話す", "はなす")
    config = _project(tmp_path, [item])
    provider = RunnableProvider("voicevox", 7)
    resolved: list[str | None] = []
    routed: list[tuple[tuple[str, ...], tuple[str, ...] | None, Path, bool]] = []
    real_plan = audio_application.plan_audio_records

    def resolve(config: ProjectConfig, chosen: str | None) -> Provider:
        resolved.append(chosen)
        return provider

    def plan(
        config: ProjectConfig,
        records: list[VocabularyRecord],
        revision: object,
        record_ids: list[str] | None,
        **kwargs: object,
    ) -> audio_application.AudioPlan:
        routed.append(
            (
                tuple(record.id for record in records),
                None if record_ids is None else tuple(record_ids),
                revision.path,
                "protected_records" in kwargs,
            )
        )
        return real_plan(config, records, revision, record_ids, **kwargs)

    monkeypatch.setattr(audio_application, "resolve_word_provider", resolve)
    monkeypatch.setattr(audio_application, "plan_audio_records", plan)

    assert cli.main(["--root", str(config.root), "audio", "--words", item.id]) == 0
    assert resolved == [None]
    assert routed == [
        ((item.id,), (item.id,), config.normalized_file.resolve(), True)
    ]


def test_audio_fingerprint_binds_force_paths_and_complete_provider_profile(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path, [_record("話す", "はなす")])

    def transport_a() -> None:
        return None

    def transport_b() -> None:
        return None

    base_provider = Provider(
        "voicevox",
        7,
        destination="http://localhost:50021",
        transport=transport_a,
    )

    def planned(
        *,
        force: bool = False,
        provider: Provider = base_provider,
        chosen_config: ProjectConfig = config,
    ) -> str:
        return plan_targeted_audio(
            chosen_config,
            ["word:話す:はなす"],
            words=True,
            force=force,
            word_provider=provider,
        ).fingerprint

    ledger_config = replace(
        config,
        ledger_file=tmp_path / "other-ledger.json",
    )
    media_config = replace(
        config,
        media_dir=tmp_path / "other-media",
    )
    fingerprints = {
        planned(),
        planned(force=True),
        planned(chosen_config=ledger_config),
        planned(chosen_config=media_config),
        planned(
            provider=Provider(
                "voicevox",
                8,
                destination="http://localhost:50021",
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "voicevox",
                7,
                speed=0.9,
                destination="http://localhost:50021",
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "other-local",
                7,
                destination="http://localhost:50021",
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "voicevox",
                7,
                suffix=".mp3",
                destination="http://localhost:50021",
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "voicevox",
                7,
                destination="http://localhost:50022",
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "voicevox",
                7,
                destination="http://localhost:50021",
                settings={"format": "wav"},
                transport=transport_a,
            )
        ),
        planned(
            provider=Provider(
                "voicevox",
                7,
                destination="http://localhost:50021",
                transport=transport_b,
            )
        ),
    }

    assert len(fingerprints) == 11


def test_example_audio_plan_refuses_conflicting_spoken_inputs_for_one_file(
    tmp_path: Path,
) -> None:
    item = _record("話す", "はなす")
    item.examples = [
        ExampleSentence(japanese="話す。", spoken_japanese="はなす。"),
        ExampleSentence(japanese="話す。", spoken_japanese="わす。"),
    ]
    config = _project(tmp_path, [item])

    with pytest.raises(AudioPlanError, match="different spoken Japanese"):
        plan_targeted_audio(
            config,
            [item.id],
            examples=True,
            word_provider=Provider("voicevox", 7),
            sentence_provider=Provider("openai", "onyx", suffix=".mp3"),
        )


def test_example_audio_plan_runs_provider_length_validation_without_dispatch(
    tmp_path: Path,
) -> None:
    class LengthProvider(Provider):
        def validate_utterance(self, text: str) -> None:
            raise TtsError("input is too long")

    item = _record("話す", "はなす", "話します。")
    config = _project(tmp_path, [item])

    with pytest.raises(AudioPlanError, match="input is too long"):
        plan_targeted_audio(
            config,
            [item.id],
            examples=True,
            word_provider=Provider("voicevox", 7),
            sentence_provider=LengthProvider("openai", "onyx", suffix=".mp3"),
        )


def test_targeted_executor_uses_the_fresh_exact_plan_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _record("話す", "はなす")
    outside = _record("本", "ほん")
    config = _project(tmp_path, [selected, outside])
    provider = Provider("voicevox", 7)
    displayed = plan_targeted_audio(
        config,
        [selected.id],
        words=True,
        word_provider=provider,
    )
    generated_ids: list[tuple[str, ...]] = []

    def capture(records: list[VocabularyRecord], **kwargs: object):
        generated_ids.append(tuple(kwargs["ids"]))  # type: ignore[arg-type]
        return audio_application.audio_cmd.AudioResult(records=list(records))

    monkeypatch.setattr(audio_application.audio_cmd, "generate_audio", capture)

    outcome = audio_application.execute_targeted_audio(
        config,
        [selected.id],
        words=True,
        expected_fingerprint=displayed.fingerprint,
        word_provider=provider,
    )

    assert outcome.succeeded
    assert outcome.plan is not None
    assert outcome.plan.record_ids == (selected.id,)
    assert generated_ids == [(selected.id,)]


def test_targeted_executor_replans_and_refuses_a_changed_rendered_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _record("話す", "はなす")
    config = _project(tmp_path, [selected])
    provider = Provider("voicevox", 7)
    displayed = plan_targeted_audio(
        config,
        [selected.id],
        words=True,
        word_provider=provider,
    )
    selected.usage_notes = "Changed after the page rendered."
    config.normalized_file.write_text(
        json.dumps([selected.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        audio_application.audio_cmd,
        "generate_audio",
        lambda *args, **kwargs: pytest.fail("a stale plan must not reach synthesis"),
    )

    with pytest.raises(AudioPlanError, match="changed after it was displayed"):
        audio_application.execute_targeted_audio(
            config,
            [selected.id],
            words=True,
            expected_fingerprint=displayed.fingerprint,
            prune=True,
            word_provider=provider,
        )
    assert not config.ledger_file.exists()


def test_cli_routes_audio_through_the_shared_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path, [])
    calls: list[tuple[Path, bool]] = []

    def execute(received: ProjectConfig, **kwargs: object):
        calls.append((received.root, bool(kwargs["words"])))
        return audio_application.AudioExecutionOutcome(
            state="no-records",
            plan=None,
            output_dir=received.media_dir / "audio",
            no_records=True,
        )

    monkeypatch.setattr(audio_application, "execute_corpus_audio", execute)

    assert cli.main(["--root", str(config.root), "audio", "--words"]) == 0
    assert calls == [(config.root, True)]


def test_drill_audio_plan_uses_the_deck_revision_and_distinct_audio_owner(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)

    plan = audio_application.plan_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai-realtime", "cedar"),
    )

    owner = "drill-audio:1047286103:potential:word:話す:はなす"
    assert plan.canonical_path == deck.resolve()
    assert plan.record_ids == (owner,)
    assert plan.word_counts.total == 0
    assert plan.example_counts == audio_application.AudioClipCounts(
        total=2,
        current=0,
        recoverable=0,
        provider_required=2,
    )
    assert [clip.request_input for clip in plan.clips] == [
        "日本語が話せます。",
        "英語も、話せる？",
    ]


def test_projected_drill_audio_plan_equals_the_plan_after_exact_bytes_land(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)
    alternate = replace(item, usage_notes="Projected source version.")
    alternate_path = tmp_path / "alternate.json"
    alternate_path.write_text(
        json.dumps([alternate.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    expected = records_revision(deck)
    assert expected.text is not None
    proposed_text = expected.text.replace(
        "source: ../vocabulary.json",
        "source: ../alternate.json",
    ).replace(
        "日本語が話せます。",
        "来週は日本語が話せます。",
    )
    proposed_revision = RecordsRevision(expected.path, proposed_text)
    words = Provider("voicevox", 7)
    sentences = Provider("openai-realtime", "cedar")

    projected = audio_application.plan_deck_audio_revision(
        config,
        deck,
        proposed_revision,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert deck.read_text(encoding="utf-8") == expected.text
    current = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert projected.fingerprint != current.fingerprint

    deck.write_text(proposed_text, encoding="utf-8")
    landed = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert projected == landed


def test_projected_drill_audio_plan_ignores_a_character_deck_beside_it(
    tmp_path: Path,
) -> None:
    """The revision planner censuses every other deck live, character decks
    included, while it projects only the deck the revision describes."""
    config, deck, _item = _drill_project(tmp_path)
    _character_deck(config)
    expected = records_revision(deck)
    assert expected.text is not None
    proposed = RecordsRevision(
        expected.path,
        expected.text.replace("日本語が話せます。", "来週は日本語が話せます。"),
    )

    projected = audio_application.plan_deck_audio_revision(
        config,
        deck,
        proposed,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("openai-realtime", "cedar"),
    )

    assert [clip.request_input for clip in projected.clips] == [
        "来週は日本語が話せます。",
        "英語も、話せる？",
    ]
    assert "kanji:理" not in {record.id for record in projected.protected_records}


def test_a_projected_drill_override_still_wins_over_the_file_census(
    tmp_path: Path,
) -> None:
    """The revision planner supplies the drill owners it has already projected.

    Routing the per-deck question through the capability table must not make
    the census recompute them from the file on disk: the whole point of the
    projection is that the deck bytes have not landed yet.
    """
    config, deck, _item = _drill_project(tmp_path)
    expected = records_revision(deck)
    assert expected.text is not None
    proposed = RecordsRevision(
        expected.path,
        expected.text.replace("日本語が話せます。", "来週は日本語が話せます。"),
    )
    owner = "drill-audio:1047286103:potential:word:話す:はなす"

    projected = audio_application._all_durable_audio_records(
        config,
        [deck],
        [],
        deck_revisions={deck: proposed},
        drill_overrides={deck: [_record("差替", "さしかえ", "差し替えました。")]},
    )

    assert [record.id for record in projected if record.id.startswith("drill-audio:")] == []
    assert "word:差替:さしかえ" in {record.id for record in projected}
    # Without the override the same census reads the deck's own owners.
    from_file = audio_application._all_durable_audio_records(config, [deck], [])
    assert owner in {record.id for record in from_file}


def test_every_configured_decks_source_stays_in_the_audio_lock_set(
    tmp_path: Path,
) -> None:
    """The lock and staleness set is not narrowed by kind.

    A `kanji` deck's `source:` is no longer parsed as vocabulary, but it is
    still a dependency the census read: it stays locked and revalidated, or a
    concurrent edit to it could not stale a running transaction.
    """
    config, drill_deck, _item = _drill_project(tmp_path)
    character_deck = _character_deck(config)
    plain = config.deck_dir / "plain.yaml"
    plain.write_text(
        "deck:\n"
        "  name: Plain\n"
        "  deck_id: 11\n"
        "  model_id: 12\n"
        "  source: ../vocabulary.json\n",
        encoding="utf-8",
    )

    sources = audio_application._audio_deck_source_paths(
        [character_deck, drill_deck, plain]
    )

    assert sources == {
        config.kanji_notes_file.resolve(),
        config.normalized_file.resolve(),
    }


def test_paid_deck_audio_preflight_checks_exact_required_clips_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    contacted = False

    def unexpected_transport(*_args: object, **_kwargs: object) -> object:
        nonlocal contacted
        contacted = True
        raise AssertionError("availability preflight contacted OpenAI")

    sentence_provider = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=unexpected_transport,
        operations_path=config.operations_file,
    )
    plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=sentence_provider,
    )
    checked: list[tuple[str, bool, str, str]] = []
    original = openai_realtime.OpenAiRealtimeProvider.available_for

    def recording_available_for(
        provider: openai_realtime.OpenAiRealtimeProvider,
        text: str,
        *,
        forced_accent: bool,
        source_file: str,
        source_sha256: str,
    ) -> bool:
        checked.append((text, forced_accent, source_file, source_sha256))
        return original(
            provider,
            text,
            forced_accent=forced_accent,
            source_file=source_file,
            source_sha256=source_sha256,
        )

    monkeypatch.setattr(
        openai_realtime.OpenAiRealtimeProvider,
        "available_for",
        recording_available_for,
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    audio_application.preflight_paid_deck_audio_plan(
        plan,
        sentence_provider=sentence_provider,
    )

    required = [
        clip
        for clip in plan.clips
        if clip.state == "provider-required"
        and clip.provider.access == "paid-network"
    ]
    assert checked == [
        (
            clip.request_input,
            clip.forced_accent,
            f"{clip.record_id}#{clip.kind}:{clip.target}",
            clip.content_fingerprint,
        )
        for clip in required
    ]
    assert contacted is False
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_paid_deck_audio_preflight_refuses_missing_key_without_contact_or_write(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    contacted = False

    def unexpected_transport(*_args: object, **_kwargs: object) -> object:
        nonlocal contacted
        contacted = True
        raise AssertionError("availability preflight contacted OpenAI")

    sentence_provider = openai_realtime.OpenAiRealtimePool(
        api_key="",
        transport=unexpected_transport,
        operations_path=config.operations_file,
    )
    plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=sentence_provider,
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(AudioPlanError, match="OPENAI_API_KEY"):
        audio_application.preflight_paid_deck_audio_plan(
            plan,
            sentence_provider=sentence_provider,
        )

    assert contacted is False
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_paid_deck_audio_preflight_refuses_a_provider_changed_after_planning(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)

    def planned_transport(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight contacted the planned transport")

    planned_provider = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=planned_transport,
        operations_path=config.operations_file,
    )
    plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=planned_provider,
    )
    changed_provider = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        operations_path=config.operations_file,
    )

    with pytest.raises(AudioPlanError, match="changed after this plan"):
        audio_application.preflight_paid_deck_audio_plan(
            plan,
            sentence_provider=changed_provider,
        )


def test_paid_deck_audio_preflight_skips_local_current_and_recoverable_clips(
    tmp_path: Path,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    local_plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=Provider("voicevox", 8),
    )
    audio_application.preflight_paid_deck_audio_plan(local_plan)

    paid_provider = openai_realtime.OpenAiRealtimePool(
        api_key="",
        operations_path=config.operations_file,
    )
    paid_plan = audio_application.plan_deck_audio(
        config,
        deck,
        word_provider=Provider("voicevox", 7),
        sentence_provider=paid_provider,
    )
    states = ("current", "recoverable")
    skipped = replace(
        paid_plan,
        clips=tuple(
            replace(clip, state=states[position])
            for position, clip in enumerate(paid_plan.clips)
        ),
        example_counts=audio_application.AudioClipCounts(
            total=2,
            current=1,
            recoverable=1,
            provider_required=0,
        ),
    )

    audio_application.preflight_paid_deck_audio_plan(skipped)


def test_locked_deck_audio_execution_skips_only_the_operation_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    expected = audio_application.AudioExecutionOutcome(
        state="no-records",
        plan=None,
        output_dir=config.media_dir / "audio",
        no_records=True,
    )
    calls: list[tuple[Path | None, bool, bool, bool]] = []

    def execute_locked(
        received: ProjectConfig,
        record_ids: tuple[str, ...] | None,
        **kwargs: object,
    ) -> audio_application.AudioExecutionOutcome:
        assert received is config
        calls.append(
            (
                kwargs["deck_path"],
                bool(kwargs["words"]),
                bool(kwargs["examples"]),
                bool(kwargs["prune"]),
            )
        )
        return expected

    monkeypatch.setattr(audio_application, "_execute_audio_locked", execute_locked)

    def unexpected_lock(path: Path):
        pytest.fail(f"locked deck execution tried to acquire {path}")

    monkeypatch.setattr(audio_application, "exclusive_path_lock", unexpected_lock)

    outcome = audio_application.execute_deck_audio_locked(config, deck)

    assert outcome is expected
    assert calls == [(deck.resolve(), False, True, False)]


@pytest.mark.parametrize(
    "options",
    [
        {"words": True},
        {"examples": False},
        {"prune": True},
    ],
)
def test_locked_deck_audio_execution_keeps_deck_only_constraints(
    tmp_path: Path,
    options: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, _item = _drill_project(tmp_path)
    monkeypatch.setattr(
        audio_application,
        "_execute_audio_locked",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid deck scope reached locked audio execution"
        ),
    )

    with pytest.raises(AudioPlanError, match="examples only|global media cleanup"):
        audio_application.execute_deck_audio_locked(config, deck, **options)


def test_drill_audio_uses_the_shared_transaction_and_persists_both_references(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)
    words = RecordingProvider("voicevox", 7)
    sentences = RecordingProvider("openai-realtime", "cedar")

    first = audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert first.succeeded
    assert sentences.said == [
        ("日本語が話せます。", False),
        ("英語も、話せる？", False),
    ]
    stored = load_structured(deck)["deck"]["drill_examples"][item.id]
    assert [example["register"] for example in stored] == ["polite", "casual"]
    assert all(example["audio"].startswith("audio/janki-") for example in stored)
    assert all((config.media_dir / example["audio"]).is_file() for example in stored)
    assert load_records_snapshot(config.normalized_file)[0][0].examples == []
    rewritten = deck.read_text(encoding="utf-8")
    assert "# This learner note must survive generated audio updates." in rewritten
    assert 'name: "Potential"' in rewritten
    assert '  include_ids: ["word:話す:はなす"]' in rewritten
    assert '\n    "word:話す:はなす":\n      - japanese:' in rewritten

    sentences.said.clear()
    second = audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert second.succeeded
    assert second.up_to_date == 2
    assert sentences.said == [], "current drill clips use the same no-rebill path"
    ledger = ledger_mod.load(config.ledger_file)
    assert set(ledger.records) == {
        "drill-audio:1047286103:potential:word:話す:はなす"
    }


def test_drill_audio_recovers_paid_stages_after_a_concurrent_deck_edit(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)

    class EditsDeckDuringSynthesis(RecordingProvider):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.said:
                current = deck.read_text(encoding="utf-8")
                deck.write_text(
                    current.replace(
                        "english: I can speak Japanese.",
                        "english: I am able to speak Japanese.",
                    ),
                    encoding="utf-8",
                )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    sentences = EditsDeckDuringSynthesis("openai-realtime", "cedar")
    first = audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=sentences,
    )

    assert first.state == "records-stale"
    assert first.pending_recovery is True
    assert len(sentences.said) == 2

    second = audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=sentences,
    )

    assert second.succeeded
    assert len(sentences.said) == 2, "the exact staged clips were adopted"
    stored = load_structured(deck)["deck"]["drill_examples"][item.id]
    assert stored[0]["english"] == "I am able to speak Japanese."
    assert all(example["audio"].startswith("audio/janki-") for example in stored)


def test_drill_audio_executes_against_the_decks_declared_source(
    tmp_path: Path,
) -> None:
    normalized = _record("本", "ほん")
    source_record = replace(
        _record("話す", "はなす"),
        part_of_speech="verb",
        verb_group="godan",
    )
    config = _project(tmp_path, [normalized])
    source_path = tmp_path / "other.json"
    source_path.write_text(
        json.dumps([source_record.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    config.deck_dir.mkdir(parents=True)
    deck = config.deck_dir / "potential.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential\n"
        "  deck_id: 1047286103\n"
        "  model_id: 1607392351\n"
        "  source: ../other.json\n"
        '  include_ids: ["word:話す:はなす"]\n'
        "  drill_examples:\n"
        '    "word:話す:はなす":\n'
        "      - japanese: 日本語が話せます。\n"
        "        english: I can speak Japanese.\n"
        "        register: polite\n"
        "      - japanese: 英語も話せる？\n"
        '        english: "Can you speak English too?"\n'
        "        register: casual\n",
        encoding="utf-8",
    )
    normalized_before = config.normalized_file.read_text(encoding="utf-8")
    source_before = source_path.read_text(encoding="utf-8")
    sentences = RecordingProvider("openai-realtime", "cedar")

    outcome = audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=sentences,
    )

    assert outcome.succeeded
    assert [spoken for spoken, _forced in sentences.said] == [
        "日本語が話せます。",
        "英語も話せる？",
    ]
    assert config.normalized_file.read_text(encoding="utf-8") == normalized_before
    assert source_path.read_text(encoding="utf-8") == source_before


def test_duplicate_drill_audio_owners_refuse_before_synthesis(tmp_path: Path) -> None:
    config, first, _item = _drill_project(tmp_path)
    second = config.deck_dir / "another-potential.yaml"
    second.write_text(
        first.read_text(encoding="utf-8").replace(
            "英語も話せる？", "明日も話せる？"
        ),
        encoding="utf-8",
    )
    sentences = RecordingProvider("openai-realtime", "cedar")

    with pytest.raises(AudioPlanError, match="shared by.*distinct deck_id"):
        audio_application.execute_deck_audio(
            config,
            first,
            examples=True,
            word_provider=RecordingProvider("voicevox", 7),
            sentence_provider=sentences,
        )

    assert sentences.said == []
    assert not config.ledger_file.exists()


def test_a_vocabulary_record_cannot_alias_a_synthetic_drill_audio_owner(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)
    owner = "drill-audio:1047286103:potential:word:話す:はなす"
    alias = replace(_record("別", "べつ"), id=owner)
    config.normalized_file.write_text(
        json.dumps([item.to_dict(), alias.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    sentences = RecordingProvider("openai-realtime", "cedar")

    with pytest.raises(AudioPlanError, match="collides with a durable vocabulary"):
        audio_application.execute_deck_audio(
            config,
            deck,
            examples=True,
            word_provider=RecordingProvider("voicevox", 7),
            sentence_provider=sentences,
        )

    assert sentences.said == []
    assert not config.ledger_file.exists()


def test_a_drill_owner_colliding_with_a_later_decks_inline_note_refuses(
    tmp_path: Path,
) -> None:
    """Every deck's declared ids are known before any drill owner is examined.

    The census cannot answer one deck at a time: a synthetic `drill-audio:`
    owner aliasing a record version declared by a deck read *after* it is the
    same collision, and it has to be refused with the same certainty.
    """
    config, deck, _item = _drill_project(tmp_path)
    later = config.deck_dir / "z-later.yaml"
    later.write_text(
        "deck:\n"
        "  name: Later\n"
        "  deck_id: 51\n"
        "  model_id: 52\n"
        "notes:\n"
        '  - id: "drill-audio:1047286103:potential:word:話す:はなす"\n'
        "    expression: 別\n"
        "    reading: べつ\n"
        "    meanings: [other]\n",
        encoding="utf-8",
    )
    sentences = RecordingProvider("openai-realtime", "cedar")

    with pytest.raises(AudioPlanError, match="collides with a durable vocabulary"):
        audio_application.execute_deck_audio(
            config,
            deck,
            examples=True,
            word_provider=RecordingProvider("voicevox", 7),
            sentence_provider=sentences,
        )

    assert sentences.said == []
    assert not config.ledger_file.exists()


def test_corpus_prune_preserves_audio_owned_only_by_a_drill_deck(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)
    audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=RecordingProvider("openai-realtime", "cedar"),
    )
    stored = load_structured(deck)["deck"]["drill_examples"][item.id]
    paths = [config.media_dir / example["audio"] for example in stored]

    outcome = audio_application.execute_corpus_audio(
        config,
        examples=True,
        prune=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=RecordingProvider("openai-realtime", "cedar"),
    )

    assert outcome.succeeded
    assert outcome.pruned_paths == ()
    assert all(path.is_file() for path in paths)
    owner = "drill-audio:1047286103:potential:word:話す:はなす"
    assert owner in ledger_mod.load(config.ledger_file).records


def test_corpus_prune_retires_audio_after_its_drill_owner_is_removed(
    tmp_path: Path,
) -> None:
    config, deck, item = _drill_project(tmp_path)
    audio_application.execute_deck_audio(
        config,
        deck,
        examples=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=RecordingProvider("openai-realtime", "cedar"),
    )
    stored = load_structured(deck)["deck"]["drill_examples"][item.id]
    paths = [config.media_dir / example["audio"] for example in stored]
    deck.unlink()

    outcome = audio_application.execute_corpus_audio(
        config,
        examples=True,
        prune=True,
        word_provider=RecordingProvider("voicevox", 7),
        sentence_provider=RecordingProvider("openai-realtime", "cedar"),
    )

    assert outcome.succeeded
    assert set(outcome.pruned_paths) == set(paths)
    assert all(not path.exists() for path in paths)
    owner = "drill-audio:1047286103:potential:word:話す:はなす"
    assert ledger_mod.load(config.ledger_file).records[owner]["audio"] == []
