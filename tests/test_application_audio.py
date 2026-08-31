"""Display-only planning for W5's exact audio actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import cli
from japanese_anki import ledger as ledger_mod
from japanese_anki.application import audio as audio_application
from japanese_anki.application.audio import (
    AudioPlanError,
    plan_audio_records,
    plan_corpus_audio,
    plan_targeted_audio,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records_snapshot
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import TtsError


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
        'media_dir = "media"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


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
