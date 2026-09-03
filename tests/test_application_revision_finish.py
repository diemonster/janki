"""One exact, durable apply/audio/build authority for reviewed revisions."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_application_audio import Provider
from test_application_revision_apply import _fixture as _revision_fixture

from japanese_anki import ledger as ledger_mod
from japanese_anki.application import audio as audio_application
from japanese_anki.application import revision as revision_application
from japanese_anki.application import revision_finish
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import RecordsRevision
from japanese_anki.tts import openai_realtime

_PCM = b"\x01\x00" * 240


class _RealtimeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        _endpoint: str,
        _headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append(request)
        voice = request["response"]["audio"]["output"]["voice"]
        return [
            {
                "type": "session.created",
                "session": {"model": openai_realtime.MODEL},
            },
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(_PCM).decode("ascii"),
            },
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output_modalities": ["audio"],
                    "audio": {
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": openai_realtime.SAMPLE_RATE,
                            },
                            "voice": voice,
                        }
                    },
                },
            },
        ]


def _fixture(tmp_path: Path) -> tuple[ProjectConfig, Path, Path]:
    config, deck, staging = _revision_fixture(tmp_path)
    shutil.copytree(
        Path(__file__).parents[1] / "templates" / "japanese-study",
        config.template_dir,
    )
    return config, deck, staging


def _add_unselected_audio_reference(
    config: ProjectConfig,
    deck: Path,
    staging: Path,
) -> tuple[Path, str]:
    """Keep one unreviewed card's existing media inside the whole-deck build."""

    media = config.media_dir / "audio" / "unselected-current.wav"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"existing unselected media")
    text = deck.read_text(encoding="utf-8").replace(
        "        english: I can read.\n",
        "        english: I can read.\n        audio: audio/unselected-current.wav\n",
    )
    deck.write_text(text, encoding="utf-8")
    deck_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    manifest = json.loads(staging.read_text(encoding="utf-8"))
    target = manifest["target"]
    request = manifest["request"]
    provider = request["provider_plan"]
    target["deck_sha256"] = deck_sha256
    request["plan_fingerprint"] = revision_application._plan_identity(
        deck_relative=target["deck_path"],
        deck_sha256=deck_sha256,
        selected=tuple(target["selected_record_ids"]),
        owner_instruction=request["owner_instruction"],
        provider=provider["provider"],
        model=provider["model"],
        canonical_context_fingerprint=request["canonical_context_fingerprint"],
        request_fingerprint=provider["request_fingerprint"],
        staging_relative=target["staging_path"],
        staging_revision=None,
    )
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    operations = json.loads(config.operations_file.read_text(encoding="utf-8"))
    operations["operations"][manifest["operation_id"]]["source_sha256"] = deck_sha256
    config.operations_file.write_text(
        json.dumps(operations, indent=2) + "\n",
        encoding="utf-8",
    )
    return media.resolve(), hashlib.sha256(media.read_bytes()).hexdigest()


def _providers(
    config: ProjectConfig,
    *,
    api_key: str,
    transport: _RealtimeTransport,
) -> tuple[Provider, openai_realtime.OpenAiRealtimePool]:
    return (
        Provider("voicevox", 7),
        openai_realtime.OpenAiRealtimePool(
            api_key=api_key,
            transport=transport,
            operations_path=config.operations_file,
        ),
    )


def _tree(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(openai_realtime.SAMPLE_RATE)
        stream.writeframes(_PCM)
    return output.getvalue()


def _seed_recoverable_plan(
    config: ProjectConfig,
    staging: Path,
    *,
    words: Provider,
    sentences: openai_realtime.OpenAiRealtimePool,
) -> tuple[revision_finish.RevisionFinishPlan, tuple[Path, ...]]:
    unpaid = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    pending = config.media_dir / "audio" / ".pending"
    pending.mkdir(parents=True)
    book = ledger_mod.Ledger(path=config.ledger_file)
    paid = _wav()
    digest = hashlib.sha256(paid).hexdigest()
    stages: list[Path] = []
    for clip in unpaid.audio.clips:
        provider = clip.provider
        key = book.pending_audio_key_for(
            clip.record_id,
            of=clip.kind,
            target=clip.target,
            request_input=clip.request_input,
            forced_accent=clip.forced_accent,
            content_fp=clip.content_fingerprint,
            provider=provider.name,
            voice=provider.voice,
            speed=provider.speed,
            settings=dict(provider.settings),
        )
        stage = pending / f"{key}-{digest}.stage"
        stage.write_bytes(paid)
        stages.append(stage)
    recoverable = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert recoverable.audio.clips
    assert all(clip.state == "recoverable" for clip in recoverable.audio.clips)
    assert recoverable.provider_required_count == 0
    return recoverable, tuple(stages)


def test_finish_plan_is_exact_side_effect_free_and_never_contacts_realtime(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )
    before = _tree(tmp_path)

    first = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    second = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert first == second
    assert first.revision.deck_path == deck.resolve()
    assert first.audio.canonical_path == deck.resolve()
    assert first.audio.words is False
    assert first.audio.examples is True
    assert first.audio.force is False
    assert first.audio.clips
    assert first.provider_required_count == len(first.audio.clips)
    assert first.build.deck_sha256 == first.projected_audio_deck_sha256
    assert first.build.deck_path == deck.resolve()
    assert all(item.sha256 is None for item in first.build.media_inputs)
    assert {item.path for item in first.build.media_inputs} == {
        (config.media_dir / "audio" / clip.target).resolve() for clip in first.audio.clips
    }
    assert (
        first.record_path
        == (
            config.staging_dir / "done" / "revisions" / f"finish-{first.fingerprint}.json"
        ).resolve()
    )
    assert len(first.fingerprint) == 64
    assert transport.calls == []
    assert _tree(tmp_path) == before


def test_finish_plan_binds_already_current_media_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )
    real_plan = audio_application.plan_deck_audio_revision

    def current_plan(*args: Any, **kwargs: Any) -> audio_application.AudioPlan:
        planned = real_plan(*args, **kwargs)
        current_clips = []
        for position, clip in enumerate(planned.clips):
            target = config.media_dir / "audio" / clip.target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"current clip {position}".encode())
            current_clips.append(
                replace(
                    clip,
                    state="current",
                    recovery_key=None,
                    recovery_sha256=None,
                    recovery_source=None,
                )
            )
        return replace(planned, clips=tuple(current_clips))

    monkeypatch.setattr(audio_application, "plan_deck_audio_revision", current_plan)

    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )

    expected = {
        clip.target: hashlib.sha256(
            (config.media_dir / "audio" / clip.target).read_bytes()
        ).hexdigest()
        for clip in plan.audio.clips
    }
    assert {item.path.name: item.sha256 for item in plan.build.media_inputs} == expected
    assert {
        clip["target"]: clip["initial_media_sha256"] for clip in plan.authority["audio"]["clips"]
    } == expected
    assert transport.calls == []


def test_projection_reuses_one_exact_audio_reference_for_duplicate_examples(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    manifest = json.loads(staging.read_text(encoding="utf-8"))
    selected = manifest["proposal"]["drill_examples"]["word:話す:はなす"]
    selected[1].update(
        japanese=selected[0]["japanese"],
        furigana=selected[0]["furigana"],
        english=selected[0]["english"],
    )
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )

    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )

    projected = yaml.safe_load(plan.projected_audio_deck_text)
    examples = projected["deck"]["drill_examples"]["word:話す:はなす"]
    assert len(examples) == 2
    assert examples[0]["audio"] == examples[1]["audio"]
    assert examples[0]["audio"].startswith("audio/janki-")
    assert sum(clip.request_input == selected[0]["japanese"] for clip in plan.audio.clips) == 1
    assert transport.calls == []


def test_projection_keeps_distinct_displayed_examples_with_same_spoken_request(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )

    base = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    records = pattern_cards.drill_audio_records_from_revision(
        deck,
        config,
        RecordsRevision(deck.resolve(), base.revision.intended_deck_text),
    )
    owner = records[0]
    assert owner.examples[0].japanese != owner.examples[1].japanese
    examples = tuple(
        replace(example, spoken_japanese="shared provider request") for example in owner.examples
    )
    records = [
        replace(record, examples=list(examples)) if record.id == owner.id else record
        for record in records
    ]
    clips: list[audio_application.AudioClipPlan] = []
    owner_position = 0
    for clip in base.audio.clips:
        if clip.record_id != owner.id:
            clips.append(clip)
            continue
        updated_example = examples[owner_position]
        owner_position += 1
        clips.append(
            replace(
                clip,
                request_input=ledger_mod.example_audio_request(updated_example),
                content_fingerprint=ledger_mod.example_audio_content_fingerprint(updated_example),
            )
        )
    audio = replace(base.audio, clips=tuple(clips))

    projected_records = revision_finish._project_audio_references(audio, records)
    projected_owner = next(record for record in projected_records if record.id == owner.id)
    assert len([clip for clip in audio.clips if clip.record_id == owner.id]) == 2
    assert {clip.request_input for clip in audio.clips if clip.record_id == owner.id} == {
        "shared provider request"
    }
    assert projected_owner.examples[0].audio != projected_owner.examples[1].audio
    assert {example.audio for example in projected_owner.examples} == {
        f"audio/{clip.target}" for clip in audio.clips if clip.record_id == owner.id
    }
    assert transport.calls == []


def test_unavailable_paid_preflight_refuses_before_apply_or_authority_write(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    deck_before = deck.read_bytes()
    staging_before = staging.read_bytes()
    operations_before = config.operations_file.read_bytes()

    with pytest.raises(JankiError, match="OPENAI_API_KEY"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert transport.calls == []
    assert deck.read_bytes() == deck_before
    assert staging.read_bytes() == staging_before
    assert config.operations_file.read_bytes() == operations_before
    assert not plan.record_path.exists()
    assert not plan.revision.archive_path.exists()
    assert not config.ledger_file.exists()
    assert not plan.build.output_path.exists()


def test_one_finish_executes_apply_audio_and_build_with_durable_receipts(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    unselected_media, unselected_sha256 = _add_unselected_audio_reference(
        config,
        deck,
        staging,
    )
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    before_document = yaml.safe_load(deck.read_text(encoding="utf-8"))
    unselected_before = before_document["deck"]["drill_examples"]["word:読む:よむ"]
    phases: list[str] = []

    def disconnected_progress(phase: str) -> None:
        phases.append(phase)
        raise RuntimeError("the browser disconnected")

    result = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
        progress=disconnected_progress,
    )

    assert result.succeeded
    assert result.receipt_id == plan.fingerprint
    assert result.card_count == plan.build.card_count
    assert result.package_sha256 == hashlib.sha256(plan.build.output_path.read_bytes()).hexdigest()
    assert plan.build.output_path.read_bytes().startswith(b"PK")
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == (plan.projected_audio_deck_sha256)
    assert not staging.exists()
    assert plan.revision.archive_path.is_file()
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["state"] == "complete"
    assert record["receipt_id"] == plan.fingerprint
    assert record["authority"] == plan.authority
    assert all(
        isinstance(record[key], dict) for key in ("apply_receipt", "audio_receipt", "build_receipt")
    )
    assert isinstance(record["build_binding"], dict)
    bound_media = {item["path"]: item["sha256"] for item in record["build_binding"]["media"]}
    selected_media = {
        str((config.media_dir / "audio" / clip["target"]).relative_to(config.root)): clip["sha256"]
        for clip in record["audio_receipt"]["clips"]
    }
    assert bound_media == {
        **selected_media,
        str(unselected_media.relative_to(config.root)): unselected_sha256,
    }
    assert (
        record["build_receipt"]["plan_fingerprint"] == record["build_binding"]["plan_fingerprint"]
    )
    assert len(transport.calls) == plan.provider_required_count
    assert all(
        (config.media_dir / "audio" / clip["target"]).read_bytes().startswith(b"RIFF")
        for clip in record["audio_receipt"]["clips"]
    )
    stored = yaml.safe_load(deck.read_text(encoding="utf-8"))
    selected_after = stored["deck"]["drill_examples"]["word:話す:はなす"]
    assert sorted(example["audio"] for example in selected_after) == sorted(
        f"audio/{clip.target}" for clip in plan.audio.clips
    )
    assert stored["deck"]["drill_examples"]["word:読む:よむ"] == unselected_before
    assert len(transport.calls) == 2
    assert "読め" not in json.dumps(transport.calls, ensure_ascii=False)
    assert phases == [
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    ]


def test_finish_never_completes_when_media_is_swapped_at_the_build_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    raced_media = config.media_dir / "audio" / plan.audio.clips[0].target
    real_build = pattern_cards.build_conjugation_deck

    def raced_build(
        deck_path: Path,
        current: ProjectConfig,
        records: tuple[Any, ...],
        output: Path,
        **kwargs: object,
    ) -> tuple[Path, int]:
        at_seam = json.loads(plan.record_path.read_text(encoding="utf-8"))
        assert at_seam["state"] == "audio_complete"
        assert isinstance(at_seam["build_binding"], dict)
        assert at_seam["build_receipt"] is None
        result = real_build(deck_path, current, records, output, **kwargs)
        raced_media.write_bytes(b"swapped at the finish build seam")
        return result

    monkeypatch.setattr(pattern_cards, "build_conjugation_deck", raced_build)

    with pytest.raises(JankiError, match="during package generation"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["state"] == "audio_complete"
    assert isinstance(record["build_binding"], dict)
    assert record["build_receipt"] is None
    assert plan.build.output_path.exists(), "dist is disposable after a refused build"


def test_partial_audio_resumes_exact_recovery_without_a_second_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(
        config,
        api_key="test-key",
        transport=transport,
    )
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    real_promote = audio_application.audio_cmd.promote_pending_audio

    def stop_before_publish(*_args: object, **_kwargs: object) -> None:
        raise JankiError("simulated media publication interruption")

    monkeypatch.setattr(
        audio_application.audio_cmd,
        "promote_pending_audio",
        stop_before_publish,
    )
    partial = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert partial.state == "revision_applied"
    assert partial.audio is not None
    assert partial.audio.pending_recovery is True
    assert not plan.build.output_path.exists()
    assert len(transport.calls) == plan.provider_required_count
    paid_call_count = len(transport.calls)
    monkeypatch.setattr(
        audio_application.audio_cmd,
        "promote_pending_audio",
        real_promote,
    )

    completed = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    assert len(transport.calls) == paid_call_count
    assert json.loads(plan.record_path.read_text(encoding="utf-8"))["state"] == ("complete")


def test_click_replan_refuses_reviewed_proposal_drift_before_writing(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    deck_before = deck.read_bytes()
    staging.write_text(
        staging.read_text(encoding="utf-8").replace(
            "New note.", "Edited after the finish plan was shown."
        ),
        encoding="utf-8",
    )

    with pytest.raises(revision_finish.RevisionFinishError, match="plan changed"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert transport.calls == []
    assert deck.read_bytes() == deck_before
    assert staging.exists()
    assert not plan.record_path.exists()
    assert not plan.revision.archive_path.exists()


def test_resume_refuses_recoverable_clip_widening_to_a_new_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan, stages = _seed_recoverable_plan(
        config,
        staging,
        words=words,
        sentences=sentences,
    )
    real_execute = revision_finish._execute_record_locked

    def pause_after_authority(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("simulated stop after authority")

    monkeypatch.setattr(
        revision_finish,
        "_execute_record_locked",
        pause_after_authority,
    )
    with pytest.raises(RuntimeError, match="after authority"):
        revision_finish.execute_revision_finish(
            config,
            plan,
            word_provider=words,
            sentence_provider=sentences,
        )
    monkeypatch.setattr(revision_finish, "_execute_record_locked", real_execute)
    stages[0].unlink()

    with pytest.raises(revision_finish.RevisionFinishError, match="wider"):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert transport.calls == []
    assert not stages[0].exists()


def test_recoverable_publication_preserves_the_exact_staged_bytes(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan, stages = _seed_recoverable_plan(
        config,
        staging,
        words=words,
        sentences=sentences,
    )

    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )

    assert completed.succeeded
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    initial_hashes = {
        clip["target"]: clip["initial_recovery_sha256"] for clip in plan.authority["audio"]["clips"]
    }
    assert {
        Path(item["path"]).name: item["sha256"] for item in plan.authority["build"]["media"]
    } == initial_hashes
    for clip in record["audio_receipt"]["clips"]:
        canonical = config.media_dir / "audio" / clip["target"]
        assert clip["sha256"] == initial_hashes[clip["target"]]
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == clip["sha256"]
    assert all(not stage.exists() for stage in stages)
    assert transport.calls == []


def test_resume_refuses_tampered_exact_build_media_binding(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="test-key", transport=transport)
    plan = revision_finish.plan_revision_finish(
        config,
        staging,
        word_provider=words,
        sentence_provider=sentences,
    )
    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert completed.succeeded
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    record["build_binding"]["media"][0]["sha256"] = "f" * 64
    plan.record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_finish.RevisionFinishError):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert len(transport.calls) == plan.provider_required_count


def test_resume_refuses_bytes_swapped_behind_a_receipted_current_clip(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan, _stages = _seed_recoverable_plan(
        config,
        staging,
        words=words,
        sentences=sentences,
    )
    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert completed.succeeded
    unchanged = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert unchanged.succeeded
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    target = record["audio_receipt"]["clips"][0]["target"]
    (config.media_dir / "audio" / target).write_bytes(b"swapped audio bytes")

    with pytest.raises(revision_finish.RevisionFinishError):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert transport.calls == []


def test_resume_refuses_receipt_target_tampering_against_bound_authority(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    transport = _RealtimeTransport()
    words, sentences = _providers(config, api_key="", transport=transport)
    plan, _stages = _seed_recoverable_plan(
        config,
        staging,
        words=words,
        sentences=sentences,
    )
    completed = revision_finish.execute_revision_finish(
        config,
        plan,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert completed.succeeded
    unchanged = revision_finish.resume_revision_finish(
        config,
        plan.fingerprint,
        word_provider=words,
        sentence_provider=sentences,
    )
    assert unchanged.succeeded
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    record["audio_receipt"]["clips"][0]["target"] = "janki-forged.wav"
    plan.record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_finish.RevisionFinishError):
        revision_finish.resume_revision_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    assert transport.calls == []
