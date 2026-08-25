"""`janki status`: the read-only view over records plus the ledger.

Three properties get most of the attention here, because they are what make the
command trustworthy: it works on a repo that has never run an import (a missing
ledger is an empty ledger), it never writes unless asked to rebuild, and its
duplicate detection finds the *kana-form* class of duplicate — the common real
one, and the one an expression-only check misses.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, extract, ledger, operations, patterns, staging
from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.ledger import (
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import (
    ExampleSentence,
    VocabularyRecord,
    mark_provisional,
)

TODAY = date.today().isoformat()

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
ledger_file = "ledger.json"
media_dir = "media"
"""


def _raw(expression: str, reading: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": f"word:{expression}:{reading}",
        "expression": expression,
        "reading": reading,
        "meanings": ["to speak"],
        "source": {"type": "shirabe", "imported_from": "export.csv", "row": 2},
    }
    data.update(overrides)
    return data


def _record(expression: str, reading: str, **overrides: Any) -> VocabularyRecord:
    return VocabularyRecord.from_dict(_raw(expression, reading, **overrides))


def _project(
    tmp_path: Path,
    records: list[dict[str, Any]] | None = None,
    decks: dict[str, dict[str, Any]] | None = None,
) -> Path:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(records or [], ensure_ascii=False), encoding="utf-8"
    )
    deck_dir = tmp_path / "decks"
    deck_dir.mkdir()
    for stem, deck in (decks or {}).items():
        (deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump(deck, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    return tmp_path


def _sourced_deck(name: str = "Vocabulary") -> dict[str, Any]:
    """A deck built from the normalized file, the way a real one is."""
    return {"deck": {"name": name, "source": "../vocabulary.json"}, "notes": []}


def _status(root: Path, *flags: str) -> int:
    return cli.main(["--root", str(root), "status", *flags])


# --- the summary ------------------------------------------------------------


def test_status_runs_on_a_repo_that_has_never_imported_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert "Records: 0" in out
    assert "no ledger file yet" in out
    # Reading a status report must not create the file it reports on.
    assert not (root / "ledger.json").exists()


def test_status_surfaces_pending_paid_audio_and_rebuild_preserves_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    item = _record("話す", "はなす")
    root = _project(tmp_path, [item.to_dict()])
    book = ledger.load(root / "ledger.json")
    content_fp = word_audio_content_fingerprint(item)
    arguments = {
        "of": "word",
        "target": "janki-pending.wav",
        "request_input": item.reading,
        "forced_accent": False,
        "content_fp": content_fp,
        "provider": "voicevox",
        "voice": 53,
        "speed": 1.0,
        "settings": {},
    }
    key = book.pending_audio_key_for(item.id, **arguments)
    stage = root / "media" / "audio" / ".pending" / f"{key}-{'f' * 64}.stage"
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"paid")
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=f".pending/{key}-{'f' * 64}.stage",
        staged_sha256="f" * 64,
    )
    book.save()
    before = json.loads((root / "ledger.json").read_text(encoding="utf-8"))[
        "pending_audio"
    ]

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "1 pending paid audio" in captured.err
    assert "janki audio --words" in captured.err
    assert "Pending audio recovery: 1" in captured.out

    assert _status(root, "--rebuild") == 0
    after = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert after["pending_audio"] == before


def test_status_prescribes_prune_for_pending_audio_whose_record_was_deleted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _project(tmp_path)
    item = _record("話す", "はなす")
    book = ledger.load(root / "ledger.json")
    arguments = {
        "of": "word",
        "target": "janki-pending.wav",
        "request_input": item.reading,
        "forced_accent": False,
        "content_fp": word_audio_content_fingerprint(item),
        "provider": "voicevox",
        "voice": 53,
        "speed": 1.0,
        "settings": {},
    }
    key = book.pending_audio_key_for(item.id, **arguments)
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=f".pending/{key}-{'f' * 64}.stage",
        staged_sha256="f" * 64,
    )
    book.save()

    assert _status(root) == 0

    warning = capsys.readouterr().err
    assert "record was deleted" in warning
    assert "janki audio --words --prune" in warning


def test_build_refuses_while_paid_audio_transaction_is_pending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    book = ledger.load(root / "ledger.json")
    item = _record("話す", "はなす")
    arguments = {
        "of": "word",
        "target": "janki-pending.wav",
        "request_input": item.reading,
        "forced_accent": False,
        "content_fp": word_audio_content_fingerprint(item),
        "provider": "voicevox",
        "voice": 53,
        "speed": 1.0,
        "settings": {},
    }
    key = book.pending_audio_key_for(item.id, **arguments)
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=f".pending/{key}-{'f' * 64}.stage",
        staged_sha256="f" * 64,
    )
    book.save()
    deck = root / "decks" / "vocabulary.yaml"
    deck.write_text(
        yaml.safe_dump(_sourced_deck(), allow_unicode=True), encoding="utf-8"
    )

    assert cli.main(["--root", str(root), "build", str(deck)]) == 1

    err = capsys.readouterr().err
    assert "pending audio" in err.lower()
    assert "janki audio" in err
    assert not any(root.glob("dist/*.apkg"))


def test_an_unavailable_audio_profile_warns_without_taking_status_down(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record("話す", "はなす", audio="audio/janki-word.wav")
    root = _project(tmp_path, [record.to_dict()])
    (root / "janki.toml").write_text(
        CONFIG + '\n[tts]\nprovider = "azure"\n',
        encoding="utf-8",
    )
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp="old-content",
    )
    book.save()

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "Stale audio: 1" in captured.out, "content remains comparable"
    assert "could not compare the configured word-audio profile" in captured.err
    assert "Azure provider was evaluated and dropped" in captured.err


def test_the_record_universe_is_the_normalized_file_plus_inline_deck_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],
        {
            "vocabulary": _sourced_deck(),
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [_raw("食べる", "たべる", source={"type": "manual"})],
            },
        },
    )

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert "Records: 2 (1 in vocabulary.json, 1 inline in deck files)" in out
    assert "By source type: manual 1, shirabe 1" in out


def test_stale_audio_sees_each_durable_inline_render_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_id = "word:話す:はなす"
    sentence = "毎日話す。"
    base = _record("話す", "はなす")
    filename = (
        "janki-"
        f"{example_audio_filename_fingerprint(base, ExampleSentence(japanese=sentence))}"
        ".mp3"
    )

    def inline(instructions: str) -> dict[str, Any]:
        return _raw(
            "話す",
            "はなす",
            examples=[
                {
                    "japanese": sentence,
                    "audio": f"audio/{filename}",
                    "instructions": instructions,
                }
            ],
        )

    root = _project(
        tmp_path,
        [],
        {
            "a": {"deck": {"name": "A"}, "notes": [inline("A.")]},
            "b": {"deck": {"name": "B"}, "notes": [inline("B.")]},
        },
    )
    (root / "janki.toml").write_text(
        CONFIG
        + '\n[tts]\nsentence_provider = "openai"\n'
        + 'openai_instructions = "Global."\n',
        encoding="utf-8",
    )
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record_id,
        file=filename,
        of="example",
        provider="openai",
        voice="onyx",
        speed=1.0,
        settings={
            "model": "gpt-4o-mini-tts",
            "instructions": "Global.\n\nB.",
        },
        content_fp=example_audio_content_fingerprint(
            ExampleSentence(japanese=sentence)
        ),
    )
    book.save()

    assert _status(root) == 0

    assert "Stale audio: 1" in capsys.readouterr().out


def test_summary_reports_exports_audio_enrichment_and_staleness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spoken = _record(
        "話す",
        "はなす",
        usage_notes="Everyday verb.",
        examples=[{"japanese": "毎日話す。", "english": "I speak every day."}],
    )
    root = _project(
        tmp_path,
        [spoken.to_dict(), _raw("食べる", "たべる")],
        {"vocabulary": _sourced_deck()},
    )

    book = ledger.load(root / "ledger.json")
    book.record_export(spoken.id, "vocabulary", at="2026-08-12")
    book.record_audio(
        spoken.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp="stale-fingerprint",
        at="2026-08-11",
    )
    book.save()

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert "Ledger: ledger.json — 1 record entry" in out
    assert "Never exported: vocabulary 1 of 2" in out
    assert "Missing word audio: 1 of 2" in out
    assert "Stale audio: 1" in out
    # 話す has both an example and usage notes; 食べる has neither.
    assert "Missing enrichment: 1" in out


def test_status_reports_word_audio_from_the_previous_configured_voice_as_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Status includes the configured render profile in its currency answer.

    The content fingerprint deliberately excludes the voice: changing a voice
    rewrites the same stable file in place. The ledger is therefore the only
    place that can prove this clip was spoken by speaker 13 while the project
    now asks for 53.
    """
    old_voice = _record("話す", "はなす", audio="audio/janki-word.wav")
    current_voice = _record("食べる", "たべる", audio="audio/janki-current.wav")
    root = _project(tmp_path, [old_voice.to_dict(), current_voice.to_dict()])
    (root / "janki.toml").write_text(
        CONFIG + "\n[tts]\nvoicevox_speaker = 53\nvoicevox_speed = 0.7\n",
        encoding="utf-8",
    )
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        old_voice.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=13,
        speed=0.7,
        content_fp=word_audio_content_fingerprint(old_voice),
    )
    book.record_audio(
        current_voice.id,
        file="janki-current.wav",
        of="word",
        provider="voicevox",
        voice=53,
        speed=0.7,
        content_fp=word_audio_content_fingerprint(current_voice),
    )
    book.save()

    assert _status(root) == 0

    assert "Stale audio: 1" in capsys.readouterr().out


def test_word_voice_does_not_decide_whether_openai_examples_are_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = ExampleSentence(
        japanese="毎日話す。",
        audio="audio/janki-example.mp3",
    )
    record = _record(
        "話す",
        "はなす",
        examples=[{"japanese": example.japanese, "audio": example.audio}],
    )
    root = _project(tmp_path, [record.to_dict()])
    (root / "janki.toml").write_text(
        CONFIG
        + "\n[tts]\nvoicevox_speaker = 53\n"
        + 'sentence_provider = "openai"\nopenai_voice = "onyx"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(root)
    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-example.mp3",
        of="example",
        provider=sentences.name,
        voice=sentences.voice,
        speed=sentences.speed,
        settings=sentences.settings,
        content_fp=example_audio_content_fingerprint(example),
    )
    book.save()

    assert _status(root) == 0

    assert "Stale audio: 0" in capsys.readouterr().out


def test_status_compares_each_openai_examples_effective_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = ExampleSentence(
        japanese="毎日話す。",
        audio="audio/janki-example.mp3",
        instructions="Pronounce 毎日 as まいにち.",
    )
    record = _record("話す", "はなす")
    # ExampleSentence intentionally has no public serializer of its own; the
    # vocabulary record owns sparse serialization.
    record.examples = [example]
    root = _project(tmp_path, [record.to_dict()])
    (root / "janki.toml").write_text(
        CONFIG
        + '\n[tts]\nsentence_provider = "openai"\n'
        + 'openai_instructions = "Global."\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(root)
    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)
    prepared = sentences.for_clip(example.instructions)
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-example.mp3",
        of="example",
        provider=prepared.name,
        voice=prepared.voice,
        speed=prepared.speed,
        settings=prepared.settings,
        content_fp=example_audio_content_fingerprint(example),
    )
    book.save()

    assert _status(root) == 0
    assert "Stale audio: 0" in capsys.readouterr().out

    record.examples[0].instructions = "Changed."
    (root / "vocabulary.json").write_text(
        json.dumps([record.to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    assert _status(root) == 0
    assert "Stale audio: 1" in capsys.readouterr().out


def test_status_explains_when_the_sentence_engine_cannot_honor_clip_instructions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = ExampleSentence(
        japanese="毎日話す。",
        audio="audio/janki-example.wav",
        instructions="Pronounce 毎日 as まいにち.",
    )
    record = _record("話す", "はなす")
    record.examples = [example]
    root = _project(tmp_path, [record.to_dict()])
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-example.wav",
        of="example",
        provider="voicevox",
        voice=46,
        speed=1.0,
        settings={},
        content_fp=example_audio_content_fingerprint(example),
    )
    book.save()

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "Stale audio: 1" in captured.out
    assert "cannot honor 1 example-audio profile" in captured.err
    assert record.id in captured.err
    assert "OpenAI" in captured.err


def test_status_checks_normalized_audio_even_when_an_inline_note_shadows_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = _raw(
        "話す",
        "はなす",
        examples=[
            {
                "japanese": "毎日話す。",
                "instructions": "Pronounce 毎日 as まいにち.",
            }
        ],
    )
    inline = _raw(
        "話す",
        "はなす",
        examples=[{"japanese": "毎日話す。"}],
    )
    root = _project(
        tmp_path,
        [normalized],
        {"inline": {"deck": {"name": "Inline"}, "notes": [inline]}},
    )

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "cannot honor 1 example-audio profile" in captured.err
    assert normalized["id"] in captured.err
    assert "'janki audio --examples' will refuse before synthesis" in captured.err


def test_status_does_not_apply_audio_refusals_to_inline_only_notes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inline = _raw(
        "話す",
        "はなす",
        examples=[
            {
                "japanese": "毎日話す。",
                "instructions": "Pronounce 毎日 as まいにち.",
            }
        ],
    )
    root = _project(
        tmp_path,
        [],
        {"inline": {"deck": {"name": "Inline"}, "notes": [inline]}},
    )

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "cannot honor" not in captured.err
    assert "'janki audio --examples' will refuse" not in captured.err


def test_status_warns_when_shadowed_normalized_openai_input_exceeds_the_api_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _raw(
        "話す",
        "はなす",
        examples=[{"japanese": "長" * 4097}],
    )
    inline = _raw(
        "話す",
        "はなす",
        examples=[{"japanese": "毎日話す。"}],
    )
    root = _project(
        tmp_path,
        [record],
        {"inline": {"deck": {"name": "Inline"}, "notes": [inline]}},
    )
    (root / "janki.toml").write_text(
        CONFIG + '\n[tts]\nsentence_provider = "openai"\n',
        encoding="utf-8",
    )

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "cannot honor 1 example-audio profile" in captured.err
    assert record["id"] in captured.err
    assert "4096-character limit" in captured.err
    assert "'janki audio --examples' will refuse before synthesis" in captured.err


def test_pitch_accent_is_counted_now_that_the_schema_carries_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Before M2.2 the honest answer was that the question could not be asked
    # yet. The field exists now, so the line is a real count —
    # `pitch_accent_supported` reads `dataclasses.fields`, so it flipped on its
    # own when the field landed.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす", pitch_accent=["LHHH"]), _raw("食べる", "たべる")],
    )

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert status_module.pitch_accent_supported() is True
    assert "Missing pitch accent: 1" in out
    assert "n/a until the pitch-accent schema lands" not in out


# --- the review queue -------------------------------------------------------


def _stage(root: Path, name: str, ids: list[str]) -> Path:
    staging = root / "data" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / name
    path.write_text(
        yaml.safe_dump(
            {
                "source_file": "export.csv",
                "records": [
                    {
                        "id": record_id,
                        "expression": record_id.split(":")[1],
                        "reading": "",
                        "source": {"raw_fields": {"hold_reason": "missing reading"}},
                    }
                    for record_id in ids
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _stage_pattern_only_extraction(
    root: Path, *, blocking_coverage: bool = False
) -> Path:
    source = "teform_song.pdf"
    run_id = "11111111-1111-4111-8111-111111111111"
    mode = "table" if blocking_coverage else "auto"
    provenance = {
        "source_sha256": "1" * 64,
        "mode": mode,
        "provider": "anthropic",
        "model": "claude-opus-5",
        "response_schema_version": 3,
        "system_prompt_fingerprint": "2" * 64,
        "style_guide_fingerprint": "3" * 64,
        "user_prompt_fingerprint": "4" * 64,
        "response_schema_fingerprint": "5" * 64,
        "request_fingerprint": "6" * 64,
    }
    proposed = patterns.PatternSet(
        source=source,
        kind="pattern",
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    bound = replace(
        patterns.with_prompt_provenance(proposed, provenance),
        review_run_id=run_id,
    )
    result = extract.ExtractionResult(
        candidates=(), source_units=(), model_reported_unit_count=0
    )
    meta = {
        "source_file": source,
        "review_run_id": run_id,
        "prompt_provenance": provenance,
        "pattern_set": bound.to_dict(),
        "coverage": extract.coverage_block(
            result,
            source_sha256=provenance["source_sha256"],
            mode=mode if blocking_coverage else None,
        ),
    }
    # Keep the fixture honest: this is the same validated schema-v3 rich
    # extraction shape as the live te-form run, not merely a lookalike key.
    pending_coverage = staging.validate_coverage_facts(meta)
    assert (pending_coverage is not None) is blocking_coverage
    path = root / "data" / "staging" / f"{source}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging.write_staging(path, [], meta)
    patterns.save_store(root / "data" / "patterns.json", {source: bound})
    return path


def test_status_says_when_nothing_is_waiting_for_a_human(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])

    assert _status(root) == 0

    assert "Staged for review: none" in capsys.readouterr().out


def test_status_counts_the_rows_an_import_held_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Held rows are the one category of record that is not in the collection
    # and needs a person. A user who misses the one-off line in the import
    # output had no other way to find out they exist.
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage(root, "shirabe-export-needs-reading.yaml", ["word:食べ物:", "word:本:"])

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert "Staged for review: 2 row(s) in 1 file(s) under data/staging" in out


def test_the_staged_detail_flag_names_the_ids_and_why_they_are_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage(root, "shirabe-export-needs-reading.yaml", ["word:食べ物:"])

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "word:食べ物: — missing reading" in out
    # The exit that re-mints these malformed IDs and checks the readings — not
    # a hand-copy recipe that skips both.
    assert "janki promote" in out
    assert "status --rebuild" not in out


def test_staged_ids_pipe_like_every_other_detail_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage(root, "shirabe-export-needs-reading.yaml", ["word:食べ物:"])

    assert _status(root, "--staged", "--format", "ids") == 0

    assert capsys.readouterr().out == "word:食べ物:\n"


def test_an_unreadable_staging_file_is_a_warning_not_a_dead_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    staging = root / "data" / "staging"
    staging.mkdir(parents=True)
    (staging / "broken.yaml").write_text("records: not-a-list\n", encoding="utf-8")

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "warning: skipping staging file" in captured.err
    assert "Records: 1" in captured.out


def test_a_staging_dir_that_is_not_a_directory_is_a_warning_not_silence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `is_dir()` conflated "no staging directory yet" (report none) with
    # "staging_dir points at a file" (a misconfiguration). The second reported
    # none too, so the user found out only when the next import that had to
    # stage a row died on `File exists`.
    root = _project(tmp_path, [_raw("話す", "はなす")])
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "staging").write_text("not a directory\n", encoding="utf-8")

    assert _status(root, "--staged") == 0

    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert "is not a directory" in captured.err
    assert "Records: 1" in captured.out


def test_a_staging_file_with_no_rows_is_not_a_review_queue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # README step 3 is "delete the rows not worth keeping"; a reviewer who kept
    # none leaves `records: []`. The file itself is still tracked project data,
    # so status must report no waiting rows without inventing a deletion route.
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = root / "data" / "staging" / "shirabe-export-needs-reading.yaml"
    path.parent.mkdir(parents=True)
    staging.write_staging(
        path,
        [],
        {
            "source_file": "export.csv",
            "extracted_at": "2026-08-20",
            "review_notes": "Fill the missing reading, then promote.",
        },
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "Staged for review: none (1 tracked empty file(s) under data/staging" in out
    assert "no review rows are waiting" in out
    assert "must be preserved as tracked staging data" in out
    assert "status has no completion action" in out
    assert "rich extraction evidence" not in out
    assert "delete" not in out
    assert "(0):" not in out


def test_a_pattern_only_rich_extraction_is_review_work_not_a_disposable_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage_pattern_only_extraction(root)

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "Staged for review: no rows; 1 pattern-only extraction file(s)" in out
    assert f"janki --root {root} patterns --review teform_song.pdf" in out
    assert f"janki --root {root} promote {path}" in out
    assert "preserve its extraction evidence" in out
    assert "can be deleted" not in out

    # The two advertised commands are an executable route from review to the
    # zero-record archive, even when status was invoked with --root elsewhere.
    assert cli.main(["--root", str(root), "patterns", "--review", "teform_song.pdf"]) == 0
    capsys.readouterr()
    assert _status(root, "--staged") == 0
    reviewed_out = capsys.readouterr().out
    assert "remain live until promotion or recovery" in reviewed_out
    assert "matching pattern set is already reviewed" in reviewed_out
    assert f"janki --root {root} promote {path}" in reviewed_out
    assert "still need review" not in reviewed_out
    assert cli.main(["--root", str(root), "promote", str(path)]) == 0
    assert not path.exists()
    assert (root / "data" / "staging" / "done" / path.name).is_file()


def test_pattern_only_summary_is_visible_without_the_staged_detail_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage_pattern_only_extraction(root)

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert (
        "Staged for review: no rows; 1 pattern-only extraction file(s) under "
        "data/staging remain live until promotion or recovery"
    ) in out
    assert "patterns --review" not in out


def test_nested_source_cannot_replace_the_top_level_source_promote_requires(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage_pattern_only_extraction(root)
    assert cli.main(["--root", str(root), "patterns", "--review", "teform_song.pdf"]) == 0
    capsys.readouterr()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["source_file"]
    payload["pattern_set"]["source"] = "teform_song.pdf"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "extraction metadata is incomplete or invalid" in out
    assert "source_file" in out
    assert "patterns --review" not in out
    assert " promote " not in out
    assert cli.main(["--root", str(root), "promote", str(path)]) == 1
    assert "needs its source_file and pattern_set" in capsys.readouterr().err


def test_source_file_is_not_stripped_to_a_different_pattern_store_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage_pattern_only_extraction(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_file"] = " teform_song.pdf "
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "matching ' teform_song.pdf ' pattern-store entry is missing" in out
    assert "patterns --review" not in out
    assert " promote " not in out


@pytest.mark.parametrize(
    "remnant",
    [
        {"review_run_id": "11111111-1111-4111-8111-111111111111"},
        {"candidate_accounting": {"version": 1}},
        {"coverage": {}},
        {"prompt_provenance": "damaged"},
    ],
)
def test_damaged_extraction_only_metadata_is_preserved_as_review_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    remnant: dict[str, Any],
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage(root, "damaged-rich.yaml", [])
    path.write_text(
        yaml.safe_dump({**remnant, "records": []}, sort_keys=False), encoding="utf-8"
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "carries rich extraction evidence" in out
    assert "extraction metadata is incomplete or invalid" in out
    assert "tracked empty file" not in out


def test_damaged_rich_extraction_metadata_is_never_called_disposable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage_pattern_only_extraction(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["pattern_set"]
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "extraction metadata is incomplete or invalid" in out
    assert "Restore this exact staging artifact from version control" in out
    assert "can be deleted" not in out


def test_malformed_nested_pattern_answer_is_not_reported_as_promotion_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    path = _stage_pattern_only_extraction(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["pattern_set"]["patterns"] = "not-a-list"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "patterns must be a list" in out
    assert " promote " not in out
    assert "can be deleted" not in out


def test_unresolved_coverage_is_an_owner_prerequisite_not_a_promote_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage_pattern_only_extraction(root, blocking_coverage=True)

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "requires the repository owner's resolution" in out
    assert "Status cannot grant that approval" in out
    assert " promote " not in out
    assert "can be deleted" not in out


def test_unreadable_pattern_store_blocks_commands_but_preserves_staging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage_pattern_only_extraction(root)
    (root / "data" / "patterns.json").write_text("[]\n", encoding="utf-8")

    assert _status(root, "--staged") == 0

    captured = capsys.readouterr()
    assert "could not inspect pattern review state" in captured.err
    assert "matching pattern-store state could not be verified" in captured.out
    assert " promote " not in captured.out
    assert "can be deleted" not in captured.out


@pytest.mark.parametrize("store_state", ["missing", "stale"])
def test_missing_or_stale_pattern_store_gets_recovery_guidance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    store_state: str,
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage_pattern_only_extraction(root)
    store_path = root / "data" / "patterns.json"
    if store_state == "missing":
        store_path.unlink()
    else:
        store = patterns.load_store(store_path)
        store["teform_song.pdf"] = replace(
            store["teform_song.pdf"],
            review_run_id="22222222-2222-4222-8222-222222222222",
            reviewed=True,
        )
        patterns.save_store(store_path, store)

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    expected = "is missing" if store_state == "missing" else "different extraction run"
    assert expected in out
    assert "There is no automatic recovery command" in out
    assert "patterns --review" not in out
    assert "can be deleted" not in out


def test_waiting_rows_do_not_hide_pattern_only_extraction_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage(root, "shirabe-export-needs-reading.yaml", ["word:食べ物:"])
    _stage(root, "ordinary-empty.yaml", [])
    _stage_pattern_only_extraction(root)

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "Staged for review: 1 row(s) in 1 file(s)" in out
    assert "Pattern-only extraction review: 1 file(s)" in out
    assert "word:食べ物: — missing reading" in out
    assert "patterns --review teform_song.pdf" in out
    assert "ordinary-empty.yaml holds no rows" in out
    assert "must be preserved as tracked staging data" in out
    assert "status has no completion action" in out


def test_an_empty_staging_file_does_not_hide_the_rows_still_waiting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")])
    _stage(root, "shirabe-done-needs-reading.yaml", [])
    _stage(root, "shirabe-export-needs-reading.yaml", ["word:食べ物:"])

    assert _status(root, "--staged") == 0

    out = capsys.readouterr().out
    assert "Staged for review: 1 row(s) in 1 file(s) under data/staging" in out
    assert "word:食べ物: — missing reading" in out
    assert "shirabe-done-needs-reading.yaml holds no rows" in out


def test_a_deck_that_cannot_be_read_is_a_warning_not_a_dead_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    (root / "decks" / "broken.yaml").write_text("deck:\nnotes: not-a-list\n", encoding="utf-8")

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "warning: skipping deck" in captured.err
    assert "broken.yaml" in captured.err
    assert "Records: 1" in captured.out


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("examples", "毎日食べる。"),  # a string instead of a list
        ("conjugations", "dict form"),  # a scalar instead of a mapping
        ("source", "shirabe"),  # a scalar instead of a mapping
    ],
)
def test_a_note_with_a_malformed_nested_field_is_skipped_not_a_dead_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field_name: str,
    value: str,
) -> None:
    # These used to escape resolve_deck_records as raw AttributeErrors, so the
    # very command whose docstring says "run it to find out what is wrong"
    # died with a traceback instead of warning and skipping the deck.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],
        {
            "broken": {
                "deck": {"name": "Broken"},
                "notes": [_raw("食べる", "たべる", **{field_name: value})],
            }
        },
    )

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "warning: skipping deck" in captured.err
    assert "broken.yaml" in captured.err
    assert field_name in captured.err
    assert "Records: 1" in captured.out


def test_a_malformed_note_still_leaves_the_id_pipeline_usable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],
        {
            "broken": {
                "deck": {"name": "Broken"},
                "notes": [_raw("食べる", "たべる", examples="全部壊れた。")],
            }
        },
    )

    assert _status(root, "--format", "ids") == 0

    captured = capsys.readouterr()
    assert captured.out == "word:話す:はなす\n"
    assert "warning: skipping deck" in captured.err


def test_build_reports_a_malformed_note_as_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Same root cause, different contract: build may fail, but with janki's
    # one-line error naming the deck and the field, never a traceback.
    root = _project(
        tmp_path,
        [],
        {
            "broken": {
                "deck": {"name": "Broken"},
                "notes": [_raw("食べる", "たべる", examples="毎日食べる。")],
            }
        },
    )

    assert cli.main(["--root", str(root), "build", str(root / "decks" / "broken.yaml")]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "broken.yaml" in err
    assert "examples" in err


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("include_ids", 5),
        ("exclude_ids", "word:話す:はなす"),
        ("include_tags", "n5"),
        ("exclude_tags", 1.5),
        ("exclude_ids", 0),  # falsy, but still not a list
        ("cards", 1.5),
        ("cards", ["recognition"]),
        ("cards", ""),
    ],
)
def test_a_deck_filter_of_the_wrong_shape_is_skipped_not_a_dead_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, value: Any
) -> None:
    # `{str(v) for v in deck_config.get("include_ids") or []}` over a scalar
    # raises TypeError, which is not a JankiError — so the command contracted
    # to warn and skip one broken deck reported nothing at all.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],
        {"broken": {"deck": {"name": "Broken", key: value}, "notes": []}},
    )

    assert _status(root) == 0

    captured = capsys.readouterr()
    assert "warning: skipping deck" in captured.err
    assert "broken.yaml" in captured.err
    assert f"deck.{key}" in captured.err
    assert "Records: 1" in captured.out


@pytest.mark.parametrize("key", ["include_ids", "exclude_tags", "cards"])
def test_build_and_validate_name_the_deck_file_for_a_malformed_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    root = _project(
        tmp_path, [], {"broken": {"deck": {"name": "Broken", key: 5}, "notes": []}}
    )
    deck = str(root / "decks" / "broken.yaml")

    assert cli.main(["--root", str(root), "build", deck]) == 1
    build = capsys.readouterr()
    assert cli.main(["--root", str(root), "validate", deck]) == 1
    validate = capsys.readouterr()

    # The build refuses outright; `validate` reports it against the file and
    # carries on, so that one deck cannot cancel a sweep of every other. Both
    # name the deck, and neither shows a traceback.
    assert "Traceback" not in build.err + validate.err + validate.out
    assert "error:" in build.err and "broken.yaml" in build.err
    assert "broken.yaml" in validate.out + validate.err


def test_real_deck_filters_still_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす", tags=["n5"]), _raw("食べる", "たべる", tags=["n4"])],
        {
            "vocabulary": {
                "deck": {
                    "name": "Vocabulary",
                    "source": "../vocabulary.json",
                    "include_tags": ["n5"],
                    "exclude_ids": [],
                    "cards": {"reading": True},
                },
                "notes": [],
            }
        },
    )

    assert _status(root, "--unexported") == 0

    out = capsys.readouterr().out
    assert "vocabulary 1 of 1" in out


# --- detail flags -----------------------------------------------------------


def test_unexported_lists_the_ids_each_deck_still_owes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす"), _raw("食べる", "たべる")],
        {"vocabulary": _sourced_deck()},
    )
    book = ledger.load(root / "ledger.json")
    book.record_export("word:話す:はなす", "vocabulary", at="2026-08-12")
    book.save()

    assert _status(root, "--unexported") == 0

    out = capsys.readouterr().out
    detail = out.split("Never exported by", 1)[1]
    assert detail.startswith(" vocabulary (1):")
    assert "  word:食べる:たべる" in detail
    assert "word:話す:はなす" not in detail


def test_missing_audio_lists_records_without_word_audio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす"), _raw("食べる", "たべる")])
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        "word:話す:はなす",
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp=word_audio_content_fingerprint(_record("話す", "はなす")),
    )
    book.save()

    assert _status(root, "--missing-audio") == 0

    out = capsys.readouterr().out
    assert "Missing word audio (1):" in out
    assert "  word:食べる:たべる" in out


def test_format_ids_prints_bare_ids_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす"), _raw("食べる", "たべる")],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root, "--format", "ids") == 0

    captured = capsys.readouterr()
    assert captured.out == "word:話す:はなす\nword:食べる:たべる\n"


def test_format_ids_narrows_to_the_flags_and_deduplicates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [_raw("話す", "はなす"), _raw("食べる", "たべる")],
        {"vocabulary": _sourced_deck()},
    )
    book = ledger.load(root / "ledger.json")
    book.record_export("word:食べる:たべる", "vocabulary", at="2026-08-12")
    book.record_audio(
        "word:食べる:たべる",
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp=word_audio_content_fingerprint(_record("食べる", "たべる")),
    )
    book.save()

    # 話す is both unexported and unvoiced; it must appear once.
    assert _status(root, "--unexported", "--missing-audio", "--format", "ids") == 0

    assert capsys.readouterr().out == "word:話す:はなす\n"


# --- duplicates -------------------------------------------------------------


def test_same_expression_under_two_ids_is_a_duplicate() -> None:
    # The historical shape: a record minted before its reading was known.
    groups = status_module.find_duplicates(
        [_record("話す", "はなす"), _record("話す", "")]
    )

    assert [(group.kind, group.key) for group in groups] == [("expression", "話す")]
    assert groups[0].ids == ["word:話す:", "word:話す:はなす"]
    assert groups[0].reason == "same expression, different id"


def test_a_kana_spelling_beside_its_kanji_spelling_is_a_duplicate() -> None:
    # The common real case: a Shirabe bookmark of わかる and a jpdb mining of
    # 分かる are one word under two ids.
    groups = status_module.find_duplicates(
        [_record("わかる", "わかる"), _record("分かる", "わかる")]
    )

    assert [group.kind for group in groups] == ["reading"]
    assert groups[0].key == "わかる"
    assert "one is the kana form" in groups[0].reason
    assert sorted(groups[0].ids) == ["word:わかる:わかる", "word:分かる:わかる"]


def test_two_spellings_sharing_a_jpdb_vid_are_a_duplicate() -> None:
    vid = {"source": {"type": "jpdb", "raw_fields": {"vid": "1577980"}}}
    groups = status_module.find_duplicates(
        [_record("引っ越す", "ひっこす", **vid), _record("引越す", "ひっこす", **vid)]
    )

    assert [group.kind for group in groups] == ["reading"]
    assert "same jpdb vid 1577980" in groups[0].reason


def test_a_shared_vid_with_diverging_readings_is_a_duplicate() -> None:
    # DESIGN_V2 rule 2 forbids auto-correcting readings, so hand-corrected
    # readings happen — and a shared vid is definitionally the same word.
    vid = {"source": {"type": "jpdb", "raw_fields": {"vid": "42"}}}
    groups = status_module.find_duplicates(
        [_record("引っ越す", "ひっこす", **vid), _record("引越す", "ひきこす", **vid)]
    )

    assert [(group.kind, group.key) for group in groups] == [("vid", "42")]
    assert "same jpdb vid 42" in groups[0].reason
    assert groups[0].ids == sorted(["word:引っ越す:ひっこす", "word:引越す:ひきこす"])


def test_a_shared_vid_with_an_empty_reading_is_a_duplicate() -> None:
    # Empty readings never reach the reading pass at all — this is the legacy
    # malformed-id class the vid pass exists to catch.
    vid = {"source": {"type": "jpdb", "raw_fields": {"vid": "42"}}}
    groups = status_module.find_duplicates(
        [_record("分かる", "わかる", **vid), _record("わかる", "", **vid)]
    )

    assert [group.kind for group in groups] == ["vid"]
    assert groups[0].ids == sorted(["word:分かる:わかる", "word:わかる:"])


def test_vids_compare_numerically_across_typing_accidents() -> None:
    # A round-tripped file can hold 1577980.0 where the import wrote 1577980.
    first = _record(
        "引っ越す", "ひっこす", source={"type": "jpdb", "raw_fields": {"vid": "1577980.0"}}
    )
    second = _record(
        "引越す", "ひっこす", source={"type": "jpdb", "raw_fields": {"vid": "1577980"}}
    )

    groups = status_module.find_duplicates([first, second])

    assert len(groups) == 1
    assert "same jpdb vid 1577980" in groups[0].reason


def test_a_pair_already_grouped_is_not_reported_again_by_the_vid_pass() -> None:
    # One duplicate, one group: the vid pass must not make the same pair look
    # like two findings.
    vid = {"source": {"type": "jpdb", "raw_fields": {"vid": "9"}}}
    groups = status_module.find_duplicates(
        [_record("分かる", "わかる", **vid), _record("分かる", "わかる", id="word:分かる:", **vid)]
    )

    assert [group.kind for group in groups] == ["expression"]


def test_a_kana_word_under_two_ids_is_one_expression_group() -> None:
    # The same-expression skip in the reading pass: without it every kana-word
    # duplicate is reported twice, once per pass.
    groups = status_module.find_duplicates(
        [_record("わかる", "わかる"), _record("わかる", "わかる", id="word:わかる:")]
    )

    assert [group.kind for group in groups] == ["expression"]
    assert groups[0].ids == sorted(["word:わかる:わかる", "word:わかる:"])


def test_homophones_are_not_duplicates() -> None:
    # 箸 and 橋 share a reading and nothing else: neither is the other's kana
    # form and there is no vid to tie them together.
    assert status_module.find_duplicates([_record("箸", "はし"), _record("橋", "はし")]) == []


def test_different_jpdb_vids_do_not_make_a_duplicate() -> None:
    first = _record("箸", "はし", source={"type": "jpdb", "raw_fields": {"vid": "1"}})
    second = _record("橋", "はし", source={"type": "jpdb", "raw_fields": {"vid": "2"}})

    assert status_module.find_duplicates([first, second]) == []


def test_a_null_vid_column_does_not_collapse_the_file_into_one_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `str(None)` used to be a grouping key, so every record with a null vid
    # landed in one group whose printed remedy is "delete the other" — said
    # over three live, unrelated records.
    root = _project(
        tmp_path,
        [
            _raw("行く", "いく", source={"type": "jpdb", "raw_fields": {"vid": None}}),
            _raw("話す", "はなす", source={"type": "jpdb", "raw_fields": {"vid": None}}),
            _raw("食べる", "たべる", source={"type": "jpdb", "raw_fields": {"vid": None}}),
        ],
    )

    assert _status(root, "--duplicates") == 0

    out = capsys.readouterr().out
    assert "Duplicate candidates: none." in out
    assert "delete the other" not in out


@pytest.mark.parametrize("placeholder", ["None", "-", "n/a", "unknown", "0x10", "-5"])
def test_a_placeholder_in_the_vid_column_is_not_a_vid(placeholder: str) -> None:
    # An import column of placeholders is shared by every record that has no
    # vid at all; grouping on one names unrelated records as copies of each
    # other, under a report whose remedy is deletion.
    vid = {"source": {"type": "jpdb", "raw_fields": {"vid": placeholder}}}
    records = [
        _record("行く", "いく", **vid),
        _record("話す", "はなす", **vid),
        _record("食べる", "たべる", **vid),
    ]

    assert status_module.find_duplicates(records) == []


def test_a_real_vid_still_groups_when_placeholders_are_around() -> None:
    real = {"source": {"type": "jpdb", "raw_fields": {"vid": "1577980"}}}
    none = {"source": {"type": "jpdb", "raw_fields": {"vid": "None"}}}
    groups = status_module.find_duplicates(
        [
            _record("引っ越す", "ひっこす", **real),
            _record("引越す", "ひきこす", **real),
            _record("行く", "いく", **none),
            _record("話す", "はなす", **none),
        ]
    )

    assert [(group.kind, group.key) for group in groups] == [("vid", "1577980")]


def test_duplicates_output_says_resolution_is_manual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("わかる", "わかる"), _raw("分かる", "わかる")])

    assert _status(root, "--duplicates") == 0

    out = capsys.readouterr().out
    assert "Duplicate candidates (1 group(s)):" in out
    assert "  わかる — same reading, different expression: one is the kana form" in out
    assert "    word:分かる:わかる" in out
    assert "Resolution is manual" in out
    assert "never re-IDed" in out


def test_duplicate_ids_can_be_piped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("わかる", "わかる"), _raw("分かる", "わかる")])

    assert _status(root, "--duplicates", "--format", "ids") == 0

    assert capsys.readouterr().out == "word:わかる:わかる\nword:分かる:わかる\n"


def test_duplicate_detection_sees_inline_deck_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Until M1.7 migrates them, a deck note is as real as a normalized record —
    # including when it duplicates one.
    root = _project(
        tmp_path,
        [_raw("分かる", "わかる")],
        {"verbs": {"deck": {"name": "Verbs"}, "notes": [_raw("わかる", "わかる")]}},
    )

    assert _status(root, "--duplicates") == 0

    assert "one is the kana form" in capsys.readouterr().out


# --- --rebuild --------------------------------------------------------------


def _media_file(root: Path, fingerprint: str) -> Path:
    path = root / "media" / "audio" / f"{status_module.MEDIA_PREFIX}{fingerprint}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def test_rebuild_recovers_sources_and_audio_and_admits_what_it_cannot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record(
        "話す", "はなす", examples=[{"japanese": "毎日話す。", "english": "I speak every day."}]
    )
    root = _project(tmp_path, [record.to_dict()])
    _media_file(root, word_audio_filename_fingerprint(record))
    _media_file(root, example_audio_filename_fingerprint(record, record.examples[0]))
    _media_file(root, "0123456789ab")  # belongs to no record

    assert _status(root, "--rebuild") == 0

    out = capsys.readouterr().out
    assert "Export state is NOT reconstructible" in out
    assert "--only-new" in out
    assert "1 janki-* file(s) matched no record" in out
    assert "dates recorded are today's" in out

    entry = ledger.load(root / "ledger.json").records[record.id]
    assert entry["sources"] == [
        {"type": "shirabe", "ref": "export.csv", "seen_at": TODAY}
    ]
    audio = sorted(entry["audio"], key=lambda item: item["of"])
    assert [item["of"] for item in audio] == ["example", "word"]
    assert all(item["provider"] == "unknown" and item["rebuilt"] is True for item in audio)
    # Negative sentinels, and load-bearing: a rebuilt entry knows neither the
    # voice nor the rate, and `_is_current` compares both. A plausible 1.0 here
    # would let `janki audio` call a clip current whose rate nothing knows.
    assert all(item["voice"] == status_module.REBUILT_VOICE == -1 for item in audio)
    assert all(item["speed"] == status_module.REBUILT_SPEED == -1.0 for item in audio)
    # Both filenames pin their content here: the word file's address embeds the
    # record id (and so the reading), the example file's embeds the sentence.
    assert audio[1]["content_fp"] == word_audio_content_fingerprint(record)
    assert audio[0]["content_fp"] == example_audio_content_fingerprint(record.examples[0])
    assert "Stale audio: 1" in out, "unknown render metadata requires regeneration"


def test_rebuild_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    _media_file(root, word_audio_filename_fingerprint(record))

    assert _status(root, "--rebuild") == 0
    first = (root / "ledger.json").read_text(encoding="utf-8")
    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    assert (root / "ledger.json").read_text(encoding="utf-8") == first


def test_rebuild_leaves_a_real_audio_entry_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    filename = _media_file(root, word_audio_filename_fingerprint(record)).name
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file=filename,
        of="word",
        provider="voicevox",
        voice=46,
        speed=0.85,
        content_fp=word_audio_content_fingerprint(record),
        at="2026-08-11",
    )
    book.save()

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entries = ledger.load(root / "ledger.json").records[record.id]["audio"]
    # A rebuilt entry knows neither the engine, the voice, nor the rate; it must
    # never replace an entry that does.
    assert entries == [
        {
            "file": filename,
            "of": "word",
            "provider": "voicevox",
            "voice": 46,
            "speed": 0.85,
            "content_fp": word_audio_content_fingerprint(record),
            "at": "2026-08-11",
        }
    ]


def test_rebuilt_word_audio_claims_no_content_it_cannot_prove(tmp_path: Path) -> None:
    # A record carrying an accent pattern: the filename fingerprint covers the
    # record id, so it proves the reading but not the accent the audio on disk
    # was generated with.
    record = _record("話す", "はなす")
    accented = _record("話す", "はなす", pitch_accent=["LHH"])
    media = tmp_path / "media"
    media.mkdir()
    fingerprint = word_audio_filename_fingerprint(record)
    (media / f"{status_module.MEDIA_PREFIX}{fingerprint}.wav").write_bytes(b"RIFF")

    book = ledger.load(tmp_path / "ledger.json")
    summary = status_module.rebuild(book, [accented], media)

    assert summary.unprovable_audio == 1
    assert book.records[record.id]["audio"][0]["content_fp"] == ""


def test_rebuild_accounts_for_every_twin_that_claims_one_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A provider switch can leave janki-<fp>.mp3 beside janki-<fp>.wav. The
    # rebuilt entry binds the documented preference (.wav — what janki
    # generates); the loser must show up in the accounting, not vanish.
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    wav = _media_file(root, word_audio_filename_fingerprint(record))
    wav.with_suffix(".mp3").write_bytes(b"ID3")  # sorts before .wav by name
    _media_file(root, "0123456789ab")  # belongs to no record

    assert _status(root, "--rebuild") == 0

    out = capsys.readouterr().out
    assert "1 janki-* file(s) share a fingerprint" in out
    assert "1 janki-* file(s) matched no record" in out

    audio = ledger.load(root / "ledger.json").records[record.id]["audio"]
    assert [item["file"] for item in audio] == [wav.name]


def test_rebuild_counts_every_unclaimed_twin_of_an_unmatched_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The other half of the list `_media_by_fingerprint` returns: two files
    # claim one fingerprint and no record wants either. Reporting "1 file"
    # while two sit on disk is the disappearance the list exists to prevent.
    root = _project(tmp_path, [])
    orphan = _media_file(root, "0123456789ab")
    orphan.with_suffix(".mp3").write_bytes(b"ID3")

    assert _status(root, "--rebuild") == 0

    assert "2 janki-* file(s) matched no record" in capsys.readouterr().out


def test_rebuild_sees_twins_that_share_a_basename_in_different_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `media` is built with rglob, so one basename can occur twice. Accounting
    # keyed by name called the loser claimed — counted as neither ambiguous nor
    # unmatched, and so absent from the summary entirely.
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    first = _media_file(root, word_audio_filename_fingerprint(record))
    second = root / "media" / "elsewhere" / first.name
    second.parent.mkdir(parents=True)
    second.write_bytes(b"RIFF")

    assert _status(root, "--rebuild") == 0

    out = capsys.readouterr().out
    assert "1 janki-* file(s) share a fingerprint" in out
    assert "matched no record" not in out


def test_rebuild_keeps_a_source_a_deck_note_states_deliberately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The normalized fallback is for the note that said *nothing* about its
    # source and so carries SourceReference() by construction. A note that
    # re-sources a record said something, into a file committed to git.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],  # source: shirabe, imported_from export.csv
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [
                    {
                        "id": "word:話す:はなす",
                        "expression": "話す",
                        "reading": "はなす",
                        "meanings": ["to speak"],
                        "source": {"type": "jpdb", "imported_from": "jpdb-deck-42"},
                    }
                ],
            }
        },
    )

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entry = ledger.load(root / "ledger.json").records["word:話す:はなす"]
    assert entry["sources"] == [{"type": "jpdb", "ref": "jpdb-deck-42", "seen_at": TODAY}]


def test_rebuild_records_the_normalized_records_own_provenance_for_shadowed_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The deck-resolved copy wins for content, but its default-manual source is
    # not the record's provenance — and the ledger is committed to git.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],  # source: shirabe, imported_from export.csv
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [
                    {
                        "id": "word:話す:はなす",
                        "expression": "話す",
                        "reading": "はなす",
                        "meanings": ["to speak"],
                        "usage_notes": "Inline copy.",
                    }
                ],
            }
        },
    )

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entry = ledger.load(root / "ledger.json").records["word:話す:はなす"]
    assert entry["sources"] == [{"type": "shirabe", "ref": "export.csv", "seen_at": TODAY}]


def test_status_without_rebuild_never_writes_the_ledger(tmp_path: Path) -> None:
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    _media_file(root, word_audio_filename_fingerprint(record))
    book = ledger.load(root / "ledger.json")
    book.record_added(record.id, at="2026-08-06")
    book.save()
    before = (root / "ledger.json").read_text(encoding="utf-8")

    for flags in ([], ["--unexported"], ["--missing-audio"], ["--duplicates"]):
        assert _status(root, *flags) == 0

    assert (root / "ledger.json").read_text(encoding="utf-8") == before


def test_rebuild_prose_stays_off_a_piped_id_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])

    assert _status(root, "--rebuild", "--format", "ids") == 0

    captured = capsys.readouterr()
    assert captured.out == "word:話す:はなす\n"
    assert "Export state is NOT reconstructible" in captured.err


# --- collection ------------------------------------------------------------


def test_an_inline_note_overrides_the_normalized_record_it_names(tmp_path: Path) -> None:
    # This is the record the deck actually exports today, and the one M1.7 will
    # write back into vocabulary.json.
    root = _project(
        tmp_path,
        [_raw("話す", "はなす")],
        {
            "verbs": {
                "deck": {"name": "Verbs", "source": "../vocabulary.json"},
                "notes": [{"id": "word:話す:はなす", "usage_notes": "Everyday verb."}],
            }
        },
    )
    config = ProjectConfig.load(root)

    universe = status_module.collect_records(config)

    assert [record.id for record in universe.records] == ["word:話す:はなす"]
    assert universe.records[0].usage_notes == "Everyday verb."
    assert universe.normalized_count == 1
    assert universe.inline_count == 0
    assert [deck.stem for deck in universe.decks] == ["verbs"]


def test_a_record_whose_example_was_edited_after_its_audio_reads_as_stale(
    tmp_path: Path,
) -> None:
    record = _record(
        "話す",
        "はなす",
        examples=[{"japanese": "毎日話す。", "audio": "audio/janki-example.wav"}],
    )
    root = _project(tmp_path, [record.to_dict()])
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-example.wav",
        of="example",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp=example_audio_content_fingerprint(ExampleSentence(japanese="毎日話した。")),
    )
    book.save()
    config = ProjectConfig.load(root)

    words = cli._speech_provider(config, None)
    report = status_module.build_report(
        config,
        status_module.collect_records(config),
        ledger.load(root / "ledger.json"),
        word_provider=words,
        example_provider=cli._sentence_provider(config, None, words),
    )

    assert report.stale_audio == [record.id]


def test_rebuild_binds_the_file_the_record_actually_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Switching a sentence to OpenAI leaves `janki-<fp>.wav` beside the new
    `janki-<fp>.mp3` until a prune. Ranking by extension bound the stale WAV —
    then reported the mp3 the record actually plays as the ambiguous one,
    telling the user to delete the file that is correct."""
    record = _record("話す", "はなす")
    fingerprint = word_audio_filename_fingerprint(record)
    # The record plays the mp3 — the fact the rebuild has to defer to.
    root = _project(
        tmp_path, [record.to_dict() | {"audio": f"audio/janki-{fingerprint}.mp3"}]
    )
    media = root / "media" / "audio"
    media.mkdir(parents=True, exist_ok=True)
    (media / f"janki-{fingerprint}.wav").write_bytes(b"RIFF stale")
    (media / f"janki-{fingerprint}.mp3").write_bytes(b"ID3 current")

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entries = ledger.load(root / "ledger.json").records[record.id]["audio"]
    assert [e["file"] for e in entries] == [f"janki-{fingerprint}.mp3"]


def test_rebuild_falls_back_to_extension_when_the_record_names_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The old rule, kept for the case it was written for: no record preference
    to defer to, so the ranking decides and stays reproducible."""
    record = _record("話す", "はなす")
    root = _project(tmp_path, [record.to_dict()])
    fingerprint = word_audio_filename_fingerprint(record)
    media = root / "media" / "audio"
    media.mkdir(parents=True, exist_ok=True)
    (media / f"janki-{fingerprint}.wav").write_bytes(b"RIFF")
    (media / f"janki-{fingerprint}.mp3").write_bytes(b"ID3")

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entries = ledger.load(root / "ledger.json").records[record.id]["audio"]
    assert [e["file"] for e in entries] == [f"janki-{fingerprint}.wav"]


def test_a_silent_example_sentence_is_counted_on_its_own_line(tmp_path: Path) -> None:
    """188 of 343 sentences shipped mute and every indicator read green.

    `missing_audio` counts word audio only, and defends that in its docstring:
    a record with one silent example among nine is not deficient the way a
    record with no word clip is. The conclusion was right and the consequence
    was that nothing counted example audio at all. It gets its own line, which
    keeps the word-level signal unburied and still answers the question.
    """
    from japanese_anki import ledger
    from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

    voiced = ExampleSentence(japanese="毎日話します。", audio="a.mp3")
    silent = ExampleSentence(japanese="毎日話すよ。")
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="x.csv"),
        examples=[voiced, silent],
    )

    book = ledger.load(tmp_path / "ledger.json")
    book.record_audio(
        record.id, file="a.mp3", of="example", provider="openai", voice="onyx",
        speed=1.0, content_fp=example_audio_content_fingerprint(voiced),
    )

    assert book.unvoiced_examples([record]) == [("word:話す:はなす", 1)]
    assert book.missing_audio([record]) == ["word:話す:はなす"], (
        "and the word-level count is untouched by either example"
    )


def test_status_prints_the_example_audio_count_on_its_own_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The line itself, from the command, not the method behind it.

    The first version of this test asserted `Ledger.unvoiced_examples` and
    stopped there. Deleting the whole `lines.append(...)` from `format_report`
    left the suite green — the count was computed, carried on the report, and
    printed nowhere, which is indistinguishable from the bug the commit was
    written to fix.

    Its own line, and not folded into the word-level count above it: a record
    with one silent example among nine is not deficient the way a record with
    no word clip is, which is what `missing_audio`'s docstring argues and why
    that count must not move.
    """
    root = _project(tmp_path, [
        _raw("話す", "はなす", examples=[
            {"japanese": "毎日話します。", "audio": "audio/janki-voiced.mp3"},
            {"japanese": "毎日話すよ。"},
        ]),
    ])
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        "word:話す:はなす", file="janki-voiced.mp3", of="example", provider="openai",
        voice="onyx", speed=1.0,
        content_fp=example_audio_content_fingerprint(
            ExampleSentence(japanese="毎日話します。")
        ),
    )
    book.save()

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert "Missing example audio: 1 of 2 sentence(s)" in out
    assert "Missing word audio: 1 of 1" in out, "the word count is its own line"


def test_a_sentence_naming_a_clip_the_ledger_never_wrote_is_not_voiced(
    tmp_path: Path,
) -> None:
    """The count answers to the ledger, like the word-level one above it.

    A first version read `example.audio`'s truthiness alone, so a record
    pointing at a file nothing ever produced reported as voiced — green while
    mute, which is the exact failure this count was added to end, one level
    along. `missing_audio` has always consulted the ledger; this now does too.
    """
    from japanese_anki import ledger as ledger_module
    from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="x.csv"),
        examples=[ExampleSentence(japanese="毎日話します。", audio="audio/ghost.mp3")],
    )
    book = ledger_module.load(tmp_path / "ledger.json")

    assert book.unvoiced_examples([record]) == [("word:話す:はなす", 0)]

    book.record_audio(
        record.id, file="ghost.mp3", of="example", provider="openai",
        voice="onyx", speed=1.0,
        content_fp=example_audio_content_fingerprint(record.examples[0]),
    )

    assert book.unvoiced_examples([record]) == [], "and voiced once the ledger says so"


# --- unsettled model claims ---------------------------------------------------
#
# A mark nobody can see does not do its job. Extraction marks the fields a model
# wrote so that a later pass — or a person — knows they are guesses, but until
# `--unsettled` the mark was visible only to the code acting on it. A wrong
# guess therefore looked exactly like curated content: that is how a medical
# sheet teaching おたふく = "mumps" shipped a card reading "homely woman".


def _guessed(expression: str, reading: str, **overrides: Any) -> dict[str, Any]:
    """A record whose meanings (and part of speech, if given) are model claims."""
    raw = _raw(expression, reading, **overrides)
    raw["source"] = {"type": "extract", "imported_from": "sheet.pdf"}
    return mark_provisional(VocabularyRecord.from_dict(raw)).to_dict()


def test_the_summary_counts_unsettled_claims_by_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [
            _guessed("おたふく", "おたふく", meanings=["mumps"]),
            _guessed("話す", "はなす", part_of_speech="noun"),
            _raw("食べる", "たべる"),
        ],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root) == 0

    out = capsys.readouterr().out
    # Per field, because the remedy differs: a meaning is settled by reading
    # the card, a part of speech by a dictionary that resolved the exact word.
    assert "Unsettled model claims: meanings 2, part_of_speech 1" in out


def test_a_collection_with_nothing_unsettled_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silence would be indistinguishable from the line not existing."""
    root = _project(
        tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()}
    )

    assert _status(root) == 0

    assert "Unsettled model claims: none" in capsys.readouterr().out


def test_a_field_edited_after_extraction_is_not_a_model_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mark is bound to the value it was written for. Editing the field
    breaks that binding, which is what makes the content curated — reporting it
    as a guess would be the exact opposite of the truth, and would send a
    person to re-check the one record they had already fixed."""
    marked = _guessed("おたふく", "おたふく", meanings=["mumps"])
    marked["meanings"] = ["mumps (hand-checked)"]
    root = _project(tmp_path, [marked], {"vocabulary": _sourced_deck()})

    assert _status(root) == 0

    assert "Unsettled model claims: none" in capsys.readouterr().out


def test_the_detail_flag_groups_the_records_by_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        [
            _guessed("おたふく", "おたふく", meanings=["mumps"]),
            _guessed("話す", "はなす", part_of_speech="noun"),
        ],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root, "--unsettled") == 0

    out = capsys.readouterr().out
    assert "Unsettled meanings (2):" in out
    assert "Unsettled part_of_speech (1):" in out
    assert "  word:おたふく:おたふく" in out


def test_unsettled_ids_pipe_into_the_pass_that_settles_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the surface. Without this the answer to "which records are
    guesses" is a number a person cannot act on; with it,
    `janki status --unsettled --format ids` is the argument list for the pass
    that settles them.

    A record marked in two fields appears once — `selected_ids` deduplicates
    every detail flag alike, so this observes that rather than proving
    `provisional_ids` does it. The property's own guarantee is pinned below.
    """
    root = _project(
        tmp_path,
        [
            _guessed("おたふく", "おたふく", meanings=["mumps"], part_of_speech="noun"),
            _raw("食べる", "たべる"),
        ],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root, "--unsettled", "--format", "ids") == 0

    assert capsys.readouterr().out == "word:おたふく:おたふく\n"


def test_provisional_ids_names_each_record_once(tmp_path: Path) -> None:
    """`selected_ids` deduplicates whatever it is handed, so a duplicate here
    is invisible from the CLI — and would surface only in whichever caller
    reads the report object directly and trusted the property to mean
    "the records to re-check"."""
    root = _project(
        tmp_path,
        [_guessed("おたふく", "おたふく", meanings=["mumps"], part_of_speech="noun")],
        {"vocabulary": _sourced_deck()},
    )
    config = ProjectConfig.load(root)
    report = status_module.build_report(
        config,
        status_module.collect_records(config),
        ledger.load(root / "ledger.json"),
        word_provider=None,
        example_provider=None,
    )

    assert sorted(report.provisional) == ["meanings", "part_of_speech"]
    assert report.provisional_ids == ["word:おたふく:おたふく"]


# --- paid calls still waiting on a person -----------------------------------
#
# `janki extract` tells somebody to run this command after a call that may have
# been billed — "Nothing was staged. Run 'janki status' to see it" — and until
# these lines `status` did not read the operations journal at all. The
# instruction was an unhonoured promise, which is the worst kind to make about
# money.


def _operations_path(root: Path) -> Path:
    return ProjectConfig.load(root).operations_file


def _stranded(root: Path, **overrides: Any) -> str:
    """One dispatched call whose answer never came back."""
    journal = operations.OperationJournal.load(_operations_path(root))
    operation_id = "0192f232-0000-7000-8000-00000000abcd"
    journal.authorize(
        operation_id,
        kind=overrides.get("kind", "extract"),
        source_file=overrides.get("source_file", "lesson.pdf"),
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )
    journal.advance(operation_id, "dispatching")
    journal.advance(operation_id, "outcome_unknown", detail="connection reset")
    return operation_id


def test_the_summary_counts_paid_calls_that_need_a_decision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    _stranded(root)

    assert _status(root) == 0

    assert "Paid calls not accounted for: 1" in capsys.readouterr().out


def test_the_summary_counts_a_call_killed_while_it_was_in_flight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Money that may already be gone. Counting only the calls that need a
    *decision* let this print nothing at all — and `--operations` then said
    every call janki made either landed or is recorded as finished."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-killed", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-killed", "dispatching")

    assert _status(root) == 0
    assert "Paid calls not accounted for: 1" in capsys.readouterr().out

    assert _status(root, "--operations") == 0
    listed = capsys.readouterr().out
    assert "op-killed" in listed
    assert "state: dispatching" in listed
    # The exact sentence this used to print over a call that may have been
    # billed. "none" alone would match "Unsettled model claims: none".
    assert "not accounted for: none" not in listed


def test_a_project_with_no_stranded_calls_says_nothing_about_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The line appears only when there is something to say. A permanent
    "Paid calls needing a person: 0" trains people to skip the line that
    matters."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})

    assert _status(root) == 0

    assert "Paid calls needing a person" not in capsys.readouterr().out


def test_the_detail_says_which_call_and_whether_a_retry_costs_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one thing a person needs from this screen: re-running a call that
    was sent and never answered may pay for it twice."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    operation_id = _stranded(root)

    assert _status(root, "--operations") == 0

    out = capsys.readouterr().out
    assert operation_id in out
    assert "lesson.pdf" in out
    assert "outcome_unknown" in out
    assert "risks a second charge" in out
    assert "connection reset" in out


def test_a_saved_reply_is_named_so_it_can_be_looked_at(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An answer that arrived and was refused leaves bytes on disk. Those have
    been paid for, so the file is the point of the entry."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    operation_id = "0192f232-0000-7000-8000-0000000000ff"
    journal.authorize(
        operation_id, kind="extract", source_file="chart.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance(operation_id, "dispatching")
    artifact = operations.capture_artifact(
        _operations_path(root), operation_id, b'{"content": []}'
    )
    journal.advance(operation_id, "result_captured", artifact=artifact)

    assert _status(root, "--operations") == 0

    out = capsys.readouterr().out
    assert artifact in out
    # Not the retry warning: this one was answered, and the answer is on disk.
    assert "risks a second charge" not in out


def test_the_pipe_can_be_narrowed_to_one_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What settles each field differs, so one list piped into one pass is
    wrong for half of it.

    `enrich --ai` writes only `AI_FIELDS`, which excludes `part_of_speech` — a
    pos-marked record piped there buys a call that cannot clear the mark. And
    piping it with `--force-fields meanings` puts a *curated* meaning up for
    replacement on a record that was only ever marked for its part of speech.
    """
    root = _project(
        tmp_path,
        [
            _guessed("おたふく", "おたふく", meanings=["mumps"]),
            # Marked for its part of speech alone: `mark_provisional` marks
            # every non-empty semantic field, so a record with meanings would
            # legitimately be in both lists and could not show the narrowing.
            _guessed("話す", "はなす", meanings=[], part_of_speech="noun"),
        ],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root, "--unsettled", "meanings", "--format", "ids") == 0
    meanings = capsys.readouterr().out.split()

    assert _status(root, "--unsettled", "part_of_speech", "--format", "ids") == 0
    parts = capsys.readouterr().out.split()

    assert "word:おたふく:おたふく" in meanings
    assert "word:話す:はなす" in parts
    assert set(meanings) & set(parts) == set()

    # And unnarrowed is still the union, for the person who wants to see them all.
    assert _status(root, "--unsettled", "--format", "ids") == 0
    assert set(capsys.readouterr().out.split()) == set(meanings) | set(parts)


def test_the_detail_names_what_settles_each_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A count somebody cannot act on is barely better than silence, and the
    action is not the same for both fields."""
    root = _project(
        tmp_path,
        [
            _guessed("おたふく", "おたふく", meanings=["mumps"]),
            _guessed("話す", "はなす", part_of_speech="noun"),
        ],
        {"vocabulary": _sourced_deck()},
    )

    assert _status(root, "--unsettled") == 0

    out = capsys.readouterr().out
    assert "enrich --ai --force-fields meanings" in out
    assert "enrich --jpdb" in out


def test_operations_have_no_ids_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operation id is not a record id, and the two pipe into entirely
    different things. The shared fallback printed the whole collection's record
    ids for an operations query — an answer that looks valid and is wrong in a
    way nothing downstream would catch."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    _stranded(root)

    assert _status(root, "--operations", "--format", "ids") == 1

    captured = capsys.readouterr()
    assert "word:話す:はなす" not in captured.out
    assert "not a record id" in captured.err


def test_the_detail_says_so_when_nothing_is_unsettled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--unsettled` on a clean collection has to answer. Printing nothing is
    indistinguishable from the flag not working, and this branch had never
    been executed by a test."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})

    assert _status(root, "--unsettled") == 0

    # The detail formatter's own sentence, not the summary line's — those
    # share an opening, and asserting the shared half passes when the detail
    # returns nothing at all.
    assert (
        "every extracted field has been confirmed, edited, or settled"
        in capsys.readouterr().out
    )


def test_the_fields_are_grouped_in_a_stable_order(tmp_path: Path) -> None:
    """Grouped by sorted field name, not by whichever record happened to come
    first. Left in encounter order the same collection prints its groups in a
    different order after an unrelated edit, and every existing test happened
    to have an encounter order that was already sorted."""
    root = _project(
        tmp_path,
        [
            # `collect_records` sorts by id, so file order proves nothing. To
            # tell sorted-by-field from encounter order, the record that sorts
            # *first* (おたふく < 話す) has to carry the field that sorts last.
            _guessed("おたふく", "おたふく", meanings=[], part_of_speech="noun"),
            _guessed("話す", "はなす", meanings=["to speak"]),
        ],
        {"vocabulary": _sourced_deck()},
    )
    config = ProjectConfig.load(root)
    report = status_module.build_report(
        config,
        status_module.collect_records(config),
        ledger.load(root / "ledger.json"),
        word_provider=None,
        example_provider=None,
    )

    assert list(report.provisional) == ["meanings", "part_of_speech"]


# --- ending a call that will never finish -----------------------------------


def _operations(root: Path, *flags: str) -> int:
    return cli.main(["--root", str(root), "operations", *flags])


def test_a_stuck_call_is_listed_with_the_way_out_of_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A killed process leaves a call marked in flight, and janki will not
    spend again until it is settled. Listing only calls that *need a decision*
    answered "why is this blocked?" with "nothing is blocked"."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-stuck", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-stuck", "dispatching")

    assert _operations(root) == 0

    out = capsys.readouterr().out
    assert "op-stuck" in out
    assert "state: dispatching" in out
    assert "janki operations --end op-stuck" in out


def test_ending_a_call_then_forgetting_it_unblocks_the_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two steps on purpose. Ending says the process is gone; forgetting says
    the money question has been dealt with. Collapsing them would drop the
    record of possible spending in the same breath as noticing it."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-stuck", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-stuck", "dispatching")

    assert _operations(root, "--end", "op-stuck") == 0
    ended = capsys.readouterr().out
    assert "outcome_unknown" in ended
    assert "cannot know whether that call was billed" in ended
    # Ended is not gone: it still counts, and the message says so.
    assert "still counts as needing a person" in ended

    assert _operations(root, "--forget", "op-stuck") == 0
    assert "Forgot op-stuck" in capsys.readouterr().out
    assert not operations.OperationJournal.load(
        _operations_path(root)
    ).operations


def test_forgetting_a_call_still_in_flight_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropping the entry would delete the only record that money may be
    moving right now."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-live", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-live", "dispatching")

    assert _operations(root, "--forget", "op-live") == 1

    assert "not finished" in capsys.readouterr().err
    assert "op-live" in operations.OperationJournal.load(
        _operations_path(root)
    ).operations


def test_forgetting_a_call_holding_an_unread_reply_needs_saying_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """That reply was paid for and never became staging, and this deletes
    it."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    path = _operations_path(root)
    journal = operations.OperationJournal.load(path)
    journal.authorize(
        "op-paid", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-paid", "dispatching")
    artifact = operations.capture_artifact(path, "op-paid", b'{"content": []}')
    journal.advance("op-paid", "result_captured", artifact=artifact)

    assert _operations(root, "--forget", "op-paid") == 1
    assert "paid for and never became staging" in capsys.readouterr().err
    assert (path.parent / artifact).exists()

    assert _operations(root, "--forget", "op-paid", "--force") == 0
    assert not (path.parent / artifact).exists()


def test_a_reason_given_for_ending_a_call_is_kept_with_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Why a call was ended by hand is the one thing the journal cannot work
    out for itself, and it is what a person reading this months later has to
    go on."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-stuck", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-stuck", "dispatching")

    assert _operations(
        root, "--end", "op-stuck", "--reason", "laptop lid closed mid-call"
    ) == 0
    capsys.readouterr()

    assert _operations(root) == 0
    assert "laptop lid closed mid-call" in capsys.readouterr().out


def test_an_orphaned_authority_is_listed_and_can_be_cleared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A process killed between writing the authority and marking it sent
    leaves an entry that blocks everything. It has to be visible, and it has
    to be described as what it is — nothing was sent."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    journal = operations.OperationJournal.load(_operations_path(root))
    journal.authorize(
        "op-orphan", kind="extract", source_file="lesson.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )

    assert _operations(root) == 0
    listed = capsys.readouterr().out
    assert "op-orphan" in listed
    assert "state: authorized" in listed
    # Never "risks a second charge": nothing left the computer.
    assert "second charge" not in listed

    assert _operations(root, "--end", "op-orphan") == 0
    assert "canceled_before_send" in capsys.readouterr().out


def test_the_oldest_blocking_call_is_the_one_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two entries can pile up — a call ends as an unknown outcome, and an
    orphaned authority survives beside it. The refusal names one, and it must
    be the one that has been waiting longest rather than whichever the
    dictionary happened to yield."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    path = _operations_path(root)
    journal = operations.OperationJournal.load(path)
    journal.authorize(
        "op-older", kind="extract", source_file="first.pdf",
        source_sha256="a" * 64, request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-older", "dispatching")
    journal.advance("op-older", "outcome_unknown", detail="lost")
    # Authorized *after*, and the journal is a dict — insertion order alone
    # would name whichever was written last.
    reloaded = operations.OperationJournal.load(path)
    reloaded.operations["op-newer"] = operations.Operation(
        operation_id="op-newer", kind="extract", state="authorized",
        source_file="second.pdf", source_sha256="c" * 64, request_fp="d" * 64,
        model="claude-opus-5", authorized_at="2099-01-01T00:00:00+00:00",
        updated_at="2099-01-01T00:00:00+00:00",
    )
    reloaded._write()

    assert _operations(root) == 0
    listed = capsys.readouterr().out
    assert listed.index("op-older") < listed.index("op-newer")


def test_operations_refuses_flags_that_would_be_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flag that is silently dropped is a flag that lied about what ran."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})

    assert _operations(root, "--end", "a", "--forget", "b") == 1
    assert "not both" in capsys.readouterr().err

    assert _operations(root, "--reason", "because") == 1
    assert "only means anything with --end" in capsys.readouterr().err

    assert _operations(root, "--force") == 1
    assert "only means anything with --forget" in capsys.readouterr().err


def test_operations_on_a_project_that_never_paid_for_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No journal file at all is the ordinary state of a new project."""
    root = _project(tmp_path, [_raw("話す", "はなす")], {"vocabulary": _sourced_deck()})
    assert not _operations_path(root).exists()

    assert _operations(root) == 0

    assert "none" in capsys.readouterr().out
