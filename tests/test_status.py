"""`janki status`: the read-only view over records plus the ledger.

Three properties get most of the attention here, because they are what make the
command trustworthy: it works on a repo that has never run an import (a missing
ledger is an empty ledger), it never writes unless asked to rebuild, and its
duplicate detection finds the *kana-form* class of duplicate — the common real
one, and the one an expression-only check misses.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from japanese_anki import cli, ledger
from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.ledger import (
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord

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


def test_pitch_accent_counts_wait_for_the_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # M2.2 adds pitch_accent to the record schema; until then the honest
    # answer is that the question cannot be asked yet.
    root = _project(tmp_path, [_raw("話す", "はなす")])

    assert _status(root) == 0

    out = capsys.readouterr().out
    assert status_module.pitch_accent_supported() is False
    assert "Missing pitch accent: n/a until the pitch-accent schema lands (M2.2)" in out


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
    # Both filenames pin their content here: the word file's address embeds the
    # record id (and so the reading), the example file's embeds the sentence.
    assert audio[1]["content_fp"] == word_audio_content_fingerprint(record)
    assert audio[0]["content_fp"] == example_audio_content_fingerprint(record.examples[0])
    assert ledger.load(root / "ledger.json").stale_audio([record]) == []


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
        content_fp=word_audio_content_fingerprint(record),
        at="2026-08-11",
    )
    book.save()

    assert _status(root, "--rebuild") == 0

    capsys.readouterr()
    entries = ledger.load(root / "ledger.json").records[record.id]["audio"]
    # A rebuilt entry knows neither the engine nor the voice; it must never
    # replace an entry that does.
    assert entries == [
        {
            "file": filename,
            "of": "word",
            "provider": "voicevox",
            "voice": 46,
            "content_fp": word_audio_content_fingerprint(record),
            "at": "2026-08-11",
        }
    ]


def test_rebuilt_word_audio_claims_no_content_it_cannot_prove(tmp_path: Path) -> None:
    # Post-M2.2 shape: the filename fingerprint covers the record id, so it
    # proves the reading but not the accent the audio was generated with.
    record = _record("話す", "はなす")
    accented = SimpleNamespace(
        id=record.id,
        reading=record.reading,
        examples=[],
        source=record.source,
        pitch_accent=["LHH"],
    )
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
    record = _record("話す", "はなす", examples=[{"japanese": "毎日話す。"}])
    root = _project(tmp_path, [record.to_dict()])
    book = ledger.load(root / "ledger.json")
    book.record_audio(
        record.id,
        file="janki-example.wav",
        of="example",
        provider="azure",
        voice=0,
        content_fp=example_audio_content_fingerprint(ExampleSentence(japanese="毎日話した。")),
    )
    book.save()
    config = ProjectConfig.load(root)

    report = status_module.build_report(
        config, status_module.collect_records(config), ledger.load(root / "ledger.json")
    )

    assert report.stale_audio == [record.id]
