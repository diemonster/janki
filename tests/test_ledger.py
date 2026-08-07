"""The ledger: operational metadata about records, and nothing else.

See docs/DESIGN_V2.md "The ledger". Two properties get the most attention here
because everything downstream leans on them: every mutator is idempotent (a
re-run must not grow the file), and the fingerprint formulas live in exactly
one module.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import ledger as ledger_module
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.ledger import (
    LedgerError,
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord

TODAY = date.today().isoformat()


def _record(
    expression: str = "話す",
    reading: str = "はなす",
    *,
    examples: list[ExampleSentence] | None = None,
    usage_notes: str = "",
) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["to speak"],
        examples=examples if examples is not None else [],
        usage_notes=usage_notes,
    )


def _enriched(record: VocabularyRecord) -> VocabularyRecord:
    """A record with the content that makes it 'enriched' by M1.3's definition."""
    record.examples = [ExampleSentence(japanese="毎日話す。", english="I speak every day.")]
    record.usage_notes = "Everyday verb."
    return record


def _accented(record: VocabularyRecord, pattern: str) -> SimpleNamespace:
    """A stand-in for the post-M2.2 record shape (``pitch_accent`` lands in M2.2)."""
    return SimpleNamespace(
        id=record.id,
        reading=record.reading,
        examples=record.examples,
        usage_notes=record.usage_notes,
        pitch_accent=[pattern],
    )


# -- loading and saving ------------------------------------------------------


def test_a_missing_ledger_file_is_an_empty_ledger(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    assert book.records == {}
    assert book.pending_batches == {}


def test_an_empty_ledger_file_is_an_empty_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("\n", encoding="utf-8")

    assert ledger_module.load(path).records == {}


def test_mutations_persist_only_when_saved(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    book = ledger_module.load(path)
    book.record_added("word:話す:はなす", at="2026-08-06")

    assert not path.exists()

    book.save()

    assert ledger_module.load(path).records["word:話す:はなす"]["added_at"] == "2026-08-06"


def test_saved_file_is_sorted_by_record_id_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    book = ledger_module.load(path)
    for expression, reading in (("話す", "はなす"), ("会う", "あう"), ("食べる", "たべる")):
        book.record_added(f"word:{expression}:{reading}", at="2026-08-06")
    book.save()

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text.endswith("\n")
    assert "\r" not in text
    assert "話す" in text  # not escaped past recognition
    assert list(payload["records"]) == sorted(payload["records"])
    assert payload["version"] == 1
    assert payload["pending_batches"] == {}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_full_entry_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    book = ledger_module.load(path)
    book.record_added("word:話す:はなす", at="2026-08-06")
    book.record_source_seen("word:話す:はなす", "shirabe", "export.csv", seen_at="2026-08-06")
    book.record_enriched(
        "word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-08-10"
    )
    book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp="1a2b3c",
        at="2026-08-11",
    )
    book.record_export("word:話す:はなす", "personal-vocabulary", at="2026-08-12")
    book.pending_batches["msgbatch_1"] = {"submitted_at": "2026-08-12", "pending_ids": ["x"]}
    book.save()

    reloaded = ledger_module.load(path)

    assert reloaded.records == book.records
    assert reloaded.pending_batches == book.pending_batches


def test_unknown_keys_written_by_a_future_janki_survive_a_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "records": {"word:話す:はなす": {"added_at": "2026-08-06", "future": "keep"}},
                "pending_batches": {},
            }
        ),
        encoding="utf-8",
    )

    book = ledger_module.load(path)
    book.record_export("word:話す:はなす", "verbs", at="2026-08-12")
    book.save()

    assert ledger_module.load(path).records["word:話す:はなす"]["future"] == "keep"


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '["a list"]',
        '{"records": []}',
        '{"version": 2, "records": {}}',
        '{"records": {"word:話す:はなす": "not an object"}}',
    ],
)
def test_an_unreadable_ledger_is_a_clean_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(LedgerError):
        ledger_module.load(path)


def test_dates_must_be_plain_iso_days(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError):
        book.record_added("word:話す:はなす", at="2026/08/06")
    with pytest.raises(LedgerError):
        book.record_added("word:話す:はなす", at="2026-08-06T09:30:00")


def test_dates_default_to_today(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    book.record_added("word:話す:はなす")

    assert book.records["word:話す:はなす"]["added_at"] == TODAY


# -- mutators are idempotent -------------------------------------------------


def test_record_added_keeps_the_first_sighting(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    assert book.record_added("word:話す:はなす", at="2026-08-06") is True
    assert book.record_added("word:話す:はなす", at="2026-09-01") is False
    assert book.records["word:話す:はなす"]["added_at"] == "2026-08-06"


def test_re_importing_the_same_file_appends_no_second_source(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_source_seen("word:話す:はなす", "shirabe", "export.csv", seen_at="2026-08-06")

    # Same file, later day: the reference is the same reference.
    assert (
        book.record_source_seen(
            "word:話す:はなす", "shirabe", "export.csv", seen_at="2026-09-01"
        )
        is False
    )

    sources = book.records["word:話す:はなす"]["sources"]
    assert sources == [{"type": "shirabe", "ref": "export.csv", "seen_at": "2026-08-06"}]


def test_a_different_source_is_kept_alongside_the_first(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_source_seen("word:話す:はなす", "shirabe", "export.csv", seen_at="2026-08-06")

    assert (
        book.record_source_seen(
            "word:話す:はなす", "jpdb", "deck:Mining", vid=1577980, seen_at="2026-08-10"
        )
        is True
    )

    types = [source["type"] for source in book.records["word:話す:はなす"]["sources"]]
    assert types == ["shirabe", "jpdb"]
    assert book.records["word:話す:はなす"]["sources"][1]["vid"] == 1577980


def test_a_source_sighting_registers_a_record_the_ledger_had_not_seen(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    book.record_source_seen("word:話す:はなす", "shirabe", "export.csv", seen_at="2026-08-06")

    assert book.records["word:話す:はなす"]["added_at"] == "2026-08-06"


def test_jpdb_and_ai_enrichment_do_not_overwrite_each_other(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    book.record_enriched(
        "word:話す:はなす", kind="jpdb", fields=["frequency_rank", "pitch_accent"], at="2026-08-10"
    )
    book.record_enriched(
        "word:話す:はなす",
        kind="ai",
        model="claude-opus-5",
        fields=["examples", "usage_notes"],
        at="2026-08-11",
    )

    assert book.records["word:話す:はなす"]["enriched"] == [
        {
            "at": "2026-08-10",
            "kind": "jpdb",
            "model": "jpdb",
            "fields": ["frequency_rank", "pitch_accent"],
        },
        {
            "at": "2026-08-11",
            "kind": "ai",
            "model": "claude-opus-5",
            "fields": ["examples", "usage_notes"],
        },
    ]


def test_re_running_the_same_enrichment_pass_records_nothing_new(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_enriched("word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-08-10")

    assert (
        book.record_enriched(
            "word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-08-10"
        )
        is False
    )
    assert len(book.records["word:話す:はなす"]["enriched"]) == 1


def test_enrichment_arguments_are_checked(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="human", fields=["examples"])
    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="ai", fields=["examples"])
    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="jpdb", fields=[])
    assert book.records == {}


def test_regenerated_audio_replaces_the_entry_for_that_file(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp="1a2b3c",
        at="2026-08-11",
    )

    unchanged = book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp="1a2b3c",
        at="2026-08-11",
    )
    revoiced = book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=8,
        content_fp="1a2b3c",
        at="2026-09-01",
        accent_unverified=True,
    )

    assert unchanged is False
    assert revoiced is True
    entries = book.records["word:話す:はなす"]["audio"]
    assert len(entries) == 1
    assert entries[0]["voice"] == 8
    assert entries[0]["accent_unverified"] is True


def test_audio_arguments_are_checked(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    arguments = {
        "file": "janki-abc.wav",
        "of": "word",
        "provider": "voicevox",
        "voice": 46,
        "content_fp": "1a2b3c",
    }

    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "of": "sentence"})
    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "file": "  "})
    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "voice": "nanami"})


def test_rebuilding_the_same_deck_twice_records_one_export(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    assert book.record_export("word:話す:はなす", "verbs", at="2026-08-12") is True
    assert book.record_export("word:話す:はなす", "verbs", at="2026-08-12") is False
    assert book.record_export("word:話す:はなす", "verbs", at="2026-08-20") is True
    assert book.record_export("word:話す:はなす", "personal-vocabulary", at="2026-08-20") is True
    assert book.records["word:話す:はなす"]["exports"] == {
        "verbs": "2026-08-20",
        "personal-vocabulary": "2026-08-20",
    }


def test_remove_drops_the_entry_and_reports_whether_there_was_one(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_added("word:話す:はなす", at="2026-08-06")

    assert book.remove("word:話す:はなす") is True
    assert book.remove("word:話す:はなす") is False
    assert book.records == {}


def test_an_empty_record_id_is_refused(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError):
        book.record_added("   ")


# -- fingerprints ------------------------------------------------------------


def test_filename_fingerprints_address_by_identity() -> None:
    record = _record()
    example = ExampleSentence(japanese="毎日話す。")

    assert word_audio_filename_fingerprint(record) == short_fingerprint(record.id)
    assert example_audio_filename_fingerprint(record, example) == short_fingerprint(
        record.id + example.japanese
    )


def test_content_fingerprints_cover_what_is_spoken() -> None:
    record = _record()
    example = ExampleSentence(japanese="毎日話す。")

    assert word_audio_content_fingerprint(record) == short_fingerprint(record.reading)
    assert example_audio_content_fingerprint(example) == short_fingerprint(example.japanese)


def test_the_two_fingerprint_families_are_different_addresses() -> None:
    record = _record()

    # Same subject, different questions: "which file is this?" vs "is that
    # file still current?". Collapsing them would make staleness undetectable.
    assert word_audio_filename_fingerprint(record) != word_audio_content_fingerprint(record)


def test_example_audio_filenames_are_per_record_but_content_is_shared() -> None:
    first = _record()
    second = _record("喋る", "しゃべる")
    example = ExampleSentence(japanese="毎日話す。")

    assert example_audio_filename_fingerprint(first, example) != (
        example_audio_filename_fingerprint(second, example)
    )
    assert example_audio_content_fingerprint(example) == example_audio_content_fingerprint(
        ExampleSentence(japanese="毎日話す。")
    )


def test_word_content_fingerprint_follows_the_selected_accent() -> None:
    record = _record()
    plain = word_audio_content_fingerprint(record)

    # pitch_accent lands in M2.2; today's records answer "" and are
    # fingerprinted on the reading alone. Once a pattern exists, audio
    # generated without one is detectably out of date.
    assert word_audio_content_fingerprint(_accented(record, "LHHH")) != plain
    assert word_audio_content_fingerprint(_accented(record, "LHHH")) == short_fingerprint(
        record.reading + "LHHH"
    )


def test_an_audio_accent_override_wins_over_the_jpdb_pattern() -> None:
    record = _record()
    accented = _accented(record, "LHHH")
    accented.audio_accent = "HLL"

    assert word_audio_content_fingerprint(accented) == short_fingerprint(record.reading + "HLL")


# -- queries -----------------------------------------------------------------


def test_unexported_reports_ids_this_deck_has_never_built(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_export("word:話す:はなす", "verbs", at="2026-08-12")
    book.record_export("word:会う:あう", "personal-vocabulary", at="2026-08-12")

    ids = ["word:話す:はなす", "word:会う:あう", "word:食べる:たべる", "word:話す:はなす"]

    assert book.unexported("verbs", ids) == ["word:会う:あう", "word:食べる:たべる"]
    assert book.unexported("personal-vocabulary", ids) == [
        "word:話す:はなす",
        "word:食べる:たべる",
    ]


def test_missing_audio_is_about_word_audio(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    spoken = _record()
    example_only = _record("会う", "あう", examples=[ExampleSentence(japanese="友達に会う。")])
    silent = _record("食べる", "たべる")

    book.record_audio(
        spoken.id,
        file=f"janki-{word_audio_filename_fingerprint(spoken)}.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp=word_audio_content_fingerprint(spoken),
        at="2026-08-11",
    )
    book.record_audio(
        example_only.id,
        file="janki-example.wav",
        of="example",
        provider="azure",
        voice=0,
        content_fp=example_audio_content_fingerprint(example_only.examples[0]),
        at="2026-08-11",
    )

    assert book.missing_audio([spoken, example_only, silent]) == [example_only.id, silent.id]


def test_stale_audio_catches_an_edited_reading_and_an_edited_example(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record(examples=[ExampleSentence(japanese="毎日話す。")])
    book.record_audio(
        record.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp=word_audio_content_fingerprint(record),
        at="2026-08-11",
    )
    book.record_audio(
        record.id,
        file="janki-example.wav",
        of="example",
        provider="azure",
        voice=0,
        content_fp=example_audio_content_fingerprint(record.examples[0]),
        at="2026-08-11",
    )

    assert book.stale_audio([record]) == []

    record.examples[0].japanese = "毎日日本語で話す。"
    assert book.stale_audio([record]) == [record.id]

    record.examples[0].japanese = "毎日話す。"
    record.reading = "はなーす"
    assert book.stale_audio([record]) == [record.id]


def test_records_with_no_audio_are_missing_not_stale(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record()

    assert book.stale_audio([record]) == []
    assert book.missing_audio([record]) == [record.id]


def test_word_audio_goes_stale_once_a_pitch_pattern_arrives(tmp_path: Path) -> None:
    # The M2.2 schema change is exactly this transition: audio synthesized
    # before the accent was known must be regenerated with it.
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record()
    book.record_audio(
        record.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        content_fp=word_audio_content_fingerprint(record),
        at="2026-08-11",
    )

    assert book.stale_audio([_accented(record, "LHHH")]) == [record.id]


def test_missing_enrichment_reads_records_not_the_ledger(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    bare = _record()
    examples_only = _record("会う", "あう", examples=[ExampleSentence(japanese="友達に会う。")])
    notes_only = _record("食べる", "たべる", usage_notes="Ichidan verb.")
    done = _enriched(_record("読む", "よむ"))

    # A jpdb pass wrote pitch accent for the bare record and none of the
    # content enrichment actually produces — it must stay a target.
    book.record_enriched(bare.id, kind="jpdb", fields=["pitch_accent"], at="2026-08-10")

    assert book.missing_enrichment([bare, examples_only, notes_only, done]) == [
        bare.id,
        examples_only.id,
        notes_only.id,
    ]


def test_an_empty_example_does_not_count_as_enrichment(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record(examples=[ExampleSentence()], usage_notes="Everyday verb.")

    assert book.missing_enrichment([record]) == [record.id]
