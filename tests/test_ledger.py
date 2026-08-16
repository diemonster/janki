"""The ledger: operational metadata about records, and nothing else.

See docs/DESIGN_V2.md "The ledger". Two properties get the most attention here
because everything downstream leans on them: every mutator is idempotent (a
re-run must not grow the file), and the fingerprint formulas live in exactly
one module.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from japanese_anki import ledger as ledger_module
from japanese_anki import pitch
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.ledger import (
    LedgerError,
    example_audio_content_fingerprint,
    example_audio_filename_fingerprint,
    word_audio_content_fingerprint,
    word_audio_filename_fingerprint,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.pitch import to_aquestalk

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


def _accented(record: VocabularyRecord, pattern: str) -> VocabularyRecord:
    """``record`` with a jpdb accent pattern on it."""
    return replace(record, pitch_accent=[pattern])


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
        speed=1.0,
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
    # Both levels: a key inside a record entry, and a whole top-level section.
    # save() rebuilds the payload from its own keys, so the section is the one
    # that gets silently deleted — and it is the one a later milestone adds.
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "records": {"word:話す:はなす": {"added_at": "2026-08-06", "future": "keep"}},
                "pending_batches": {},
                "promote_queue": {"shirabe-export-needs-reading.yaml": ["word:話す:"]},
            }
        ),
        encoding="utf-8",
    )

    book = ledger_module.load(path)
    book.record_export("word:話す:はなす", "verbs", at="2026-08-12")
    book.save()

    reloaded = ledger_module.load(path)
    written = json.loads(path.read_text(encoding="utf-8"))
    queue = {"shirabe-export-needs-reading.yaml": ["word:話す:"]}

    assert reloaded.records["word:話す:はなす"]["future"] == "keep"
    assert written["promote_queue"] == queue
    assert reloaded.extra == {"promote_queue": queue}


def test_the_ledger_is_written_through_the_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Asserted directly, not by "no .tmp file is left behind": a plain
    # write_text leaves none either, so that proves nothing. A crash or a full
    # disk mid-write is what truncates data/ledger.json, and a truncated JSON
    # file is not recoverable — it just fails to parse on the next run.
    path = tmp_path / "ledger.json"
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        ledger_module,
        "atomic_write_text",
        lambda target, text: calls.append((target, text)),
    )

    book = ledger_module.load(path)
    book.record_added("word:話す:はなす", at="2026-08-06")
    book.save()

    assert [target for target, _ in calls] == [path]
    assert json.loads(calls[0][1])["records"]["word:話す:はなす"]["added_at"] == "2026-08-06"
    # Nothing else wrote the file behind the helper's back.
    assert not path.exists()


def test_a_filesystem_failure_to_save_is_a_ledger_error(tmp_path: Path) -> None:
    # The shape a real save failure actually has. The atomic writer reports
    # every filesystem failure (permission denied, read-only mount, ENOSPC) as
    # DataError, a *sibling* of LedgerError under JankiError. Left unwrapped it
    # sails past every `except LedgerError` its callers wrote — including
    # `cli._save_ledger`, whose whole job is to stop a failed ledger write from
    # printing a bare `error:` over a command that wrote everything else.
    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    directory = tmp_path / "unwritable"
    directory.mkdir()
    path = directory / "ledger.json"
    book = ledger_module.load(path)
    book.record_added("word:話す:はなす", at="2026-08-06")
    directory.chmod(0o555)

    try:
        with pytest.raises(LedgerError) as error:
            book.save()
    finally:
        directory.chmod(0o755)

    assert str(path) in str(error.value)
    assert not path.exists()


def test_saving_over_a_ledger_that_changed_on_disk_is_refused(tmp_path: Path) -> None:
    # Two live ledgers on one path: whichever saves second would otherwise
    # overwrite everything the first wrote, and the atomic writer guarantees the
    # loser still finds a perfectly well-formed file, so the loss is silent.
    path = tmp_path / "ledger.json"
    first = ledger_module.load(path)
    second = ledger_module.load(path)

    first.record_added("word:話す:はなす", at="2026-08-06")
    first.save()
    second.record_added("word:食べる:たべる", at="2026-08-06")

    with pytest.raises(LedgerError) as error:
        second.save()

    assert "changed on disk" in str(error.value)
    assert list(ledger_module.load(path).records) == ["word:話す:はなす"]


def test_saving_twice_from_one_ledger_is_fine(tmp_path: Path) -> None:
    # The guard is about a *second* ledger, not a second save: a command that
    # saves, does more work, and saves again must not trip over its own write.
    path = tmp_path / "ledger.json"
    book = ledger_module.load(path)

    book.record_added("word:話す:はなす", at="2026-08-06")
    book.save()
    book.record_added("word:食べる:たべる", at="2026-08-06")
    book.save()

    assert sorted(ledger_module.load(path).records) == ["word:話す:はなす", "word:食べる:たべる"]


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


@pytest.mark.parametrize(
    "value",
    [
        "2026/08/06",  # rejected by fromisoformat itself
        "2026-08-06T09:30:00",  # ditto, on 3.11+
        # These two are the reason the round-trip check exists: fromisoformat
        # accepts both and quietly normalises them to 2026-08-06, so without
        # the check a compact or week date would enter the ledger as a plain
        # day and the stored format would be enforced by nothing.
        "20260806",
        "2026-W32-4",
    ],
)
def test_dates_must_be_plain_iso_days(tmp_path: Path, value: str) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError):
        book.record_added("word:話す:はなす", at=value)
    assert book.records == {}


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


def test_re_running_the_same_pass_months_later_still_records_nothing(tmp_path: Path) -> None:
    # A pass is identified by what it did, never by when. Comparing the date
    # too makes every mutator idempotent for one calendar day only, so a
    # monthly enrich leaves `janki status` reporting one record enriched five
    # times by the same pass with the same fields.
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_enriched("word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-08-01")

    assert (
        book.record_enriched(
            "word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-09-01"
        )
        is False
    )

    entries = book.records["word:話す:はなす"]["enriched"]
    assert len(entries) == 1
    assert entries[0]["at"] == "2026-08-01"  # the first run, as with sources


def test_enrichment_identity_ignores_field_order_and_repeats(tmp_path: Path) -> None:
    # The identity is (kind, model, fields) as a *set* of names: a caller that
    # builds the list in dict order must not re-record the same pass forever.
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_enriched(
        "word:話す:はなす",
        kind="ai",
        model="claude-opus-5",
        fields=["usage_notes", "examples"],
        at="2026-08-01",
    )

    assert (
        book.record_enriched(
            "word:話す:はなす",
            kind="ai",
            model="claude-opus-5",
            fields=["examples", "usage_notes", "examples"],
            at="2026-09-01",
        )
        is False
    )

    entries = book.records["word:話す:はなす"]["enriched"]
    assert len(entries) == 1
    assert entries[0]["fields"] == ["examples", "usage_notes"]
    assert entries[0]["at"] == "2026-08-01"


def test_a_pass_that_wrote_different_fields_is_a_different_pass(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_enriched("word:話す:はなす", kind="jpdb", fields=["pitch_accent"], at="2026-08-01")

    assert (
        book.record_enriched(
            "word:話す:はなす",
            kind="jpdb",
            fields=["pitch_accent", "frequency_rank"],
            at="2026-09-01",
        )
        is True
    )
    assert len(book.records["word:話す:はなす"]["enriched"]) == 2


def test_enrichment_arguments_are_checked(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="human", fields=["examples"])
    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="ai", fields=["examples"])
    with pytest.raises(LedgerError):
        book.record_enriched("word:話す:はなす", kind="jpdb", fields=[])
    assert book.records == {}


def test_a_jpdb_pass_cannot_be_attributed_to_a_model(tmp_path: Path) -> None:
    # jpdb is a dictionary lookup, not one model among many. An AI pass
    # copy-pasted from the jpdb call keeps `model=config.enrich_model`, and the
    # ledger would then record AI work as jpdb work for `status` to misreport.
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError) as error:
        book.record_enriched(
            "word:話す:はなす", kind="jpdb", model="claude-opus-5", fields=["pitch_accent"]
        )

    assert "claude-opus-5" in str(error.value)
    assert book.records == {}
    # Saying it explicitly is still allowed: it is what gets written anyway.
    assert (
        book.record_enriched(
            "word:話す:はなす", kind="jpdb", model="jpdb", fields=["pitch_accent"]
        )
        is True
    )


def test_regenerated_audio_replaces_the_entry_for_that_file(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")
    book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp="1a2b3c",
        at="2026-08-11",
    )

    unchanged = book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
        content_fp="1a2b3c",
        at="2026-08-11",
    )
    revoiced = book.record_audio(
        "word:話す:はなす",
        file="janki-abc.wav",
        of="word",
        provider="voicevox",
        voice=8,
        speed=1.0,
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


def test_re_recording_identical_audio_months_later_reports_no_change(tmp_path: Path) -> None:
    # Same file, same voice, same content fingerprint: nothing was regenerated,
    # so nothing changed and the ledger must not be marked dirty over a date.
    book = ledger_module.load(tmp_path / "ledger.json")
    arguments = {
        "file": "janki-abc.wav",
        "of": "word",
        "provider": "voicevox",
        "voice": 46,
        "content_fp": "1a2b3c",
    }
    book.record_audio("word:話す:はなす", **arguments, at="2026-08-01", speed=1.0)

    assert book.record_audio("word:話す:はなす", **arguments, at="2026-09-01", speed=1.0) is False

    entries = book.records["word:話す:はなす"]["audio"]
    assert len(entries) == 1
    assert entries[0]["at"] == "2026-08-01"


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
        book.record_audio("word:話す:はなす", **{**arguments, "of": "sentence"}, speed=1.0)
    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "file": "  "}, speed=1.0)
    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "voice": "   "}, speed=1.0)
    with pytest.raises(LedgerError):
        book.record_audio("word:話す:はなす", **{**arguments, "voice": True}, speed=1.0)
    # None and any other object: `str(voice)` for anything at all would write
    # `"voice": "None"` — or `"{'id': 13}"` — into a committed file and report
    # success, where the `int()` this replaced raised.
    for bad in (None, {"id": 13}, 1.5):
        with pytest.raises(LedgerError, match="must be an id or a name"):
            book.record_audio("word:話す:はなす", **{**arguments, "voice": bad}, speed=1.0)


def test_a_named_voice_is_stored_as_its_name(tmp_path: Path) -> None:
    """Engines disagree about what a voice is: VOICEVOX numbers its speakers,
    OpenAI names them. Mapping `onyx` onto an index would put a number in a
    committed file that means nothing outside janki and would change meaning the
    day the voice list grows."""
    book = ledger_module.load(tmp_path / "ledger.json")

    book.record_audio(
        "word:話す:はなす", file="janki-abc.mp3", of="example", provider="openai",
        voice="onyx", speed=1.0, content_fp="1a2b3c",
    )

    entry = book.records["word:話す:はなす"]["audio"][0]
    assert entry["voice"] == "onyx"
    assert book.audio_file_for(
        "word:話す:はなす", of="example", content_fp="1a2b3c", voice="onyx", speed=1.0
    ) == "janki-abc.mp3"
    assert book.audio_file_for(
        "word:話す:はなす", of="example", content_fp="1a2b3c", voice="ash", speed=1.0
    ) is None, "a different voice is not current"


def test_detail_the_ledger_could_not_write_is_refused_at_the_call_site(tmp_path: Path) -> None:
    # Unchecked, a Path or a datetime in **details surfaces at save() as a bare
    # TypeError from json.dumps — not a JankiError, so cli.main() lets it out as
    # a traceback, which is exactly what M1.2 hardened the CLI against.
    book = ledger_module.load(tmp_path / "ledger.json")

    with pytest.raises(LedgerError) as audio_error:
        book.record_audio(
            "word:話す:はなす",
            file="janki-abc.wav",
            of="word",
            provider="voicevox",
            voice=46,
            speed=1.0,
            content_fp="1a2b3c",
            src=Path("/x/y.wav"),
        )
    with pytest.raises(LedgerError) as source_error:
        book.record_source_seen("word:話す:はなす", "jpdb", "deck:Mining", fetched=date.today())

    # The message names the caller's own key, not a line inside save().
    assert "'src'" in str(audio_error.value)
    assert "'fetched'" in str(source_error.value)
    assert book.records == {}
    book.save()  # nothing was stored, so the file is still writable


def test_serialisable_detail_is_still_accepted(tmp_path: Path) -> None:
    book = ledger_module.load(tmp_path / "ledger.json")

    book.record_source_seen("word:話す:はなす", "jpdb", "deck:Mining", vid=1577980, tags=["a"])
    book.save()

    assert ledger_module.load(tmp_path / "ledger.json").records["word:話す:はなす"]["sources"][
        0
    ]["tags"] == ["a"]


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

    # A record with no pattern is fingerprinted on the reading alone. Once a
    # pattern exists, audio generated without one is detectably out of date.
    assert word_audio_content_fingerprint(_accented(record, "LHHH")) != plain
    # Over the *utterance*, not the stored pattern: the AquesTalk string the
    # engine is actually sent. Fingerprinting the raw pattern let a fullwidth
    # ＬＨＬ and an ASCII LHL collide under NFKC, so correcting one to the other
    # left the guessed clip permanently current.
    assert word_audio_content_fingerprint(_accented(record, "LHHH")) == short_fingerprint(
        record.reading + to_aquestalk(record.reading, "LHHH")
    )


def test_an_audio_accent_override_wins_over_the_jpdb_pattern() -> None:
    record = _record()
    accented = _accented(record, "LHHH")
    # A *usable* override — one per kana plus the particle slot. An unfitting
    # one is not "the override winning", it is the record having no pattern the
    # engine can force, and the fingerprint now says so by matching the
    # no-pattern case.
    accented.audio_accent = "HLLL"

    assert word_audio_content_fingerprint(accented) == short_fingerprint(
        record.reading + to_aquestalk(record.reading, "HLLL")
    )


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
        speed=1.0,
        content_fp=word_audio_content_fingerprint(spoken),
        at="2026-08-11",
    )
    book.record_audio(
        example_only.id,
        file="janki-example.wav",
        of="example",
        provider="azure",
        voice=0,
        speed=1.0,
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
        speed=1.0,
        content_fp=word_audio_content_fingerprint(record),
        at="2026-08-11",
    )
    book.record_audio(
        record.id,
        file="janki-example.wav",
        of="example",
        provider="azure",
        voice=0,
        speed=1.0,
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
    # The transition an enrichment pass creates: audio synthesized before the
    # accent was known must be regenerated with it.
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record()
    book.record_audio(
        record.id,
        file="janki-word.wav",
        of="word",
        provider="voicevox",
        voice=46,
        speed=1.0,
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


def test_an_accepted_extracted_example_with_annotation_holes_still_needs_ai(
    tmp_path: Path,
) -> None:
    from japanese_anki.identifiers import short_fingerprint

    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record(
        examples=[ExampleSentence(japanese="日本語を話します。")],
        usage_notes="A useful sentence.",
    )
    record.source.type = "extract"
    record.source.raw_fields["example_authority"] = short_fingerprint(
        "日本語を話します。"
    )

    assert book.missing_enrichment([record]) == [record.id]


def test_an_unaccepted_extracted_example_is_not_a_target(tmp_path: Path) -> None:
    # The prompt refuses to pin it and the absorb refuses to overwrite it, so
    # selecting on it made every run pay a model call that could write
    # nothing — forever. Its remedy is a user decision, not another attempt.
    book = ledger_module.load(tmp_path / "ledger.json")
    record = _record(
        examples=[ExampleSentence(japanese="日本語を話します。")],
        usage_notes="A useful sentence.",
    )
    record.source.type = "extract"

    assert book.missing_enrichment([record]) == []


def test_the_word_audio_fingerprint_uses_pitch_select_pattern() -> None:
    """One definition of "which pattern does audio use". The fingerprint is
    computed *over* that choice, so a second definition would mean audio
    generated under one rule is fingerprinted under the other and never reads as
    stale when the accent changes.

    A leading blank entry discriminates: taking `pitch_accent[0]` literally
    gives the reading alone, while `select_pattern` skips to the first real
    pattern."""
    record = VocabularyRecord(
        id="word:橋:はし",
        expression="橋",
        reading="はし",
        meanings=["bridge"],
        pitch_accent=["", "LHL"],
    )

    assert ledger_module.word_audio_content_fingerprint(record) == short_fingerprint(
        "はし" + to_aquestalk("はし", "LHL")
    )
    assert pitch.select_pattern(record) == "LHL"


def test_a_case_only_edit_does_not_make_word_audio_look_stale() -> None:
    """The fingerprint covers what was spoken, and 'LHLL' and 'lhll' are the
    same utterance — to_aquestalk renders them byte for byte alike."""
    upper = VocabularyRecord(
        id="word:卵:たまご",
        expression="卵",
        reading="たまご",
        meanings=["egg"],
        pitch_accent=["LHLL"],
    )

    assert ledger_module.word_audio_content_fingerprint(
        upper
    ) == ledger_module.word_audio_content_fingerprint(
        replace(upper, pitch_accent=["lhll"])
    )


def test_batch_recovery_never_deletes_an_unknown_same_named_artifact(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    recovery_path = ledger_module.batch_recovery_path(ledger_path)
    recovery_path.write_text("important unrelated content\n", encoding="utf-8")

    with pytest.raises(LedgerError, match="recovery journal"):
        ledger_module.clear_batch_recovery(ledger_path, "msgbatch_polish")

    assert recovery_path.read_text(encoding="utf-8") == "important unrelated content\n"


def test_a_second_fetch_cannot_replace_different_recovery_for_the_same_batch(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    entry = {
        "kind": "polish",
        "model": "claude-opus-5",
        "submitted_at": "2026-08-11",
        "pending_ids": ["one", "two"],
        "force_fields": [],
        "prompt_fingerprints": {"one": "fp-one", "two": "fp-two"},
    }
    first = ledger_module.write_batch_recovery(
        ledger_path,
        "msgbatch_polish",
        entry,
        accepted_meanings={"one": ["first review"]},
        retry_ids=["two"],
    )

    with pytest.raises(LedgerError, match="Another fetch"):
        ledger_module.write_batch_recovery(
            ledger_path,
            "msgbatch_polish",
            entry,
            accepted_meanings={"two": ["second review"]},
            retry_ids=["one"],
        )

    current = ledger_module.load_batch_recovery(ledger_path)
    assert current is not None
    assert current.text == first.text
    assert current.accepted_meanings == {"one": ["first review"]}


def test_a_stale_fetch_cannot_clear_a_newer_same_batch_journal(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    entry = {
        "kind": "polish",
        "model": "claude-opus-5",
        "submitted_at": "2026-08-11",
        "pending_ids": ["one"],
        "force_fields": [],
        "prompt_fingerprints": {"one": "fp-one"},
    }
    stale = ledger_module.write_batch_recovery(
        ledger_path,
        "msgbatch_polish",
        entry,
        accepted_meanings={"one": ["older review"]},
        retry_ids=[],
    )
    assert ledger_module.clear_batch_recovery(
        ledger_path, "msgbatch_polish", expected=stale
    )
    newer = ledger_module.write_batch_recovery(
        ledger_path,
        "msgbatch_polish",
        entry,
        accepted_meanings={"one": ["newer review"]},
        retry_ids=[],
    )

    with pytest.raises(LedgerError, match="newer journal was kept intact"):
        ledger_module.clear_batch_recovery(
            ledger_path, "msgbatch_polish", expected=stale
        )

    current = ledger_module.load_batch_recovery(ledger_path)
    assert current is not None
    assert current.text == newer.text
    assert current.accepted_meanings == {"one": ["newer review"]}


def test_the_tracked_ledger_agrees_with_the_records_it_describes() -> None:
    """`status --rebuild` must stay a no-op on a healthy repository.

    The cross-module contract stated at `cli.run_import` and `migrate.migrate_inline`:
    a source reference's `ref` is the same string the record carries as
    `source.imported_from`, because rebuild reconstructs the reference from the
    record. A reference's identity is every key but the date, so any divergence
    has the documented recovery command *append* a second near-duplicate
    reference to each affected record and report it as a recovery.

    Nothing pinned this against the real committed ledger, and it drifted: a
    commit that filled `imported_from` on the three hand-seeded records left
    their ledger refs empty, so those three would have doubled on the next
    rebuild.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    records = json.loads(
        (root / "data" / "normalized" / "vocabulary.json").read_text(encoding="utf-8")
    )
    book = json.loads((root / "data" / "ledger.json").read_text(encoding="utf-8"))

    diverging = []
    for record in records:
        stated = record["source"].get("imported_from", "")
        if not stated:
            continue
        entry = book["records"].get(record["id"], {})
        refs = {source.get("ref", "") for source in entry.get("sources", [])}
        if refs and stated not in refs:
            diverging.append((record["id"], stated, sorted(refs)))

    assert diverging == [], (
        "these records would gain a duplicate source reference on "
        f"`janki status --rebuild`: {diverging}"
    )
