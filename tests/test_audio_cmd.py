"""``janki audio`` — and mostly, the things it declines to say.

No engine: the provider is a fake that records what it was asked to speak and
whether the accent was forced. That distinction is the point of the whole
command, and it is invisible in the bytes that come back, so asserting on the
returned WAV would pass just as well against a version that never forced
anything.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import audio_cmd, cli
from japanese_anki import ledger as ledger_mod
from japanese_anki.audio_cmd import AudioError, generate_audio, prune_unreferenced
from japanese_anki.config import ProjectConfig
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

WAV = b"RIFF....WAVEfake"


class FakeVoice:
    """Stands in for a speech engine, recording what it was told to say."""

    def __init__(
        self, *, reachable: bool = True, audio: bytes = WAV, voice: int = 7,
        speed: float = 1.0,
    ) -> None:
        self.said: list[tuple[str, bool]] = []
        self.reachable = reachable
        self.audio = audio
        self.voice = voice
        self.speed = speed

    name = "fakevox"
    launch_hint = "start the fake engine"

    def available(self) -> bool:
        return self.reachable

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        self.said.append((text_or_kana, forced_accent))
        return self.audio


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:橋:はし",
        "expression": "橋",
        "reading": "はし",
        "meanings": ["bridge"],
        "pitch_accent": ["LHL"],
        "source": SourceReference(type="jpdb", imported_from="deck"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def run(records: list[VocabularyRecord], tmp_path: Path, **kwargs: Any):
    book = kwargs.pop("book", None) or ledger_mod.Ledger(path=tmp_path / "ledger.json")
    provider = kwargs.pop("provider", None) or FakeVoice()
    result = generate_audio(
        records, provider=provider, book=book, media_dir=tmp_path / "media", **kwargs
    )
    return result, provider, book


# ---------------------------------------------------------------------------
# Words: the accent is forced, or nothing is said
# ---------------------------------------------------------------------------


def test_a_word_is_spoken_as_aquestalk_kana_with_the_accent_forced(tmp_path: Path) -> None:
    """橋 is odaka: LHL becomes ハシ' and goes out with forced_accent set. Left
    to guess, the engine renders it the same as 箸 — the pair the card exists
    to tell apart."""
    result, provider, _ = run([record()], tmp_path, words=True)

    assert provider.said == [("ハシ'", True)]
    assert result.file_count == 1


def test_the_clip_lands_in_the_media_dir_and_on_the_record(tmp_path: Path) -> None:
    result, _, _ = run([record()], tmp_path, words=True)

    stored = result.records[0].audio
    assert stored.startswith("audio/janki-") and stored.endswith(".wav")
    assert (tmp_path / "media" / stored).read_bytes() == WAV
    assert "/" in stored and not stored.startswith("/"), "relative to media_dir"


def test_a_record_with_no_pattern_is_skipped_and_reported(tmp_path: Path) -> None:
    """The rule the whole command is arranged around. An engine's guess is
    wrong exactly for the homographs a pitch card is for, so silence is the
    honest answer — but it has to be a visible one."""
    result, provider, _ = run([record(pitch_accent=[])], tmp_path, words=True)

    assert provider.said == []
    assert result.no_pattern == ["word:橋:はし"]
    assert result.file_count == 0


def test_allow_default_accent_opts_into_the_guess_and_marks_it(tmp_path: Path) -> None:
    result, provider, book = run(
        [record(pitch_accent=[])], tmp_path, words=True, allow_default_accent=True
    )

    # Read naturally, not forced — there is no accent to force.
    assert provider.said == [("はし", False)]
    entry = book.records["word:橋:はし"]["audio"][0]
    assert entry[audio_cmd.ACCENT_UNVERIFIED] is True
    assert result.file_count == 1


def test_a_pattern_that_does_not_fit_its_reading_is_not_guessed_around(
    tmp_path: Path,
) -> None:
    """`to_aquestalk` refuses a mismatched pattern rather than inventing an
    alignment; this command must not turn that refusal into a guess."""
    result, provider, _ = run([record(pitch_accent=["LH"])], tmp_path, words=True)

    assert provider.said == []
    assert result.no_pattern == ["word:橋:はし"]
    assert any("does not fit" in w for w in result.warnings)


def test_the_override_pattern_is_the_one_spoken(tmp_path: Path) -> None:
    # audio_accent exists for the reader who listened and disagreed.
    _, provider, _ = run(
        [record(pitch_accent=["LHL"], audio_accent="HLL")], tmp_path, words=True
    )

    assert provider.said == [("ハ'シ", True)]


# ---------------------------------------------------------------------------
# Examples: read naturally, and only when the furigana was confirmed
# ---------------------------------------------------------------------------


def test_an_example_is_read_naturally(tmp_path: Path) -> None:
    with_example = record(examples=[ExampleSentence(japanese="橋を渡ります。")])

    result, provider, _ = run([with_example], tmp_path, words=False, examples=True)

    assert provider.said == [("橋を渡ります。", False)]
    assert result.records[0].examples[0].audio.startswith("audio/janki-")


def test_an_example_with_unconfirmed_furigana_is_not_voiced(tmp_path: Path) -> None:
    """M4.2 flagged it because nobody checked its segmentation. Speaking it
    would turn an open question into a recording."""
    from japanese_anki.identifiers import short_fingerprint

    sentence = "橋を渡ります。"
    flagged = record(
        examples=[ExampleSentence(japanese=sentence)],
        source=SourceReference(
            type="jpdb",
            imported_from="deck",
            raw_fields={"furigana_unverified": short_fingerprint(sentence)},
        ),
    )

    result, provider, _ = run([flagged], tmp_path, words=False, examples=True)

    assert provider.said == []
    assert result.unverified == [f"word:橋:はし: {sentence}"]


def test_a_confirmed_example_beside_a_flagged_one_is_still_voiced(tmp_path: Path) -> None:
    from japanese_anki.identifiers import short_fingerprint

    flagged_text, fine_text = "橋を渡ります。", "橋が長いです。"
    both = record(
        examples=[
            ExampleSentence(japanese=flagged_text),
            ExampleSentence(japanese=fine_text),
        ],
        source=SourceReference(
            type="jpdb",
            imported_from="deck",
            raw_fields={"furigana_unverified": short_fingerprint(flagged_text)},
        ),
    )

    result, provider, _ = run([both], tmp_path, words=False, examples=True)

    assert provider.said == [(fine_text, False)]
    assert result.records[0].examples[0].audio == ""
    assert result.records[0].examples[1].audio.startswith("audio/")


# ---------------------------------------------------------------------------
# The config reaches the engine
# ---------------------------------------------------------------------------


def test_the_configured_voice_and_rate_reach_the_provider(tmp_path: Path) -> None:
    """`_speech_provider` is the only join between janki.toml and the engine, and
    both halves of it are silent when wrong: a dropped speed gives audio at the
    wrong rate with no error, and — since the ledger now records what the
    provider reports — a ledger that agrees with the audio and with nothing the
    user asked for."""
    (tmp_path / "janki.toml").write_text(
        '[tts]\nvoicevox_speaker = 13\nvoicevox_speed = 0.7\n', encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    provider = cli._speech_provider(config, None)

    assert (provider.voice, provider.speed) == (13, 0.7)


# ---------------------------------------------------------------------------
# Doing it twice
# ---------------------------------------------------------------------------


def test_a_second_run_writes_nothing_new(tmp_path: Path) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)

    second, provider, _ = run(first.records, tmp_path, words=True, book=book)

    assert provider.said == []
    assert second.file_count == 0
    assert second.up_to_date == 1


def test_a_new_voice_is_not_current_without_force(tmp_path: Path) -> None:
    """Voice is audible and deliberately not part of the content fingerprint, so
    the ledger has to carry it. Without that, changing `voicevox_speaker` and
    re-running reported everything "already current" and re-voiced nothing —
    the change only took effect under `--force`, which is a bigger hammer than
    the situation calls for."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)

    second, provider, _ = run(
        first.records, tmp_path, words=True, book=book, provider=FakeVoice(voice=13)
    )

    assert provider.said == [("ハシ'", True)], "re-voiced in the new voice"
    assert second.up_to_date == 0
    entries = book.records[record().id]["audio"]
    assert [entry["voice"] for entry in entries] == [13], "and the ledger says so"


def test_a_new_speed_is_not_current_either(tmp_path: Path) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)

    second, provider, _ = run(
        first.records, tmp_path, words=True, book=book, provider=FakeVoice(speed=0.7)
    )

    assert provider.said == [("ハシ'", True)]
    assert second.up_to_date == 0
    assert [e["speed"] for e in book.records[record().id]["audio"]] == [0.7]


def test_an_entry_from_before_speed_was_recorded_is_regenerated(tmp_path: Path) -> None:
    """A ledger written by an older janki has no `speed` key, so nothing can say
    how its clips were spoken. Regenerating is seconds; keeping audio janki
    cannot describe is a collection that quietly speaks at two rates."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    for entry in book.records[record().id]["audio"]:
        del entry["speed"]

    second, provider, _ = run(first.records, tmp_path, words=True, book=book)

    assert provider.said == [("ハシ'", True)]
    assert second.up_to_date == 0


def test_a_re_voice_interrupted_part_way_resumes(tmp_path: Path) -> None:
    """The scenario the missing rate made unrecoverable: a re-voice of two
    records dies after the first, and the re-run has to finish the second rather
    than call the whole collection current. Both clips must end up in one voice,
    and the ledger must be able to prove it."""
    records = [record(), record(id="word:箸:はし", expression="箸", pitch_accent=["HLL"])]
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run(records, tmp_path, words=True, book=book)

    # The engine dies after the first clip of the new voice.
    dying = FakeVoice(voice=13)
    original = dying.synthesize

    def die_after_one(text: str, *, forced_accent: bool) -> bytes:
        if dying.said:
            raise ledger_mod.LedgerError("engine went away")
        return original(text, forced_accent=forced_accent)

    dying.synthesize = die_after_one  # type: ignore[method-assign]
    interrupted, _, _ = run(first.records, tmp_path, words=True, book=book, provider=dying)
    assert interrupted.stopped_by, "the run reported that it stopped"

    resumed, provider, _ = run(
        interrupted.records, tmp_path, words=True, book=book, provider=FakeVoice(voice=13)
    )

    assert len(provider.said) == 1, "only the record the failure skipped"
    assert resumed.up_to_date == 1, "the one already re-voiced is left alone"
    voices = {e["voice"] for r in records for e in book.records[r.id]["audio"]}
    assert voices == {13}, "and the collection ends in one voice"


def test_force_regenerates_under_the_same_name(tmp_path: Path) -> None:
    """Same content, same address. Anki's media sync notices content rather
    than names, so rewriting in place is what a re-voice should do."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    name = first.records[0].audio

    second, provider, _ = run(
        first.records, tmp_path, words=True, book=book, force=True
    )

    assert provider.said == [("ハシ'", True)]
    assert second.records[0].audio == name


def test_a_changed_accent_needs_new_audio(tmp_path: Path) -> None:
    """The content fingerprint covers the accent, so re-accenting a word makes
    its recorded clip detectably out of date. Word audio is addressed by record
    id, so the *file* keeps its name and is rewritten in place."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    reaccented = [record(pitch_accent=["HLL"], audio=first.records[0].audio)]

    second, provider, _ = run(reaccented, tmp_path, words=True, book=book)

    assert provider.said == [("ハ'シ", True)], "the new accent is spoken"
    assert second.file_count == 1
    assert second.records[0].audio == first.records[0].audio, "same address"


def test_a_changed_reading_lands_under_a_new_name(tmp_path: Path) -> None:
    """A word clip is addressed by record id, and janki's id is
    `word:<expression>:<reading>` — so changing the reading changes the id and
    therefore the file. Not a rewrite in place, which is what a changed *accent*
    gets."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    # きょう: three kana, so four pattern positions.
    renamed = [record(id="word:橋:きょう", reading="きょう", pitch_accent=["LHHH"])]

    second, _, _ = run(renamed, tmp_path, words=True, book=book)

    assert second.records[0].audio != first.records[0].audio


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_prune_removes_only_what_nothing_references(tmp_path: Path) -> None:
    result, _, _ = run([record()], tmp_path, words=True)
    audio_dir = tmp_path / "media" / "audio"
    (audio_dir / "janki-orphan.wav").write_bytes(b"stale")

    removed = prune_unreferenced(result.records, tmp_path / "media")

    assert [p.name for p in removed] == ["janki-orphan.wav"]
    assert (audio_dir / Path(result.records[0].audio).name).exists()


def test_prune_leaves_files_janki_did_not_write(tmp_path: Path) -> None:
    """A clip somebody dropped in by hand is theirs. A sweep that deleted it
    would be janki removing a file it never created."""
    result, _, _ = run([record()], tmp_path, words=True)
    mine = tmp_path / "media" / "audio" / "my-own-recording.wav"
    mine.write_bytes(b"hand made")

    prune_unreferenced(result.records, tmp_path / "media")

    assert mine.exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_neither_kind_asked_for_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AudioError) as caught:
        run([record()], tmp_path, words=False, examples=False)

    assert "--words" in str(caught.value)


def test_an_unknown_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        run([record()], tmp_path, words=True, ids=["word:nope:nope"])


def test_a_provider_that_returns_nothing_stops_the_run_rather_than_unwinding(
    tmp_path: Path,
) -> None:
    """Zero bytes is not a clip. But it is a *provider* failure on one record,
    not a caller error — folding it into AudioError sent an empty response from
    record 40 of 300 unwinding past the code that saves the 39 already written,
    leaving files with no reference and no ledger entry for --prune to delete."""
    result, _, _ = run(
        [record(), record(id="word:箸:はし", expression="箸")],
        tmp_path, words=True, provider=FakeVoice(audio=b""),
    )

    assert "returned no audio" in result.stopped_by
    assert result.file_count == 0


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'media_dir = "media"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_the_command_refuses_before_spending_when_the_engine_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that voices forty clips and then fails on the forty-first has
    written forty files and half a ledger — and "the engine is not running" is
    the ordinary reason."""
    root = project(tmp_path, [record()])
    down = FakeVoice(reachable=False)
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: down)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert "start the fake engine" in capsys.readouterr().err
    assert down.said == []


def test_the_command_writes_records_media_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: FakeVoice())

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert stored[0]["audio"].startswith("audio/janki-")
    assert (root / "media" / stored[0]["audio"]).is_file()
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entry = book["records"]["word:橋:はし"]["audio"][0]
    assert entry["of"] == "word" and entry["provider"] == "fakevox" and entry["voice"] == 7
    assert "Wrote 1 clip(s)" in capsys.readouterr().out


def test_azure_is_refused_by_name_rather_than_falling_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Falling back to VOICEVOX would record Azure's name in the ledger against
    VOICEVOX's audio."""
    root = project(tmp_path, [record()])

    assert cli.main(["--root", str(root), "audio", "--words", "--provider", "azure"]) == 1

    assert "M5.7" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# What the review found
# ---------------------------------------------------------------------------


def test_a_dropped_reference_is_regenerated_not_called_current(tmp_path: Path) -> None:
    """The ledger alone cannot answer "is this current?". A record whose audio
    field was cleared — a reverted vocabulary.json, a migrate-inline that
    rewrote examples — still matches by content fingerprint, so the run would
    report it up to date while the field stayed empty. Then --prune, which reads
    the records, deletes the clip nothing appears to want."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    dropped = [record()]  # same content, audio field empty

    second, provider, _ = run(dropped, tmp_path, words=True, book=book)

    assert provider.said == [("ハシ'", True)], "re-voiced rather than skipped"
    assert second.records[0].audio.startswith("audio/janki-")
    assert second.up_to_date == 0


def test_a_missing_file_is_regenerated_too(tmp_path: Path) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    (tmp_path / "media" / first.records[0].audio).unlink()

    second, provider, _ = run(first.records, tmp_path, words=True, book=book)

    assert provider.said == [("ハシ'", True)]
    assert second.up_to_date == 0


def test_pruning_takes_the_ledger_entries_with_it(tmp_path: Path) -> None:
    """Or the ledger outlives the media it describes, and goes on answering
    "already recorded" for a file that is gone."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    orphaned = [record()]  # reference dropped, so the clip is unreferenced

    removed = prune_unreferenced(orphaned, tmp_path / "media", book)

    assert len(removed) == 1
    assert not book.records["word:橋:はし"].get("audio")


def test_an_edited_sentence_supersedes_its_old_entry(tmp_path: Path) -> None:
    """Example clips are addressed by record id + sentence, so an edit makes a
    new file and a new entry while the old one lingers — reported stale forever,
    and on a revert it answers "already recorded" while the record points at the
    other sentence's clip: the card shows one sentence and plays another."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run(
        [record(examples=[ExampleSentence(japanese="橋を渡ります。")])],
        tmp_path, words=False, examples=True, book=book,
    )
    edited = [record(examples=[ExampleSentence(japanese="橋を渡りました。")])]

    run(edited, tmp_path, words=False, examples=True, book=book)

    entries = book.records["word:橋:はし"]["audio"]
    assert len(entries) == 1, "the superseded entry is gone"
    assert book.stale_audio(edited) == [], "and nothing is reported stale"


def test_a_record_with_no_reading_is_reported_rather_than_dropped(
    tmp_path: Path,
) -> None:
    result, provider, _ = run([record(reading="")], tmp_path, words=True)

    assert provider.said == []
    assert result.no_reading == ["word:橋:はし"]


def test_a_provider_failure_keeps_what_was_already_written(tmp_path: Path) -> None:
    """300 records and a failure at 299 must not throw away 298 clips: they are
    on disk, their entries are in hand, and the names are deterministic so a
    re-run skips exactly what landed."""
    from japanese_anki.errors import JankiError

    class DiesOnSecond(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if self.said:
                raise JankiError("engine went away")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    records = [record(), record(id="word:箸:はし", expression="箸")]

    result, _, _ = run(records, tmp_path, words=True, book=book, provider=DiesOnSecond())

    assert result.file_count == 1, "the first record's clip survived"
    assert result.records[0].audio.startswith("audio/")
    assert "engine went away" in result.stopped_by
    assert book.records["word:橋:はし"]["audio"], "and its ledger entry"


def test_bare_janki_audio_refuses_rather_than_voicing_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--words or not --examples` made the refusal unreachable: a bare
    invocation silently voiced every record in the collection."""
    root = project(tmp_path, [record()])
    voice = FakeVoice()
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: voice)

    assert cli.main(["--root", str(root), "audio"]) == 1

    assert "--words" in capsys.readouterr().err
    assert voice.said == []


def test_a_superseded_entry_survives_while_its_clip_is_still_referenced(
    tmp_path: Path,
) -> None:
    """The mismatch has to stay visible until it is actually fixed. Editing a
    sentence whose replacement is flagged unverified writes nothing — dropping
    the old entry first would leave the record naming clip A with sentence B and
    no evidence anywhere that the card shows one and plays the other."""
    from japanese_anki.identifiers import short_fingerprint

    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run(
        [record(examples=[ExampleSentence(japanese="橋を渡ります。")])],
        tmp_path, words=False, examples=True, book=book,
    )
    edited_text = "橋を渡りました。"
    edited = [
        record(
            examples=[
                replace(first.records[0].examples[0], japanese=edited_text)
            ],
            source=SourceReference(
                type="jpdb",
                imported_from="deck",
                raw_fields={"furigana_unverified": short_fingerprint(edited_text)},
            ),
        )
    ]

    result, provider, _ = run(edited, tmp_path, words=False, examples=True, book=book)

    assert provider.said == [], "flagged, so nothing was written"
    assert book.records["word:橋:はし"]["audio"], "and the entry stayed"
    assert book.stale_audio(result.records) == ["word:橋:はし"], "still reported"


def test_clips_written_before_a_failure_stay_referenced(tmp_path: Path) -> None:
    """Or the record does not name them, --prune deletes them as unreferenced,
    and the run says "the clips written before this are saved" while it does."""
    from japanese_anki.errors import JankiError

    class DiesOnSecond(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if self.said:
                raise JankiError("engine went away")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    three = [
        record(
            examples=[
                ExampleSentence(japanese="橋を渡ります。"),
                ExampleSentence(japanese="橋が長いです。"),
                ExampleSentence(japanese="橋の上です。"),
            ]
        )
    ]

    result, _, _ = run(
        three, tmp_path, words=False, examples=True, book=book, provider=DiesOnSecond()
    )

    assert result.file_count == 1
    assert result.records[0].examples[0].audio.startswith("audio/"), "named by the record"
    assert prune_unreferenced(result.records, tmp_path / "media", book) == []


def test_the_command_prunes_the_saved_ledger_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durability half: `prune_unreferenced` taking a Ledger object proves
    nothing about `janki audio --prune` writing that ledger back to disk. Drop
    the book argument at the call site, or the save after it, and ledger.json
    keeps entries for files that are gone."""
    root = project(tmp_path, [record()])
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: FakeVoice())
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    # A clip nothing will ever regenerate: an example entry whose sentence the
    # record does not have. Dropping a *word* reference would only re-voice it
    # under the same address, so it would never be orphaned.
    orphan = root / "media" / "audio" / "janki-orphan.wav"
    orphan.write_bytes(b"stale")
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    book["records"]["word:橋:はし"]["audio"].append(
        {
            "at": "2026-08-08", "file": "janki-orphan.wav", "of": "example",
            "provider": "fakevox", "voice": 7, "content_fp": "deadbeef",
        }
    )
    (root / "ledger.json").write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 0

    assert not orphan.exists(), "the file went"
    saved = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    files = [a["file"] for a in saved["records"]["word:橋:はし"].get("audio", [])]
    assert "janki-orphan.wav" not in files, "and its entry left the saved ledger"


def test_a_stopped_run_is_reported_even_when_the_ledger_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ledger warning on its own reads as a successful partial run, and the
    records the run never reached would be invisible."""
    from japanese_anki.errors import JankiError

    class Dies(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            raise JankiError("engine went away")

    root = project(tmp_path, [record()])
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: Dies())
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("read-only")),
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    err = capsys.readouterr().err
    assert "engine went away" in err
    assert "read-only" in err


def test_a_ledger_whose_audio_is_not_a_list_does_not_traceback(tmp_path: Path) -> None:
    """A hand-edited ledger gave a TypeError out of --prune instead of the clean
    LedgerError this module promises. Guarding readers one at a time missed the
    writer, so the shape is refused where entries are handed out."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    book.records["word:橋:はし"] = {"audio": None}

    assert book.forget_audio_files({"janki-abc.wav"}) == 0
    with pytest.raises(ledger_mod.LedgerError) as caught:
        book.record_audio(
            "word:橋:はし", file="x.wav", of="word", provider="p", voice=1,
            content_fp="fp",
        )
    assert "not a list" in str(caught.value)


def test_an_entry_the_ledger_cannot_read_is_kept_rather_than_erased(
    tmp_path: Path,
) -> None:
    """Writing the recognised subset back deletes what these methods cannot
    read — from a committed file that `status --rebuild` cannot reconstruct,
    with no error and no mention in any summary."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    book.records["word:箸:はし"] = {"audio": ["janki-legacy.wav"]}

    assert book.forget_audio_files({"janki-abc.wav"}) == 0
    assert book.records["word:箸:はし"]["audio"] == ["janki-legacy.wav"]

    assert book.drop_superseded_audio("word:箸:はし", of="example", keep=set()) == 0
    assert book.records["word:箸:はし"]["audio"] == ["janki-legacy.wav"]
