"""``janki audio`` — and mostly, the things it declines to say.

No engine: the provider is a fake that records what it was asked to speak and
whether the accent was forced. That distinction is the point of the whole
command, and it is invisible in the bytes that come back, so asserting on the
returned WAV would pass just as well against a version that never forced
anything.
"""

from __future__ import annotations

import json
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
    suffix = ".wav"
    settings: dict[str, str] = {}

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
    kwargs.setdefault("sentence_provider", None)
    result = generate_audio(
        records, provider=provider, book=book, media_dir=tmp_path / "media", **kwargs
    )
    return result, provider, book


# ---------------------------------------------------------------------------
# Words: the accent is forced when known, and marked when guessed
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


def test_a_record_with_no_pattern_is_voiced_with_the_engines_own_accent(
    tmp_path: Path,
) -> None:
    """The owner's decision (2026-08-15): a clip carrying an unverified accent
    beats no clip. It has to be a *marked* one — a listener cannot tell it from
    a forced clip, and the ledger is where later work finds out."""
    result, provider, book = run([record(pitch_accent=[])], tmp_path, words=True)

    # Read naturally, not forced — there is no accent to force.
    assert provider.said == [("はし", False)]
    entry = book.records["word:橋:はし"]["audio"][0]
    assert entry[audio_cmd.ACCENT_UNVERIFIED] is True
    assert result.guessed_accent == ["word:橋:はし"]
    assert result.file_count == 1


def test_a_forced_clip_carries_no_unverified_mark(tmp_path: Path) -> None:
    """The other direction, and the one that matters for the mark's meaning: a
    tag written unconditionally would make every clip look guessed and the
    upgrade below unfindable."""
    result, provider, book = run([record(pitch_accent=["LHL"])], tmp_path, words=True)

    assert provider.said == [("ハシ'", True)]
    assert audio_cmd.ACCENT_UNVERIFIED not in book.records["word:橋:はし"]["audio"][0]
    assert result.guessed_accent == []


def test_a_guessed_accent_is_reported_on_every_run(tmp_path: Path) -> None:
    """Not only on the run that wrote the clip.

    A guessed accent is a standing property of the record until the dictionary
    supplies a pattern, so the report has to survive the currency check that
    skips re-synthesizing. Appended after that early return — where it first
    was — this fired once and then went quiet while the clips went on carrying
    a guess, which is silence about the one thing the fallback trades away.
    """
    first, _, book = run([record(pitch_accent=[])], tmp_path, words=True)
    stored = f"audio/{first.written['word:橋:はし'][0]}"
    assert first.guessed_accent == ["word:橋:はし"]

    again, provider, _ = run(
        [record(pitch_accent=[], audio=stored)], tmp_path, words=True, book=book
    )

    assert provider.said == [], "nothing was re-voiced"
    assert again.up_to_date == 1
    assert again.guessed_accent == ["word:橋:はし"], "still carrying a guess"


def test_a_mismatched_pattern_warns_on_every_run(tmp_path: Path) -> None:
    """Same rule for the bad-data half. The clip settles, the fault does not:
    a warning that fires once leaves a record whose stored pattern contradicts
    its reading looking resolved forever."""
    first, _, book = run([record(pitch_accent=["LH"])], tmp_path, words=True)
    stored = f"audio/{first.written['word:橋:はし'][0]}"

    again, provider, _ = run(
        [record(pitch_accent=["LH"], audio=stored)], tmp_path, words=True, book=book
    )

    assert provider.said == []
    assert any("does not fit" in w for w in again.warnings)


def test_a_guessed_clip_is_replaced_once_the_pattern_arrives(tmp_path: Path) -> None:
    """What makes the fallback safe rather than a silent lock-in. Word audio is
    fingerprinted over reading + pattern, so a clip voiced without an accent
    reads as stale the day `enrich --jpdb` fills one — no --force needed. If
    that ever stopped holding, the records voiced by guess today would keep
    their guess forever.

    The record has to carry its audio reference forward, and the control half
    is why: `_is_current` short-circuits on an empty `audio` field, so a
    re-run built from a bare record re-voices whatever the fingerprint says and
    proves nothing. Measured — the first draft of this test passed with the
    pattern dropped from the fingerprint entirely.
    """
    guessed, _, book = run([record(pitch_accent=[])], tmp_path, words=True)
    stored = f"audio/{guessed.written['word:橋:はし'][0]}"

    # Control: same content, reference carried, nothing to do.
    same, provider, book = run(
        [record(pitch_accent=[], audio=stored)], tmp_path, words=True, book=book
    )
    assert provider.said == [], "an unchanged guess is not re-voiced"
    assert same.up_to_date == 1

    # Now the dictionary answers, and the same clip is no longer current.
    result, provider, book = run(
        [record(pitch_accent=["LHL"], audio=stored)], tmp_path, words=True, book=book
    )

    assert provider.said == [("ハシ\'", True)], "re-voiced, and forced this time"
    assert result.up_to_date == 0, "the guessed clip did not answer for it"
    assert result.guessed_accent == []


def test_a_pattern_that_does_not_fit_its_reading_falls_back_and_warns(
    tmp_path: Path,
) -> None:
    """`to_aquestalk` refuses a mismatched pattern rather than inventing an
    alignment, so this is the no-pattern case wearing different clothes and it
    takes the same fallback — but the warning still names the bad data, because
    voicing the card must not hide the fault."""
    result, provider, _ = run([record(pitch_accent=["LH"])], tmp_path, words=True)

    assert provider.said == [("はし", False)], "the engine's own, not forced"
    assert result.guessed_accent == ["word:橋:はし"]
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


def test_the_command_tells_the_shell_which_records_carry_a_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through the CLI, because `AudioResult.guessed_accent` proves nothing
    about anyone being told. Deleting the whole report block left every test
    green — the list was asserted only in-process, and this message is the one
    surface a person ever sees for a clip that sounds right and is not."""
    root = project(tmp_path, [record(pitch_accent=[])])
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: FakeVoice())

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    err = capsys.readouterr().err
    assert "accent_unverified" in err
    assert "word:橋:はし" in err


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
            "word:橋:はし", file="x.wav", of="word", provider="p", voice=1, speed=1.0,
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


# ---------------------------------------------------------------------------
# A separate voice for sentences
# ---------------------------------------------------------------------------


def test_examples_take_the_sentence_provider_and_words_do_not(tmp_path: Path) -> None:
    """Two providers rather than one provider with two voices: everything
    downstream — which voice the ledger records, which voice `_is_current`
    compares against — already asks *a provider*, and one that answered
    differently per utterance would make both wrong."""
    words = FakeVoice(voice=13)
    sentences = FakeVoice(voice=52)
    record_with_example = record(
        examples=[ExampleSentence(japanese="橋を渡る。", english="Cross the bridge.")]
    )
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    generate_audio(
        [record_with_example], provider=words, sentence_provider=sentences,
        book=book, media_dir=tmp_path / "media", words=True, examples=True,
    )

    assert words.said == [("ハシ'", True)], "the word, accent forced"
    assert sentences.said == [("橋を渡る。", False)], "the sentence, read naturally"
    voices = {e["of"]: e["voice"] for e in book.records[record().id]["audio"]}
    assert voices == {"word": 13, "example": 52}, "and the ledger says which is which"


def test_changing_only_the_sentence_voice_revoices_only_the_sentences(
    tmp_path: Path,
) -> None:
    """The currency check reads the provider that made each kind, so a new
    sentence voice must not re-synthesize every word as well."""
    example = [ExampleSentence(japanese="橋を渡る。", english="Cross the bridge.")]
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run(
        [record(examples=example)], tmp_path, words=True, examples=True, book=book,
        provider=FakeVoice(voice=13), sentence_provider=FakeVoice(voice=52),
    )

    words = FakeVoice(voice=13)
    sentences = FakeVoice(voice=99)
    second, _, _ = run(
        first.records, tmp_path, words=True, examples=True, book=book,
        provider=words, sentence_provider=sentences,
    )

    assert words.said == [], "the word was left alone"
    assert len(sentences.said) == 1, "only the sentence was re-voiced"
    assert second.up_to_date == 1


def test_one_voice_by_default(tmp_path: Path) -> None:
    """An unset sentence voice must not change what anything records — the
    ledger entries have to keep meaning exactly what they meant before."""
    words = FakeVoice(voice=13)
    example = [ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    generate_audio(
        [record(examples=example)], provider=words, book=book,
        media_dir=tmp_path / "media", words=True, examples=True,
    )

    assert len(words.said) == 2, "one engine spoke both"
    assert {e["voice"] for e in book.records[record().id]["audio"]} == {13}


def test_the_configured_sentence_voice_reaches_its_own_provider(tmp_path: Path) -> None:
    (tmp_path / "janki.toml").write_text(
        "[tts]\nvoicevox_speaker = 13\nvoicevox_sentence_speaker = 52\n"
        "voicevox_speed = 0.7\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)

    assert (words.voice, sentences.voice) == (13, 52)
    assert sentences.speed == 0.7, "and takes the same rate"


def test_an_unset_sentence_voice_returns_the_word_provider(tmp_path: Path) -> None:
    (tmp_path / "janki.toml").write_text(
        "[tts]\nvoicevox_speaker = 13\n", encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, None)

    assert cli._sentence_provider(config, None, words) is words, "the same object"


class FakeMp3Voice(FakeVoice):
    """A second engine whose audio is not WAV — the whole point of `suffix`."""

    name = "fakemp3"
    suffix = ".mp3"


def test_the_file_is_named_for_the_format_the_provider_returns(tmp_path: Path) -> None:
    """A hard-coded `.wav` would name an mp3 file `.wav` and hand Anki a lie
    about its contents. Every earlier filename assertion used a `.wav` fake, so
    reverting the suffix threading left the whole suite green."""
    example = [ExampleSentence(japanese="橋を渡る。", english="Cross the bridge.")]
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    result = generate_audio(
        [record(examples=example)],
        provider=FakeVoice(voice=13),
        sentence_provider=FakeMp3Voice(voice=52),
        book=book, media_dir=tmp_path / "media", words=True, examples=True,
    )

    word, sentence = result.records[0].audio, result.records[0].examples[0].audio
    assert word.endswith(".wav") and sentence.endswith(".mp3")
    assert not word.endswith(".mp3") and not sentence.endswith(".wav")
    assert (tmp_path / "media" / word).is_file()
    assert (tmp_path / "media" / sentence).is_file(), "on disk under the same name"
    files = {e["of"]: e["file"] for e in book.records[record().id]["audio"]}
    assert files["word"].endswith(".wav") and files["example"].endswith(".mp3")


class FakeStyledVoice(FakeVoice):
    """An engine whose delivery is set by something other than voice and speed."""

    def __init__(self, *, style: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.style = style

    name = "fakestyled"

    @property
    def settings(self) -> dict[str, str]:
        return {"instructions": self.style}


def test_changed_engine_settings_make_a_clip_stale(tmp_path: Path) -> None:
    """`instructions` is the only pace control the OpenAI API has, and it is in
    neither the content fingerprint nor the voice. Unrecorded, rewriting it to
    ask for a slower delivery left every clip "already current" and nothing was
    re-voiced — the setting that exists to change the audio changed nothing."""
    example = [ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run(
        [record(examples=example)], tmp_path, words=False, examples=True, book=book,
        provider=FakeStyledVoice(style="normally"),
    )

    engine = FakeStyledVoice(style="very slowly")
    second, _, _ = run(
        first.records, tmp_path, words=False, examples=True, book=book, provider=engine,
    )

    assert len(engine.said) == 1, "the new instructions re-voiced it"
    assert second.up_to_date == 0
    entry = book.records[record().id]["audio"][0]
    assert entry["settings"] == {"instructions": "very slowly"}


def test_an_engine_with_nothing_to_say_records_no_settings(tmp_path: Path) -> None:
    """So a VOICEVOX entry keeps exactly the shape it has always had, and no
    committed ledger moves."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    run([record()], tmp_path, words=True, book=book)

    assert "settings" not in book.records[record().id]["audio"][0]


def test_the_cli_gives_sentences_their_own_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI is the only entry point a user has, and nothing drove it: every
    other test called `generate_audio` or `_sentence_provider` directly, so
    deleting the wiring left the feature a no-op with the suite green."""
    root = project(tmp_path, [record(
        examples=[ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    )])
    words, sentences = FakeVoice(voice=13), FakeVoice(voice=52)
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: words)
    monkeypatch.setattr(cli, "_sentence_provider", lambda config, chosen, w: sentences)

    assert cli.main(["--root", str(root), "audio", "--words", "--examples"]) == 0

    book = ledger_mod.load(root / "ledger.json")
    voices = {e["of"]: e["voice"] for e in book.records[record().id]["audio"]}
    assert voices == {"word": 13, "example": 52}, "saved to disk, not just in memory"


def test_the_word_suffix_follows_the_word_provider_too(tmp_path: Path) -> None:
    """The mirror of the test above. With a `.wav` word fake, hard-coding
    `.wav` on the word path leaves that test green — the assertion is satisfied
    by the constant rather than by the provider."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    result = generate_audio(
        [record()], provider=FakeMp3Voice(voice=13), book=book,
        media_dir=tmp_path / "media", words=True,
    )

    assert result.records[0].audio.endswith(".mp3")
    assert book.records[record().id]["audio"][0]["file"].endswith(".mp3")


def test_a_sentence_voice_of_zero_is_a_real_voice(tmp_path: Path) -> None:
    """0 is 四国めたん・あまあま, and the audition page prints it as a copyable
    value. Treating it as "unset" silently discards a valid configuration —
    the exact failure `_int` exists to prevent, and the asymmetry is worse:
    `voicevox_speaker = 0` worked while `voicevox_sentence_speaker = 0` did
    not."""
    (tmp_path / "janki.toml").write_text(
        "[tts]\nvoicevox_speaker = 13\nvoicevox_sentence_speaker = 0\n", encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)

    assert sentences is not words, "0 selected a different speaker"
    assert sentences.voice == 0


def test_an_absent_sentence_speaker_is_the_only_way_to_opt_out(tmp_path: Path) -> None:
    (tmp_path / "janki.toml").write_text("[tts]\nvoicevox_speaker = 13\n", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, None)

    assert config.voicevox_sentence_speaker is None
    assert cli._sentence_provider(config, None, words) is words


def test_a_words_run_ignores_an_unavailable_sentence_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--words` run never reaches the sentence provider, so refusing to
    start because *that* engine has no key aborts a job it plays no part in —
    and the message blames the wrong thing, since nothing is unreachable."""
    root = project(tmp_path, [record()])
    down = FakeVoice(reachable=False)
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: FakeVoice())
    monkeypatch.setattr(cli, "_sentence_provider", lambda config, chosen, w: down)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0


def test_an_examples_run_still_refuses_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half: when the sentence engine *is* needed, an unusable one
    stops the run before it writes a single clip."""
    root = project(tmp_path, [record(
        examples=[ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    )])
    monkeypatch.setattr(cli, "_speech_provider", lambda config, chosen: FakeVoice())
    monkeypatch.setattr(
        cli, "_sentence_provider", lambda config, chosen, w: FakeVoice(reachable=False)
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1

    assert "fakevox" in capsys.readouterr().err


@pytest.mark.parametrize("written", ["Voicevox", "VOICEVOX", " voicevox ", ""])
def test_the_provider_name_is_normalised_once(tmp_path: Path, written: str) -> None:
    """Two normalisations disagreed: `_speech_provider` lower-cased and
    defaulted while `_sentence_provider` compared the raw string, so
    `provider = "Voicevox"` built a VOICEVOX word engine and then silently
    discarded the configured sentence voice — every sentence in the word voice,
    with no message."""
    (tmp_path / "janki.toml").write_text(
        f'[tts]\nprovider = "{written}"\nvoicevox_speaker = 13\n'
        "voicevox_sentence_speaker = 52\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, None)
    sentences = cli._sentence_provider(config, None, words)

    assert (words.voice, sentences.voice) == (13, 52)


def test_a_provider_flag_overriding_the_file_still_selects_the_sentence_voice(
    tmp_path: Path,
) -> None:
    """`--provider voicevox` over an `azure` file is a valid invocation, and
    `_sentence_provider` read the file rather than the flag."""
    (tmp_path / "janki.toml").write_text(
        '[tts]\nprovider = "azure"\nvoicevox_speaker = 13\n'
        "voicevox_sentence_speaker = 52\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    words = cli._speech_provider(config, "voicevox")

    assert cli._sentence_provider(config, "voicevox", words).voice == 52


def test_the_word_rate_does_not_reach_the_openai_sentence_provider(tmp_path: Path) -> None:
    """`voicevox_speed` is a VOICEVOX knob and that API has no rate parameter, so
    passing it through would tie every OpenAI clip's staleness to the *word*
    pace: tuning that would re-synthesize and re-bill every sentence in the
    collection, rewrite each mp3 in place under its content-addressed name, and
    leave nothing visible but a changed number in the ledger."""
    (tmp_path / "janki.toml").write_text(
        '[tts]\nsentence_provider = "openai"\nvoicevox_speed = 0.7\n', encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    sentences = cli._sentence_provider(config, None, object())

    assert sentences.name == "openai"
    assert sentences.speed == 1.0, "the word engine's rate stayed out of it"




def test_correcting_an_unusable_pattern_revoices_the_clip(tmp_path: Path) -> None:
    """The trap the fallback created, and the reason the fingerprint describes
    the utterance rather than the stored pattern.

    A curator typing an accent in a Japanese IME gets fullwidth ＬＨＬ.
    `to_aquestalk` refuses it, so the word takes the guess path. They then
    retype it in ASCII — which is exactly what `janki build`'s
    `invalid-pitch-accent` error asks for. Fingerprinted over the raw pattern
    the two collide under NFKC, so the clip stayed current, no warning fired
    (the pattern parses now), and the ledger said `accent_unverified` forever.
    """
    guessed, provider, book = run(
        [record(pitch_accent=[], audio_accent="ＬＨＬ")], tmp_path, words=True
    )
    stored = f"audio/{guessed.written['word:橋:はし'][0]}"
    assert provider.said == [("はし", False)], "refused, so voiced as a guess"

    fixed, provider, book = run(
        [record(pitch_accent=[], audio_accent="LHL", audio=stored)],
        tmp_path, words=True, book=book,
    )

    assert provider.said == [("ハシ'", True)], "the correction is re-voiced, forced"
    assert fixed.up_to_date == 0


def test_the_upgrade_clears_the_unverified_mark_from_the_ledger(tmp_path: Path) -> None:
    """The durable half. A clip re-voiced as forced must stop being *recorded*
    as a guess, or every later reader — and any future report — still believes
    it is one.

    Measured: merging a new audio reference over the old entry rather than
    replacing it keeps the stale tag through a forced re-voice, and the whole
    suite stayed green.
    """
    guessed, _, book = run([record(pitch_accent=[])], tmp_path, words=True)
    stored = f"audio/{guessed.written['word:橋:はし'][0]}"
    entry = book.records["word:橋:はし"]["audio"][0]
    assert entry[audio_cmd.ACCENT_UNVERIFIED] is True, "the premise"

    run(
        [record(pitch_accent=["LHL"], audio=stored)], tmp_path, words=True, book=book
    )

    [entry] = [a for a in book.records["word:橋:はし"]["audio"] if a["of"] == "word"]
    assert audio_cmd.ACCENT_UNVERIFIED not in entry, "the mark is gone, not merged"
