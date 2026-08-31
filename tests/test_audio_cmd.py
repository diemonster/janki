"""``janki audio`` — and mostly, the things it declines to say.

No engine: the provider is a fake that records what it was asked to speak and
whether the accent was forced. That distinction is the point of the whole
command, and it is invisible in the bytes that come back, so asserting on the
returned WAV would pass just as well against a version that never forced
anything.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import audio_cmd, cli
from japanese_anki import ledger as ledger_mod
from japanese_anki.application import audio as audio_application
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
    fingerprinted over the exact bare-reading or forced-AquesTalk utterance, so
    a clip voiced without an accent reads as stale the day `enrich --jpdb`
    fills one — no --force needed. If that ever stopped holding, the records
    voiced by guess today would keep their guess forever.

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
    """`resolve_word_provider` is the join between janki.toml and the engine, and
    both halves of it are silent when wrong: a dropped speed gives audio at the
    wrong rate with no error, and — since the ledger now records what the
    provider reports — a ledger that agrees with the audio and with nothing the
    user asked for."""
    (tmp_path / "janki.toml").write_text(
        '[tts]\nvoicevox_speaker = 13\nvoicevox_speed = 0.7\n', encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    provider = audio_application.resolve_word_provider(config, None)

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


def test_a_new_engine_is_not_current_even_when_its_profile_otherwise_matches(
    tmp_path: Path,
) -> None:
    """Provider is durable metadata for the same reason voice and rate are.

    Two engines can name a voice alike. If engine is omitted from the shared
    render-profile comparison, status and audio can disagree about whether that
    configured synthesis choice changed.
    """

    class OtherEngine(FakeVoice):
        name = "other-engine"

    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)

    second, provider, _ = run(
        first.records,
        tmp_path,
        words=True,
        book=book,
        provider=OtherEngine(),
    )

    assert provider.said == [("ハシ'", True)]
    assert second.up_to_date == 0
    assert book.records[record().id]["audio"][0]["provider"] == "other-engine"


def test_an_old_engine_entry_does_not_leave_the_replacement_stale(
    tmp_path: Path,
) -> None:
    """Switching sentence engines can change the stable file's extension.

    The old entry can remain beside the replacement because both describe the
    same sentence content. Currency belongs to the file the record now names;
    otherwise status reports this record stale forever while the next audio run
    correctly leaves the replacement alone.
    """

    class Mp3Engine(FakeVoice):
        name = "mp3-engine"
        suffix = ".mp3"

    source = record(examples=[ExampleSentence(japanese="橋を渡ります。")])
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([source], tmp_path, words=False, examples=True, book=book)
    switched, provider, _ = run(
        first.records,
        tmp_path,
        words=False,
        examples=True,
        book=book,
        provider=Mp3Engine(),
    )

    entries = book.records[source.id]["audio"]
    assert {Path(entry["file"]).suffix for entry in entries} == {".wav", ".mp3"}
    assert book.stale_audio(
        switched.records,
        word_provider=provider,
        example_provider=provider,
    ) == []

    again, provider, _ = run(
        switched.records,
        tmp_path,
        words=False,
        examples=True,
        book=book,
        provider=Mp3Engine(),
    )
    assert provider.said == []
    assert again.up_to_date == 1


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


def test_a_disk_failure_salvages_earlier_paid_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = 0

    def fail_the_second_write(path: Path, data: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    monkeypatch.setattr(
        audio_cmd,
        "atomic_write_bytes",
        fail_the_second_write,
        raising=False,
    )
    provider = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    item = record(
        examples=[
            ExampleSentence(japanese="一。"),
            ExampleSentence(japanese="二。"),
        ]
    )

    result = generate_audio(
        [item],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )

    assert provider.said == [("一。", False), ("二。", False)]
    assert "disk full" in result.stopped_by
    assert result.file_count == 1
    assert result.records[0].examples[0].audio
    assert result.records[0].examples[1].audio == ""
    assert len(book.records[item.id]["audio"]) == 1


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
    """Static engine unavailability is knowable before the first paid call."""
    root = project(tmp_path, [record()])
    down = FakeVoice(reachable=False)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: down)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert "start the fake engine" in capsys.readouterr().err
    assert down.said == []


def test_the_command_writes_records_media_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert stored[0]["audio"].startswith("audio/janki-")
    assert (root / "media" / stored[0]["audio"]).is_file()
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entry = book["records"]["word:橋:はし"]["audio"][0]
    assert entry["of"] == "word" and entry["provider"] == "fakevox" and entry["voice"] == 7
    assert "Wrote 1 clip(s)" in capsys.readouterr().out


def test_a_concurrent_record_edit_does_not_make_the_rerun_pay_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The revision guard wins without orphaning a paid clip.

    Speech calls are slow enough that a human edit during synthesis is ordinary.
    The first command must refuse to overwrite it, durably record what it paid
    for, and let the next command attach that exact clip without another call.
    """
    root = project(tmp_path, [record()])

    class EditsDuringSynthesis(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.said:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["usage_notes"] = "Concurrent human edit."
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = EditsDuringSynthesis()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1
    assert len(provider.said) == 1
    assert "changed on disk" in capsys.readouterr().err

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    assert len(provider.said) == 1, "the durable first clip was adopted"
    saved = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert saved[0]["usage_notes"] == "Concurrent human edit."
    assert saved[0]["audio"].startswith("audio/janki-")


def test_same_address_cas_failure_keeps_old_media_and_canonical_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-voice is prepared beside the live clip until its record CAS wins."""
    root = project(tmp_path, [record()])
    old = FakeVoice(audio=b"old voice", voice=7)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: old)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    [before_record] = json.loads(
        (root / "vocabulary.json").read_text(encoding="utf-8")
    )
    target = root / "media" / before_record["audio"]
    before_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))

    class EditsDuringRevoice(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            payload = json.loads(
                (root / "vocabulary.json").read_text(encoding="utf-8")
            )
            payload[0]["usage_notes"] = "Concurrent note-only edit."
            (root / "vocabulary.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    replacement = EditsDuringRevoice(audio=b"new voice", voice=13)
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda config, chosen: replacement,
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert target.read_bytes() == b"old voice"
    failed_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert failed_ledger["records"] == before_ledger["records"]
    assert all(
        "record_ref_pending" not in entry
        for entry in failed_ledger["records"][record().id]["audio"]
    )
    [pending] = failed_ledger["pending_audio"].values()
    assert pending["target"] == target.name
    assert pending["request"] == {"forced_accent": True, "input": "ハシ'"}
    assert pending["profile"]["voice"] == 13
    staged = root / "media" / "audio" / pending["staged_file"]
    assert staged.is_file() and staged.read_bytes() == b"new voice"

    real_save_records = audio_application.save_records_json_locked
    guarded_writes = 0

    def count_guarded_write(*args: object, **kwargs: object) -> None:
        nonlocal guarded_writes
        guarded_writes += 1
        real_save_records(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        audio_application, "save_records_json_locked", count_guarded_write
    )
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    assert len(replacement.said) == 1, "the exact pending request was adopted"
    assert guarded_writes == 1, "adoption still CAS-checks an unchanged reference"
    assert target.read_bytes() == b"new voice"
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert saved["usage_notes"] == "Concurrent note-only edit."
    final_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert final_ledger["records"][record().id]["audio"][0]["voice"] == 13
    assert "pending_audio" not in final_ledger


def test_new_address_cas_failure_does_not_publish_or_forget_the_old_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = record(examples=[ExampleSentence(japanese="橋を渡る。")])
    root = project(tmp_path, [original])
    old = FakeVoice(audio=b"old sentence", voice=52)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: old)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: old,
    )
    assert cli.main(["--root", str(root), "audio", "--examples"]) == 0

    [payload] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    old_name = Path(payload["examples"][0]["audio"]).name
    old_path = root / "media" / "audio" / old_name
    payload["examples"][0]["japanese"] = "毎日話す。"
    (root / "vocabulary.json").write_text(
        json.dumps([payload], ensure_ascii=False), encoding="utf-8"
    )
    before_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))

    class EditsDuringSynthesis(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            current = json.loads(
                (root / "vocabulary.json").read_text(encoding="utf-8")
            )
            current[0]["usage_notes"] = "Keep this concurrent edit."
            (root / "vocabulary.json").write_text(
                json.dumps(current, ensure_ascii=False), encoding="utf-8"
            )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    replacement = EditsDuringSynthesis(audio=b"new sentence", voice=53)
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda config, chosen: replacement,
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: replacement,
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1

    failed_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert failed_ledger["records"] == before_ledger["records"]
    [pending] = failed_ledger["pending_audio"].values()
    new_path = root / "media" / "audio" / pending["target"]
    assert old_path.read_bytes() == b"old sentence"
    assert not new_path.exists(), "the new address is not published before CAS"

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 0

    assert len(replacement.said) == 1
    assert new_path.read_bytes() == b"new sentence"
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert saved["usage_notes"] == "Keep this concurrent edit."
    assert Path(saved["examples"][0]["audio"]).name == new_path.name
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert [entry["file"] for entry in final["records"][record().id]["audio"]] == [
        new_path.name
    ]


def test_failed_final_ledger_commit_recovers_promoted_audio_without_rebilling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    provider = FakeVoice(audio=b"paid once", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    real_save = cli.ledger.Ledger.save
    saves = 0

    def fail_the_canonical_commit(book: ledger_mod.Ledger) -> None:
        nonlocal saves
        saves += 1
        if saves == 2:
            raise ledger_mod.LedgerError("interrupted final ledger commit")
        real_save(book)

    monkeypatch.setattr(cli.ledger.Ledger, "save", fail_the_canonical_commit)

    outcome = audio_application.execute_corpus_audio(
        ProjectConfig.load(root), words=True
    )

    assert outcome.state == "ledger-incomplete"
    assert outcome.record_references_written is True
    assert outcome.media_published is True
    assert outcome.ledger_committed is False
    assert outcome.pending_recovery is True

    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    target = root / "media" / saved["audio"]
    assert target.read_bytes() == b"paid once", "record CAS won before promotion"
    interrupted = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert interrupted["pending_audio"]
    assert not interrupted.get("records", {}).get(record().id, {}).get("audio")

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    assert len(provider.said) == 1
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final
    assert final["records"][record().id]["audio"][0]["voice"] == 53


def test_shared_executor_reports_provider_partial_state_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from japanese_anki.errors import JankiError

    class DiesOnSecond(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if self.said:
                raise JankiError("engine went away")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    root = project(
        tmp_path,
        [record(), record(id="word:箸:はし", expression="箸")],
    )
    provider = DiesOnSecond(audio=b"first paid clip", voice=53)
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda config, chosen: provider,
    )

    outcome = audio_application.execute_corpus_audio(
        ProjectConfig.load(root), words=True
    )

    assert outcome.state == "generation-stopped"
    assert outcome.file_count == 1
    assert "engine went away" in (outcome.stopped_by or "")
    assert outcome.record_references_written is True
    assert outcome.media_published is True
    assert outcome.ledger_committed is True
    assert outcome.pending_recovery is False


@pytest.mark.parametrize(
    ("failure", "recovery"),
    [("provider", False), ("wal", True)],
)
def test_shared_executor_does_not_claim_an_unfinished_phase_landed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    recovery: bool,
) -> None:
    from japanese_anki.errors import JankiError

    class FailsImmediately(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            raise JankiError("engine went away")

    root = project(tmp_path, [record()])
    provider = FailsImmediately() if failure == "provider" else FakeVoice()
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda config, chosen: provider,
    )
    if failure == "wal":
        monkeypatch.setattr(
            audio_application,
            "_persist_audio_wal",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ledger_mod.LedgerError("ledger became read-only")
            ),
        )

    outcome = audio_application.execute_corpus_audio(
        ProjectConfig.load(root), words=True
    )

    assert outcome.state == "generation-stopped"
    assert outcome.file_count == 0
    assert outcome.record_references_written is False
    assert outcome.media_published is False
    assert outcome.ledger_committed is False
    assert outcome.pending_recovery is recovery


def test_concurrent_ledger_change_merges_the_additive_wal_without_losing_paid_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])

    class EditsLedgerDuringSynthesis(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            concurrent = ledger_mod.load(root / "ledger.json")
            concurrent.record_export(record().id, "concurrent-deck")
            concurrent.save()
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = EditsLedgerDuringSynthesis(audio=b"one paid render", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    assert len(provider.said) == 1
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert (root / "media" / saved["audio"]).read_bytes() == b"one paid render"
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entry = final["records"][record().id]
    assert "concurrent-deck" in entry["exports"]
    assert entry["audio"][0]["voice"] == 53
    assert "pending_audio" not in final


def test_each_completed_clip_is_durable_before_the_next_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record(
        examples=[
            ExampleSentence(japanese="一つ目。"),
            ExampleSentence(japanese="二つ目。"),
        ]
    )
    root = project(tmp_path, [item])

    class InterruptsSecondClip(FakeVoice):
        interrupt = True

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            self.said.append((text_or_kana, forced_accent))
            if self.interrupt and text_or_kana == "二つ目。":
                raise KeyboardInterrupt
            return self.audio

    provider = InterruptsSecondClip(audio=b"paid sentence", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application, "resolve_sentence_provider", lambda config, chosen, words: provider
    )

    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(root), "audio", "--examples"])

    interrupted = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [pending] = interrupted["pending_audio"].values()
    stage = root / "media" / "audio" / pending["staged_file"]
    assert stage.read_bytes() == b"paid sentence"

    provider.interrupt = False
    assert cli.main(["--root", str(root), "audio", "--examples"]) == 0

    utterances = [utterance for utterance, _ in provider.said]
    assert utterances.count("一つ目。") == 1
    assert utterances.count("二つ目。") == 2


def test_audio_operation_lock_is_held_before_records_and_ledger_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    events: list[str] = []

    class Guard:
        def __enter__(self) -> None:
            events.append("locked")

        def __exit__(self, *exc: object) -> None:
            events.append("unlocked")

    monkeypatch.setattr(
        audio_application, "exclusive_path_lock", lambda path: Guard()
    )

    def inspect_snapshots(*args: object, **kwargs: object):
        assert events == ["locked"]
        events.append("snapshots")
        return audio_application.AudioExecutionOutcome(
            state="no-records",
            plan=None,
            output_dir=root / "media" / "audio",
            no_records=True,
        )

    monkeypatch.setattr(audio_application, "_execute_audio_locked", inspect_snapshots)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0
    assert events == ["locked", "snapshots", "unlocked"]


@pytest.mark.parametrize("kind", ["word", "example"])
def test_force_rerun_adopts_its_exact_paid_stage_instead_of_rebilling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    item = record(
        examples=[ExampleSentence(japanese="橋を渡る。")]
        if kind == "example"
        else []
    )
    root = project(tmp_path, [item])
    flag = "--words" if kind == "word" else "--examples"
    original = FakeVoice(audio=b"old current bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: original)
    monkeypatch.setattr(
        audio_application, "resolve_sentence_provider", lambda config, chosen, words: original
    )
    assert cli.main(["--root", str(root), "audio", flag]) == 0

    class EditsOnce(FakeVoice):
        edit = True

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if self.edit:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["usage_notes"] = "Concurrent note."
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = EditsOnce(audio=b"forced paid bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application, "resolve_sentence_provider", lambda config, chosen, words: provider
    )
    assert cli.main(["--root", str(root), "audio", flag, "--force"]) == 1

    provider.edit = False
    # Status and build deliberately prescribe the ordinary matching command,
    # not a repetition of the implementation detail that created the WAL.
    assert cli.main(["--root", str(root), "audio", flag]) == 0

    assert len(provider.said) == 1
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    relative = saved["audio"] if kind == "word" else saved["examples"][0]["audio"]
    assert (root / "media" / relative).read_bytes() == b"forced paid bytes"
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final


def test_stage_is_self_describing_if_wal_persistence_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider result is recoverable at every boundary after staging."""
    root = project(tmp_path, [record()])
    provider = FakeVoice(audio=b"paid exactly once", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    real_persist = audio_application._persist_audio_wal

    def interrupt_after_stage(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(audio_application, "_persist_audio_wal", interrupt_after_stage)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(root), "audio", "--words"])

    interrupted = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert not interrupted.get("pending_audio")
    stages = list((root / "media" / "audio" / ".pending").glob("*.stage"))
    assert len(stages) == 1 and stages[0].read_bytes() == b"paid exactly once"

    monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)
    provider.reachable = False
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0

    assert len(provider.said) == 1
    assert not list((root / "media" / "audio" / ".pending").glob("*.stage"))


@pytest.mark.parametrize("retirement", ["changed", "deleted"])
def test_unregistered_stage_is_retired_when_its_request_can_no_longer_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retirement: str,
) -> None:
    item = record()
    root = project(tmp_path, [item])
    provider = FakeVoice(audio=b"old completed render", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    real_persist = audio_application._persist_audio_wal
    monkeypatch.setattr(
        audio_application,
        "_persist_audio_wal",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(root), "audio", "--words"])
    [old_stage] = list((root / "media" / "audio" / ".pending").glob("*.stage"))

    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    if retirement == "changed":
        payload[0]["pitch_accent"] = ["HLL"]
    else:
        payload = []
    (root / "vocabulary.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)
    command = ["--root", str(root), "audio", "--words"]
    if retirement == "deleted":
        command.append("--prune")

    assert cli.main(command) == 0

    assert not old_stage.exists()


def test_targeted_run_preserves_another_current_requests_unregistered_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = record()
    second = record(id="word:箸:はし", expression="箸", pitch_accent=["HLL"])
    root = project(tmp_path, [first, second])
    provider = FakeVoice(audio=b"paid render", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    real_persist = audio_application._persist_audio_wal
    monkeypatch.setattr(
        audio_application,
        "_persist_audio_wal",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(root), "audio", first.id, "--words"])
    [first_stage] = list(
        (root / "media" / "audio" / ".pending").glob("*.stage")
    )

    monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)
    assert cli.main(["--root", str(root), "audio", second.id, "--words"]) == 0

    assert first_stage.is_file()


def test_corrupt_completed_wal_refuses_before_rebilling_exact_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    item = record()
    root = project(tmp_path, [item])
    provider = FakeVoice(audio=b"replacement would cost", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    utterance, forced, _ = ledger_mod.word_audio_request(item)
    arguments = {
        "of": "word",
        "target": f"janki-{ledger_mod.word_audio_filename_fingerprint(item)}.wav",
        "request_input": utterance,
        "forced_accent": forced,
        "content_fp": ledger_mod.word_audio_content_fingerprint(item),
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    book = ledger_mod.load(root / "ledger.json")
    key = book.pending_audio_key_for(item.id, **arguments)
    claimed_sha = "a" * 64
    stage = (
        root
        / "media"
        / "audio"
        / ".pending"
        / f"{key}-{claimed_sha}.stage"
    )
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"corrupt bytes")
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=f".pending/{key}-{claimed_sha}.stage",
        staged_sha256=claimed_sha,
    )
    book.save()

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert provider.said == []
    assert "refusing to bill" in capsys.readouterr().err
    assert key in json.loads((root / "ledger.json").read_text(encoding="utf-8"))[
        "pending_audio"
    ]

    assert cli.main(["--root", str(root), "audio", "--words", "--force"]) == 0
    assert len(provider.said) == 1
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert (root / "media" / saved["audio"]).read_bytes() == b"replacement would cost"
    assert "pending_audio" not in json.loads(
        (root / "ledger.json").read_text(encoding="utf-8")
    )
    assert not stage.exists()


def test_record_lock_stays_held_from_cas_through_publish_and_ledger_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    provider = FakeVoice(audio=b"new canonical bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    real_promote = audio_cmd.promote_pending_audio
    writer_done = threading.Event()
    writer_failures: list[BaseException] = []
    writer: threading.Thread | None = None

    def promote_while_another_writer_waits(*args: object, **kwargs: object):
        nonlocal writer
        revision = cli.records_revision(root / "vocabulary.json")
        current = cli.load_records(root / "vocabulary.json")
        edited = [replace(current[0], usage_notes="Concurrent human edit.")]

        def write_edit() -> None:
            try:
                cli.save_records_json(
                    root / "vocabulary.json", edited, expected=revision
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                writer_failures.append(exc)
            finally:
                writer_done.set()

        writer = threading.Thread(target=write_edit)
        writer.start()
        assert not writer_done.wait(timeout=0.1), (
            "a cooperating record writer entered between CAS and media promotion"
        )
        return real_promote(*args, **kwargs)

    monkeypatch.setattr(audio_cmd, "promote_pending_audio", promote_while_another_writer_waits)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0
    assert writer is not None
    writer.join(timeout=2)
    assert writer_done.is_set() and not writer_failures
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert saved["usage_notes"] == "Concurrent human edit."
    assert (root / "media" / saved["audio"]).read_bytes() == b"new canonical bytes"


def test_deck_source_owner_revision_is_rechecked_before_media_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = record()
    root = project(tmp_path, [selected])
    deck = root / "data" / "decks" / "custom.yaml"
    source = root / "data" / "decks" / "custom.json"
    deck.parent.mkdir(parents=True)
    source.write_text("[]\n", encoding="utf-8")
    deck.write_text(
        json.dumps(
            {
                "deck": {"name": "Custom", "source": "custom.json"},
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    target = (
        root
        / "media"
        / "audio"
        / f"janki-{ledger_mod.word_audio_filename_fingerprint(selected)}.wav"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"other owner old bytes")

    class AddsSourceOwner(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            other = record(
                id="word:箸:はし",
                expression="箸",
                audio=f"audio/{target.name}",
            )
            source.write_text(
                json.dumps([other.to_dict()], ensure_ascii=False), encoding="utf-8"
            )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = AddsSourceOwner(audio=b"selected new bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    outcome = audio_application.execute_corpus_audio(
        ProjectConfig.load(root), words=True
    )

    assert outcome.state == "finalization-stopped"
    assert outcome.record_references_written is True
    assert outcome.media_published is False
    assert outcome.ledger_committed is False
    assert outcome.pending_recovery is True

    assert target.read_bytes() == b"other owner old bytes"
    assert json.loads(source.read_text(encoding="utf-8"))[0]["audio"].endswith(
        target.name
    )


def test_pending_audio_owner_participates_in_address_collision_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = record()
    second = record(id="word:箸:はし", expression="箸", pitch_accent=["HL"])
    root = project(tmp_path, [first, second])
    provider = FakeVoice(audio=b"must not be called", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    target = (
        f"janki-{ledger_mod.word_audio_filename_fingerprint(second)}.wav"
    )
    utterance, forced, _ = ledger_mod.word_audio_request(first)
    arguments = {
        "of": "word",
        "target": target,
        "request_input": utterance,
        "forced_accent": forced,
        "content_fp": ledger_mod.word_audio_content_fingerprint(first),
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    book = ledger_mod.load(root / "ledger.json")
    key = book.pending_audio_key_for(first.id, **arguments)
    paid = b"first owner's paid stage"
    staged_sha = hashlib.sha256(paid).hexdigest()
    stage = root / "media" / "audio" / ".pending" / f"{key}-{staged_sha}.stage"
    stage.parent.mkdir(parents=True)
    stage.write_bytes(paid)
    book.record_pending_audio(
        first.id,
        **arguments,
        staged_file=f".pending/{key}-{staged_sha}.stage",
        staged_sha256=staged_sha,
    )
    book.save()

    assert cli.main(["--root", str(root), "audio", second.id, "--words"]) == 1

    assert provider.said == []
    assert stage.read_bytes() == paid
    after = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert key in after["pending_audio"]


def test_prune_persists_ledger_forget_before_deleting_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    orphan = root / "media" / "audio" / "janki-orphan.wav"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"canonical until ledger forget commits")
    book = ledger_mod.load(root / "ledger.json")
    book.record_audio(
        record().id,
        file=orphan.name,
        of="word",
        provider="fakevox",
        voice=7,
        speed=1.0,
        content_fp=ledger_mod.word_audio_content_fingerprint(record()),
    )
    book.save()
    real_save = ledger_mod.Ledger.save
    saves = 0

    def fail_forget(current: ledger_mod.Ledger) -> None:
        nonlocal saves
        saves += 1
        if saves == 2:
            raise ledger_mod.LedgerError("interrupted prune ledger commit")
        real_save(current)

    monkeypatch.setattr(ledger_mod.Ledger, "save", fail_forget)

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 1

    assert orphan.read_bytes() == b"canonical until ledger forget commits"
    on_disk = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert on_disk["records"][record().id]["audio"][0]["file"] == orphan.name


def test_changed_sentence_retires_nonmatching_pending_wal_and_unblocks_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record(examples=[ExampleSentence(japanese="古い文。")])
    root = project(tmp_path, [item])

    class ChangesSentence(FakeVoice):
        changed = False

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.changed:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["examples"][0]["japanese"] = "新しい文。"
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                self.changed = True
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = ChangesSentence(audio=b"sentence", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application, "resolve_sentence_provider", lambda config, chosen, words: provider
    )
    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1
    pending_before = json.loads(
        (root / "ledger.json").read_text(encoding="utf-8")
    )["pending_audio"]
    [old_pending] = pending_before.values()
    old_stage = root / "media" / "audio" / old_pending["staged_file"]
    assert old_stage.is_file()

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 0

    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final
    assert not old_stage.exists()
    monkeypatch.setattr(cli, "_build_one", lambda *args, **kwargs: False)
    deck = root / "data" / "decks" / "vocabulary.yaml"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text("deck:\n  name: Vocabulary\nnotes: []\n", encoding="utf-8")
    assert cli.main(["--root", str(root), "build", str(deck)]) == 0


def test_changed_word_request_supersedes_same_slot_pending_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record(pitch_accent=["LHL"])])

    class ChangesPitchOnce(FakeVoice):
        changed = False

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.changed:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["pitch_accent"] = ["HLL"]
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                self.changed = True
            self.audio = text_or_kana.encode("utf-8")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = ChangesPitchOnce(voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 1
    interrupted = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [old_pending] = interrupted["pending_audio"].values()
    old_stage = root / "media" / "audio" / old_pending["staged_file"]
    assert old_stage.is_file()

    assert cli.main(["--root", str(root), "audio", "--words", "--force"]) == 0

    assert len(provider.said) == 2
    assert provider.said[0][0] != provider.said[1][0]
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final
    assert not old_stage.exists()


def test_changed_spoken_japanese_supersedes_same_slot_pending_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = "雨、まだ止まないの？"
    spoken = "雨、まだやまないの？"
    root = project(
        tmp_path,
        [record(examples=[ExampleSentence(japanese=display)])],
    )

    class ChangesSpokenInputOnce(FakeVoice):
        changed = False

        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.changed:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["examples"][0]["spoken_japanese"] = spoken
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                self.changed = True
            self.audio = text_or_kana.encode("utf-8")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = ChangesSpokenInputOnce(voice=53)
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: provider
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: provider,
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1
    interrupted = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [old_pending] = interrupted["pending_audio"].values()
    old_stage = root / "media" / "audio" / old_pending["staged_file"]
    assert old_stage.is_file()

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 0

    assert provider.said == [(display, False), (spoken, False)]
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final
    assert not old_stage.exists()


def test_force_retry_adopts_valid_sibling_of_corrupt_same_key_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record()
    root = project(tmp_path, [item])
    provider = FakeVoice(audio=b"one valid replacement", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    utterance, forced, _ = ledger_mod.word_audio_request(item)
    arguments = {
        "of": "word",
        "target": f"janki-{ledger_mod.word_audio_filename_fingerprint(item)}.wav",
        "request_input": utterance,
        "forced_accent": forced,
        "content_fp": ledger_mod.word_audio_content_fingerprint(item),
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    book = ledger_mod.load(root / "ledger.json")
    key = book.pending_audio_key_for(item.id, **arguments)
    corrupt_sha = "a" * 64
    corrupt_stage = (
        root / "media" / "audio" / ".pending" / f"{key}-{corrupt_sha}.stage"
    )
    corrupt_stage.parent.mkdir(parents=True)
    corrupt_stage.write_bytes(b"corrupt")
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=f".pending/{key}-{corrupt_sha}.stage",
        staged_sha256=corrupt_sha,
    )
    book.save()
    real_persist = audio_application._persist_audio_wal
    monkeypatch.setattr(
        audio_application,
        "_persist_audio_wal",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(root), "audio", "--words", "--force"])
    assert len(provider.said) == 1
    assert len(list(corrupt_stage.parent.glob(f"{key}-*.stage"))) == 2

    monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)
    assert cli.main(["--root", str(root), "audio", "--words", "--force"]) == 0

    assert len(provider.said) == 1
    assert not list(corrupt_stage.parent.glob(f"{key}-*.stage"))
    assert "pending_audio" not in json.loads(
        (root / "ledger.json").read_text(encoding="utf-8")
    )


def test_prune_retires_pending_audio_for_a_deleted_record_and_unblocks_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])

    class DeletesRecord(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            (root / "vocabulary.json").write_text("[]\n", encoding="utf-8")
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = DeletesRecord(audio=b"orphaned paid bytes")
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 1
    pending = json.loads((root / "ledger.json").read_text(encoding="utf-8"))[
        "pending_audio"
    ]
    [entry] = pending.values()
    stage = root / "media" / "audio" / entry["staged_file"]
    assert stage.is_file()

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 0

    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert "pending_audio" not in final
    assert not stage.exists()
    monkeypatch.setattr(cli, "_build_one", lambda *args, **kwargs: False)
    deck = root / "data" / "decks" / "vocabulary.yaml"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text("deck:\n  name: Vocabulary\nnotes: []\n", encoding="utf-8")
    assert cli.main(["--root", str(root), "build", str(deck)]) == 0


def test_build_holds_audio_operation_lock_through_output_and_ledger_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    deck = root / "deck.yaml"
    deck.write_text("deck:\n  name: Vocabulary\nnotes: []\n", encoding="utf-8")
    acquired = threading.Event()
    worker: threading.Thread | None = None

    def build_while_audio_waits(*args: object, **kwargs: object) -> bool:
        nonlocal worker

        def take_audio_lock() -> None:
            with cli.exclusive_path_lock(root / ".janki-audio-operation"):
                acquired.set()

        worker = threading.Thread(target=take_audio_lock)
        worker.start()
        assert not acquired.wait(timeout=0.1), (
            "an audio transaction entered after build's pending gate"
        )
        return True

    monkeypatch.setattr(cli, "_build_one", build_while_audio_waits)

    assert cli.main(["--root", str(root), "build", str(deck)]) == 0
    assert worker is not None
    worker.join(timeout=2)
    assert acquired.is_set()


def test_prune_keeps_a_paid_clip_with_a_pending_record_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = record()
    second = record(id="word:箸:はし", expression="箸", pitch_accent=["HL"])
    root = project(tmp_path, [first, second])

    class EditsTheFirstRun(FakeVoice):
        def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
            if not self.said:
                payload = json.loads(
                    (root / "vocabulary.json").read_text(encoding="utf-8")
                )
                payload[0]["usage_notes"] = "Concurrent human edit."
                (root / "vocabulary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            return super().synthesize(text_or_kana, forced_accent=forced_accent)

    provider = EditsTheFirstRun()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", first.id, "--words"]) == 1
    pending = root / "media" / "audio" / (
        f"janki-{ledger_mod.word_audio_filename_fingerprint(first)}.wav"
    )
    assert not pending.exists(), "a failed record CAS does not publish the target"
    pending_book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [pending_entry] = pending_book["pending_audio"].values()
    staged = root / "media" / "audio" / pending_entry["staged_file"]
    assert staged.is_file()

    assert (
        cli.main(
            ["--root", str(root), "audio", second.id, "--words", "--prune"]
        )
        == 0
    )
    assert staged.is_file(), "another target's prune must not erase paid recovery"

    assert cli.main(["--root", str(root), "audio", first.id, "--words"]) == 0
    assert len(provider.said) == 2, "first and second were each synthesized once"
    saved = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in saved}
    assert by_id[first.id]["audio"].endswith(pending.name)
    assert pending.is_file()


def test_stale_prune_snapshot_refuses_before_deleting_a_newly_committed_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    stale_records = [record()]
    stale_book = ledger_mod.load(root / "ledger.json")
    provider = FakeVoice(audio=b"new canonical clip")
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    target = root / "media" / saved["audio"]

    with pytest.raises(ledger_mod.LedgerError, match="changed on disk"):
        prune_unreferenced(stale_records, root / "media", stale_book)

    assert target.read_bytes() == b"new canonical clip"
    canonical = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert canonical["records"][record().id]["audio"][0]["file"] == target.name


@pytest.mark.parametrize(
    "owner", ["normalized", "inline-deck", "new-inline-deck", "deck-source"]
)
def test_prune_revalidates_every_record_owner_before_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
) -> None:
    root = project(tmp_path, [])
    deck = root / "data" / "decks" / "inline.yaml"
    source = root / "data" / "decks" / "custom.json"
    if owner in {"inline-deck", "deck-source"}:
        deck.parent.mkdir(parents=True)
        if owner == "deck-source":
            source.write_text("[]\n", encoding="utf-8")
            deck_payload = {
                "deck": {"name": "Inline", "source": "custom.json"},
                "notes": [],
            }
        else:
            deck_payload = {"deck": {"name": "Inline"}, "notes": []}
        deck.write_text(json.dumps(deck_payload), encoding="utf-8")
    orphan = root / "media" / "audio" / "janki-new.wav"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"newly referenced bytes")
    real_prune = audio_cmd.prune_unreferenced

    def add_reference_then_prune(*args: object, **kwargs: object):
        referring = record(audio="audio/janki-new.wav").to_dict()
        if owner == "normalized":
            (root / "vocabulary.json").write_text(
                json.dumps([referring], ensure_ascii=False), encoding="utf-8"
            )
        elif owner in {"inline-deck", "new-inline-deck"}:
            deck.parent.mkdir(parents=True, exist_ok=True)
            deck.write_text(
                json.dumps(
                    {"deck": {"name": "Inline"}, "notes": [referring]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            source.write_text(
                json.dumps([referring], ensure_ascii=False), encoding="utf-8"
            )
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(audio_cmd, "prune_unreferenced", add_reference_then_prune)

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 1

    assert orphan.read_bytes() == b"newly referenced bytes"


def test_pending_cleanup_never_unlinks_a_canonical_target(tmp_path: Path) -> None:
    audio_dir = tmp_path / "media" / "audio"
    audio_dir.mkdir(parents=True)
    target = audio_dir / "janki-live.wav"
    target.write_bytes(b"keep canonical")

    warnings = audio_cmd.cleanup_pending_stages(
        [{"staged_file": target.name, "target": target.name}], audio_dir
    )

    assert target.read_bytes() == b"keep canonical"
    assert warnings and "unsafe pending stage" in warnings[0]


def test_pending_stage_symlink_cannot_overwrite_canonical_media_before_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [record()])
    original = FakeVoice(audio=b"old canonical", voice=7)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: original)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    canonical = root / "media" / saved["audio"]

    replacement = FakeVoice(audio=b"new paid bytes", voice=13)
    utterance, forced, _ = ledger_mod.word_audio_request(record())
    book = ledger_mod.load(root / "ledger.json")
    arguments = {
        "of": "word",
        "target": canonical.name,
        "request_input": utterance,
        "forced_accent": forced,
        "content_fp": ledger_mod.word_audio_content_fingerprint(record()),
        "provider": replacement.name,
        "voice": replacement.voice,
        "speed": replacement.speed,
        "settings": replacement.settings,
    }
    key = book.pending_audio_key_for(record().id, **arguments)
    staged_sha = hashlib.sha256(replacement.audio).hexdigest()
    staged = (
        root / "media" / "audio" / ".pending" / f"{key}-{staged_sha}.stage"
    )
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.symlink_to(canonical)
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda config, chosen: replacement,
    )

    assert cli.main(["--root", str(root), "audio", "--words", "--force"]) == 1

    assert canonical.read_bytes() == b"old canonical"
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert final["records"][record().id]["audio"][0]["voice"] == 7


def test_canonical_target_symlink_is_refused_during_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record()
    root = project(tmp_path, [item])
    target = (
        root
        / "media"
        / "audio"
        / f"janki-{ledger_mod.word_audio_filename_fingerprint(item)}.wav"
    )
    target.parent.mkdir(parents=True)
    outside = root / "outside.wav"
    outside.write_bytes(b"do not overwrite")
    target.symlink_to(outside)
    provider = FakeVoice(audio=b"new paid bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert provider.said == []
    assert outside.read_bytes() == b"do not overwrite"
    assert target.is_symlink()
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert not final.get("pending_audio")
    assert not final.get("records", {}).get(item.id, {}).get("audio")


def test_matching_canonical_symlink_cannot_bypass_bound_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record()
    root = project(tmp_path, [item])
    provider = FakeVoice(audio=b"same bytes", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    target_name = f"janki-{ledger_mod.word_audio_filename_fingerprint(item)}.wav"
    audio_dir = root / "media" / "audio"
    audio_dir.mkdir(parents=True)
    outside = root / "outside-same.wav"
    outside.write_bytes(provider.audio)
    target = audio_dir / target_name
    target.symlink_to(outside)
    utterance, forced, _ = ledger_mod.word_audio_request(item)
    arguments = {
        "of": "word",
        "target": target_name,
        "request_input": utterance,
        "forced_accent": forced,
        "content_fp": ledger_mod.word_audio_content_fingerprint(item),
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    book = ledger_mod.load(root / "ledger.json")
    key = book.pending_audio_key_for(item.id, **arguments)
    staged_sha = hashlib.sha256(provider.audio).hexdigest()
    staged_name = f".pending/{key}-{staged_sha}.stage"
    stage = audio_dir / staged_name
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_bytes(provider.audio)
    book.record_pending_audio(
        item.id,
        **arguments,
        staged_file=staged_name,
        staged_sha256=staged_sha,
    )
    book.save()

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert provider.said == []
    assert target.is_symlink() and outside.read_bytes() == provider.audio
    final = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert key in final["pending_audio"]


@pytest.mark.parametrize("outside_bytes", [b"old canonical", b"arbitrary outside"])
def test_current_currency_refuses_a_canonical_symlink_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside_bytes: bytes,
) -> None:
    root = project(tmp_path, [record()])
    provider = FakeVoice(audio=b"old canonical", voice=53)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    assert cli.main(["--root", str(root), "audio", "--words"]) == 0
    [saved] = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    target = root / "media" / saved["audio"]
    target.unlink()
    outside = root / "outside-current.wav"
    outside.write_bytes(outside_bytes)
    target.symlink_to(outside)
    provider.said.clear()

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    assert provider.said == []
    assert target.is_symlink() and outside.read_bytes() == outside_bytes


def test_cleanup_refuses_a_cross_stage_symlink_without_unlinking_its_referent(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "media" / "audio"
    pending = audio_dir / ".pending"
    pending.mkdir(parents=True)
    key = "a" * 64
    digest = "b" * 64
    referent = pending / f"{'c' * 64}-{'d' * 64}.stage"
    referent.write_bytes(b"keep other paid stage")
    link = pending / f"{key}-{digest}.stage"
    link.symlink_to(referent)

    warnings = audio_cmd.cleanup_pending_stages(
        [{"staged_file": f".pending/{link.name}"}], audio_dir
    )

    assert warnings and "could not remove" in warnings[0]
    assert referent.read_bytes() == b"keep other paid stage"
    assert link.is_symlink()


def test_azure_is_refused_by_name_rather_than_falling_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Falling back to VOICEVOX would record Azure's name in the ledger against
    VOICEVOX's audio."""
    root = project(tmp_path, [record()])
    (root / "janki.toml").write_text(
        (root / "janki.toml").read_text(encoding="utf-8")
        + '\n[tts]\nprovider = "azure"\n',
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    error = capsys.readouterr().err
    assert "Azure" in error and "VOICEVOX" in error and "openai" in error


def test_azure_is_not_an_audio_provider_choice() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["audio", "--words", "--provider", "azure"])


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

    _, provider, _ = run(edited, tmp_path, words=False, examples=True, book=book)

    entries = book.records["word:橋:はし"]["audio"]
    assert len(entries) == 1, "the superseded entry is gone"
    assert book.stale_audio(
        edited,
        word_provider=provider,
        example_provider=provider,
    ) == [], "and nothing is reported stale"


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
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: voice)

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
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )

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
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )
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


def test_an_unwritable_ledger_refuses_before_a_paid_audio_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A static durability failure is knowable before synthesis, so ask first."""
    root = project(tmp_path, [record()])
    provider = FakeVoice()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("read-only")),
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1

    err = capsys.readouterr().err
    assert "read-only" in err
    assert provider.said == []
    assert not (root / "media").exists()


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

    words = audio_application.resolve_word_provider(config, None)
    sentences = audio_application.resolve_sentence_provider(config, None, words)

    assert (words.voice, sentences.voice) == (13, 52)
    assert sentences.speed == 0.7, "and takes the same rate"


def test_an_unset_sentence_voice_returns_the_word_provider(tmp_path: Path) -> None:
    (tmp_path / "janki.toml").write_text(
        "[tts]\nvoicevox_speaker = 13\n", encoding="utf-8"
    )
    config = ProjectConfig.load(tmp_path)

    words = audio_application.resolve_word_provider(config, None)

    assert (
        audio_application.resolve_sentence_provider(config, None, words) is words
    ), "the same object"


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
    """`instructions` is the pace control janki sends to OpenAI, and it is in
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
    other test called `generate_audio` or `resolve_sentence_provider` directly, so
    deleting the wiring left the feature a no-op with the suite green."""
    root = project(tmp_path, [record(
        examples=[ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    )])
    words, sentences = FakeVoice(voice=13), FakeVoice(voice=52)
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: words)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, w: sentences,
    )

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

    words = audio_application.resolve_word_provider(config, None)
    sentences = audio_application.resolve_sentence_provider(config, None, words)

    assert sentences is not words, "0 selected a different speaker"
    assert sentences.voice == 0


def test_an_absent_sentence_speaker_is_the_only_way_to_opt_out(tmp_path: Path) -> None:
    (tmp_path / "janki.toml").write_text("[tts]\nvoicevox_speaker = 13\n", encoding="utf-8")
    config = ProjectConfig.load(tmp_path)

    words = audio_application.resolve_word_provider(config, None)

    assert config.voicevox_sentence_speaker is None
    assert audio_application.resolve_sentence_provider(config, None, words) is words


def test_a_words_run_ignores_an_unavailable_sentence_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--words` run never reaches the sentence provider, so refusing to
    start because *that* engine has no key aborts a job it plays no part in —
    and the message blames the wrong thing, since nothing is unreachable."""
    root = project(tmp_path, [record()])
    down = FakeVoice(reachable=False)
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, w: down,
    )

    assert cli.main(["--root", str(root), "audio", "--words"]) == 0


def test_an_examples_run_still_refuses_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half: when the sentence engine *is* needed, an unusable one
    stops the run before it writes a single clip."""
    root = project(tmp_path, [record(
        examples=[ExampleSentence(japanese="橋を渡る。", english="Cross.")]
    )])
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, w: FakeVoice(reachable=False),
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1

    assert "fakevox" in capsys.readouterr().err


@pytest.mark.parametrize("written", ["Voicevox", "VOICEVOX", " voicevox ", ""])
def test_the_provider_name_is_normalised_once(tmp_path: Path, written: str) -> None:
    """Two normalisations once disagreed: word resolution lower-cased and
    defaulted while sentence resolution compared the raw string, so
    `provider = "Voicevox"` built a VOICEVOX word engine and then silently
    discarded the configured sentence voice — every sentence in the word voice,
    with no message."""
    (tmp_path / "janki.toml").write_text(
        f'[tts]\nprovider = "{written}"\nvoicevox_speaker = 13\n'
        "voicevox_sentence_speaker = 52\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    words = audio_application.resolve_word_provider(config, None)
    sentences = audio_application.resolve_sentence_provider(config, None, words)

    assert (words.voice, sentences.voice) == (13, 52)


def test_a_provider_flag_overriding_the_file_still_selects_the_sentence_voice(
    tmp_path: Path,
) -> None:
    """An explicit word-provider flag does not discard the configured
    sentence speaker."""
    (tmp_path / "janki.toml").write_text(
        '[tts]\nprovider = "voicevox"\nvoicevox_speaker = 13\n'
        "voicevox_sentence_speaker = 52\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    words = audio_application.resolve_word_provider(config, "voicevox")

    assert audio_application.resolve_sentence_provider(config, "voicevox", words).voice == 52


def test_words_only_ignores_dormant_spoken_japanese(tmp_path: Path) -> None:
    words = FakeVoice(voice=13)

    result = generate_audio(
        [
            record(
                examples=[
                    ExampleSentence(
                        japanese="橋を渡る。", spoken_japanese="はしをわたる。"
                    )
                ]
            )
        ],
        provider=words,
        book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
        media_dir=tmp_path / "media",
        words=True,
        examples=False,
    )

    assert result.file_count == 1
    assert words.said == [("ハシ'", True)]


def test_an_unselected_spoken_override_does_not_block_a_targeted_run(
    tmp_path: Path,
) -> None:
    selected = record(id="word:橋:はし")
    unrelated = record(
        id="word:箸:はし",
        expression="箸",
        examples=[
            ExampleSentence(japanese="箸を使う。", spoken_japanese="はしをつかう。")
        ],
    )
    words = FakeVoice(voice=13)

    result = generate_audio(
        [selected, unrelated],
        provider=words,
        sentence_provider=FakeVoice(voice=52),
        book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
        media_dir=tmp_path / "media",
        words=True,
        examples=True,
        ids=[selected.id],
    )

    assert result.file_count == 1
    assert words.said == [("ハシ'", True)]


def test_conflicting_spoken_japanese_for_one_clip_refuses_before_synthesis(
    tmp_path: Path,
) -> None:
    provider = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    item = record(
        examples=[
            ExampleSentence(
                japanese="雨、まだ止まないの？",
                spoken_japanese="雨、まだやまないの？",
            ),
            ExampleSentence(
                japanese="雨、まだ止まないの？",
                spoken_japanese="雨、まだとまないの？",
            ),
        ]
    )

    with pytest.raises(AudioError, match="same audio file.*different spoken Japanese"):
        generate_audio(
            [item],
            provider=FakeVoice(),
            sentence_provider=provider,
            book=book,
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
        )

    assert provider.said == []
    assert book.records == {}
    assert not (tmp_path / "media").exists()


def test_spoken_japanese_is_the_exact_paid_request_and_wal_input(
    tmp_path: Path,
) -> None:
    display = "雨、まだ止まないの？"
    spoken = "雨、まだやまないの？"
    example = ExampleSentence(japanese=display, spoken_japanese=spoken)
    item = record(examples=[example])
    provider = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    result = generate_audio(
        [item],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
        stage_only=True,
    )

    assert provider.said == [(spoken, False)]
    assert result.records[0].examples[0].japanese == display
    expected_name = (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(item, example)}.wav"
    )
    assert Path(result.records[0].examples[0].audio).name == expected_name
    [pending] = book.pending_audio.values()
    assert pending["request"] == {"forced_accent": False, "input": spoken}
    assert pending["content_fp"] == ledger_mod.example_audio_content_fingerprint(
        example
    )


def test_current_pending_audio_key_binds_the_exact_spoken_japanese(
    tmp_path: Path,
) -> None:
    display = "雨、まだ止まないの？"
    spoken = "雨、まだやまないの？"
    example = ExampleSentence(japanese=display, spoken_japanese=spoken)
    item = record(reading="", examples=[example])
    words = FakeVoice(voice=13)
    sentences = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    target = (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(item, example)}.wav"
    )

    keys = audio_cmd.current_pending_audio_keys(
        [item],
        book=book,
        word_provider=words,
        prepared_examples={item.id: [sentences]},
    )

    assert keys == {
        book.pending_audio_key_for(
            item.id,
            of="example",
            target=target,
            request_input=spoken,
            forced_accent=False,
            content_fp=ledger_mod.example_audio_content_fingerprint(example),
            provider=sentences.name,
            voice=sentences.voice,
            speed=sentences.speed,
            settings=sentences.settings,
        )
    }


def test_two_distinct_example_identities_cannot_resolve_to_one_filename(
    tmp_path: Path,
) -> None:
    """The frozen formula concatenates id + Japanese without framing. These
    two distinct pairs therefore have the same hash input; the safe pre-release
    response is a refusal, not changing every existing filename."""
    first = record(
        id="word:何:な",
        expression="何",
        reading="な",
        examples=[ExampleSentence(japanese="に行く。")],
    )
    second = record(
        id="word:何:なに",
        expression="何",
        reading="なに",
        examples=[ExampleSentence(japanese="行く。")],
    )
    sentences = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [first, second],
            provider=FakeVoice(),
            sentence_provider=sentences,
            book=book,
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
        )

    assert sentences.said == []
    assert book.records == {}
    assert not (tmp_path / "media").exists()


def test_word_and_example_identities_cannot_overwrite_one_voicevox_file(
    tmp_path: Path,
) -> None:
    example_owner = record(
        id="word:何:な",
        expression="何",
        reading="な",
        examples=[ExampleSentence(japanese="に行く。")],
    )
    word_owner = record(
        id="word:何:なに行く。",
        expression="行く",
        reading="いく",
    )
    voice = FakeVoice(voice=52)

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [example_owner, word_owner],
            provider=voice,
            sentence_provider=voice,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=True,
            examples=True,
        )

    assert voice.said == []


def test_selected_example_cannot_overwrite_a_foreign_referenced_word_file(
    tmp_path: Path,
) -> None:
    """A stale or hand-edited reference still owns the bytes it names.

    Looking only at the filenames this run would compute misses that durable
    reference, so an examples-only run could overwrite a word clip that a
    different card still plays.
    """
    selected = record(
        id="word:B:x",
        expression="B",
        reading="x",
        examples=[ExampleSentence(japanese="二。")],
    )
    target = (
        "audio/janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(selected, selected.examples[0])}"
        ".wav"
    )
    protected = record(
        id="word:A:x",
        expression="A",
        reading="x",
        audio=target,
    )
    sentences = FakeVoice(voice=52)

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [protected, selected],
            provider=FakeVoice(voice=13),
            sentence_provider=sentences,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            ids=[selected.id],
        )

    assert sentences.said == []
    assert not (tmp_path / "media").exists()


def test_collision_comparison_protects_case_insensitive_filesystems(
    tmp_path: Path,
) -> None:
    selected = record(
        id="word:B:x",
        expression="B",
        reading="x",
        examples=[ExampleSentence(japanese="二。")],
    )
    target = (
        "audio/janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(selected, selected.examples[0])}"
        ".wav"
    )
    protected = record(id="word:A:x", expression="A", reading="x", audio=target.upper())
    sentences = FakeVoice(voice=52)

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [protected, selected],
            provider=FakeVoice(voice=13),
            sentence_provider=sentences,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
            ids=[selected.id],
        )

    assert sentences.said == []


def test_nfkc_distinct_sentences_that_share_an_address_refuse_before_synthesis(
    tmp_path: Path,
) -> None:
    provider = FakeVoice(voice=52)
    item = record(
        examples=[
            ExampleSentence(japanese="Ａ。"),
            ExampleSentence(japanese="A。"),
        ]
    )

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [item],
            provider=FakeVoice(),
            sentence_provider=provider,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=False,
            examples=True,
        )

    assert provider.said == []


@pytest.mark.parametrize(
    ("protected_reading", "protected_pattern", "selected_reading"),
    [
        ("あ", ["HL"], "あア'"),
        ("カナ", [], "ｶﾅ"),
    ],
)
def test_word_collision_identity_uses_the_exact_provider_request(
    tmp_path: Path,
    protected_reading: str,
    protected_pattern: list[str],
    selected_reading: str,
) -> None:
    """A truncated, normalized content digest is not an ownership identity."""
    selected = record(
        id="word:shared",
        expression="x",
        reading=selected_reading,
        pitch_accent=[],
    )
    target = (
        "audio/janki-"
        f"{ledger_mod.word_audio_filename_fingerprint(selected)}.wav"
    )
    protected = record(
        id=selected.id,
        expression="x",
        reading=protected_reading,
        pitch_accent=protected_pattern,
        audio=target,
    )
    provider = FakeVoice()

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [selected],
            protected_records=[protected],
            provider=provider,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=True,
            examples=False,
        )

    assert provider.said == []


def test_nfkc_word_edit_revoices_even_when_the_filename_is_unchanged(
    tmp_path: Path,
) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    original = record(reading="カナ", pitch_accent=[])
    first, _, _ = run([original], tmp_path, words=True, book=book)
    edited = replace(first.records[0], reading="ｶﾅ")

    second, provider, _ = run([edited], tmp_path, words=True, book=book)

    assert provider.said == [("ｶﾅ", False)]
    assert second.file_count == 1


def test_nfkc_example_edit_revoices_even_when_the_filename_is_unchanged(
    tmp_path: Path,
) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    original = record(examples=[ExampleSentence(japanese="Ａ。")])
    first, _, _ = run(
        [original], tmp_path, words=False, examples=True, book=book
    )
    edited = replace(
        first.records[0],
        examples=[
            replace(first.records[0].examples[0], japanese="A。")
        ],
    )

    second, provider, _ = run(
        [edited], tmp_path, words=False, examples=True, book=book
    )

    assert provider.said == [("A。", False)]
    assert second.file_count == 1


def test_selected_word_cannot_overwrite_a_foreign_referenced_example_file(
    tmp_path: Path,
) -> None:
    selected = record(id="word:B:x", expression="B", reading="x")
    target = (
        "audio/janki-"
        f"{ledger_mod.word_audio_filename_fingerprint(selected)}.wav"
    )
    protected = record(
        id="word:A:x",
        expression="A",
        reading="x",
        examples=[ExampleSentence(japanese="一。", audio=target)],
    )
    words = FakeVoice(voice=13)

    with pytest.raises(AudioError, match="Different audio identities.*same filename"):
        generate_audio(
            [protected, selected],
            provider=words,
            sentence_provider=FakeVoice(),
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=True,
            examples=False,
            ids=[selected.id],
        )

    assert words.said == []
    assert not (tmp_path / "media").exists()


def test_the_command_protects_audio_referenced_only_by_an_inline_deck_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = record(
        id="word:B:x",
        expression="B",
        reading="x",
        examples=[ExampleSentence(japanese="二。")],
    )
    target_name = (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(selected, selected.examples[0])}"
        ".wav"
    )
    inline = record(
        id="word:A:x",
        expression="A",
        reading="x",
        examples=[
            ExampleSentence(japanese="一。", audio=f"audio/{target_name}")
        ],
    )
    root = project(tmp_path, [selected])
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "inline.yaml").write_text(
        json.dumps(
            {"deck": {"name": "Inline"}, "notes": [inline.to_dict()]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    target = root / "media" / "audio" / target_name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"OLD-INLINE-CLIP")
    provider = FakeVoice(voice=52, audio=b"NEW-NORMALIZED-CLIP")
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: provider,
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1

    assert provider.said == []
    assert target.read_bytes() == b"OLD-INLINE-CLIP"
    assert "Different audio identities resolve to the same filename" in capsys.readouterr().err


def test_an_inline_override_with_different_spoken_japanese_cannot_share_a_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = record(
        examples=[
            ExampleSentence(japanese="橋を渡る。", spoken_japanese="はしをわたる。")
        ]
    )
    target_name = (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(selected, selected.examples[0])}"
        ".wav"
    )
    inline = replace(
        selected,
        examples=[
            ExampleSentence(
                japanese="橋を渡る。",
                audio=f"audio/{target_name}",
                spoken_japanese="きょうをわたる。",
            )
        ],
    )
    root = project(tmp_path, [selected])
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "inline.yaml").write_text(
        json.dumps(
            {"deck": {"name": "Inline"}, "notes": [inline.to_dict()]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider = FakeVoice(voice=52)
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: provider,
    )

    assert cli.main(["--root", str(root), "audio", "--examples"]) == 1
    assert provider.said == []


def test_an_inline_word_variant_cannot_share_the_normalized_word_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = record(pitch_accent=["LHL"])
    target_name = (
        f"janki-{ledger_mod.word_audio_filename_fingerprint(selected)}.wav"
    )
    inline = replace(
        selected,
        pitch_accent=["HLL"],
        audio=f"audio/{target_name}",
    )
    root = project(tmp_path, [selected])
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "inline.yaml").write_text(
        json.dumps(
            {"deck": {"name": "Inline"}, "notes": [inline.to_dict()]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider = FakeVoice()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", "--words"]) == 1
    assert provider.said == []


def test_prune_preserves_a_noncolliding_inline_only_audio_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = record()
    inline = record(
        id="word:一:いち",
        expression="一",
        reading="いち",
        audio="audio/janki-inline-only.wav",
    )
    root = project(tmp_path, [selected])
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "inline.yaml").write_text(
        json.dumps(
            {"deck": {"name": "Inline"}, "notes": [inline.to_dict()]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    protected = root / "media" / "audio" / "janki-inline-only.wav"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"INLINE")
    provider = FakeVoice()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 0
    assert protected.read_bytes() == b"INLINE"


def test_prune_uses_the_saved_source_record_after_revoicing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collision census must be pre-write, but the prune census must not.

    A source-backed deck sees the old reference before generation. Keeping that
    snapshot alive through prune makes an edited sentence's old clip survive
    the first run even though the just-saved source no longer references it.
    """
    old_example = ExampleSentence(japanese="古い文。")
    old_record = record(examples=[old_example])
    old_name = (
        "janki-"
        f"{ledger_mod.example_audio_filename_fingerprint(old_record, old_example)}"
        ".wav"
    )
    changed = replace(
        old_record,
        examples=[ExampleSentence(japanese="新しい文。", audio=f"audio/{old_name}")],
    )
    root = project(tmp_path, [changed])
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "source.yaml").write_text(
        json.dumps(
            {
                "deck": {
                    "name": "Source",
                    "source": "../../vocabulary.json",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    old_path = root / "media" / "audio" / old_name
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"OLD")
    book = ledger_mod.Ledger(path=root / "ledger.json")
    book.record_audio(
        changed.id,
        file=old_name,
        of="example",
        provider="fakevox",
        voice=7,
        speed=1.0,
        content_fp=ledger_mod.example_audio_content_fingerprint(old_example),
    )
    book.save()
    provider = FakeVoice(audio=b"NEW")
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: provider,
    )

    assert cli.main(["--root", str(root), "audio", "--examples", "--prune"]) == 0

    assert not old_path.exists()
    saved = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    files = [item["file"] for item in saved["records"][changed.id]["audio"]]
    assert old_name not in files


def test_a_partial_prune_saves_the_ledger_before_reporting_the_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [record()])
    audio_dir = root / "media" / "audio"
    audio_dir.mkdir(parents=True)
    first = audio_dir / "janki-000-first.wav"
    second = audio_dir / "janki-001-second.wav"
    first.write_bytes(b"FIRST")
    second.write_bytes(b"SECOND")
    book = ledger_mod.Ledger(path=root / "ledger.json")
    for record_id, path, fingerprint in (
        ("word:old:first", first, "a" * 12),
        ("word:old:second", second, "b" * 12),
    ):
        book.record_audio(
            record_id,
            file=path.name,
            of="word",
            provider="fakevox",
            voice=7,
            speed=1.0,
            content_fp=fingerprint,
        )
    book.save()
    real_unlink = Path.unlink

    def fail_second(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == second.name:
            raise OSError("disk refuses unlink")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_second)
    monkeypatch.setattr(
        audio_application, "resolve_word_provider", lambda config, chosen: FakeVoice()
    )

    assert cli.main(["--root", str(root), "audio", "--words", "--prune"]) == 1

    assert not first.exists()
    assert second.exists()
    saved = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert saved["records"]["word:old:first"].get("audio", []) == []
    assert saved["records"]["word:old:second"].get("audio", []) == []
    assert "disk refuses unlink" in capsys.readouterr().err


def test_a_dormant_word_address_does_not_block_a_selected_example(
    tmp_path: Path,
) -> None:
    """A word with no reading cannot write its computed address.

    It is not an owner unless it already carries a reference; treating the
    impossible future write as one makes an unrelated valid run refuse.
    """
    selected = record(
        id="word:何:な",
        expression="何",
        reading="な",
        pitch_accent=[],
        examples=[ExampleSentence(japanese="に行く。")],
    )
    dormant = record(
        id="word:何:なに行く。",
        expression="行く",
        reading="",
    )
    voice = FakeVoice(voice=52)

    result = generate_audio(
        [selected, dormant],
        provider=voice,
        sentence_provider=voice,
        book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
        media_dir=tmp_path / "media",
        words=True,
        examples=True,
        ids=[selected.id],
    )

    assert result.file_count == 2
    assert voice.said == [("な", False), ("に行く。", False)]


def test_duplicate_selected_record_ids_refuse_before_any_synthesis(tmp_path: Path) -> None:
    first = record(
        examples=[
            ExampleSentence(japanese="一。"),
            ExampleSentence(japanese="二。"),
        ]
    )
    second = record(examples=[ExampleSentence(japanese="三。")])
    sentences = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(AudioError, match="duplicate record id"):
        generate_audio(
            [first, second],
            provider=FakeVoice(),
            sentence_provider=sentences,
            book=book,
            media_dir=tmp_path / "media",
            words=True,
            examples=True,
        )

    assert sentences.said == []
    assert book.records == {}
    assert not (tmp_path / "media").exists()


def test_duplicate_examples_with_one_spoken_input_share_one_paid_clip(
    tmp_path: Path,
) -> None:
    provider = FakeVoice(voice=52)
    item = record(
        examples=[
            ExampleSentence(japanese="橋を渡る。", spoken_japanese="はしをわたる。"),
            ExampleSentence(japanese="橋を渡る。", spoken_japanese="はしをわたる。"),
        ]
    )

    result = generate_audio(
        [item],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )

    assert provider.said == [("はしをわたる。", False)]
    assert result.file_count == 1
    assert result.records[0].examples[0].audio == result.records[0].examples[1].audio


@pytest.mark.parametrize("blank_position", [0, 1])
def test_duplicate_examples_reuse_a_current_reference_regardless_of_order(
    tmp_path: Path,
    blank_position: int,
) -> None:
    provider = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    original = record(
        examples=[
            ExampleSentence(japanese="橋を渡る。", spoken_japanese="はしをわたる。"),
            ExampleSentence(japanese="橋を渡る。", spoken_japanese="はしをわたる。"),
        ]
    )
    first = generate_audio(
        [original],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )
    current = first.records[0].examples[0].audio
    examples = list(first.records[0].examples)
    examples[blank_position] = replace(examples[blank_position], audio="")
    provider.said.clear()

    second = generate_audio(
        [replace(first.records[0], examples=examples)],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )

    assert provider.said == []
    assert second.file_count == 0
    assert second.up_to_date == 1
    assert [example.audio for example in second.records[0].examples] == [
        current,
        current,
    ]


def test_the_command_persists_a_duplicate_reference_repair_without_a_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeVoice(voice=7)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    original = record(
        examples=[
            ExampleSentence(japanese="橋を渡る。"),
            ExampleSentence(japanese="橋を渡る。"),
        ]
    )
    first = generate_audio(
        [original],
        provider=provider,
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )
    current = first.records[0].examples[0].audio
    broken = replace(
        first.records[0],
        examples=[first.records[0].examples[0], replace(first.records[0].examples[1], audio="")],
    )
    project(tmp_path, [broken])
    book.save()
    provider.said.clear()
    monkeypatch.setattr(audio_application, "resolve_word_provider", lambda config, chosen: provider)
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda config, chosen, words: provider,
    )

    assert cli.main(["--root", str(tmp_path), "audio", "--examples"]) == 0

    [stored] = json.loads((tmp_path / "vocabulary.json").read_text(encoding="utf-8"))
    assert [example.get("audio", "") for example in stored["examples"]] == [
        current,
        current,
    ]
    assert provider.said == []


def test_invalid_utf8_spoken_japanese_refuses_the_whole_run_in_preflight(
    tmp_path: Path,
) -> None:
    from japanese_anki.tts.openai_realtime import OpenAiRealtimeProvider

    requests: list[object] = []

    def encoding_transport(*args: object) -> list[dict[str, object]]:
        requests.append(args)
        return []

    words = FakeVoice()
    item = record(
        examples=[
            ExampleSentence(japanese="一。"),
            ExampleSentence(japanese="二。", spoken_japanese="\ud800"),
        ]
    )
    sentences = OpenAiRealtimeProvider(
        record_id=item.id,
        api_key="not-a-real-key",
        transport=encoding_transport,
    )

    with pytest.raises(AudioError, match="Realtime input.*valid UTF-8"):
        generate_audio(
            [item],
            provider=words,
            sentence_provider=sentences,
            book=ledger_mod.Ledger(path=tmp_path / "ledger.json"),
            media_dir=tmp_path / "media",
            words=True,
            examples=True,
        )

    assert words.said == []
    assert requests == []
    assert not (tmp_path / "media").exists()


def test_editing_one_spoken_japanese_revoices_only_that_clip_in_place(
    tmp_path: Path,
) -> None:
    provider = FakeVoice(voice=52)
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    original = record(
        examples=[
            ExampleSentence(japanese="橋を渡る。"),
            ExampleSentence(japanese="毎日話す。"),
        ]
    )
    first = generate_audio(
        [original],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )
    before = [example.audio for example in first.records[0].examples]
    provider.said.clear()
    edited = record(
        examples=[
            replace(first.records[0].examples[0], spoken_japanese="はしをわたる。"),
            first.records[0].examples[1],
        ]
    )

    second = generate_audio(
        [edited],
        provider=FakeVoice(),
        sentence_provider=provider,
        book=book,
        media_dir=tmp_path / "media",
        words=False,
        examples=True,
    )

    assert provider.said == [("はしをわたる。", False)]
    assert second.up_to_date == 1
    assert [example.audio for example in second.records[0].examples] == before
    entries = {entry["file"]: entry for entry in book.records[original.id]["audio"]}
    assert entries[Path(before[0]).name]["content_fp"] == (
        ledger_mod.example_audio_content_fingerprint(edited.examples[0])
    )


def test_the_word_rate_does_not_reach_the_realtime_sentence_provider(
    tmp_path: Path,
) -> None:
    """`voicevox_speed` is a VOICEVOX knob, not an OpenAI sentence setting.

    Realtime pace is fixed by the reviewed learner prompt. Passing the word
    setting through would tie every sentence's staleness to an unrelated knob
    and re-bill the collection.
    """
    (tmp_path / "janki.toml").write_text(
        '[tts]\nsentence_provider = "openai-realtime"\nvoicevox_speed = 0.7\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)

    sentences = audio_application.resolve_sentence_provider(config, None, object())

    assert sentences.name == "openai-realtime"
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
