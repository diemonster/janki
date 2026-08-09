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
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

WAV = b"RIFF....WAVEfake"


class FakeVoice:
    """Stands in for a speech engine, recording what it was told to say."""

    def __init__(self, *, reachable: bool = True, audio: bytes = WAV) -> None:
        self.said: list[tuple[str, bool]] = []
        self.reachable = reachable
        self.audio = audio

    name = "fakevox"
    voice = 7
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
# Doing it twice
# ---------------------------------------------------------------------------


def test_a_second_run_writes_nothing_new(tmp_path: Path) -> None:
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)

    second, provider, _ = run(first.records, tmp_path, words=True, book=book)

    assert provider.said == []
    assert second.file_count == 0
    assert second.up_to_date == 1


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


def test_a_changed_reading_needs_new_audio(tmp_path: Path) -> None:
    """Staleness falls out of content addressing rather than being tracked:
    the clip for the old reading is simply not the clip this record needs."""
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    first, _, _ = run([record()], tmp_path, words=True, book=book)
    moved = [record(id="word:橋:はし", reading="はし", pitch_accent=["HLL"])]

    second, provider, _ = run(moved, tmp_path, words=True, book=book)

    assert provider.said == [("ハ'シ", True)], "the new accent is spoken"
    assert second.file_count == 1


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


def test_a_provider_that_returns_nothing_is_refused(tmp_path: Path) -> None:
    # Zero bytes is not a clip; writing it would put a silent file on a card.
    with pytest.raises(AudioError):
        run([record()], tmp_path, words=True, provider=FakeVoice(audio=b""))


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
